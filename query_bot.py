# query_bot.py — strict RAG + streaming
# Handles context retrieval with true async concurrency for classroom scale
# Dynamic chat-history budgeting to avoid max-context overflows

import json
import os
import asyncio
import torch
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from functools import lru_cache
import re
import time
from typing import List, Dict, Optional, Tuple, AsyncIterator
import logging
from collections import deque
import hashlib
from transformers import AutoTokenizer  # NEW: tokenizer for token budgeting

load_dotenv()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
CHUNK_FILE = "chunks.jsonl"  # fallback if ever needed
REGISTRY_DIR = "registry"

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Request queue (classroom scale)
# ---------------------------------------------------------------------
class RequestQueue:
    def __init__(self, max_concurrent=25, max_queue_size=75):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.active_requests = 0
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.not_full = asyncio.Condition(self.lock)

    async def acquire(self, priority=0):
        async with self.lock:
            if self.active_requests >= self.max_concurrent and len(self.queue) >= self.max_queue_size:
                raise Exception("Server at capacity. Please try again later.")
            while self.active_requests >= self.max_concurrent:
                fut = asyncio.Future()
                self.queue.append((priority, time.time(), fut))
                await self.not_full.wait()
                if fut.done():
                    break
            self.active_requests += 1
            logger.info(f"Request acquired. Active: {self.active_requests}/{self.max_concurrent}, Queue: {len(self.queue)}")

    async def release(self):
        async with self.lock:
            self.active_requests -= 1
            if self.queue:
                _, _, fut = self.queue.popleft()
                fut.set_result(True)
            self.not_full.notify()
            logger.info(f"Request released. Active: {self.active_requests}/{self.max_concurrent}")

request_queue = RequestQueue(max_concurrent=25, max_queue_size=75)

# ---------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.eval()
    return model

model = get_embedding_model()

# ---------------------------------------------------------------------
# FAISS load
# ---------------------------------------------------------------------
def load_faiss_index(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            metadata = data['metadata']
            texts = data.get('texts', [])
            stats = data.get('stats', {})
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
        if stats:
            print(f"📊 Document types in index: {stats.get('by_doc_type', {})}")
        return index, metadata, texts
    except FileNotFoundError:
        print(f"⚠️ FAISS index files not found at {index_path}")
        print("Please run: python embed_chunks_faiss.py")
        return None, [], []
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None, [], []

# ---------------------------------------------------------------------
# Vector store abstraction
# ---------------------------------------------------------------------
class FAISSVectorStore:
    def __init__(self, embedding_dim=384):
        self.index = None
        self.metadata = []
        self.texts = []
        self.embedding_dim = embedding_dim

    def initialize_empty(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.texts = []

    def set_index(self, index, metadata, texts):
        self.index = index
        self.metadata = metadata
        self.texts = texts

    def search(self, query_vector, k=2):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    'metadata': self.metadata[idx],
                    'text': self.texts[idx] if idx < len(self.texts) else '',
                    'score': float(score)
                })
        return results

faiss_store = FAISSVectorStore()
try:
    idx, md, tx = load_faiss_index()
    if idx is not None:
        faiss_store.set_index(idx, md, tx)
        print(f"✅ Successfully initialized FAISS with {len(md)} chunks")
    else:
        print("⚠️ No FAISS index found. Initializing empty store (RAG disabled).")
        faiss_store.initialize_empty()
except Exception as e:
    print(f"❌ Error initializing FAISS: {e}")
    faiss_store.initialize_empty()

# ---------------------------------------------------------------------
# Retrieval logic
# ---------------------------------------------------------------------
def classify_question_type(question: str) -> str:
    q = question.lower().strip()
    casual_patterns = [
        r'\b(hi|hello|hey|thanks|thank you|goodbye|bye)\b',
        r'how are you',
        r'what\'s up',
        r'good (morning|afternoon|evening)',
        r'nice to meet',
        r'see you later'
    ]
    academic_patterns = [
        r'\b(explain|describe|define|what is|how does|compare|contrast)\b',
        r'\b(theory|concept|process|mechanism|principle)\b',
        r'\b(perception|sensation|visual|auditory|cognitive|neural)\b',
        r'\b(color|vision|hearing|attention|memory|learning)\b',
        r'difference between',
        r'relationship between',
        r'example of'
    ]
    test_patterns = [
        r'\b(test|quiz|exam|assessment|homework|assignment)\b',
        r'correct answer',
        r'multiple choice',
        r'true or false',
        r'which of the following'
    ]
    for p in test_patterns:
        if re.search(p, q):
            return "test_question"
    for p in casual_patterns:
        if re.search(p, q):
            return "casual"
    for p in academic_patterns:
        if re.search(p, q):
            return "academic"
    if len(q) > 50 and any(w in q for w in ['psychology','brain','mind','behavior','study','research']):
        return "academic"
    return "general"

def should_use_rag(question: str, question_type: str, has_custom_prompt: bool = False) -> bool:
    if question_type in ("casual", "test_question"):
        return False
    if question_type == "academic":
        return True
    if question_type == "general":
        return any(k in question.lower() for k in
                   ['perception','sensation','visual','auditory','attention',
                    'memory','learning','brain','neural','cognitive','psychology'])
    return False

def retrieve_relevant_chunks(query, k=2):
    try:
        vec = model.encode([query])[0]
        results = faiss_store.search(vec, k)
        return results
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []

def enhance_lab_query(query: str) -> str:
    """
    Enhance lab queries with semantic keywords to improve retrieval.
    Based on actual course structure discovered during testing.
    """
    q_lower = query.lower()
    
    # Short Labs enhancement
    if any(term in q_lower for term in ['lab 1', 'lab1', 'first lab', 'short lab 1']):
        # Lab 1 is about Signal Detection Theory
        return query + " signal detection theory SDT thresholds perception decisions"
    
    elif any(term in q_lower for term in ['lab 0', 'lab0', 'short lab 0']):
        # Lab 0 is about Method of Limits
        return query + " method of limits thresholds psychophysics"
    
    elif any(term in q_lower for term in ['lab 3', 'lab3', 'third lab', 'short lab 3']):
        # Lab 3 is about Visual Angle
        return query + " visual angle perception size distance"
    
    # Long Labs enhancement
    elif any(term in q_lower for term in ['ll1', 'long lab 1', 'visual search']):
        return query + " visual search attention targets distractors"
    
    elif any(term in q_lower for term in ['ll2', 'long lab 2']):
        return query + " visual attention selective sustained divided"
    
    elif any(term in q_lower for term in ['ll3', 'long lab 3']):
        return query + " final exam preparation review"
    
    # General lab queries - add context
    elif 'lab' in q_lower or 'assignment' in q_lower:
        return query + " experiment report methods results"
    
    return query

def retrieve_relevant_chunks_enhanced(query, k=2):
    """
    Enhanced retrieval that detects lab/assignment queries and improves them.
    This wraps the existing retrieve_relevant_chunks function.
    """
    q_lower = query.lower()
    
    # Detect if this is a lab/assignment query
    is_lab_query = any(term in q_lower for term in [
        'lab', 'assignment', 'homework', 'll0', 'll1', 'll2', 'll3', 
        'short lab', 'long lab', 'what is', 'tell me about', 'help with'
    ])
    
    if is_lab_query:
        # Enhance the query with semantic keywords
        enhanced_query = enhance_lab_query(query)
        # Increase k to get more relevant chunks
        k = max(k, 6)
        logger.info(f"Lab query detected. Enhanced: '{query}' -> '{enhanced_query}', k={k}")
    else:
        enhanced_query = query
    
    # Use the existing retrieval function
    try:
        vec = model.encode([enhanced_query])[0]
        results = faiss_store.search(vec, k)
        
        # Optional: Re-rank results if it's a lab query to prioritize lab-specific sources
        if is_lab_query and results:
            # Boost scores for chunks from lab-related files
            for result in results:
                source = result.get('metadata', {}).get('source', '').lower()
                # Boost lab-specific files
                if any(indicator in source for indicator in ['lab', 'introduction', 'sdt', 'signal']):
                    result['score'] *= 0.9  # Lower score is better in your system
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []

def get_adaptive_chunks(question: str, question_type: str) -> Tuple[List[Dict], List[float]]:
    """
    Enhanced adaptive chunk retrieval with better lab/assignment support.
    """
    q_lower = question.lower()
    
    # Check if this is a lab/assignment query first
    if any(term in q_lower for term in ['lab', 'assignment', 'homework', 'll0', 'll1', 'll2', 'll3']):
        # Use enhanced retrieval for lab queries
        k = 6  # More chunks for lab queries
        results = retrieve_relevant_chunks_enhanced(question, k=k)
    elif question_type == "academic":
        if any(w in q_lower for w in ['compare','contrast','difference','relationship']):
            k = 4
        elif any(w in q_lower for w in ['explain','describe','how']):
            k = 3
        else:
            k = 2
        results = retrieve_relevant_chunks_enhanced(question, k=k)
    else:
        k = 2
        results = retrieve_relevant_chunks(question, k=k)
    
    chunks = []
    scores = []
    for r in results:
        meta = r.get('metadata', {})
        chunks.append({
            'source': meta.get('source', 'unknown'),
            'chunk_id': meta.get('chunk_id', 0),
            'doc_type': meta.get('doc_type', 'general'),
            'text': r.get('text', '')
        })
        scores.append(r.get('score', 0.0))
    
    # Log what we're retrieving for lab queries (helpful for debugging)
    if any(term in q_lower for term in ['lab', 'assignment']):
        logger.info(f"Retrieved {len(chunks)} chunks for lab query: '{question}'")
        for i, chunk in enumerate(chunks[:3]):  # Log first 3
            logger.info(f"  {i+1}. {chunk['source']} (score: {scores[i]:.4f})")
    
    return chunks, scores

def load_text_for_chunks(chunks):
    if not chunks:
        return []
    return [c.get('text','') for c in chunks]

# ---------------------------------------------------------------------
# NEW: token & chat-history budgeting helpers
# ---------------------------------------------------------------------
def count_tokens(tok, text: str) -> int:
    if tok is None or not text:
        # fallback heuristic
        return max(1, len(text) // 4)
    return len(tok.encode(text))

def join_blocks(*blocks: str) -> str:
    return "".join([b for b in blocks if b])

def trim_to_token_budget(tok, text: str, budget: int) -> str:
    if count_tokens(tok, text) <= budget:
        return text
    # binary trim from the left, keep the tail (most recent info)
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[-mid:]
        if count_tokens(tok, cand) <= budget:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def format_history_budgeted(tok, chat_history: List[Dict], hard_cap_tokens: int, max_turns_soft: int = 8) -> str:
    """Prioritize most recent messages; keep under hard token cap, soft cap by last N messages."""
    if not chat_history or hard_cap_tokens <= 0:
        return ""
    msgs = chat_history[-max_turns_soft:]
    pieces_rev = []
    used = 0
    for m in reversed(msgs):
        role = "user" if m.get("role", "user") == "user" else "assistant"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        block = f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        t = count_tokens(tok, block)
        if used + t > hard_cap_tokens:
            if used == 0:
                header = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                footer = "<|eot_id|>"
                remaining = max(0, hard_cap_tokens - count_tokens(tok, header) - count_tokens(tok, footer))
                trimmed_content = trim_to_token_budget(tok, content, remaining)
                if trimmed_content:
                    pieces_rev.append(f"{header}{trimmed_content}{footer}")
                    used = hard_cap_tokens
            break
        pieces_rev.append(block)
        used += t
    pieces = list(reversed(pieces_rev))
    return "".join(pieces)

def build_retrieval_query(question: str, chat_history: List[Dict], max_chars: int = 400) -> str:
    if not chat_history:
        return question
    last_user = ""
    for m in reversed(chat_history):
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
            if last_user:
                break
    composed = (last_user + " " + question).strip() if last_user else question
    return composed[-max_chars:]

def fingerprint_history_for_cache(chat_history: List[Dict], max_turns: int = 8) -> str:
    try:
        short = (chat_history or [])[-max_turns:]
        j = json.dumps(short, ensure_ascii=False, separators=(",", ":"))
        return hashlib.md5(j.encode("utf-8")).hexdigest()[:10]
    except Exception:
        return "nohist"

# ---------------------------------------------------------------------
# vLLM (prod) + Transformers CPU fallback (dev) with tokenizer for budgeting
# ---------------------------------------------------------------------
class AsyncLlamaService:
    def __init__(self):
        self.engine = None
        self.engine_args = None
        self.initialization_lock = asyncio.Lock()
        self.is_initialized = False
        self.warmup_done = False

        self.dev_fallback = False
        self.hf_model = None
        self.hf_tokenizer = None

        self.prompt_tokenizer = None  # NEW: tokenizer for prompt token counts

        self.prod_model_id = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
        self.dev_model_id = os.getenv("DEV_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    async def initialize(self):
        async with self.initialization_lock:
            if self.is_initialized:
                return
            logger.info("🚀 Initializing AsyncLLMEngine...")
            os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
            os.environ["CUDA_CACHE_PATH"] = "/tmp/.cuda_cache"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"  # V100

            ENV = os.getenv("ENVIRONMENT", "").lower()

            if ENV == "development":
                # DEV: smaller CPU model via transformers
                self.dev_fallback = True
                from transformers import AutoModelForCausalLM, AutoTokenizer as DevTokenizer
                model_id = self.dev_model_id
                logger.info(f"DEV mode: loading HF model on CPU: {model_id}")
                self.hf_tokenizer = DevTokenizer.from_pretrained(model_id)
                self.prompt_tokenizer = self.hf_tokenizer  # reuse for budgeting
                from transformers import AutoModelForCausalLM as DevModel
                self.hf_model = DevModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                self.is_initialized = True
                return

            # PROD / non-dev: vLLM path
            self.dev_fallback = False
            self.engine_args = AsyncEngineArgs(
                model=self.prod_model_id,
                dtype="float16",
                gpu_memory_utilization=0.88,
                max_model_len=16384,  # keep as in your original file
                max_num_seqs=4,
                max_num_batched_tokens=16384,
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                trust_remote_code=True,
                tokenizer_mode="auto",
                disable_log_stats=False,
                enforce_eager=True,
            )
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

            # Load a HF tokenizer for budgeting on CPU
            try:
                self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.prod_model_id)
            except Exception as e:
                logger.warning(f"Tokenizer load failed for budgeting; falling back to length heuristic: {e}")
                self.prompt_tokenizer = None

            self.is_initialized = True

    async def generate_stream(self, prompt: str, temperature: float = 0.7) -> AsyncIterator[str]:
        if not self.is_initialized:
            await self.initialize()
        await request_queue.acquire()
        try:
            if self.dev_fallback:
                # Transformers CPU generation with simple chunked streaming
                from transformers import StoppingCriteria, StoppingCriteriaList

                class StopOnTokens(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return False

                stopping_criteria = StoppingCriteriaList([StopOnTokens()])
                inputs = self.hf_tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                with torch.no_grad():
                    output_ids = self.hf_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        stopping_criteria=stopping_criteria,
                    )[0]
                full_text = self.hf_tokenizer.decode(
                    output_ids[input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                chunk_size = max(24, len(full_text) // 50)
                for i in range(0, len(full_text), chunk_size):
                    yield full_text[i:i + chunk_size]
                return

            # vLLM path (prod)
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=1024,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                repetition_penalty=1.1,
                top_p=0.95,
            )
            request_id = f"req_{time.time()}_{hash(prompt)}"
            results = self.engine.generate(prompt, sampling_params, request_id)

            emitted = ""
            def lcp_len(a: str, b: str) -> int:
                m = min(len(a), len(b)); i = 0
                while i < m and a[i] == b[i]: i += 1
                return i

            async for out in results:
                if out.outputs and len(out.outputs) > 0:
                    new_text = out.outputs[0].text
                    if new_text.startswith(emitted):
                        delta = new_text[len(emitted):]
                    else:
                        common = lcp_len(emitted, new_text)
                        delta = new_text[common:]
                    if delta:
                        emitted = new_text
                        yield delta
                if out.finished:
                    logger.info(f"Request {request_id[:20]}... completed successfully")
                    break
        finally:
            await request_queue.release()

llama_service = AsyncLlamaService()

# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------
response_cache: Dict[str, Dict] = {}
CACHE_TTL = 3600

def get_cache_key(question: str, system_prompt: str = None, chat_history: List[Dict] = None) -> str:
    h = fingerprint_history_for_cache(chat_history or [])
    key_content = f"{question.lower().strip()}_{(system_prompt or 'default').strip()}_{h}"
    return str(hash(key_content))

def is_cacheable_question(question: str, question_type: str) -> bool:
    if question_type == "academic":
        pats = [r'what is', r'define', r'explain the difference between', r'how does.*work']
        return any(re.search(p, question.lower()) for p in pats)
    return False

# ---------------------------------------------------------------------
# Public API (dynamic budgeting in ask_question_stream)
# ---------------------------------------------------------------------
async def ask_question_stream(
    question: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    chat_history: List[Dict] = None
) -> AsyncIterator[str]:
    logger.info(f"Processing question: {question[:80]}...")
    question_core = question.strip()
    question_type = classify_question_type(question_core)
    if chat_history is None:
        chat_history = []

    cache_key = get_cache_key(question_core, system_prompt, chat_history)
    if cache_key in response_cache and (time.time() - response_cache[cache_key]['timestamp'] < CACHE_TTL):
        response = response_cache[cache_key]['response']
        for w in response.split():
            yield w + " "
        return

    has_custom_prompt = bool(system_prompt and system_prompt.strip())
    retrieval_query = build_retrieval_query(question_core, chat_history)

    # Build raw RAG context; trimming happens after budgeting
    if should_use_rag(question_core, question_type, has_custom_prompt):
        top_chunks, _scores = get_adaptive_chunks(retrieval_query, question_type)
        passages = load_text_for_chunks(top_chunks)
        combined_context_raw = "\n\n".join([p for p in passages if p])
    else:
        combined_context_raw = ""

    # ----------- Dynamic token budgeting -----------
    await llama_service.initialize()
    tok = llama_service.prompt_tokenizer
    max_ctx = llama_service.engine_args.max_model_len if llama_service.engine_args else 16384

    RESPONSE_BUDGET = 1024          # tokens reserved for model output
    PROMPT_SAFETY  = 128            # guard band
    prompt_budget  = max(512, max_ctx - RESPONSE_BUDGET - PROMPT_SAFETY)

    default_sys = (
        "You are a step-by-step Socratic psychology tutor. Your goal is to guide students to think critically "
        "and build their understanding through structured dialogue.\n\n"
        "Teaching style:\n"
        "- Always follow this sequence:\n"
        "  1. Give a short, natural affirmation of the student’s response (e.g., “Nice start!” or “Good point”). Correct gently if needed.\n"
        "  2. Ask ONE open-ended guiding question that builds directly on their answer. Avoid leading or stacked phrasing.\n"
        "  3. After they respond, expand or clarify with evidence, examples, or research findings.\n\n"
        "- Let students attempt their own definitions or reasoning first before refining or adding detail.\n"
        "- Anchor questions tightly to the same topic or scenario; do not jump to unrelated examples.\n"
        "- Use concrete, everyday scenarios as scaffolds (e.g., studying for a test, remembering a name, recognizing a face).\n\n"
        "Handling challenges:\n"
        "- If the student is vague, off-topic, or says “I don’t know,” follow this rhythm:\n"
        "  1. Give a short, supportive affirmation (“That’s okay — this can be tricky”).\n"
        "  2. Provide a small, direct, on-topic hint or example.\n"
        "  3. Ask one simple follow-up question tied to the hint.\n"
        "- For lazy answers, do not escalate difficulty too quickly. Start with process-level steps (e.g., encoding vs retrieval) before introducing more advanced ideas (e.g., brain regions).\n"
        "- Build success quickly by offering strong hints when needed (e.g., “This brain area’s name starts with ‘hippo…’”). Once the student answers, affirm and expand.\n\n"
        "- If the student tries to derail, acknowledge their input but bring them back to the topic.\n"
        "- If the student asks about assignments (e.g., lab reports, introductions, research papers), walk them through step by step: purpose/goal → structure → examples → refinements.\n\n"
        "Tone:\n"
        "- Keep responses concise, supportive, and encouraging.\n"
        "- Favor exploration, but ground reasoning in known psychological findings when appropriate."
        "- If there is no context provided, or the question is not related to the course, kindly tell them you can't answer unrelated questions."
    )
    sys_text = (system_prompt.strip() if has_custom_prompt else default_sys)

    sys_block = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{sys_text}\n<|eot_id|>"
    )
    sys_tokens = count_tokens(tok, sys_block)
    remaining_for_prompt = max(0, prompt_budget - sys_tokens)

    # Allocate ~70% of remaining prompt to recent chat history
    hist_budget = int(remaining_for_prompt * 0.7)
    history_block = format_history_budgeted(tok, chat_history, hist_budget, max_turns_soft=8)
    history_tokens = count_tokens(tok, history_block)

    # Remaining goes to RAG context (if any)
    remaining_after_hist = max(0, remaining_for_prompt - history_tokens)
    if combined_context_raw:
        ctx_body = "Additional context:\n" + combined_context_raw
        ctx_block = trim_to_token_budget(tok, ctx_body, remaining_after_hist)
        ctx_block = f"{ctx_block}<|eot_id|>" if ctx_block else ""
    else:
        ctx_block = ""

    user_block = f"<|start_header_id|>user<|end_header_id|>\n\n{question_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt = join_blocks(sys_block, ctx_block, history_block, user_block)

    # Final guard: if still oversized, re-trim history then context
    total_tokens = count_tokens(tok, prompt)
    if total_tokens > (prompt_budget + PROMPT_SAFETY):
        over = total_tokens - (prompt_budget + PROMPT_SAFETY)
        new_hist_budget = max(0, hist_budget - over)
        history_block = format_history_budgeted(tok, chat_history, new_hist_budget, max_turns_soft=8)
        spent = count_tokens(tok, sys_block) + count_tokens(tok, history_block) + count_tokens(tok, user_block)
        ctx_budget2 = max(0, prompt_budget - spent)
        if combined_context_raw and ctx_budget2 > 0:
            ctx_body = "Additional context:\n" + combined_context_raw
            ctx_block = trim_to_token_budget(tok, ctx_body, ctx_budget2)
            ctx_block = f"{ctx_block}<|eot_id|>" if ctx_block else ""
        else:
            ctx_block = ""
        prompt = join_blocks(sys_block, ctx_block, history_block, user_block)
    # ----------- End dynamic budgeting -----------

    response_text = ""
    async for token in llama_service.generate_stream(prompt, temperature):
        response_text += token
        yield token

    if not has_custom_prompt and is_cacheable_question(question_core, question_type):
        response_cache[cache_key] = {'response': response_text.strip(), 'timestamp': time.time()}

async def ask_question(
    question: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    chat_history: List[Dict] = None
) -> str:
    out = ""
    async for tok in ask_question_stream(question, system_prompt, temperature, chat_history):
        out += tok
    return out.strip()

# Initialization helper
async def initialize_llm():
    await llama_service.initialize()

def cleanup_cache():
    now = time.time()
    for k in list(response_cache.keys()):
        if now - response_cache[k]['timestamp'] > CACHE_TTL:
            del response_cache[k]

def get_queue_status():
    return {
        "active_requests": request_queue.active_requests,
        "max_concurrent": request_queue.max_concurrent,
        "queue_length": len(request_queue.queue),
        "max_queue_size": request_queue.max_queue_size,
        "capacity_percentage": (request_queue.active_requests / request_queue.max_concurrent) * 100
    }

print("🚀 Async query system with FAISS loaded!")
if faiss_store.index and faiss_store.index.ntotal > 0:
    print(f"✅ FAISS index loaded with {faiss_store.index.ntotal} vectors")
else:
    print("⚠️  No FAISS index loaded - RAG disabled. Run: python embed_chunks_faiss.py")

