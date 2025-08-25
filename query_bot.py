# query_bot.py — strict RAG + streaming
# Handles context retrieval with true async concurrency for classroom scale

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

def get_adaptive_chunks(question: str, question_type: str) -> Tuple[List[Dict], List[float]]:
    if question_type == "academic":
        if any(w in question.lower() for w in ['compare','contrast','difference','relationship']):
            k = 4
        elif any(w in question.lower() for w in ['explain','describe','how']):
            k = 3
        else:
            k = 2
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
    return chunks, scores

def load_text_for_chunks(chunks):
    if not chunks:
        return []
    return [c.get('text','') for c in chunks]

# ---------------------------------------------------------------------
# vLLM (prod) + Transformers CPU fallback (dev)
# - PROD uses big model (default: meta-llama/Llama-3.1-8B-Instruct) via vLLM
# - DEV uses a smaller model via transformers on CPU for fast local boot
# ---------------------------------------------------------------------
class AsyncLlamaService:
    def __init__(self):
        self.engine = None
        self.engine_args = None
        self.initialization_lock = asyncio.Lock()
        self.is_initialized = False
        self.warmup_done = False

        # dev fallback bits
        self.dev_fallback = False
        self.hf_model = None
        self.hf_tokenizer = None

        # model IDs
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
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except Exception:
                    logger.error("Transformers not installed; install 'transformers' for dev CPU mode or use GPU.")
                    raise

                model_id = self.dev_model_id
                logger.info(f"DEV mode: loading HF model on CPU: {model_id}")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                self.is_initialized = True
                return

            # PROD / non-dev: vLLM path (unchanged, big model)
            self.dev_fallback = False
            self.engine_args = AsyncEngineArgs(
                model=self.prod_model_id,
                dtype="float16",
                gpu_memory_utilization=0.88,
                max_model_len=16384,
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
                        return False  # rely on max_new_tokens / eos from tokenizer

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
                # crude chunked streaming to mimic server-sent tokens
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
                m = min(len(a), len(b))
                i = 0
                while i < m and a[i] == b[i]:
                    i += 1
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

def get_cache_key(question: str, system_prompt: str = None) -> str:
    key_content = f"{question.lower().strip()}_{system_prompt or 'default'}"
    return str(hash(key_content))

def is_cacheable_question(question: str, question_type: str) -> bool:
    if question_type == "academic":
        pats = [r'what is', r'define', r'explain the difference between', r'how does.*work']
        return any(re.search(p, question.lower()) for p in pats)
    return False

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_prompt(system_prompt: Optional[str],
                 combined_context: str,
                 question_core: str,
                 body_noisy: bool) -> str:
    if system_prompt and system_prompt.strip():
        if combined_context:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt.strip()}\n\n"
                "Additional context:\n"
                f"{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{question_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{question_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
    else:
        sys = (
            "You are a Socratic-style psychology tutor. Your goal is to help students think critically "
            "and arrive at answers themselves through guided questioning. Ask probing, open-ended questions "
            "instead of giving direct answers immediately. Encourage students to explain their reasoning, "
            "consider alternatives, and make connections to prior knowledge. Provide clarifications or hints "
            "when they are stuck, and only supply direct explanations or definitions as a last resort. "
            "Always keep responses clear, concise, and supportive, while maintaining an encouraging tone."
        )
        if combined_context:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{sys}\n"
                "Course Materials:\n"
                f"{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{question_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{question_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

async def ask_question_stream(
    question: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    chat_history: List[Dict] = None
) -> AsyncIterator[str]:
    logger.info(f"Processing question: {question[:80]}...")
    question_core = question.strip()
    question_type = classify_question_type(question_core)

    cache_key = get_cache_key(question_core, system_prompt)
    if cache_key in response_cache and (time.time() - response_cache[cache_key]['timestamp'] < CACHE_TTL):
        response = response_cache[cache_key]['response']
        for w in response.split():
            yield w + " "
        return

    if chat_history is None:
        chat_history = []

    has_custom_prompt = bool(system_prompt and system_prompt.strip())

    if should_use_rag(question_core, question_type, has_custom_prompt):
        top_chunks, _scores = get_adaptive_chunks(question_core, question_type)
        passages = load_text_for_chunks(top_chunks)
        MAX_CONTEXT_CHARS = 1800
        buf, used = [], 0
        for p in passages:
            if not p:
                continue
            if used + len(p) + 2 > MAX_CONTEXT_CHARS:
                break
            buf.append(p)
            used += len(p) + 2
        combined_context = "\n\n".join(buf)
    else:
        combined_context = ""

    prompt = build_prompt(system_prompt, combined_context, question_core, False)

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
