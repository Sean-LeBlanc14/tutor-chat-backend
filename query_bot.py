# Optimized query_bot.py - Fixed Triton issues with request queuing for 50+ users
""" Module handles context retrieval with true async concurrency for classroom scale """
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
import threading

load_dotenv()

# File paths for FAISS index and chunks
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
CHUNK_FILE = "chunks.jsonl"  # Kept for fallback if needed

# Response cache for common questions (saves 30-40% compute)
response_cache = {}
CACHE_TTL = 3600  # 1 hour

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request queue for managing high load
class RequestQueue:
    """Priority queue for managing requests when at capacity"""
    def __init__(self, max_concurrent=15, max_queue_size=100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.active_requests = 0
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.not_full = asyncio.Condition(self.lock)
        
    async def acquire(self, priority=0):
        """Acquire a slot for processing"""
        async with self.lock:
            # If queue is full, reject request
            if self.active_requests >= self.max_concurrent and len(self.queue) >= self.max_queue_size:
                raise Exception("Server at capacity. Please try again later.")
            
            # Wait if at concurrent limit
            while self.active_requests >= self.max_concurrent:
                future = asyncio.Future()
                self.queue.append((priority, time.time(), future))
                await self.not_full.wait()
                if future.done():
                    break
            
            self.active_requests += 1
            logger.info(f"Request acquired. Active: {self.active_requests}/{self.max_concurrent}, Queue: {len(self.queue)}")
    
    async def release(self):
        """Release a slot after processing"""
        async with self.lock:
            self.active_requests -= 1
            
            # Process next in queue if any
            if self.queue:
                _, _, future = self.queue.popleft()
                future.set_result(True)
            
            self.not_full.notify()
            logger.info(f"Request released. Active: {self.active_requests}/{self.max_concurrent}")

# Global request queue - increase concurrent limit since we have GPU headroom
request_queue = RequestQueue(max_concurrent=25, max_queue_size=75)

# Load sentence-transformer embedding model (optimized)
@lru_cache(maxsize=1)
def get_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    # Optimize for inference
    model.eval()
    return model

model = get_embedding_model()

def load_faiss_index(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH):
    """Load existing FAISS index and metadata from our new format"""
    try:
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata and texts from pickle
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
        print(f"Please run: python embed_chunks_faiss.py")
        return None, [], []
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return None, [], []

class FAISSVectorStore:
    def __init__(self, embedding_dim=384):
        self.index = None
        self.metadata = []
        self.texts = []
        self.embedding_dim = embedding_dim

    def initialize_empty(self):
        """Initialize empty index if no saved index exists"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.texts = []

    def set_index(self, index, metadata, texts):
        """Set pre-loaded index and metadata"""
        self.index = index
        self.metadata = metadata
        self.texts = texts

    def search(self, query_vector, k=2):
        """Search the index and return results with metadata and text"""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):
                result = {
                    'metadata': self.metadata[idx],
                    'text': self.texts[idx] if idx < len(self.texts) else '',
                    'score': float(score)
                }
                results.append(result)
        return results

# Initialize FAISS store
faiss_store = FAISSVectorStore()

# Load existing index or create empty one
try:
    loaded_index, loaded_metadata, loaded_texts = load_faiss_index()
    if loaded_index is not None:
        faiss_store.set_index(loaded_index, loaded_metadata, loaded_texts)
        print(f"✅ Successfully initialized FAISS with {len(loaded_metadata)} chunks")
    else:
        print("⚠️ No FAISS index found. Initializing empty store.")
        print("Run 'python embed_chunks_faiss.py' to create the index.")
        faiss_store.initialize_empty()
except Exception as e:
    print(f"❌ Error initializing FAISS: {e}")
    faiss_store.initialize_empty()

# Smart Question Classification functions
def classify_question_type(question: str) -> str:
    """Classify question to determine RAG strategy"""
    question_lower = question.lower().strip()

    # Casual/greeting patterns
    casual_patterns = [
        r'\b(hi|hello|hey|thanks|thank you|goodbye|bye)\b',
        r'how are you',
        r'what\'s up',
        r'good (morning|afternoon|evening)',
        r'nice to meet',
        r'see you later'
    ]

    # Academic question patterns
    academic_patterns = [
        r'\b(explain|describe|define|what is|how does|compare|contrast)\b',
        r'\b(theory|concept|process|mechanism|principle)\b',
        r'\b(perception|sensation|visual|auditory|cognitive|neural)\b',
        r'\b(color|vision|hearing|attention|memory|learning)\b',
        r'difference between',
        r'relationship between',
        r'example of'
    ]

    # Test/assessment patterns (avoid giving answers)
    test_patterns = [
        r'\b(test|quiz|exam|assessment|homework|assignment)\b',
        r'correct answer',
        r'multiple choice',
        r'true or false',
        r'which of the following'
    ]

    # Check patterns in order of priority
    for pattern in test_patterns:
        if re.search(pattern, question_lower):
            return "test_question"

    for pattern in casual_patterns:
        if re.search(pattern, question_lower):
            return "casual"

    for pattern in academic_patterns:
        if re.search(pattern, question_lower):
            return "academic"

    # Default: check if question is academic-related by length and complexity
    if len(question) > 50 and any(word in question_lower for word in
                                  ['psychology', 'brain', 'mind', 'behavior', 'study', 'research']):
        return "academic"

    return "general"

def should_use_rag(question: str, question_type: str, has_custom_prompt: bool = False) -> bool:
    """Intelligent decision on whether to use RAG - only for psychology content in regular chat"""
    if has_custom_prompt:
        return False
    if question_type == "casual":
        return False
    if question_type == "test_question":
        return False
    if question_type == "academic":
        return True
    if question_type == "general":
        academic_keywords = [
            'perception', 'sensation', 'visual', 'auditory', 'attention',
            'memory', 'learning', 'brain', 'neural', 'cognitive', 'psychology'
        ]
        return any(keyword in question.lower() for keyword in academic_keywords)
    return False

def get_adaptive_chunks(question: str, question_type: str) -> Tuple[List, List]:
    """Get different numbers of chunks based on question complexity"""
    if question_type == "academic":
        if any(word in question.lower() for word in ['compare', 'contrast', 'difference', 'relationship']):
            return retrieve_relevant_chunks(question, k=4)
        elif any(word in question.lower() for word in ['explain', 'describe', 'how']):
            return retrieve_relevant_chunks(question, k=3)
        else:
            return retrieve_relevant_chunks(question, k=2)
    else:
        return retrieve_relevant_chunks(question, k=2)

def retrieve_relevant_chunks(query, k=2):
    """Retrieves the top k most relevant chunks from FAISS with text included"""
    try:
        # Generate query embedding
        query_embedding = model.encode([query])[0]
        
        # Search FAISS index
        results = faiss_store.search(query_embedding, k)
        
        # Extract chunks and scores
        chunks = []
        scores = []
        
        for result in results:
            # Create chunk with text included
            chunk = {
                'source': result['metadata'].get('source', 'unknown'),
                'chunk_id': result['metadata'].get('chunk_id', 0),
                'doc_type': result['metadata'].get('doc_type', 'general'),
                'text': result['text']  # Text is already included
            }
            chunks.append(chunk)
            scores.append(result['score'])
        
        return chunks, scores
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return [], []

def load_text_for_chunks(chunks, chunk_file_path=None):
    """Extract text from chunks - text is already included in chunks"""
    if not chunks:
        return []
    
    # Text is already in the chunks from retrieve_relevant_chunks
    return [chunk.get('text', '') for chunk in chunks if 'text' in chunk]

class AsyncLlamaService:
    """Async service using vLLM's AsyncLLMEngine with Triton compilation fix"""
    
    def __init__(self):
        self.engine = None
        self.engine_args = None
        self.initialization_lock = asyncio.Lock()
        self.is_initialized = False
        self.warmup_done = False
        
    async def initialize(self):
        """Initialize the async engine with proper warmup to prevent Triton issues"""
        async with self.initialization_lock:
            if self.is_initialized:
                return
                
            logger.info("🚀 Initializing AsyncLLMEngine with Triton compilation fix...")
            
            # Set environment variables for Triton
            os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
            os.environ["CUDA_CACHE_PATH"] = "/tmp/.cuda_cache"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"  # V100 architecture
            
            try:
                self.engine_args = AsyncEngineArgs(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    dtype="float16",
                    gpu_memory_utilization=0.80,  # Increased since we have headroom
                    max_model_len=2048,
                    max_num_seqs=30,  # Increased to match queue capacity
                    max_num_batched_tokens=8192,  # Increased for better batching
                    enable_prefix_caching=False,
                    enable_chunked_prefill=False,
                    trust_remote_code=True,
                    tokenizer_mode="auto",
                    disable_log_stats=False,
                    enforce_eager=True,  # Keep for stability
                )
                
                self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
                self.is_initialized = True
                logger.info("✅ AsyncLLMEngine initialized successfully!")
                
                # Perform warmup to compile kernels
                await self._warmup()
                
            except Exception as e:
                logger.error(f"Failed to initialize AsyncLLMEngine: {e}")
                raise
    
    async def _warmup(self):
        """Warmup the model to pre-compile Triton kernels"""
        if self.warmup_done:
            return
            
        logger.info("🔥 Warming up model to compile Triton kernels...")
        
        warmup_prompts = [
            "Hello, how are you?",
            "Explain the concept of perception.",
            "What is the difference between sensation and perception?",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=50,
            stop=["<|eot_id|>"],
        )
        
        try:
            # Run warmup requests sequentially to avoid compilation conflicts
            for i, prompt in enumerate(warmup_prompts):
                logger.info(f"Warmup {i+1}/{len(warmup_prompts)}...")
                request_id = f"warmup_{i}"
                
                # Format prompt properly
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
                
                # Generate and consume the output
                async for _ in self.engine.generate(formatted_prompt, sampling_params, request_id):
                    pass
                
                # Small delay between warmup requests
                await asyncio.sleep(0.1)
            
            self.warmup_done = True
            logger.info("✅ Model warmup complete! Triton kernels compiled.")
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            # Continue anyway - warmup is not critical
            self.warmup_done = True
    
    async def generate_stream(self, prompt: str, temperature: float = 0.7) -> AsyncIterator[str]:
        """Async streaming generation with request queuing"""
        if not self.is_initialized:
            await self.initialize()
        
        # Acquire a slot from the request queue
        await request_queue.acquire()
        
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=1024,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                repetition_penalty=1.1,
                top_p=0.95,
            )
            
            # Generate unique request ID
            request_id = f"req_{time.time()}_{hash(prompt)}"
            
            # Generate and stream tokens
            full_response = ""
            results_generator = self.engine.generate(prompt, sampling_params, request_id)
            
            async for request_output in results_generator:
                if request_output.outputs and len(request_output.outputs) > 0:
                    text = request_output.outputs[0].text
                    new_content = text[len(full_response):]
                    if new_content:
                        full_response = text
                        yield new_content
                    
                    # Check if generation is complete
                    if request_output.finished:
                        logger.info(f"Request {request_id[:20]}... completed successfully")
                        break
                        
        except asyncio.CancelledError:
            # Handle client disconnection gracefully
            logger.info(f"Request cancelled by client")
            raise
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}")
            import traceback
            traceback.print_exc()
            yield f"Error generating response: {str(e)}"
        finally:
            # Always release the queue slot
            await request_queue.release()

# Global instance (will be initialized at startup)
llama_service = AsyncLlamaService()

def get_cache_key(question: str, system_prompt: str = None) -> str:
    """Generate cache key for common questions"""
    key_content = f"{question.lower().strip()}_{system_prompt or 'default'}"
    return str(hash(key_content))

def is_cacheable_question(question: str, question_type: str) -> bool:
    """Determine if question should be cached"""
    if question_type == "academic":
        academic_cache_patterns = [
            r'what is',
            r'define',
            r'explain the difference between',
            r'how does.*work'
        ]
        return any(re.search(pattern, question.lower()) for pattern in academic_cache_patterns)
    return False

async def ask_question_stream(
    question: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    chat_history: List[Dict] = None
) -> AsyncIterator[str]:
    """True async streaming with request queuing"""
    
    logger.info(f"Processing question: {question[:50]}...")
    
    # Check cache for common questions
    cache_key = get_cache_key(question, system_prompt)
    if cache_key in response_cache:
        cache_entry = response_cache[cache_key]
        if time.time() - cache_entry['timestamp'] < CACHE_TTL:
            logger.info(f"Cache hit for question")
            response = cache_entry['response']
            # Stream cached response word by word
            words = response.split()
            for word in words:
                yield word + " "
            return
    
    # Classify question type
    question_type = classify_question_type(question)
    
    # Default to empty history if none provided
    if chat_history is None:
        chat_history = []
    
    # Check if we have a custom system prompt (sandbox mode)
    has_custom_prompt = system_prompt and system_prompt.strip()
    
    # Handle different question types (casual, test questions)
    if question_type == "casual" and not has_custom_prompt:
        simple_responses = {
            "hi": "Hello! I'm here to help you with psychology concepts. What would you like to learn about?",
            "hello": "Hi there! I'm your psychology tutor assistant. How can I help you today?",
            "thanks": "You're welcome! Feel free to ask any psychology-related questions.",
            "bye": "Goodbye! Good luck with your psychology studies!"
        }
        
        for greeting, response in simple_responses.items():
            if greeting in question.lower():
                words = response.split()
                for word in words:
                    yield word + " "
                return
    
    elif question_type == "test_question" and not has_custom_prompt:
        response = "I can help you understand concepts and provide explanations, but I can't give direct answers to test questions. Instead, let me help you understand the underlying concepts. What specific topic would you like me to explain?"
        words = response.split()
        for word in words:
            yield word + " "
        return
    
    # For academic questions, use smart RAG
    combined_context = ""
    if should_use_rag(question, question_type, has_custom_prompt):
        top_chunks, scores = get_adaptive_chunks(question, question_type)
        context_passages = load_text_for_chunks(top_chunks)
        
        if scores and len(scores) > 0:
            relevant_passages = []
            for i, score in enumerate(scores):
                if score > 0.3:  # Relevance threshold
                    if i < len(context_passages):
                        relevant_passages.append(context_passages[i])
            
            combined_context = "\n\n".join(relevant_passages) if relevant_passages else ""
    
    # Build prompt
    if has_custom_prompt:
        if combined_context:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt.strip()}

Additional context:
{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        if combined_context:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful psychology tutor. Answer the student's question using the provided course materials when relevant. Keep your response clear, educational, and engaging.

Course Materials:
{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful psychology tutor assistant. Answer the student's question clearly and educationally.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Generate response using async streaming with queuing
    response_text = ""
    async for token in llama_service.generate_stream(prompt, temperature):
        response_text += token
        yield token
    
    # Cache common academic questions
    if not has_custom_prompt and is_cacheable_question(question, question_type):
        response_cache[cache_key] = {
            'response': response_text.strip(),
            'timestamp': time.time()
        }

async def ask_question(
    question: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    chat_history: List[Dict] = None
) -> str:
    """Non-streaming async version"""
    response = ""
    async for token in ask_question_stream(question, system_prompt, temperature, chat_history):
        response += token
    return response.strip()

# Initialize the async engine at module import
async def initialize_llm():
    """Initialize the LLM engine - call this at app startup"""
    await llama_service.initialize()

def cleanup_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = [
        key for key, value in response_cache.items()
        if current_time - value['timestamp'] > CACHE_TTL
    ]
    for key in expired_keys:
        del response_cache[key]

# Status endpoint helper
def get_queue_status():
    """Get current queue status for monitoring"""
    return {
        "active_requests": request_queue.active_requests,
        "max_concurrent": request_queue.max_concurrent,
        "queue_length": len(request_queue.queue),
        "max_queue_size": request_queue.max_queue_size,
        "capacity_percentage": (request_queue.active_requests / request_queue.max_concurrent) * 100
    }

print("🚀 Async query system with FAISS loaded!")
print("📊 Model: Llama 3.2-3B | Max concurrent: 25 | Queue size: 75")
print("🎯 Features: FAISS RAG, Response Caching, Request Queuing, Triton Fix")
if faiss_store.index and faiss_store.index.ntotal > 0:
    print(f"✅ FAISS index loaded with {faiss_store.index.ntotal} vectors")
else:
    print("⚠️  No FAISS index loaded - RAG disabled. Run: python embed_chunks_faiss.py")