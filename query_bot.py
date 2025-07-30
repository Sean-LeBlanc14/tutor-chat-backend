# Optimized query_bot.py - Classroom Scale with Smart RAG
""" Module handles context retrieval with intelligent question classification """
import json
import os
import asyncio
import requests
import torch
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from functools import lru_cache
import re
import time
from typing import List, Dict, Optional, Tuple
import logging

load_dotenv()

API_KEY = os.getenv("API_KEY")
INDEX_NAME = "tutor-bot-index"
CHUNK_FILE = "chunks.jsonl"

# Response cache for common questions (saves 30-40% compute)
response_cache = {}
CACHE_TTL = 3600  # 1 hour

# Request semaphore for controlled concurrency
MAX_CONCURRENT_REQUESTS = 15  # Optimized for your GPU
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Load sentence-transformer embedding model (optimized)
@lru_cache(maxsize=1)
def get_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    # Optimize for inference
    model.eval()
    return model

model = get_embedding_model()

def load_faiss_index(index_path="chunk_index.faiss", metadata_path="chunk_metadata.json"):
    """Load existing FAISS index and metadata"""
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, metadata
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None, []

class FAISSVectorStore:
    def __init__(self, embedding_dim=384):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []

    def add_vectors(self, vectors, metadata):
        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector, k=2):
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                results.append({
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        return results

# Initialize FAISS store
try:
    loaded_index, loaded_metadata = load_faiss_index()
    if loaded_index is not None:
        faiss_store = FAISSVectorStore()
        faiss_store.index = loaded_index
        faiss_store.metadata = loaded_metadata
        print("Successfully loaded existing FAISS index")
    else:
        faiss_store = FAISSVectorStore()
        print("Created new empty FAISS index")
except:
    faiss_store = FAISSVectorStore()
    print("Created new empty FAISS index")

# Smart Question Classification
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

def should_use_rag(question: str, question_type: str) -> bool:
    """Intelligent decision on whether to use RAG"""
    # Never use RAG for casual conversation
    if question_type == "casual":
        return False
    
    # Never give direct answers to test questions
    if question_type == "test_question":
        return False
    
    # Always use RAG for academic questions
    if question_type == "academic":
        return True
    
    # For general questions, use semantic similarity
    if question_type == "general":
        # Quick semantic check
        academic_keywords = [
            'perception', 'sensation', 'visual', 'auditory', 'attention',
            'memory', 'learning', 'brain', 'neural', 'cognitive', 'psychology'
        ]
        return any(keyword in question.lower() for keyword in academic_keywords)
    
    return False

def get_adaptive_chunks(question: str, question_type: str) -> Tuple[List, List]:
    """Get different numbers of chunks based on question complexity"""
    if question_type == "academic":
        # Complex academic questions need more context
        if any(word in question.lower() for word in ['compare', 'contrast', 'difference', 'relationship']):
            return retrieve_relevant_chunks(question, k=4)  # Comparison questions
        elif any(word in question.lower() for word in ['explain', 'describe', 'how']):
            return retrieve_relevant_chunks(question, k=3)  # Explanation questions
        else:
            return retrieve_relevant_chunks(question, k=2)  # Definition questions
    else:
        return retrieve_relevant_chunks(question, k=2)  # Default

def format_chat_history(messages, max_history=6):  # Reduced for efficiency
    """Format recent chat messages for context"""
    if not messages:
        return "No previous conversation."

    recent_messages = messages[-max_history:]
    formatted_history = []
    for msg in recent_messages:
        role = "Student" if msg.get('role') == 'user' else "Assistant"
        content = msg.get('content', '').strip()
        if content:
            formatted_history.append(f"{role}: {content}")

    return "\n\n".join(formatted_history) if formatted_history else "No previous conversation."

def retrieve_relevant_chunks(query, k=2):
    """Retrieves the top k most relevant chunks from FAISS"""
    try:
        query_embedding = model.encode([query])[0]
        results = faiss_store.search(query_embedding, k)
        chunks = [result['metadata'] for result in results]
        scores = [result['score'] for result in results]
        return chunks, scores
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return [], []

def load_text_for_chunks(chunks, chunk_file_path):
    """Load the text associated with the chunks"""
    if not chunks:
        return []

    texts_by_source = {}
    try:
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = (obj["source"], obj["chunk_id"])
                texts_by_source[key] = obj["text"]

        return [texts_by_source[(c["source"], c["chunk_id"])] 
                for c in chunks if (c["source"], c["chunk_id"]) in texts_by_source]
    except FileNotFoundError:
        print(f"Warning: {chunk_file_path} not found. Using empty context.")
        return []
    except Exception as e:
        print(f"Error loading chunk texts: {e}")
        return []

class OptimizedLlamaService:
    def __init__(self):
        print("Loading OPT-2.7B model for classroom scale...")
        hf_token = os.getenv("HF_TOKEN")

        try:
            print("Attempting vLLM with optimized settings...")
            self.llm = LLM(
                model="meta-llama/Llama-3.2-3B-Instruct",  # Smaller model for scale
                gpu_memory_utilization=0.4,  # Much lower memory usage
                max_model_len=1024,  # Shorter context for speed
                tensor_parallel_size=1,
                trust_remote_code=True,
                tokenizer_mode="auto",
                enforce_eager=False,  # Allow CUDA graphs for speed
                max_num_batched_tokens=2048,  # Enable batching
            )
            self.use_vllm = True
            print("vLLM 3B model loaded successfully for classroom scale!")

        except Exception as e:
            import traceback
            print(f"vLLM failed: {e}")
            print("Falling back to transformers...")

            self.use_vllm = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/opt-2.7b",
                token=hf_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-2.7b",
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                token=hf_token
            )
            print("Transformers 3B model loaded successfully!")

    def generate_stream(self, prompt, temperature=0.7):
        """Optimized streaming with better parameters"""
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=150,  # Shorter responses for speed
                stop=["<|eot_id|>", "<|end_of_text|>"],
                repetition_penalty=1.1,
            )

            outputs = self.llm.generate([prompt], sampling_params)
            full_response = outputs[0].outputs[0].text.strip()

            if not full_response:
                yield "I'm having trouble generating a response right now."
                return

            # Stream by words for smooth user experience
            words = full_response.split()
            for word in words:
                yield word + " "
        else:
            response = self._generate_transformers(prompt, temperature)
            words = response.split()
            for word in words:
                yield word + " "

    def _generate_transformers(self, prompt, temperature):
        """Fallback transformers generation"""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

# Initialize service
llama_service = OptimizedLlamaService()

def get_cache_key(question: str, system_prompt: str = None) -> str:
    """Generate cache key for common questions"""
    key_content = f"{question.lower().strip()}_{system_prompt or 'default'}"
    return str(hash(key_content))

def is_cacheable_question(question: str, question_type: str) -> bool:
    """Determine if question should be cached"""
    # Cache academic definitions and common questions
    if question_type == "academic":
        academic_cache_patterns = [
            r'what is',
            r'define',
            r'explain the difference between',
            r'how does.*work'
        ]
        return any(re.search(pattern, question.lower()) for pattern in academic_cache_patterns)
    return False

def ask_question_stream(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Smart streaming with concurrency control and caching (SYNC generator)"""
    import asyncio
    import time
    
    # Get the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async function and return generator
    return loop.run_until_complete(
        _ask_question_stream_async(question, system_prompt, temperature, chat_history)
    )

async def _ask_question_stream_async(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Internal async implementation"""
    # Acquire semaphore for controlled concurrency
    async with request_semaphore:
        # Check cache for common questions
        cache_key = get_cache_key(question, system_prompt)
        if cache_key in response_cache:
            cache_entry = response_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < CACHE_TTL:
                # Return cached response as generator
                def cached_generator():
                    words = cache_entry['response'].split()
                    for word in words:
                        yield word + " "
                return cached_generator()

        # Classify question type
        question_type = classify_question_type(question)
        
        # Default to empty history if none provided
        if chat_history is None:
            chat_history = []

        # Handle different question types
        if question_type == "casual":
            # Simple response without RAG
            simple_responses = {
                "hi": "Hello! I'm here to help you with psychology concepts. What would you like to learn about?",
                "hello": "Hi there! I'm your psychology tutor assistant. How can I help you today?",
                "thanks": "You're welcome! Feel free to ask any psychology-related questions.",
                "bye": "Goodbye! Good luck with your psychology studies!"
            }
            
            for greeting, response in simple_responses.items():
                if greeting in question.lower():
                    def simple_generator():
                        words = response.split()
                        for word in words:
                            yield word + " "
                    return simple_generator()

        elif question_type == "test_question":
            # Provide guidance instead of answers
            response = "I can help you understand concepts and provide explanations, but I can't give direct answers to test questions. Instead, let me help you understand the underlying concepts. What specific topic would you like me to explain?"
            def guidance_generator():
                words = response.split()
                for word in words:
                    yield word + " "
            return guidance_generator()

        # For academic questions, use smart RAG
        if should_use_rag(question, question_type):
            top_chunks, scores = get_adaptive_chunks(question, question_type)
            context_passages = load_text_for_chunks(top_chunks, CHUNK_FILE)
            
            # Filter low-relevance chunks
            if scores and len(scores) > 0:
                relevant_passages = []
                for i, score in enumerate(scores):
                    if score > 0.3:  # Only use relevant chunks
                        if i < len(context_passages):
                            relevant_passages.append(context_passages[i])
                
                combined_context = "\n\n".join(relevant_passages) if relevant_passages else ""
            else:
                combined_context = ""
        else:
            combined_context = ""

        # Build prompt based on context availability
        if combined_context:
            if system_prompt:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt.strip()}

Relevant course materials:
{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful psychology tutor. Answer the student's question using the provided course materials when relevant. Keep your response clear, educational, and engaging.

Course Materials:
{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # No context needed - direct conversation
            if system_prompt:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt.strip()}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful psychology tutor assistant. Answer the student's question clearly and educationally.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # Generate response using the sync generator
        def model_generator():
            response_text = ""
            for token in llama_service.generate_stream(prompt, temperature):
                response_text += token
                yield token
            
            # Cache common academic questions
            if is_cacheable_question(question, question_type):
                response_cache[cache_key] = {
                    'response': response_text.strip(),
                    'timestamp': time.time()
                }
        
        return model_generator()

# Backward compatibility
def ask_question(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Legacy function for backward compatibility"""
    response = ""
    for token in ask_question_stream(question, system_prompt, temperature, chat_history):
        response += token
    return response

# Clean up cache periodically
def cleanup_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = [
        key for key, value in response_cache.items()
        if current_time - value['timestamp'] > CACHE_TTL
    ]
    for key in expired_keys:
        del response_cache[key]

# Batch processing for high load (future enhancement)
class BatchProcessor:
    def __init__(self, batch_size=5, timeout=2.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.batch_lock = asyncio.Lock()

    async def add_request(self, request_data):
        """Add request to batch (future enhancement)"""
        async with self.batch_lock:
            self.pending_requests.append(request_data)
            if len(self.pending_requests) >= self.batch_size:
                return await self._process_batch()
        
        # Process batch after timeout
        await asyncio.sleep(self.timeout)
        async with self.batch_lock:
            if self.pending_requests:
                return await self._process_batch()

    async def _process_batch(self):
        """Process batched requests (future enhancement)"""
        if not self.pending_requests:
            return []
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        
        # Process batch with vLLM
        return batch

# Initialize batch processor (for future use)
batch_processor = BatchProcessor()

print("🚀 Classroom-optimized query system loaded!")
print(f"📊 Model: Llama 3.2-3B | Max concurrent: {MAX_CONCURRENT_REQUESTS}")
print("🎯 Features: Smart RAG, Response Caching, Request Queuing")
