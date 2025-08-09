# Optimized query_bot.py - True Async Concurrency with AsyncLLMEngine
""" Module handles context retrieval with true async concurrency for classroom scale """
import json
import os
import asyncio
import torch
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from functools import lru_cache
import re
import time
from typing import List, Dict, Optional, Tuple, AsyncIterator
import logging

load_dotenv()

INDEX_NAME = "tutor-bot-index"
CHUNK_FILE = "chunks.jsonl"

# Response cache for common questions (saves 30-40% compute)
response_cache = {}
CACHE_TTL = 3600  # 1 hour

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

def should_use_rag(question: str, question_type: str, has_custom_prompt: bool = False) -> bool:
    """Intelligent decision on whether to use RAG - only for psychology content in regular chat"""
    # Never use RAG if there's a custom system prompt (sandbox mode)
    if has_custom_prompt:
        return False

    # Never use RAG for casual conversation
    if question_type == "casual":
        return False

    # Never give direct answers to test questions
    if question_type == "test_question":
        return False

    # Always use RAG for academic questions (but only in regular chat)
    if question_type == "academic":
        return True

    # For general questions, use semantic similarity
    if question_type == "general":
        # Quick semantic check for psychology content
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

class AsyncLlamaService:
    """Async service using vLLM's AsyncLLMEngine for true concurrency"""
    
    def __init__(self):
        self.engine = None
        self.engine_args = None
        
    async def initialize(self):
        """Initialize the async engine - call this at startup"""
        if self.engine is not None:
            return  # Already initialized
            
        print("🚀 Initializing AsyncLLMEngine for true concurrency...")
        
        self.engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.2-3B-Instruct",
            dtype="auto",  # Let vLLM choose optimal dtype
            gpu_memory_utilization=0.75,  # Reduced for better concurrency
            max_model_len=2048,  # Increased for better responses
            max_num_seqs=64,  # Maximum concurrent sequences
            max_num_batched_tokens=8192,  # Larger batching
            enable_prefix_caching=True,  # Cache common prefixes
            trust_remote_code=True,
            tokenizer_mode="auto",
            disable_log_stats=False,  # Enable logging for debugging
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        print("✅ AsyncLLMEngine initialized successfully!")
        
    async def generate_stream(self, prompt: str, temperature: float = 0.7) -> AsyncIterator[str]:
        """Async streaming generation - yields complete tokens only"""
        if self.engine is None:
            await self.initialize()
            
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=512,  # Reasonable default, adjust as needed
            stop=["<|eot_id|>", "<|end_of_text|>"],
            repetition_penalty=1.1,
        )
        
        # Generate unique request ID
        request_id = f"req_{time.time()}_{hash(prompt)}"
        
        # Add request to engine
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        # Stream the raw tokens as vLLM provides them
        full_response = ""
        async for request_output in results_generator:
            if request_output.outputs:
                text = request_output.outputs[0].text
                # Get only the new content since last yield
                new_content = text[len(full_response):]
                if new_content:
                    full_response = text
                    # Yield the new content directly without splitting
                    yield new_content

# Global instance (will be initialized at startup)
llama_service = AsyncLlamaService()

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

async def ask_question_stream(
    question: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    chat_history: List[Dict] = None
) -> AsyncIterator[str]:
    """True async streaming with no blocking"""
    
    # Check cache for common questions
    cache_key = get_cache_key(question, system_prompt)
    if cache_key in response_cache:
        cache_entry = response_cache[cache_key]
        if time.time() - cache_entry['timestamp'] < CACHE_TTL:
            # Stream cached response
            words = cache_entry['response'].split()
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
    
    # Handle different question types
    if question_type == "casual" and not has_custom_prompt:
        # Regular chat - use default psychology tutor responses
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
        # Regular chat - provide default psychology tutor guidance
        response = "I can help you understand concepts and provide explanations, but I can't give direct answers to test questions. Instead, let me help you understand the underlying concepts. What specific topic would you like me to explain?"
        words = response.split()
        for word in words:
            yield word + " "
        return
    
    # For academic questions, use smart RAG (but respect custom prompts)
    combined_context = ""
    if should_use_rag(question, question_type, has_custom_prompt):
        # Only use psychology-focused RAG for regular chat
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
    
    # Build prompt based on context availability and system prompt
    if has_custom_prompt:
        # Custom system prompt (sandbox) - use it directly
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
        # Regular chat - use default psychology tutor behavior
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
    
    # Generate response using async streaming
    response_text = ""
    async for token in llama_service.generate_stream(prompt, temperature):
        response_text += token
        yield token
    
    # Cache common academic questions (only for regular chat)
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

# Initialize the async engine at module import (will be called from main.py)
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

print("🚀 Async query system loaded - waiting for initialization...")
print("📊 Model: Llama 3.2-3B | True async concurrency")
print("🎯 Features: Smart RAG, Response Caching, AsyncLLMEngine")
