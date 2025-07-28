# Unified query_bot.py with single intelligent system prompt
""" Module handles context retrieval with unified question type handling """
import json
import os
import requests
import torch
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from functools import lru_cache

load_dotenv()

API_KEY = os.getenv("API_KEY")
INDEX_NAME = "tutor-bot-index"

# Define file path for full chunk text
CHUNK_FILE = "chunks.jsonl"

# Load sentence-transformer embedding model
@lru_cache(maxsize=1)
def get_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Usage:
model = get_embedding_model()

def load_faiss_index(index_path="chunk_index.faiss", metadata_path="chunk_metadata.json"):
    """Load existing FAISS index and metadata"""
    try:
        # Load the FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, metadata
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None, []

class FAISSVectorStore:
    def __init__(self, embedding_dim=384):  # all-MiniLM-L6-v2 dimension
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine sim
        self.metadata = []
        
    def add_vectors(self, vectors, metadata):
        """Add vectors to the index"""
        vectors = np.array(vectors).astype('float32')
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata)
    
    def search(self, query_vector, k=2):
        """Search for similar vectors"""
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
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
# Unified system prompt that handles all question types
UNIFIED_SYSTEM_PROMPT = """
You are a helpful and friendly course assistant for psychology students. 

For each question the student asks, you must first identify it as one of three types:

1. **Psychology Question**: New questions about psychology concepts, theories, research, or topics
2. **Follow-up Question**: Questions that refer to your previous responses, asking for clarification, simplification, elaboration, examples, or step-by-step explanations
3. **Irrelevant Question**: Questions not related to psychology or course content

**Instructions for each type:**

**Psychology Questions:**
- Use the provided course context to give comprehensive, accurate answers
- Draw from the course materials below to support your explanations
- Provide clear examples and explanations appropriate for students
- Reference relevant theories, researchers, or studies from the context

**Follow-up Questions:**
- Focus ONLY on your previous responses from the conversation history
- Do NOT introduce new information from course context
- If asked to simplify: make your previous explanation simpler
- If asked to elaborate: add more detail to what you already said
- If asked for examples: provide examples related to your previous response
- If asked for steps: break down your previous explanation into clear steps

**Irrelevant Questions:**
- Politely redirect: "I'm here to help with psychology-related topics. Could you ask a psychology question instead?"
- Do not attempt to answer non-psychology questions
- Encourage the student to ask about psychology concepts

**Available Information:**

Previous Conversation:
{chat_history}

Course Context (use only for psychology questions):
{context}

Current Question: {question}

**Your Response:**"""

def format_chat_history(messages, max_history=8):
    """Format recent chat messages for context"""
    if not messages:
        return "No previous conversation."

    # Take last max_history messages
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

        return [texts_by_source[(
            c["source"], c["chunk_id"])] for c in chunks if (c["source"], c["chunk_id"]) in texts_by_source]
    except FileNotFoundError:
        print(f"Warning: {chunk_file_path} not found. Using empty context.")
        return []
    except Exception as e:
        print(f"Error loading chunk texts: {e}")
        return []
        
class OptimizedLlamaService:
    def __init__(self):
        print("Loading Llama 3.1 8B model with vLLM...")
        self.llm = LLM(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                gpu_memory_utilization=0.85,
                max_model_len=4096,
                tensor_parallel_size=1,
                trust_remote_code=True,
                tokenizer_mode="auto"
        )

        self.sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=512,
                stream=True
        )
        print("vLLM model loaded successfully")


    def generate_stream(self, prompt, temperature=0.7):
        """Stream response tokens"""
        sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=512,
                stream=True
        )

        request_id = random_uuid()
        results_generator = self.llm.generate(
                [prompt],
                sampling_params,
                request_id=request_id
        )

        for request_output in results_generator:
            for output in request_output.outputs:
                yield output.text

    
    def generate_batch(self, prompts, temperature=0.7):
        """Handle multiple requests simultaneously"""
        sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=512,
                stream=False
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

llama_service = OptimizedLlamaService()


def ask_question_stream(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Streaming version of ask_question"""
    # Default to empty history if none provided
    if chat_history is None:
        chat_history = []

    # Format chat history
    formatted_history = format_chat_history(chat_history)

    # Get relevant chunks from RAG system for potential psychology questions
    top_chunks, _ = retrieve_relevant_chunks(question)
    context_passages = load_text_for_chunks(top_chunks, CHUNK_FILE)
    combined_context = "\n\n".join(context_passages) if context_passages else "No relevant course material found."

    # Use custom prompt if provided (for sandbox), otherwise use unified prompt
    if system_prompt:
        # For sandbox: use custom prompt but with same structure
        prompt = f"""{system_prompt.strip()}

Previous Conversation:
{formatted_history}

Course Context (if relevant):
{combined_context}

Current Question: {question}

Your Response:"""
    else:
        # Use unified system prompt
        prompt = UNIFIED_SYSTEM_PROMPT.format(
            chat_history=formatted_history,
            context=combined_context,
            question=question
        )

    for token in llama_service.generate_stream(prompt, temperature):
        yield token

# Backward compatibility
def ask_question(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Legacy function for backward compatibility"""
    response = ""
    for token in ask_question_stream(question, system_prompt, temperature, chat_history):
        response += token
    return response







