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
        print("Loading Llama 3.1 8B model...")

        hf_token = os.getenv("HF_TOKEN")

        try:
            print("Attempting vLLM...")
            self.llm = LLM(
                model="meta-llama/Llama-3.1-8B-Instruct",
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                tensor_parallel_size=1,
                trust_remote_code=True,
                tokenizer_mode="auto",
            )
            self.use_vllm = True
            print("vLLM model loaded successfully!")

        except Exception as e:
            import traceback
            print(f"vLLM failed with full error: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            print("Falling back to transformers (still optimized with streaming)...")

            # Fallback to transformers
            self.use_vllm = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                token=hf_token
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                token=hf_token
            )
            print("Transformers model loaded successfully!")

    def generate_stream(self, prompt, temperature=0.7):
        """Stream response tokens"""
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=200,  # Even shorter to prevent loops
                stop=["<|eot_id|>", "<|end_of_text|>", "**Correct Answer:**", "**Identity Question:**", "**Your Response:**"],
                repetition_penalty=1.2,  # Prevent repetition
            )

            outputs = self.llm.generate([prompt], sampling_params)
            full_response = outputs[0].outputs[0].text.strip()
            
            # Clean up the response - remove unwanted patterns
            bad_patterns = [
                "*** Your Answer:**",
                "**Your Answer:**", 
                "I have written",
                "The question is:",
                "You have to answer",
                "1. The stimulus",
                "2. The stimulus",
                "If the stimulus",
                "Next Steps:",
                "--- ",
                "Please fill in",
                "What do you think",
                "Instructions:",
                "Wait for further"
            ]
            
            # Find the first occurrence of any bad pattern and cut there
            cut_index = len(full_response)
            for pattern in bad_patterns:
                index = full_response.find(pattern)
                if index != -1 and index < cut_index:
                    cut_index = index
            
            full_response = full_response[:cut_index].strip()

            if not full_response:
                yield "I apologize, but I'm having trouble generating a response right now."
                return

            # Simulate streaming by yielding words
            words = full_response.split()
            for word in words:
                yield word + " "
        else:
            # Transformers fallback - simulate streaming
            response = self._generate_transformers(prompt, temperature)
            # Simulate streaming by yielding word by word
            words = response.split()
            for word in words:
                yield word + " "

    def _generate_transformers(self, prompt, temperature):
        """Fallback transformers generation"""
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
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

    def generate_batch(self, prompts, temperature=0.7):
        """Handle multiple requests simultaneously"""
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=512,
                stream=False
            )
            outputs = self.llm.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        else:
            # Fallback for batch processing
            return [self._generate_transformers(prompt, temperature) for prompt in prompts]

llama_service = OptimizedLlamaService()


def ask_question_stream(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Streaming version of ask_question"""
    # Default to empty history if none provided
    if chat_history is None:
        chat_history = []

    # Format chat history
    formatted_history = format_chat_history(chat_history)

    # Get relevant chunks from RAG system
    top_chunks, scores = retrieve_relevant_chunks(question, k=3)  # Back to 3 chunks
    context_passages = load_text_for_chunks(top_chunks, CHUNK_FILE)
    combined_context = "\n\n".join(context_passages) if context_passages else "No relevant course material found."

    # Use custom prompt if provided (for sandbox), otherwise use unified prompt
    if system_prompt:
        # For sandbox: use custom prompt but with better context handling
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt.strip()}

IMPORTANT: Only use the course materials below if they are directly relevant to the user's question. If the course materials are not relevant to what the user asked, ignore them completely and just respond naturally to the user's question.

Course Materials (may not be relevant):
{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        # Chat-style format that Llama is trained for
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful psychology course assistant. Answer student questions clearly and concisely. Use only the most relevant information from the course materials to answer their specific question. Do not reproduce answer keys or test questions.

Course Materials:
{combined_context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    for token in llama_service.generate_stream(prompt, temperature):
        yield token

# Backward compatibility
def ask_question(question, system_prompt=None, temperature=0.7, chat_history=None):
    """Legacy function for backward compatibility"""
    response = ""
    for token in ask_question_stream(question, system_prompt, temperature, chat_history):
        response += token
    return response
