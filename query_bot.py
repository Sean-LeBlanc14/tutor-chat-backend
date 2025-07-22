# Unified query_bot.py with single intelligent system prompt
""" Module handles context retrieval with unified question type handling """
import json
import os
import requests
import torch
from functools import lru_cache
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
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

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

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

def retrieve_relevant_chunks(query, k=5):
    """Retrieves the top k most relevant chunks from DB"""
    try:
        query_embedding = model.encode([query])[0]
        response = index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )

        chunks = [match['metadata'] for match in response['matches']]
        scores = [match['score'] for match in response['matches']]
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

def call_mistral_api(prompt, temperature=0.7):
    """Call the Mistral LLM and return the response"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return "I'm having trouble connecting right now. Please try again in a moment."

def ask_question(question, system_prompt=None, temperature=0.7, chat_history=None):
    """
    Unified Q&A function with intelligent question type handling.
    
    Args:
        question (str): The user's question
        system_prompt (str, optional): Custom system prompt (for sandbox)
        temperature (float): Temperature for the AI model
        chat_history (list, optional): Previous messages in the conversation
    
    Returns:
        str: The AI's response
    """
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

    answer = call_mistral_api(prompt, temperature)
    return answer

# Backward compatibility
def ask_question_default(question):
    """Legacy function for backward compatibility"""
    return ask_question(question)