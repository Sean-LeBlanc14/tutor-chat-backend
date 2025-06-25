# Import necessary libraries for JSON handling, FAISS indexing, and embeddings
import json
import faiss
from sentence_transformers import SentenceTransformer

# Set the input/output file paths
CHUNK_FILE = r"C:\Users\SeanA\TutorChatBot\chunks.jsonl"         # Path to JSONL file with all text chunks
INDEX_FILE = r"C:\Users\SeanA\TutorChatBot\chunk_index.faiss"    # Path to save FAISS index
METADATA_FILE = r"C:\Users\SeanA\TutorChatBot\chunk_metadata.json"  # Path to save metadata for chunks

# Load the sentence transformer model (MiniLM, fast and lightweight)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize lists to hold chunk texts and their metadata
texts = []
metadata = []

# Read all chunks from the .jsonl file
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line) # Load each line as a JSON object
        texts.append(obj["text"]) # Store the chunk text
        metadata.append({  # Store chunk metadata
            "source": obj["source"],
            "chunk_id": obj["chunk_id"]
        })

# Generate embeddings for all text chunks
embeddings = model.encode(texts, show_progress_bar=True)

# Create FAISS index and add all embeddings
dimension = embeddings[0].shape[0] # Get embedding vector size
index = faiss.IndexFlatL2(dimension) # Use L2 distance index
index.add(embeddings) # Add all chunk embeddings to the index

# Save the FAISS index to disk
faiss.write_index(index, INDEX_FILE)

# Save metadata for lookup and reference during search
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Embedding and indexing complete.")
