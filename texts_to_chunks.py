import os
import json

# Parameters
INPUT_DIR = "texts"
OUTPUT_PATH = "chunks.jsonl"
CHUNK_SIZE = 500
OVERLAP = 100

# Function to split text into chunks with overlap
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range (0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# Create and write to JSONL
with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(INPUT_DIR, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                chunks = split_text(text, CHUNK_SIZE, OVERLAP)
                for i, chunk in enumerate(chunks):
                    out_file.write(json.dumps({
                        "source": filename,
                        "chunk_id": i,
                        "text": chunk
                    }) + "\n")

print(f"Chunking complete. Saved to: {OUTPUT_PATH}")
