""" Module breaks up txt files into chunks to be stored later """
import os
import json

# Parameters
INPUT_DIR = "texts"
OUTPUT_PATH = "chunks.jsonl"
CHUNK_SIZE = 500
OVERLAP = 100

# Function to split text into chunks with overlap
def split_text(txt, chunk_size=500, overlap=100):
    """ Handles splitting up the text into 500 word chunks """
    temp_chunks = []
    for i in range (0, len(txt), chunk_size - overlap):
        piece = txt[i:i + chunk_size]
        if piece.strip():
            temp_chunks.append(piece.strip())
    return temp_chunks

# Create and write to JSONL
with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(INPUT_DIR, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                chunks = split_text(text, CHUNK_SIZE, OVERLAP)
                for index, chunk in enumerate(chunks):
                    out_file.write(json.dumps({
                        "source": filename,
                        "chunk_id": index,
                        "text": chunk
                    }) + "\n")

print(f"Chunking complete. Saved to: {OUTPUT_PATH}")
