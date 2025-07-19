""" Module embeds each chunk into a Pinecone vector database """
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import ServerlessSpec, Pinecone

#Load API keys from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "tutor-bot-index"

# Set the input/output file paths
CHUNK_FILE = "chunks.jsonl"         # Path to JSONL file with all text chunks

# Load the sentence transformer model (MiniLM, fast and lightweight)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize pinecone
print("Connecting to pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

#Load chunk data
print("Loading chunk data...")
texts = []
ids = []
metadatas = []

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        text = obj["text"]
        chunk_id = obj["chunk_id"]
        source = obj["source"]

        texts.append(text)
        ids.append(f"chunk-{chunk_id}")
        metadatas.append({
            "chunk_id": chunk_id,
            "source": source,
            "text": text
        })

print("Generating embeddings and uploading to pinecone")
BATCH_SIZE = 100
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]
    batch_metas = metadatas[i:i + BATCH_SIZE]
    batch_vectors = model.encode(batch_texts)

    index.upsert(
        vectors=[
            {
                "id": id,
                "values": vector.tolist(),
                "metadata": meta
            }
            for id, vector, meta in zip(batch_ids, batch_vectors, batch_metas)
        ]
    )

print("All embeddings uploaded to pinecone")
