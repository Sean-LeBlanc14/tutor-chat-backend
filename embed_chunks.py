""" 
Enhanced embedding module for FAISS vector database
Converts chunks to embeddings and creates searchable FAISS index
"""
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
from datetime import datetime

# Configuration
CHUNK_FILE = "chunks.jsonl"                    # Input chunks
FAISS_INDEX_PATH = "faiss_index.bin"          # FAISS index file
METADATA_PATH = "faiss_metadata.pkl"          # Metadata storage
MODEL_NAME = "all-MiniLM-L6-v2"               # Same model you were using
BATCH_SIZE = 100                              # Process in batches
EMBEDDING_DIM = 384                           # Dimension for MiniLM model

# Alternative models (if you want to experiment):
# "all-mpnet-base-v2" - dimension: 768, better quality but slower
# "all-MiniLM-L12-v2" - dimension: 384, slightly better than L6
# "BAAI/bge-small-en-v1.5" - dimension: 384, good balance

class FAISSIndexer:
    """Create and manage FAISS index for academic content"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the indexer with embedding model"""
        print(f"📚 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded (embedding dimension: {self.embedding_dim})")
        
        # Storage for metadata
        self.metadata = []
        self.texts = []
        self.index = None
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'by_doc_type': {},
            'by_source': {}
        }
    
    def load_chunks(self, chunk_file: str) -> Tuple[List[str], List[Dict]]:
        """Load chunks from JSONL file"""
        print(f"\n📂 Loading chunks from {chunk_file}...")
        
        texts = []
        metadatas = []
        
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    
                    # Extract text
                    text = obj.get("text", "")
                    if not text.strip():
                        continue
                    
                    # Build metadata (including new fields from enhanced chunking)
                    metadata = {
                        "chunk_id": obj.get("chunk_id", line_num),
                        "source": obj.get("source", "unknown"),
                        "doc_type": obj.get("doc_type", "general"),
                        "char_count": obj.get("char_count", len(text)),
                        "source_file": obj.get("source_file", ""),
                        "text_preview": text[:200] + "..." if len(text) > 200 else text
                    }
                    
                    texts.append(text)
                    metadatas.append(metadata)
                    
                    # Update statistics
                    doc_type = metadata["doc_type"]
                    source = metadata["source"]
                    self.stats['by_doc_type'][doc_type] = self.stats['by_doc_type'].get(doc_type, 0) + 1
                    self.stats['by_source'][source] = self.stats['by_source'].get(source, 0) + 1
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing line {line_num}: {e}")
                    continue
        
        self.stats['total_chunks'] = len(texts)
        print(f"✅ Loaded {len(texts)} chunks")
        
        # Print statistics
        print(f"\n📊 Chunk Statistics:")
        print(f"  Total chunks: {self.stats['total_chunks']}")
        print(f"  Document types: {len(self.stats['by_doc_type'])}")
        print(f"  Unique sources: {len(self.stats['by_source'])}")
        
        return texts, metadatas
    
    def create_embeddings(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Create embeddings for all texts"""
        print(f"\n🔄 Generating embeddings (batch size: {batch_size})...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False  # We're using tqdm already
            )
            
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings).astype('float32')
        print(f"✅ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "Flat") -> faiss.Index:
        """
        Create FAISS index from embeddings
        
        Index types:
        - "Flat": Exact search (best quality, slower for large datasets)
        - "IVF": Inverted file index (faster, slight quality loss)
        - "HNSW": Hierarchical NSW (good balance)
        """
        print(f"\n🏗️  Creating FAISS index (type: {index_type})...")
        
        n_vectors, dim = embeddings.shape
        
        if index_type == "Flat":
            # Exact search - best for small-medium datasets (<10k vectors)
            index = faiss.IndexFlatL2(dim)
            
        elif index_type == "IVF":
            # Inverted file index - good for larger datasets
            nlist = min(100, n_vectors // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            print(f"  Training IVF index with {nlist} clusters...")
            index.train(embeddings)
            
        elif index_type == "HNSW":
            # Hierarchical NSW - good balance of speed and quality
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(dim, M)
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        print(f"  Adding {n_vectors} vectors to index...")
        index.add(embeddings)
        
        print(f"✅ Index created with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, metadata: List[Dict], texts: List[str]):
        """Save FAISS index and metadata to disk"""
        print(f"\n💾 Saving index and metadata...")
        
        # Save FAISS index
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"  ✅ FAISS index saved to {FAISS_INDEX_PATH}")
        
        # Save metadata and texts
        data_to_save = {
            'metadata': metadata,
            'texts': texts,
            'stats': self.stats,
            'embedding_dim': self.embedding_dim,
            'model_name': MODEL_NAME,
            'created_at': datetime.now().isoformat()
        }
        
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"  ✅ Metadata saved to {METADATA_PATH}")
        
        # Print summary
        print(f"\n📈 Index Summary:")
        print(f"  Total vectors: {index.ntotal}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Index size: {os.path.getsize(FAISS_INDEX_PATH) / 1024 / 1024:.2f} MB")
        print(f"  Metadata size: {os.path.getsize(METADATA_PATH) / 1024 / 1024:.2f} MB")
    
    def test_search(self, index: faiss.Index, texts: List[str], metadata: List[Dict], 
                   query: str = "mental rotation experiment", k: int = 5):
        """Test the index with a sample search"""
        print(f"\n🔍 Testing search with query: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        print(f"\n📊 Top {k} results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid result
                meta = metadata[idx]
                text_preview = texts[idx][:150] + "..." if len(texts[idx]) > 150 else texts[idx]
                
                print(f"\n  {i+1}. Source: {meta['source']}")
                print(f"     Type: {meta['doc_type']}")
                print(f"     Distance: {dist:.4f}")
                print(f"     Preview: {text_preview}")
    
    def process(self, chunk_file: str = CHUNK_FILE):
        """Main processing pipeline"""
        print("="*60)
        print("🚀 FAISS Embedding Pipeline Started")
        print("="*60)
        
        # Load chunks
        texts, metadata = self.load_chunks(chunk_file)
        
        if not texts:
            print("❌ No chunks to process!")
            return
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Create FAISS index
        # For ~3000 chunks, Flat index is perfect (exact search)
        # For >10k chunks, consider IVF or HNSW
        index = self.create_faiss_index(embeddings, index_type="Flat")
        
        # Save everything
        self.save_index(index, metadata, texts)
        
        # Test with sample search
        self.test_search(index, texts, metadata)
        
        print("\n" + "="*60)
        print("✅ FAISS Embedding Pipeline Complete!")
        print("="*60)
        
        # Print usage instructions
        print("\n📝 To use the index in your application:")
        print("```python")
        print("import faiss")
        print("import pickle")
        print("from sentence_transformers import SentenceTransformer")
        print("")
        print("# Load index and metadata")
        print(f"index = faiss.read_index('{FAISS_INDEX_PATH}')")
        print(f"with open('{METADATA_PATH}', 'rb') as f:")
        print("    data = pickle.load(f)")
        print("    metadata = data['metadata']")
        print("    texts = data['texts']")
        print("")
        print("# Load model for query encoding")
        print(f"model = SentenceTransformer('{MODEL_NAME}')")
        print("")
        print("# Search")
        print("query = 'your search query'")
        print("query_embedding = model.encode([query], convert_to_numpy=True)")
        print("distances, indices = index.search(query_embedding, k=5)")
        print("```")


def main():
    """Main function"""
    # Check if chunks file exists
    if not os.path.exists(CHUNK_FILE):
        print(f"❌ Chunks file not found: {CHUNK_FILE}")
        print("Please run the chunking script first.")
        return
    
    # Create indexer and process
    indexer = FAISSIndexer(model_name=MODEL_NAME)
    indexer.process(chunk_file=CHUNK_FILE)
    
    # Additional test queries for academic content
    print("\n🔬 Running additional test queries...")
    
    # Load the saved index for testing
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        data = pickle.load(f)
        metadata = data['metadata']
        texts = data['texts']
    
    test_queries = [
        "visual attention experiment methods",
        "signal detection theory",
        "lab report requirements",
        "data analysis JASP"
    ]
    
    for query in test_queries:
        print(f"\n📍 Query: '{query}'")
        query_embedding = indexer.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, k=3)
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0][:1])):  # Just show top 1
            if idx >= 0:
                meta = metadata[idx]
                print(f"   → Best match: {meta['source']} (type: {meta['doc_type']}, dist: {dist:.4f})")


if __name__ == "__main__":
    main()