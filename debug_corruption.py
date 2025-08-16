#!/usr/bin/env python3
"""
Corruption Debugging Script for Psychology Tutor RAG Pipeline
Analyzes text corruption without modifying query_bot.py
"""

import sys
import os
import re
import json
from collections import Counter
import asyncio

# Import from your existing query_bot
try:
    from query_bot import (
        CHUNK_FILE, faiss_store, model, 
        retrieve_relevant_chunks, load_text_for_chunks,
        classify_question_type, get_adaptive_chunks
    )
    print("✅ Successfully imported from query_bot.py")
except ImportError as e:
    print(f"❌ Error importing from query_bot: {e}")
    print("Make sure you're running this in the same directory as query_bot.py")
    sys.exit(1)

class CorruptionAnalyzer:
    """Comprehensive corruption analysis for the RAG pipeline"""
    
    def __init__(self):
        self.corruption_patterns = {
            'object_corruption': r'\bct recognition\b',
            'shape_corruption': r'\bape perception\b', 
            'size_corruption': r'\bize perception\b',
            'pattern_corruption': r'\brn recognition\b',
            'effect_corruption': r'\bects on\b',
            'creative_corruption': r'\bative thinking\b',
            'visual_corruption': r'\bual perception\b',
            'memory_corruption': r'\bmory\b(?!\s+foam)',  # memory but not "memory foam"
            'brain_corruption': r'\brain\b',
            'overall_corruption': r'\bverall\b',
            'general_missing_start': r'\b[a-z]{1,3}\s+[a-z]{4,}\b',  # Short words + longer words
        }
        
        self.test_question = "What are the effects of mental rotation?"
        
    def quick_corruption_test(self):
        """Quick test to identify where corruption is happening"""
        print("🔍 QUICK CORRUPTION TEST")
        print("-" * 40)
        
        corruption_found = False
        corruption_source = "unknown"
        
        # Test 1: Check chunks file directly
        print("\n1. Testing chunks.jsonl file...")
        try:
            with open(CHUNK_FILE, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 chunks
                        break
                    
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    
                    # Check for corruption patterns
                    for pattern_name, pattern in self.corruption_patterns.items():
                        if re.search(pattern, text, re.IGNORECASE):
                            print(f"❌ CORRUPTION FOUND IN CHUNK {i+1}")
                            print(f"   Pattern: {pattern_name}")
                            print(f"   Text preview: {text[:150]}...")
                            corruption_found = True
                            corruption_source = "chunks_file"
                            break
                    
                    if corruption_found:
                        break
                        
            if not corruption_found:
                print("✅ First 5 chunks look clean")
                
        except Exception as e:
            print(f"❌ Error reading chunks file: {e}")
            return "file_error"
        
        # Test 2: Check live retrieval if chunks look clean
        if not corruption_found:
            print("\n2. Testing live RAG retrieval...")
            try:
                chunks, scores = retrieve_relevant_chunks(self.test_question, k=2)
                chunk_texts = load_text_for_chunks(chunks, CHUNK_FILE)
                
                for i, text in enumerate(chunk_texts):
                    for pattern_name, pattern in self.corruption_patterns.items():
                        if re.search(pattern, text, re.IGNORECASE):
                            print(f"❌ CORRUPTION IN RETRIEVED CHUNK {i+1}")
                            print(f"   Pattern: {pattern_name}")
                            print(f"   Text: {text[:150]}...")
                            corruption_found = True
                            corruption_source = "retrieval"
                            break
                    if corruption_found:
                        break
                        
                if not corruption_found:
                    print("✅ Retrieved chunks look clean")
                    
            except Exception as e:
                print(f"❌ Error in retrieval test: {e}")
                return "retrieval_error"
        
        print(f"\n🎯 RESULT: Corruption source appears to be: {corruption_source}")
        return corruption_source
    
    def analyze_chunks_file(self, sample_size=20):
        """Detailed analysis of chunks.jsonl file"""
        print(f"\n📊 ANALYZING {CHUNK_FILE} (first {sample_size} chunks)")
        print("=" * 60)
        
        corrupted_chunks = []
        total_chunks = 0
        corruption_stats = Counter()
        
        try:
            with open(CHUNK_FILE, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                        
                    total_chunks += 1
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    chunk_id = obj.get("chunk_id", f"chunk_{i}")
                    source = obj.get("source", "unknown")
                    
                    # Check for corruption patterns
                    chunk_corruption = {}
                    for pattern_name, pattern in self.corruption_patterns.items():
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            chunk_corruption[pattern_name] = matches
                            corruption_stats[pattern_name] += len(matches)
                    
                    if chunk_corruption:
                        corrupted_chunks.append({
                            'chunk_id': chunk_id,
                            'source': source,
                            'corruption': chunk_corruption,
                            'text_length': len(text),
                            'text_preview': text[:200] + "..." if len(text) > 200 else text
                        })
                        
        except Exception as e:
            print(f"❌ Error reading chunks file: {e}")
            return None
        
        # Print results
        corruption_rate = len(corrupted_chunks) / total_chunks * 100 if total_chunks > 0 else 0
        
        print(f"📈 CORRUPTION ANALYSIS RESULTS:")
        print(f"   Total chunks analyzed: {total_chunks}")
        print(f"   Corrupted chunks: {len(corrupted_chunks)}")
        print(f"   Corruption rate: {corruption_rate:.1f}%")
        
        if corruption_stats:
            print(f"\n🔍 CORRUPTION PATTERNS FOUND:")
            for pattern, count in corruption_stats.most_common():
                print(f"   {pattern}: {count} occurrences")
        
        if corrupted_chunks:
            print(f"\n⚠️  SAMPLE CORRUPTED CHUNKS:")
            for i, chunk in enumerate(corrupted_chunks[:3]):  # Show first 3
                print(f"\n   Chunk {i+1}:")
                print(f"   ID: {chunk['chunk_id']}")
                print(f"   Source: {chunk['source']}")
                print(f"   Corruption: {list(chunk['corruption'].keys())}")
                print(f"   Preview: {chunk['text_preview'][:100]}...")
        
        return corrupted_chunks
    
    def test_live_rag_pipeline(self):
        """Test the complete RAG pipeline for corruption"""
        print(f"\n🔄 TESTING LIVE RAG PIPELINE")
        print("=" * 50)
        print(f"Question: '{self.test_question}'")
        
        try:
            # Step 1: Question classification
            question_type = classify_question_type(self.test_question)
            print(f"✅ Question type: {question_type}")
            
            # Step 2: Chunk retrieval
            chunks, scores = get_adaptive_chunks(self.test_question, question_type)
            print(f"✅ Retrieved {len(chunks)} chunks with scores: {[f'{s:.3f}' for s in scores]}")
            
            # Step 3: Load chunk texts
            chunk_texts = load_text_for_chunks(chunks, CHUNK_FILE)
            print(f"✅ Loaded {len(chunk_texts)} chunk texts")
            
            # Step 4: Analyze each chunk for corruption
            print(f"\n🔍 ANALYZING RETRIEVED CHUNKS:")
            for i, (chunk, text, score) in enumerate(zip(chunks, chunk_texts, scores)):
                print(f"\n--- CHUNK {i+1} (Score: {score:.3f}) ---")
                print(f"Source: {chunk.get('source', 'unknown')}")
                print(f"Chunk ID: {chunk.get('chunk_id', 'unknown')}")
                print(f"Text length: {len(text)} chars")
                
                # Check for corruption
                corruption_found = []
                for pattern_name, pattern in self.corruption_patterns.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        corruption_found.append(pattern_name)
                
                if corruption_found:
                    print(f"❌ CORRUPTION: {corruption_found}")
                    print(f"First 200 chars: '{text[:200]}...'")
                    
                    # Show specific corruption examples
                    for pattern_name in corruption_found:
                        pattern = self.corruption_patterns[pattern_name]
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            print(f"   {pattern_name}: {matches}")
                else:
                    print("✅ No corruption detected")
                    print(f"Preview: '{text[:100]}...'")
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline test: {e}")
            import traceback
            traceback.print_exc()
    
    def check_faiss_metadata(self):
        """Check FAISS metadata for issues"""
        print(f"\n🗂️  CHECKING FAISS METADATA")
        print("=" * 40)
        
        try:
            print(f"FAISS index size: {faiss_store.index.ntotal} vectors")
            print(f"Metadata entries: {len(faiss_store.metadata)}")
            
            if len(faiss_store.metadata) > 0:
                sample_metadata = faiss_store.metadata[:3]
                print(f"\nSample metadata entries:")
                for i, meta in enumerate(sample_metadata):
                    print(f"  {i+1}: {meta}")
            
            # Check for metadata consistency
            if faiss_store.index.ntotal != len(faiss_store.metadata):
                print(f"⚠️  Mismatch: {faiss_store.index.ntotal} vectors but {len(faiss_store.metadata)} metadata entries")
            else:
                print("✅ Vector count matches metadata count")
                
        except Exception as e:
            print(f"❌ Error checking FAISS metadata: {e}")
    
    def test_embedding_process(self):
        """Test if corruption happens during embedding"""
        print(f"\n🧮 TESTING EMBEDDING PROCESS")
        print("=" * 40)
        
        test_texts = [
            "Object recognition is important for perception.",
            "Shape perception helps us understand objects.", 
            "Effects on memory include spatial reasoning.",
            "Creative thinking involves mental rotation."
        ]
        
        corrupted_texts = [
            "ct recognition is important for perception.",
            "ape perception helps us understand objects.",
            "ects on memory include spatial reasoning.", 
            "ative thinking involves mental rotation."
        ]
        
        try:
            # Test clean embeddings
            clean_embeddings = model.encode(test_texts)
            print(f"✅ Clean text embeddings: {clean_embeddings.shape}")
            
            # Test corrupted embeddings  
            corrupt_embeddings = model.encode(corrupted_texts)
            print(f"✅ Corrupted text embeddings: {corrupt_embeddings.shape}")
            
            # Test similarity between clean and corrupted
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(clean_embeddings, corrupt_embeddings)
            print(f"\nSimilarity between clean and corrupted texts:")
            for i, sim in enumerate(similarities.diagonal()):
                print(f"  Text {i+1}: {sim.item():.3f}")
            
        except Exception as e:
            print(f"❌ Error in embedding test: {e}")
    
    def comprehensive_audit(self):
        """Run complete corruption audit"""
        print("🔍 COMPREHENSIVE CORRUPTION AUDIT")
        print("=" * 60)
        
        # 1. Quick test
        corruption_source = self.quick_corruption_test()
        
        # 2. Detailed chunks analysis
        self.analyze_chunks_file()
        
        # 3. Live RAG pipeline test
        self.test_live_rag_pipeline()
        
        # 4. FAISS metadata check
        self.check_faiss_metadata()
        
        # 5. Embedding process test
        self.test_embedding_process()
        
        # 6. Final recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 30)
        
        if corruption_source == "chunks_file":
            print("❌ CORRUPTION IS IN YOUR chunks.jsonl FILE")
            print("   → You need to re-process your source documents")
            print("   → Check your document chunking pipeline")
            print("   → Verify source document encoding")
        elif corruption_source == "retrieval":
            print("❌ CORRUPTION HAPPENS DURING RETRIEVAL")
            print("   → Check load_text_for_chunks() function")
            print("   → Verify JSON parsing in chunks file")
        elif corruption_source == "unknown":
            print("🤔 CORRUPTION SOURCE UNCLEAR")
            print("   → May need deeper investigation")
            print("   → Check document preprocessing pipeline")
        else:
            print("✅ NO OBVIOUS CORRUPTION DETECTED")
            print("   → Issue may be intermittent or query-specific")

def main():
    """Main function to run debugging"""
    print("🚀 Starting Psychology Tutor Corruption Debugging")
    print("=" * 60)
    
    analyzer = CorruptionAnalyzer()
    
    try:
        # Check if we can run the basic test
        if len(sys.argv) > 1 and sys.argv[1] == "--quick":
            # Just run quick test
            analyzer.quick_corruption_test()
        else:
            # Run comprehensive audit
            analyzer.comprehensive_audit()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Debugging interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
