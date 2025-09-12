#!/usr/bin/env python3
"""
Script to inspect FAISS index metadata and understand the structure
Run this to diagnose metadata issues before fixing retrieval
"""

import pickle
import json
from collections import Counter, defaultdict
import re

def inspect_faiss_metadata(metadata_path="faiss_metadata.pkl"):
    """Inspect the FAISS metadata to understand structure and content"""
    
    print("=" * 80)
    print("FAISS METADATA INSPECTION")
    print("=" * 80)
    
    try:
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            metadata = data.get('metadata', [])
            texts = data.get('texts', [])
            stats = data.get('stats', {})
        
        print(f"\n✅ Successfully loaded metadata from {metadata_path}")
        print(f"📊 Total chunks: {len(metadata)}")
        print(f"📝 Total texts: {len(texts)}")
        
        # Analyze metadata structure
        print("\n" + "=" * 40)
        print("METADATA STRUCTURE ANALYSIS")
        print("=" * 40)
        
        if metadata:
            # Check first few metadata entries
            print("\n🔍 Sample metadata entries (first 3):")
            for i, meta in enumerate(metadata[:3]):
                print(f"\n  Entry {i}:")
                for key, value in meta.items():
                    if key == 'text':
                        print(f"    {key}: {str(value)[:100]}...")
                    else:
                        print(f"    {key}: {value}")
            
            # Analyze all metadata keys
            all_keys = set()
            for meta in metadata:
                all_keys.update(meta.keys())
            
            print(f"\n📋 All metadata keys found: {sorted(all_keys)}")
            
            # Analyze doc_type distribution
            doc_types = Counter()
            for meta in metadata:
                doc_types[meta.get('doc_type', 'MISSING')] += 1
            
            print(f"\n📊 Document type distribution:")
            for doc_type, count in doc_types.most_common():
                print(f"    {doc_type}: {count} chunks ({count/len(metadata)*100:.1f}%)")
            
            # Analyze sources
            sources = Counter()
            source_to_type = defaultdict(set)
            for meta in metadata:
                source = meta.get('source', 'UNKNOWN')
                doc_type = meta.get('doc_type', 'UNKNOWN')
                sources[source] += 1
                source_to_type[source].add(doc_type)
            
            print(f"\n📁 Source files ({len(sources)} unique files):")
            print("    Top 10 sources by chunk count:")
            for source, count in sources.most_common(10):
                doc_types_str = ', '.join(source_to_type[source])
                print(f"      {source}: {count} chunks (types: {doc_types_str})")
            
            # Look for assignment-related content
            print("\n" + "=" * 40)
            print("ASSIGNMENT CONTENT ANALYSIS")
            print("=" * 40)
            
            assignment_sources = []
            lab_sources = []
            
            for source in sources.keys():
                source_lower = source.lower()
                if 'assignment' in source_lower:
                    assignment_sources.append(source)
                if 'lab' in source_lower:
                    lab_sources.append(source)
            
            print(f"\n🔬 Assignment-related sources:")
            if assignment_sources:
                for source in assignment_sources:
                    print(f"    ✓ {source} ({sources[source]} chunks)")
            else:
                print("    ⚠️ No sources with 'assignment' in name")
            
            print(f"\n🧪 Lab-related sources:")
            if lab_sources:
                for source in lab_sources:
                    print(f"    ✓ {source} ({sources[source]} chunks)")
            else:
                print("    ⚠️ No sources with 'lab' in name")
            
            # Search for specific assignment content
            print("\n🔍 Searching for 'signal detection' content:")
            signal_detection_chunks = []
            for i, meta in enumerate(metadata):
                text = texts[i] if i < len(texts) else ""
                if 'signal detection' in text.lower() or 'sdt' in text.lower():
                    signal_detection_chunks.append({
                        'source': meta.get('source', 'UNKNOWN'),
                        'doc_type': meta.get('doc_type', 'UNKNOWN'),
                        'chunk_id': meta.get('chunk_id', 'UNKNOWN'),
                        'text_preview': text[:150]
                    })
            
            if signal_detection_chunks:
                print(f"    Found {len(signal_detection_chunks)} chunks with signal detection content")
                print("    First 3 chunks:")
                for chunk in signal_detection_chunks[:3]:
                    print(f"\n      Source: {chunk['source']}")
                    print(f"      Type: {chunk['doc_type']}")
                    print(f"      Preview: {chunk['text_preview']}...")
            else:
                print("    ⚠️ No chunks found with 'signal detection' content")
            
            # Check for Lab 1 specifically
            print("\n🔍 Searching for 'Lab 1' content:")
            lab1_chunks = []
            for i, meta in enumerate(metadata):
                source = meta.get('source', '').lower()
                text = texts[i] if i < len(texts) else ""
                text_lower = text.lower()
                
                if ('lab 1' in source or 'lab_1' in source or 'lab1' in source or
                    'lab 1' in text_lower or 'lab1' in text_lower):
                    lab1_chunks.append({
                        'source': meta.get('source', 'UNKNOWN'),
                        'doc_type': meta.get('doc_type', 'UNKNOWN'),
                        'text_preview': text[:150]
                    })
            
            if lab1_chunks:
                print(f"    Found {len(lab1_chunks)} chunks related to Lab 1")
                for chunk in lab1_chunks[:3]:
                    print(f"\n      Source: {chunk['source']}")
                    print(f"      Type: {chunk['doc_type']}")
                    print(f"      Preview: {chunk['text_preview']}...")
            else:
                print("    ⚠️ No chunks found for Lab 1")
            
            # Recommendations
            print("\n" + "=" * 40)
            print("RECOMMENDATIONS")
            print("=" * 40)
            
            issues = []
            
            if 'MISSING' in doc_types or 'UNKNOWN' in doc_types:
                issues.append("❌ Some chunks are missing doc_type metadata")
            
            if not assignment_sources and not lab_sources:
                issues.append("❌ No assignment or lab documents detected in sources")
            
            if not signal_detection_chunks:
                issues.append("❌ No signal detection content found in chunks")
            
            if 'doc_type' not in all_keys:
                issues.append("❌ doc_type field is completely missing from metadata")
            
            if issues:
                print("\n⚠️ Issues found:")
                for issue in issues:
                    print(f"  {issue}")
                print("\n💡 You may need to re-index your documents with proper metadata tagging")
            else:
                print("\n✅ Metadata structure looks good!")
                print("  The issue might be in the retrieval logic rather than indexing")
            
        else:
            print("\n❌ No metadata found in the file!")
            
    except FileNotFoundError:
        print(f"\n❌ File not found: {metadata_path}")
        print("  Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"\n❌ Error loading metadata: {e}")

if __name__ == "__main__":
    # Run the inspection
    inspect_faiss_metadata()
    
    # Also check if chunks.jsonl exists
    print("\n" + "=" * 40)
    print("CHECKING CHUNKS FILE")
    print("=" * 40)
    
    try:
        with open("chunks.jsonl", 'r') as f:
            lines = f.readlines()
            print(f"✅ chunks.jsonl found with {len(lines)} chunks")
            
            # Check first chunk structure
            if lines:
                first_chunk = json.loads(lines[0])
                print("\n🔍 First chunk structure:")
                for key in first_chunk.keys():
                    if key == 'text':
                        print(f"    {key}: {str(first_chunk[key])[:100]}...")
                    else:
                        print(f"    {key}: {first_chunk[key]}")
    except FileNotFoundError:
        print("⚠️ chunks.jsonl not found")
    except Exception as e:
        print(f"❌ Error reading chunks.jsonl: {e}")
