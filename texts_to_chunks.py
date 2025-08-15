""" 
Fixed chunking module for academic content
Corrected document type detection and validation
"""
import os
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path

# Parameters
INPUT_DIR = "texts"
METADATA_DIR = "metadata"
OUTPUT_PATH = "chunks.jsonl"
CHUNK_SIZE = 500  # Target size in characters
MAX_CHUNK_SIZE = 600  # Hard limit
OVERLAP = 100     # Overlap in characters
MIN_CHUNK_SIZE = 100  # Don't create tiny chunks

# Academic section markers
ACADEMIC_SECTIONS = [
    'ABSTRACT', 'INTRODUCTION', 'BACKGROUND', 'LITERATURE REVIEW',
    'METHODS', 'METHODOLOGY', 'MATERIALS AND METHODS',
    'RESULTS', 'FINDINGS', 'ANALYSIS',
    'DISCUSSION', 'IMPLICATIONS',
    'CONCLUSION', 'CONCLUSIONS', 'SUMMARY',
    'REFERENCES', 'BIBLIOGRAPHY',
    'APPENDIX', 'APPENDICES'
]

class AcademicChunker:
    """Fixed chunker for academic content"""
    
    def __init__(self, chunk_size=500, overlap=100, min_chunk_size=100, max_chunk_size=600):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.stats = {
            'total_files': 0,
            'total_chunks': 0,
            'corrupted_chunks': 0,
            'section_preserved': 0,
            'data_chunks': 0,
            'by_type': {}
        }
    
    def load_metadata(self, filename: str) -> Dict:
        """Load metadata for a file if it exists"""
        base_name = filename.replace('.txt', '')
        metadata_path = os.path.join(METADATA_DIR, f"{base_name}_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                pass
        return {}
    
    def identify_document_type(self, text: str, filename: str, metadata: Dict) -> str:
        """Fixed document type identification"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check for data files first (by content patterns)
        if 'columns:' in text_lower[:500] or '=== sheet:' in text_lower[:500]:
            return 'data'
        
        # Check for presentation
        if '=== slide' in text_lower[:500]:
            return 'presentation'
        
        # Check filename patterns
        if any(x in filename_lower for x in ['data', 'alldata', '.csv', 'sartdata']):
            return 'data'
        
        # Check for research paper indicators (must have multiple sections)
        research_indicators = 0
        if 'abstract' in text_lower[:2000]:
            research_indicators += 1
        if any(x in text_lower for x in ['methods', 'methodology']):
            research_indicators += 1
        if 'results' in text_lower:
            research_indicators += 1
        if 'discussion' in text_lower:
            research_indicators += 1
        if 'references' in text_lower:
            research_indicators += 1
        
        if research_indicators >= 3:
            return 'research_paper'
        
        # Check for lab/instruction materials
        if any(x in filename_lower for x in ['lab', 'instruction', 'report']):
            return 'lab_instruction'
        
        # Check for code files
        if any(x in filename_lower for x in ['.py', '.psyexp', 'lastrun']):
            return 'code'
        
        # Default based on content
        if 'procedure' in text_lower or 'instructions' in text_lower:
            return 'instruction'
        
        return 'general'
    
    def find_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Find academic sections in the text"""
        sections = []
        
        for section in ACADEMIC_SECTIONS:
            # Look for section headers with various formats
            patterns = [
                rf'\n{section}\n',  # Standalone line
                rf'\n{section}:',   # With colon
                rf'\n\d+\.?\s*{section}',  # Numbered
                rf'^{section}\n',   # Start of document
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    sections.append((section, match.start(), match.end()))
        
        # Sort by position
        sections.sort(key=lambda x: x[1])
        return sections
    
    def chunk_with_sections(self, text: str, doc_type: str) -> List[str]:
        """Chunk text while preserving section boundaries when possible"""
        if doc_type == 'data':
            return self.chunk_data_content(text)
        
        sections = self.find_sections(text)
        
        if not sections or doc_type in ['presentation', 'code']:
            return self.chunk_by_sentences(text)
        
        chunks = []
        last_end = 0
        
        for i, (section_name, start, end) in enumerate(sections):
            # Get content before this section
            if start > last_end:
                pre_section = text[last_end:start].strip()
                if pre_section and len(pre_section) >= self.min_chunk_size:
                    section_chunks = self.chunk_by_sentences(pre_section)
                    chunks.extend(section_chunks)
            
            # Get section content
            if i < len(sections) - 1:
                section_end = sections[i + 1][1]
            else:
                section_end = len(text)
            
            section_content = text[start:section_end].strip()
            
            # If section is small enough, keep it as one chunk
            if self.min_chunk_size <= len(section_content) <= self.max_chunk_size:
                chunks.append(section_content)
                self.stats['section_preserved'] += 1
            elif len(section_content) > self.min_chunk_size:
                # Chunk the section
                section_chunks = self.chunk_by_sentences(section_content)
                chunks.extend(section_chunks)
            
            last_end = section_end
        
        # Get remaining content
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining and len(remaining) >= self.min_chunk_size:
                remaining_chunks = self.chunk_by_sentences(remaining)
                chunks.extend(remaining_chunks)
        
        return chunks
    
    def chunk_data_content(self, text: str) -> List[str]:
        """Special chunking for data/CSV content - fixed to handle large data"""
        chunks = []
        lines = text.split('\n')
        
        # Find and preserve header
        header = ""
        data_start = 0
        for i, line in enumerate(lines[:10]):  # Check first 10 lines for header
            if 'columns:' in line.lower() or any(c in line for c in [',', '\t', '|']):
                header = line + '\n'
                data_start = i + 1
                break
        
        # For very large data files, sample rather than chunk everything
        if len(lines) > 1000:
            # Take first chunk, middle chunk, and last chunk as samples
            sample_ranges = [
                (data_start, min(data_start + 20, len(lines))),
                (len(lines)//2 - 10, len(lines)//2 + 10),
                (max(0, len(lines) - 20), len(lines))
            ]
            
            for start, end in sample_ranges:
                chunk_lines = lines[start:end]
                chunk_text = header + '\n'.join(chunk_lines)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunks.append(chunk_text.strip())
                    self.stats['data_chunks'] += 1
        else:
            # For smaller data files, chunk normally
            current_chunk = header
            for line in lines[data_start:]:
                if len(current_chunk) + len(line) + 1 > self.chunk_size:
                    if len(current_chunk.strip()) >= self.min_chunk_size:
                        chunks.append(current_chunk.strip())
                        self.stats['data_chunks'] += 1
                    current_chunk = header + line
                else:
                    current_chunk += '\n' + line if current_chunk else line
            
            if len(current_chunk.strip()) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
                self.stats['data_chunks'] += 1
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk by sentences - fixed sentence detection"""
        # Better sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        if not sentences or len(sentences) == 1:
            # If no sentence breaks found, chunk by paragraphs or words
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 1:
                sentences = paragraphs
            else:
                return self.chunk_by_words(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = self.get_overlap_content(current_chunk, self.overlap) + sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_words(self, text: str) -> List[str]:
        """Chunk by words as fallback"""
        words = text.split()
        if len(words) < 20:  # Too short to chunk meaningfully
            if len(text.strip()) >= self.min_chunk_size:
                return [text.strip()]
            return []
        
        chunks = []
        current_chunk_words = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            
            if current_length + word_length > self.chunk_size and current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                
                # Calculate overlap
                overlap_words = self.calculate_word_overlap(current_chunk_words, self.overlap)
                current_chunk_words = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk_words) - 1
            else:
                current_chunk_words.append(word)
                current_length += word_length
        
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def get_overlap_content(self, text: str, target_overlap_chars: int) -> str:
        """Get overlap content"""
        if len(text) <= target_overlap_chars:
            return text + " "
        
        start_pos = len(text) - target_overlap_chars
        while start_pos > 0 and start_pos < len(text) and text[start_pos] != ' ':
            start_pos -= 1
        
        overlap = text[start_pos:].strip()
        return overlap + " " if overlap else ""
    
    def calculate_word_overlap(self, words: List[str], target_overlap_chars: int) -> List[str]:
        """Calculate word overlap"""
        if not words:
            return []
        
        overlap_words = []
        current_length = 0
        
        for word in reversed(words):
            word_length = len(word) + 1
            if current_length + word_length <= target_overlap_chars:
                overlap_words.insert(0, word)
                current_length += word_length
            else:
                break
        
        return overlap_words
    
    def clean_text(self, text: str) -> str:
        """Clean text before chunking"""
        # Fix common issues
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines
        text = re.sub(r'(?<=[a-z])-\s*\n\s*(?=[a-z])', '', text)  # Hyphenated line breaks
        # Don't collapse all spaces - keep structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        return text.strip()
    
    def validate_chunk(self, chunk: str) -> bool:
        """Fixed validation - less aggressive corruption detection"""
        # Only check for obvious OCR artifacts
        obvious_corruptions = [
            r'\b[a-z]\s+[a-z]\s+[a-z]\s+[a-z]\b',  # Single letters separated by spaces
            r'\b(?:ti|ta|te|tu|to)\s+on\b',  # Clear word breaks like "ti on" instead of "tion"
            r'\b(?:in|ing|ed|er)\s+[a-z]{1,2}\b',  # Clear suffix breaks
        ]
        
        for pattern in obvious_corruptions:
            if re.search(pattern, chunk, re.IGNORECASE):
                self.stats['corrupted_chunks'] += 1
                return False
        return True
    
    def process_file(self, file_path: str, filename: str) -> List[Dict]:
        """Process a single file into chunks"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            if not text.strip():
                print(f"  ⏭️  Skipping empty file: {filename}")
                return []
            
            # Clean text
            text = self.clean_text(text)
            
            # Load metadata
            metadata = self.load_metadata(filename)
            
            # Identify document type - FIXED
            doc_type = self.identify_document_type(text, filename, metadata)
            
            # Track document types
            self.stats['by_type'][doc_type] = self.stats['by_type'].get(doc_type, 0) + 1
            
            # Choose chunking strategy
            if doc_type == 'data':
                chunks = self.chunk_data_content(text)
            elif doc_type in ['research_paper']:
                chunks = self.chunk_with_sections(text, doc_type)
            else:
                chunks = self.chunk_by_sentences(text)
            
            # Filter by size
            chunks = [c for c in chunks if self.min_chunk_size <= len(c.strip()) <= self.max_chunk_size]
            
            # Create chunk objects
            chunk_objects = []
            for index, chunk in enumerate(chunks):
                is_valid = self.validate_chunk(chunk)
                
                chunk_obj = {
                    "source": filename,
                    "chunk_id": index,
                    "text": chunk,
                    "doc_type": doc_type,
                    "char_count": len(chunk),
                    "valid": is_valid
                }
                
                if metadata:
                    chunk_obj["source_extension"] = metadata.get("extension", "")
                    chunk_obj["source_file"] = metadata.get("source_file", "")
                
                chunk_objects.append(chunk_obj)
                self.stats['total_chunks'] += 1
            
            print(f"  ✅ {filename}: {len(chunks)} chunks (type: {doc_type})")
            self.stats['total_files'] += 1
            return chunk_objects
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            return []
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("📊 CHUNKING STATISTICS")
        print("="*60)
        print(f"Total files processed: {self.stats['total_files']}")
        print(f"Total chunks created: {self.stats['total_chunks']}")
        print(f"Sections preserved: {self.stats['section_preserved']}")
        print(f"Data chunks: {self.stats['data_chunks']}")
        
        print("\n📁 Document types detected:")
        for doc_type, count in sorted(self.stats['by_type'].items()):
            print(f"  {doc_type}: {count} files")
        
        if self.stats['total_chunks'] > 0:
            corruption_rate = (self.stats['corrupted_chunks'] / self.stats['total_chunks']) * 100
            print(f"\nCorruption rate: {corruption_rate:.1f}%")
            
            if corruption_rate < 5:
                print("✅ Excellent chunk quality!")
            elif corruption_rate < 20:
                print("⚠️  Some corruption detected")
            else:
                print("❌ High corruption rate")


def get_available_output_path(base_path: str) -> str:
    """Find available output path"""
    if not os.path.exists(base_path):
        return base_path
    
    try:
        with open(base_path, "a", encoding="utf-8") as test_file:
            pass
        
        # Create backup
        backup_path = base_path.replace('.jsonl', '_backup.jsonl')
        if os.path.exists(base_path):
            try:
                import shutil
                shutil.copy2(base_path, backup_path)
                print(f"📋 Created backup: {backup_path}")
            except:
                pass
        
        return base_path
        
    except (PermissionError, OSError):
        import time
        timestamp = int(time.time())
        fallback_path = f"{base_path.replace('.jsonl', '')}_{timestamp}.jsonl"
        print(f"✅ Using timestamped path: {fallback_path}")
        return fallback_path


def main():
    """Main function"""
    print("🚀 Starting fixed academic content chunking...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Metadata directory: {METADATA_DIR}")
    print(f"Target output: {OUTPUT_PATH}")
    print(f"Chunk size: {CHUNK_SIZE} chars (max: {MAX_CHUNK_SIZE})")
    print(f"Overlap: {OVERLAP} chars")
    print("-" * 60)
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory '{INPUT_DIR}' not found!")
        return
    
    # Initialize chunker
    chunker = AcademicChunker(
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        max_chunk_size=MAX_CHUNK_SIZE
    )
    
    # Get all text files
    txt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")])
    if not txt_files:
        print(f"❌ No .txt files found in {INPUT_DIR}")
        return
    
    print(f"📁 Found {len(txt_files)} text files to process\n")
    
    # Process all files
    all_chunks = []
    for filename in txt_files:
        file_path = os.path.join(INPUT_DIR, filename)
        chunks = chunker.process_file(file_path, filename)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print("❌ No chunks created!")
        return
    
    # Get output path
    actual_output_path = get_available_output_path(OUTPUT_PATH)
    
    # Write chunks
    print(f"\n💾 Writing {len(all_chunks)} chunks to {actual_output_path}...")
    try:
        with open(actual_output_path, "w", encoding="utf-8") as out_file:
            for chunk in all_chunks:
                out_file.write(json.dumps(chunk) + "\n")
        
        print(f"✅ Chunking complete!")
        
        # Print statistics
        chunker.print_stats()
        
        print(f"\n📁 Output saved to: {actual_output_path}")
        
        # Sample chunks
        print("\n📝 Sample chunks for verification:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"\nChunk {i} ({chunk['source']}, type: {chunk['doc_type']}):")
            print(f"  {chunk['text'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error writing chunks: {e}")


if __name__ == "__main__":
    main()