""" 
Fixed text extraction module - accepts all content and handles long paths
"""
import os
import re
import json
import hashlib
from datetime import datetime
from collections import defaultdict
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
from docx import Document
from pptx import Presentation
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = r"Dr.Mishra-materials"
OUTPUT_DIR = r"texts"
METADATA_DIR = r"metadata"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# File type categories
UNSUPPORTED_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.mp3', '.wav', '.m4a', '.flac', '.ogg'
]

# Track used filenames
used_names = defaultdict(int)

class TextExtractor:
    """Main text extraction class with metadata tracking"""
    
    def __init__(self, output_dir: str, metadata_dir: str):
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir
        self.extraction_stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'unsupported': 0,
            'by_type': defaultdict(int)
        }
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean up common PDF extraction artifacts"""
        if not text:
            return ""
        
        # Fix common OCR artifacts
        ocr_fixes = {
            r'\bA B S T R A C T\b': 'ABSTRACT',
            r'\bS T R A C T\b': 'ABSTRACT',
            r'\bI N T R O D U C T I O N\b': 'INTRODUCTION',
            r'\bM E T H O D S\b': 'METHODS',
            r'\bR E S U L T S\b': 'RESULTS',
            r'\bD I S C U S S I O N\b': 'DISCUSSION',
            r'\bC O N C L U S I O N\b': 'CONCLUSION',
            r'\bR E F E R E N C E S\b': 'REFERENCES',
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Fix spacing issues from OCR
        text = re.sub(r'\b([a-z])\s+([a-z])\s+([a-z])\s+([a-z])\b', r'\1\2\3\4', text)
        text = re.sub(r'\bﬀ\b', 'ff', text)
        text = re.sub(r'\bﬁ\b', 'fi', text)
        text = re.sub(r'\bfl\b', 'fl', text)
        text = re.sub(r'\bﬃ\b', 'ffi', text)
        
        # Fix broken words with hyphens
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
        
        # Fix multiple spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove standalone single letters at start of lines
        text = re.sub(r'\n[a-z]\s+', '\n', text)
        
        return text.strip()
    
    def extract_pdf_improved(self, file_path: str) -> Tuple[str, Dict]:
        """Improved PDF extraction with metadata"""
        metadata = {
            'extraction_method': None,
            'page_count': 0,
        }
        
        try:
            doc = fitz.open(file_path)
            metadata['page_count'] = len(doc)
            
            # Try multiple extraction methods
            extracted_texts = []
            
            # Method 1: Standard text extraction
            standard_text = "\n".join([page.get_text() for page in doc])
            if standard_text.strip():
                extracted_texts.append(("standard", standard_text))
            
            # Method 2: Block-based extraction (better for columns)
            block_text = ""
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                            block_text += "\n"
                        block_text += "\n"
            
            if block_text.strip():
                extracted_texts.append(("blocks", block_text))
            
            # Choose the best extraction (longest with least artifacts)
            best_text = ""
            best_score = 0
            best_method = "none"
            
            for method, text in extracted_texts:
                cleaned = self.clean_extracted_text(text)
                
                # Simple scoring based on length
                score = len(cleaned)
                
                # Penalize for obvious OCR problems
                score -= cleaned.count("S T R A C T") * 100
                
                if score > best_score:
                    best_score = score
                    best_text = cleaned
                    best_method = method
            
            metadata['extraction_method'] = best_method
            doc.close()
            return best_text, metadata
            
        except Exception as e:
            print(f"Error extracting PDF {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def extract_excel(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from Excel files with metadata"""
        metadata = {
            'sheet_count': 0,
            'total_rows': 0,
            'total_columns': 0,
        }
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            metadata['sheet_count'] = len(excel_file.sheet_names)
            
            text_parts = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                metadata['total_rows'] += len(df)
                metadata['total_columns'] = max(metadata['total_columns'], len(df.columns))
                
                # Convert to text with formatting
                text_parts.append(f"=== Sheet: {sheet_name} ===\n")
                text_parts.append("Columns: " + ", ".join(str(col) for col in df.columns) + "\n\n")
                
                # Convert data to string, handling NaN values
                df_clean = df.fillna('')
                text_parts.append(df_clean.to_string(index=False, max_rows=1000))
                text_parts.append("\n\n")
            
            return "\n".join(text_parts), metadata
            
        except Exception as e:
            print(f"Error extracting Excel {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def extract_docx(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from Word documents with metadata"""
        metadata = {'paragraph_count': 0}
        
        try:
            doc = Document(file_path)
            text_parts = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
                    metadata['paragraph_count'] += 1
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = '\t'.join(cell.text for cell in row.cells)
                    text_parts.append(row_text)
            
            return self.clean_extracted_text("\n".join(text_parts)), metadata
            
        except Exception as e:
            print(f"Error extracting DOCX {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def extract_csv(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from CSV files with metadata"""
        metadata = {'row_count': 0, 'column_count': 0}
        
        try:
            df = pd.read_csv(file_path)
            metadata['row_count'] = len(df)
            metadata['column_count'] = len(df.columns)
            
            # Convert to string with proper formatting
            text = f"Columns: {', '.join(df.columns)}\n\n"
            text += df.to_string(index=False, max_rows=1000)
            
            return text, metadata
            
        except Exception as e:
            print(f"Error extracting CSV {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def extract_pptx(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from PowerPoint presentations with metadata"""
        metadata = {'slide_count': 0}
        
        try:
            prs = Presentation(file_path)
            metadata['slide_count'] = len(prs.slides)
            
            text_parts = []
            
            for i, slide in enumerate(prs.slides, 1):
                text_parts.append(f"\n=== Slide {i} ===\n")
                
                # Extract text from shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text_parts.append(shape.text)
                
                # Extract notes
                if slide.notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text
                    if notes.strip():
                        text_parts.append(f"[Notes: {notes}]")
            
            return self.clean_extracted_text("\n".join(text_parts)), metadata
            
        except Exception as e:
            print(f"Error extracting PPTX {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def extract_text_file(self, file_path: str) -> Tuple[str, Dict]:
        """Extract text from plain text files with metadata"""
        metadata = {'encoding': 'utf-8'}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return self.clean_extracted_text(text), metadata
            
        except Exception as e:
            print(f"Error extracting text file {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def extract_jasp(self, file_path: str) -> Tuple[str, Dict]:
        """Extract data from JASP files"""
        metadata = {'has_data': False, 'has_results': False}
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_ref.extractall(tmpdir)
                    
                    output_parts = []
                    
                    # Extract data.csv if it exists
                    csv_path = os.path.join(tmpdir, 'data.csv')
                    if os.path.exists(csv_path):
                        metadata['has_data'] = True
                        df = pd.read_csv(csv_path)
                        output_parts.append("### Data\n" + df.to_string(index=False, max_rows=100))
                    
                    # Extract results.json if it exists
                    results_path = os.path.join(tmpdir, 'results.json')
                    if os.path.exists(results_path):
                        metadata['has_results'] = True
                        with open(results_path, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        
                        def walk_results(obj, path=""):
                            lines = []
                            if isinstance(obj, dict):
                                for key, val in obj.items():
                                    lines.extend(walk_results(val, f"{path}/{key}" if path else key))
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    lines.extend(walk_results(item, f"{path}[{i}]"))
                            else:
                                if isinstance(obj, (str, int, float)) and str(obj).strip():
                                    lines.append(f"{path}: {obj}")
                            return lines
                        
                        extracted = walk_results(results)
                        output_parts.append("### Results Summary\n" + "\n".join(extracted[:100]))
                    
                    return "\n\n".join(output_parts), metadata
                    
        except Exception as e:
            print(f"Error extracting JASP {os.path.basename(file_path)}: {e}")
            return "", metadata
    
    def simple_validate(self, text: str, filename: str) -> Tuple[bool, Dict]:
        """Simple validation - just check if text exists"""
        metrics = {
            'char_count': len(text.strip()),
            'word_count': len(text.split()) if text.strip() else 0
        }
        
        # Accept ANY non-empty text
        is_valid = len(text.strip()) > 0
        
        return is_valid, metrics
    
    def safe_file_hash(self, file_path: str) -> str:
        """Generate file hash with error handling"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                buf = f.read(65536)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(65536)
            return hasher.hexdigest()
        except:
            return "hash_error"
    
    def extract_text(self, file_path: str) -> Tuple[str, Dict]:
        """Main extraction method that routes to appropriate extractor"""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        # Initialize metadata with safe file operations
        metadata = {
            'source_file': file_path,
            'filename': filename,
            'extension': ext,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Safely get file size
        try:
            metadata['file_size'] = os.path.getsize(file_path)
            metadata['file_hash'] = self.safe_file_hash(file_path)
        except:
            metadata['file_size'] = 0
            metadata['file_hash'] = "access_error"
            print(f"⚠️  Cannot access file: {filename} (path too long or file missing)")
            return "", metadata
        
        try:
            # Route to appropriate extractor
            if ext == '.pdf':
                text, extract_meta = self.extract_pdf_improved(file_path)
            elif ext == '.docx':
                text, extract_meta = self.extract_docx(file_path)
            elif ext in ['.xlsx', '.xls']:
                text, extract_meta = self.extract_excel(file_path)
            elif ext == '.csv':
                text, extract_meta = self.extract_csv(file_path)
            elif ext == '.pptx':
                text, extract_meta = self.extract_pptx(file_path)
            elif ext in ['.txt', '.conf', '.py', '.psyexp']:
                text, extract_meta = self.extract_text_file(file_path)
            elif ext == '.jasp':
                text, extract_meta = self.extract_jasp(file_path)
            else:
                return "", metadata
            
            # Merge extraction metadata
            metadata.update(extract_meta)
            
            return text, metadata
            
        except Exception as e:
            print(f"[Error] Failed to extract {filename}: {e}")
            metadata['error'] = str(e)
            return "", metadata
    
    def save_extracted_content(self, text: str, metadata: Dict, base_name: str) -> str:
        """Save extracted text and metadata"""
        # Clean filename for Windows
        base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        base_name = base_name[:100]  # Limit length
        
        # Generate unique filename
        final_name = base_name
        while os.path.exists(os.path.join(self.output_dir, final_name + ".txt")):
            used_names[base_name] += 1
            final_name = f"{base_name}_{used_names[base_name]}"
        
        # Save text
        text_path = os.path.join(self.output_dir, final_name + ".txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, final_name + "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return text_path
    
    def process_folder(self, input_dir: str):
        """Process all files in the input directory"""
        all_files = []
        
        # Collect all files
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip if path is too long for Windows
                if len(file_path) > 250:
                    print(f"⚠️  Skipping (path too long): {file}")
                    continue
                all_files.append(file_path)
        
        print(f"Found {len(all_files)} files to process")
        
        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Processing files"):
            ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)
            
            self.extraction_stats['total_files'] += 1
            self.extraction_stats['by_type'][ext] += 1
            
            # Skip unsupported files silently
            if ext in UNSUPPORTED_EXTENSIONS:
                self.extraction_stats['unsupported'] += 1
                continue
            
            # Extract text
            text, metadata = self.extract_text(file_path)
            
            if text.strip():
                # Simple validation
                is_valid, validation_metrics = self.simple_validate(text, filename)
                metadata['validation_metrics'] = validation_metrics
                
                if is_valid:
                    # Save content
                    base_name = os.path.splitext(filename)[0]
                    saved_path = self.save_extracted_content(text, metadata, base_name)
                    
                    print(f"✅ {filename} -> {os.path.basename(saved_path)} ({len(text)} chars)")
                    self.extraction_stats['successful'] += 1
            else:
                # Only show warning for supported file types that failed
                if ext not in UNSUPPORTED_EXTENSIONS:
                    print(f"⚠️  {filename}: Empty extraction")
                self.extraction_stats['failed'] += 1
    
    def print_summary(self):
        """Print extraction summary statistics"""
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total files processed: {self.extraction_stats['total_files']}")
        print(f"Successfully extracted: {self.extraction_stats['successful']}")
        print(f"Failed/Empty: {self.extraction_stats['failed']}")
        print(f"Unsupported formats: {self.extraction_stats['unsupported']}")
        
        if self.extraction_stats['total_files'] > 0:
            supported_files = self.extraction_stats['total_files'] - self.extraction_stats['unsupported']
            if supported_files > 0:
                success_rate = (self.extraction_stats['successful'] / supported_files) * 100
                print(f"Success rate (excluding unsupported): {success_rate:.1f}%")
        
        print("\n📁 Files by type:")
        for ext, count in sorted(self.extraction_stats['by_type'].items()):
            status = "✅" if ext not in UNSUPPORTED_EXTENSIONS else "⏭️"
            print(f"  {status} {ext}: {count} files")
        
        # Save summary to file
        summary_path = os.path.join(self.metadata_dir, "extraction_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.extraction_stats, f, indent=2)
        print(f"\n💾 Summary saved to: {summary_path}")


def main():
    """Main execution function"""
    print("🚀 Starting text extraction...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata directory: {METADATA_DIR}\n")
    
    # Check if openpyxl is installed
    try:
        import openpyxl
        print("✅ openpyxl is installed\n")
    except ImportError:
        print("❌ openpyxl is not installed!")
        print("Please run: pip install openpyxl\n")
        return
    
    # Initialize extractor
    extractor = TextExtractor(OUTPUT_DIR, METADATA_DIR)
    
    # Process all files
    extractor.process_folder(INPUT_DIR)
    
    # Print summary
    extractor.print_summary()
    
    print("\n✅ Extraction complete!")
    print("\n📝 Next steps:")
    print("1. Review extracted texts in:", OUTPUT_DIR)
    print("2. Check metadata in:", METADATA_DIR)
    print("3. Run your chunking script")
    print("4. Create embeddings for FAISS vector store")


if __name__ == "__main__":
    main()