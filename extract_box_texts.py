# Import necessary libraries for handling different file types
import os
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd  # For handling CSV and Excel files
from docx import Document  # For reading .docx Word files
from pptx import Presentation  # For reading .pptx PowerPoint files
from collections import defaultdict  # For handling duplicate files

# Set the input and output directories
INPUT_DIR = r"Dr.Mishra-materials"  # Folder containing source files
OUTPUT_DIR = r"texts"  # Folder to save extracted .txt files

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionary to track used filenames and avoid overwriting
used_names = defaultdict(int)

# Function to extract text from a file based on its extension
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()  # Get file extension in lowercase

    try:
        # Extract text from PDF using PyMuPDF
        if ext == '.pdf':
            doc = fitz.open(file_path)
            return "\n".join([page.get_text() for page in doc])

        # Extract text from Word .docx documents
        elif ext == '.docx':
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])

        # Extract text from CSV using pandas
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_string(index=False)

        # Extract text from all sheets in Excel files
        elif ext == '.xlsx':
            dfs = pd.read_excel(file_path, sheet_name=None)  # Load all sheets
            return "\n\n".join(
                f"Sheet: {name}\n{df.to_string(index=False)}" for name, df in dfs.items()
            )

        # Extract visible text from PowerPoint slides
        elif ext == '.pptx':
            prs = Presentation(file_path)
            return "\n".join(
                shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")
            )

        # Read plain text directly from config, Python, or PsychoPy experiment files
        elif ext in ['.conf', '.py', '.psyexp', '.txt']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
            
        # Attempt to extract text from .jasp file (which is a ZIP archive)
        elif ext == '.jasp':
            import zipfile
            import tempfile
            import json

            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_ref.extractall(tmpdir)

                        output_parts = []

                        # Extract data.csv if it exists
                        csv_path = os.path.join(tmpdir, 'data.csv')
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            output_parts.append("### Data\n" + df.to_string(index=False))

                        # Extract results.json if it exists
                        results_path = os.path.join(tmpdir, 'results.json')
                        if os.path.exists(results_path):
                            with open(results_path, 'r', encoding='utf-8') as f:
                                results = json.load(f)

                            # Flatten the results tree into readable lines
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

                        # Optionally extract log.json (analysis steps)
                        log_path = os.path.join(tmpdir, 'log.json')
                        if os.path.exists(log_path):
                            with open(log_path, 'r', encoding='utf-8') as f:
                                log = json.load(f)
                            log_text = json.dumps(log, indent=2)
                            output_parts.append("### Analysis Log (JSON format)\n" + log_text)

                        return "\n\n".join(output_parts)

            except Exception as e:
                print(f"[Error] Could not extract .jasp file {file_path}: {e}")
                return ""


        # Skip unsupported file types
        else:
            return ""

    # Catch and report any file-specific extraction errors
    except Exception as e:
        print(f"[Error] Failed to extract {file_path}: {e}")
        return ""

# Function to walk through all files in the input directory and process them
def process_folder(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            text = extract_text(file_path)  # Try extracting text

            if text.strip():  # Only save non-empty extractions
                base_name = os.path.splitext(file)[0]  # Get filename without extension
                ext_name = ".txt"
                final_name = base_name

                # Check for existing filenames and append _1, _2, etc. to avoid overwriting
                while os.path.exists(os.path.join(output_dir, final_name + ext_name)):
                    used_names[base_name] += 1
                    final_name = f"{base_name}_{used_names[base_name]}"

                out_path = os.path.join(output_dir, final_name + ext_name)  # Build final unique path
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved: {out_path}")  # Confirm file was written
            else:
                print(f"Skipped (empty or unsupported): {file_path}")  # Report skipped file

# Call process folder
process_folder(INPUT_DIR, OUTPUT_DIR)
