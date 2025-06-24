# Import necessary libraries for handling different file types
import os
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd  # For handling CSV and Excel files
from docx import Document  # For reading .docx Word files
from pptx import Presentation  # For reading .pptx PowerPoint files

# Set the input and output directories
INPUT_DIR = r"C:\Users\SeanA\Dr.Mishra-materials"  # Folder containing source files
OUTPUT_DIR = r"C:\Users\SeanA\TutorChatBot\texts"  # Folder to save extracted .txt files

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        elif ext in ['.conf', '.py', '.psyexp']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

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
                out_path = os.path.join(output_dir, base_name + ".txt")  # Build .txt path
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved: {out_path}")  # Confirm file was written
            else:
                print(f"Skipped (empty or unsupported): {file_path}")  # Report skipped file

#Call process folder
process_folder(INPUT_DIR, OUTPUT_DIR)