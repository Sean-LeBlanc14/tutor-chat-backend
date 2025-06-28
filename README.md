# 🤖 Tutor Chatbot

Tutor Chatbot is a research-driven AI assistant designed to help students learn more effectively by responding to user questions using course materials embedded via vector search. It uses prompt engineering and retrieval techniques to simulate Socratic-style tutoring.

---

## 🚧 Current Status

- ✅ Able to generate responses using a FAISS index and SentenceTransformer embeddings  
- ❌ No front-end yet (currently backend-only)  
- 📄 Prompts are currently stored as `.txt` files and used to shape response behavior  

---

## 🎯 Planned Features

- 🌐 Build a user-friendly front-end interface  
- 👥 Scale to support multiple users  
- 🧠 Improve prompt structure and retrieval logic for more efficient and accurate tutoring  

---

## 🧪 Tech Stack

- **FAISS** – fast similarity search on vectorized text  
- **SentenceTransformers** – for embedding natural language text  
- **NumPy**, **Pandas** – for data manipulation  
- **Requests**, **OS**, **dotenv** – for API access and environment handling  
- **PyMuPDF (fitz)** – for PDF processing  
- **python-docx**, **python-pptx** – for handling Word and PowerPoint files  
- **JSON** – for metadata, prompts, and serialized data structures  
- **collections** – for efficient data structures  

---

## 📂 Repository Structure

TutorChatBot/
│
├── texts/ # Folder for .txt prompt files
├── chunks.jsonl # All document chunks as JSON lines
├── chunk_index.faiss # FAISS index of embedded chunks
├── chunk_metadata.json # Metadata mapping for each chunk
├── query_bot.py # Main script to answer queries using the index
├── texts_to_chunks.py # Converts raw .txt files into chunked format
├── extract_box_texts.py # Extracts and processes files from Box directory
├── embed_chunks.py # Embeds chunked data into the FAISS index
├── .env # API keys and environment variables
└── README.md # You're here!