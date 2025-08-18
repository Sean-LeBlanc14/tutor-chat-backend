# chunk_splitter.py
# Split already-clean texts/ into chunks.jsonl — no registry logic here.

import os, json, re
from typing import List, Dict

INPUT_DIR   = "texts"
OUTPUT_PATH = "chunks.jsonl"
CHUNK_SIZE  = 500
MAX_CHUNK   = 650
MIN_CHUNK   = 120
OVERLAP     = 80

def split_sentences(text: str) -> List[str]:
    # simple sentence splitter that respects caps start
    sents = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z])', text.strip())
    if len(sents) <= 1:
        # fallback to paragraphs
        sents = [p.strip() for p in text.split("\n\n") if p.strip()]
    return sents

def chunk_text(text: str) -> List[str]:
    sents = split_sentences(text)
    chunks=[]
    cur=""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > CHUNK_SIZE:
            if len(cur) >= MIN_CHUNK:
                chunks.append(cur.strip())
            # overlap tail words
            tail = cur[-OVERLAP:]
            cur = tail + " " + s
        else:
            cur = (cur + " " + s) if cur else s
        if len(cur) > MAX_CHUNK and len(cur) >= MIN_CHUNK:
            chunks.append(cur.strip())
            cur = ""
    if cur.strip() and len(cur) >= MIN_CHUNK:
        chunks.append(cur.strip())
    return chunks

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt")]
    files.sort()
    all_objs=[]
    total=0
    for f in files:
        p = os.path.join(INPUT_DIR,f)
        with open(p,"r",encoding="utf-8",errors="ignore") as fh:
            text = fh.read()
        parts = chunk_text(text)
        for i,ck in enumerate(parts):
            all_objs.append({
                "source": f,
                "chunk_id": i,
                "text": ck,
                "doc_type": "general",
                "char_count": len(ck),
                "valid": True
            })
        total += len(parts)

    with open(OUTPUT_PATH,"w",encoding="utf-8") as out:
        for obj in all_objs:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("============================================================")
    print("📊 CHUNKING SUMMARY")
    print("============================================================")
    print(f"Files processed: {len(files)}")
    print(f"Total chunks: {total}")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
