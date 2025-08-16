#!/bin/bash
# Stop any existing container
docker stop tutor_chatbot_backend 2>/dev/null
docker rm tutor_chatbot_backend 2>/dev/null

# Start with GPU access and UPDATED FAISS file mounts
docker run -d \
  --name tutor_chatbot_backend \
  --gpus all \
  --restart unless-stopped \
  -p 8080:8080 \
  --network host \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/faiss_index.bin:/app/faiss_index.bin \
  -v $(pwd)/faiss_metadata.pkl:/app/faiss_metadata.pkl \
  -v $(pwd)/chunks.jsonl:/app/chunks.jsonl \
  --health-cmd="curl -f http://localhost:8080/api/health" \
  --health-interval=60s \
  --health-timeout=60s \
  --health-retries=3 \
  tutor-chatbot_backend

echo "Container started with GPU access and FAISS files mounted"
echo "Testing GPU availability..."
sleep 10

# Test GPU access
docker exec tutor_chatbot_backend python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo "Testing vLLM setup..."
docker exec tutor_chatbot_backend python -c "
try:
    from vllm import LLM
    print('vLLM imported successfully')
except Exception as e:
    print(f'vLLM import error: {e}')
"

echo "Checking FAISS files..."
docker exec tutor_chatbot_backend ls -la faiss_* chunks.jsonl || echo "FAISS files not found - need to generate them"

echo "Testing FAISS loading..."
docker exec tutor_chatbot_backend python -c "
import pickle
import faiss
try:
    index = faiss.read_index('faiss_index.bin')
    with open('faiss_metadata.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'✅ FAISS index loaded: {index.ntotal} vectors')
    print(f'✅ Metadata loaded: {len(data.get(\"metadata\", []))} chunks')
except Exception as e:
    print(f'❌ FAISS loading error: {e}')
"

echo "Container logs:"
docker logs tutor_chatbot_backend --tail 20
