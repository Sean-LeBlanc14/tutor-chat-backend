#!/bin/bash
# Stop any existing container
docker stop tutor_chatbot_backend 2>/dev/null
docker rm tutor_chatbot_backend 2>/dev/null

# Start with GPU access and FAISS file mounts
docker run -d \
  --name tutor_chatbot_backend \
  --gpus all \
  --restart unless-stopped \
  -p 8080:8080 \
  --network host \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/chunk_index.faiss:/app/chunk_index.faiss \
  -v $(pwd)/chunk_metadata.json:/app/chunk_metadata.json \
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
docker exec tutor_chatbot_backend ls -la chunk_*.* || echo "FAISS files not found - add them after GitHub commit"

echo "Container logs:"
docker logs tutor_chatbot_backend --tail 20
