#!/bin/bash

# Stop any existing container
docker stop tutor_chatbot_backend 2>/dev/null
docker rm tutor_chatbot_backend 2>/dev/null

# Start with GPU access
docker run -d \
  --name tutor_chatbot_backend \
  --gpus all \
  --restart unless-stopped \
  -p 8080:8080 \
  --network host \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  --health-cmd="curl -f http://localhost:8080/api/health" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  tutor-chatbot_backend

echo "Container started with GPU access"
echo "Testing GPU availability..."
sleep 5
docker exec -it tutor_chatbot_backend python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
