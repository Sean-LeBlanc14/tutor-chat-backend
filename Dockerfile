# Use official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (for things like PyMuPDF, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies early to cache layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (optional in dev, but okay to keep)
COPY . .

# Environment variables
ENV PORT=8080

# Expose port
EXPOSE 8080

# Use Uvicorn in dev mode with --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
