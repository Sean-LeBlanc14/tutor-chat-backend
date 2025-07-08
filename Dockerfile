# Use official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Optional: install system dependencies for some packages (e.g., PyMuPDF, lxml)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set env vars (optional)
ENV PORT=8080

# Tell Docker which port your app listens on
EXPOSE 8080

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
