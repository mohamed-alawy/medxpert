# Use Python 3.10-slim for optimal size
FROM python:3.10-slim

# Set working directory to /app (Hugging Face standard)
WORKDIR /app

# Install system dependencies (minimized)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (Avoids downloading CUDA libs, saves ~2-3GB)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow CPU (Saves ~500MB+ compared to full TF)
RUN pip install --no-cache-dir tensorflow-cpu

# Copy the optimized requirements file
COPY requirements_cpu.txt requirements.txt

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories explicitly inside the container
RUN mkdir -p static/uploads

# Create a writable directory for the database if needed (Hugging Face specific)
RUN mkdir -p /data && chmod 777 /data

# Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# Run the app on host 0.0.0.0 and port 7860
CMD ["python", "app.py"]
