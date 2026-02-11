# Use Python 3.10-slim for smaller base image
FROM python:3.10-slim

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
# Using separate RUN command to ensure it installs before other packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow CPU (Saves ~500MB+ compared to full TF)
# Also helps ensure keras is compatible
RUN pip install --no-cache-dir tensorflow-cpu

# Copy the optimized requirements file
COPY requirements_cpu.txt requirements.txt

# Install remaining dependencies
# We use --no-deps for heavier libs if needed, but here pip resolves well usually
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
