#!/bin/bash

# MedXpert Startup Script for GitHub Codespaces

echo "🏥 Starting MedXpert Application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

# Install dependencies if not already installed
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p static/uploads
mkdir -p static/uploads/chatbot_docs
mkdir -p models
mkdir -p chroma_db

# Set environment variables for Codespaces
export FLASK_APP=app.py
export FLASK_ENV=development
export PORT=5000

# Start the Flask application
echo "🚀 Starting Flask application on port $PORT..."
echo "🌐 Your app will be available at the forwarded port URL"
python3 app.py
