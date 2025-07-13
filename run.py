#!/usr/bin/env python3
"""
Quick start script for MedXpert
Run this to start the application quickly
"""

import os
import subprocess
import sys

def check_python():
    """Check if Python 3 is available"""
    try:
        import sys
        if sys.version_info.major < 3:
            print("❌ Python 3 is required")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    except:
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    dirs = [
        "static/uploads",
        "static/uploads/chatbot_docs", 
        "models",
        "chroma_db"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created")

def start_app():
    """Start the Flask application"""
    print("🚀 Starting MedXpert application...")
    print("🌐 The app will be available at: http://localhost:5000")
    print("📱 In Codespaces, it will be available at the forwarded port URL")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Set environment variables
        os.environ["FLASK_APP"] = "app.py"
        os.environ["PORT"] = "5000"
        
        # Import and run the app
        from app import app
        app.run(host="0.0.0.0", port=5000, debug=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    print("🏥 MedXpert Quick Start")
    print("=" * 30)
    
    if not check_python():
        sys.exit(1)
    
    if not install_requirements():
        sys.exit(1)
    
    create_directories()
    start_app()

if __name__ == "__main__":
    main()
