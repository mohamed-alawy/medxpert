# MedXpert - AI-Powered Medical Imaging Analysis Platform 🏥

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

## 🔬 Overview

MedXpert is a comprehensive AI-powered medical imaging analysis platform that leverages state-of-the-art deep learning models to assist healthcare professionals in medical diagnosis. The platform provides automated analysis for multiple medical imaging modalities with high accuracy and detailed visualization.

## ✨ Key Features

### 🧠 **Brain Tumor Detection**
- **Technology**: U-Net architecture
- **Input**: MRI NIFTI files (.nii.gz)
- **Capability**: 3D tumor segmentation and volumetric analysis
- **Performance**: 0.73 Dice Score
- **Output**: Slice-by-slice tumor segmentation with yellow overlay visualization

### 🫁 **Chest X-ray Analysis**
- **Technology**: EfficientNetB1
- **Input**: Standard chest X-ray images (PNG, JPG, JPEG)
- **Capability**: Multi-class classification for lung pathologies
- **Performance**: 95% accuracy
- **Classes**: COVID-19, Lung Opacity, Normal, Viral Pneumonia
- **Features**: GradCAM heatmap visualization for explainable AI

### 🩺 **Skin Cancer Detection**
- **Technology**: CNN
- **Input**: Dermoscopic images (PNG, JPG, JPEG)
- **Capability**: Binary classification for melanoma screening
- **Performance**: 93% accuracy
- **Classes**: Malignant Melanoma, Benign Lesions

### 🦴 **Fracture Detection**
- **Technology**: YOLO v8 object detection
- **Input**: X-ray images (PNG, JPG, JPEG)
- **Capability**: Real-time fracture detection and localization
- **Performance**: 94% mAP50
- **Features**: Bounding box visualization with confidence scores

### 🤖 **AI Medical Chatbot**
- **Technology**: LangChain + Google Gemini 2.5 Flash
- **Capability**: Document-based Q&A system
- **Features**: 
  - Upload medical documents (PDF, DOCX, TXT)
  - Conversational memory
  - Context-aware responses
  - User-specific document collections

## 🏗️ Architecture

```
MedXpert/
├── app.py                 # Main Flask application
├── chatbot_services.py    # AI chatbot backend services
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
├── models/               # Pre-trained AI models
│   ├── best_metric_model.pth    # Brain tumor U-Net model
│   ├── best_model_chest.h5      # Chest X-ray ResNet model
│   ├── best_model_skin.h5       # Skin cancer model
│   └── best.pt                  # YOLO fracture detection
├── templates/            # HTML templates
├── static/              # CSS, JS, and uploaded images
│   ├── css/
│   ├── js/
│   ├── images/
│   └── uploads/
└── chroma_db/           # Vector database for chatbot
```

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Recommended)
1. Click the "Code" button on GitHub
2. Select "Codespaces"
3. Click "Create codespace on main"
4. Wait for environment setup
5. Run: `python app.py`
6. Open the provided URL

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- 3GB free disk space
- Optional: CUDA for GPU acceleration

#### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/mohamed-alawy/medxpert.git
cd medxpert
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
```

5. **Run the application**
```bash
python app.py
```

6. **Access the platform**
Open your browser and navigate to: `http://localhost:5000`

### Option 3: Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t medxpert .
docker run -p 5000:5000 medxpert
```

## 🎯 Usage Guide

### Medical Image Analysis
1. **Select Model**: Choose from Brain, Chest, Skin, or Fracture analysis
2. **Upload Image**: Drag and drop or select your medical image
3. **Analyze**: Click the analyze button and wait for processing
4. **Review Results**: Examine the AI analysis with confidence scores and visualizations
5. **Download Report**: Generate and download detailed analysis reports

### AI Chatbot
1. **Login/Register**: Create an account or login
2. **Upload Documents**: Add PDF, DOCX, or TXT medical documents
3. **Ask Questions**: Query the AI about your uploaded documents
4. **Conversational Context**: The chatbot maintains conversation history

### User Management
- **User Registration**: Create accounts with email verification
- **Admin Panel**: Administrative interface for user management
- **Session Management**: Secure user sessions with automatic logout

## 🔧 Technical Specifications

### Models Performance
| Model | Architecture | Accuracy/Metric | Input Size | Training Data |
|-------|-------------|-----------------|------------|---------------|
| Brain Tumor | U-Net | 0.73 Dice | 128×128×64 | 75,000+ MRI scans |
| Chest X-ray | EfficientNet | 95% Accuracy | 224×224×1 | 16K+ X-rays |
| Skin Cancer | CNN | 93% Accuracy | 300×300×3 | 21K+ dermoscopic images |
| Fracture | YOLO v8 | 94% mAP50 | 640×640×1 | 6K+ X-ray images |

### Technology Stack
- **Backend**: Flask, SQLAlchemy, Flask-Login
- **AI/ML**: PyTorch, TensorFlow, MONAI, Ultralytics YOLO
- **Computer Vision**: OpenCV, scikit-image, matplotlib
- **NLP**: LangChain, Google Gemini AI
- **Database**: SQLite, ChromaDB (vector database)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Deployment**: Docker, Docker Compose

## 📊 Features in Detail

### Advanced Visualizations
- **GradCAM Heatmaps**: Explainable AI for chest X-ray analysis
- **3D Tumor Segmentation**: Multi-slice brain tumor visualization
- **Bounding Box Detection**: Precise fracture localization
- **Interactive Image Viewer**: Zoom, pan, and slice navigation

### Security & Privacy
- Secure user authentication and session management
- File upload validation and sanitization
- User-specific data isolation
- HTTPS ready for production deployment

### Scalability
- Lazy model loading for memory efficiency
- Background task processing capability
- Docker containerization for easy deployment
- Horizontal scaling support

## 🚦 API Endpoints

### Medical Analysis
- `POST /predict/brain` - Brain tumor analysis
- `POST /predict/chest` - Chest X-ray analysis  
- `POST /predict/skin` - Skin cancer detection
- `POST /predict/fracture` - Fracture detection

### Chatbot
- `POST /api/chatbot/upload` - Upload documents
- `POST /api/chatbot/query` - Ask questions
- `POST /api/chatbot/clear-history` - Clear chat history

### User Management
- `GET /register` - User registration
- `POST /login` - User authentication
- `GET /profile` - User profile
- `GET /admin` - Admin panel (admin only)

## 🔮 Future Enhancements

- [ ] Real-time collaborative diagnosis
- [ ] Mobile application development  
- [ ] Integration with PACS systems
- [ ] Advanced reporting and analytics
- [ ] Multi-language support
- [ ] Federated learning capabilities
- [ ] 3D visualization improvements
- [ ] Integration with hospital EMR systems

## ⚠️ Important Disclaimers

**Medical Disclaimer**: This AI platform is designed for educational and research purposes only. It should not replace professional medical diagnosis, treatment, or clinical decision-making. Always consult qualified healthcare professionals for medical advice.

**Accuracy Notice**: While our models achieve high accuracy on test datasets, real-world performance may vary. The platform should be used as a decision support tool alongside professional medical expertise.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Team

Developed by the MedXpert team as part of an AI in Healthcare initiative.

**⭐ Star this repository if you find it useful!**

*Built with ❤️ for advancing AI in healthcare*