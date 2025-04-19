import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, url_for, redirect, flash
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import io
import base64
from ultralytics import YOLO
import traceback
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# MONAI imports for brain tumor segmentation
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, ToTensord
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import Dataset

app = Flask(__name__)
app.secret_key = 'medxpert_secret_key'

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii', 'nii.gz', 'dcm'}

def allowed_file(filename):
    return '.' in filename and (
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or
        filename.endswith('.nii.gz')  # Special handling for .nii.gz files
    )
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
def load_brain_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    model.load_state_dict(torch.load("models/best_metric_model.pth", map_location=device))
    model.eval()
    return model

def load_skin_model():
    model = tf.keras.models.load_model("models/best_model_skin.h5")
    return model

def load_chest_model():
    model = tf.keras.models.load_model("models/best_model_chest.h5")
    return model

def load_fracture_model():
    model = YOLO("models/best.pt")
    return model

# Load models on startup
print("Loading models...")
try:
    brain_model = load_brain_model()
    skin_model = load_skin_model()
    chest_model = load_chest_model()
    fracture_model = load_fracture_model()
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    # Continue without crashing, models will be reloaded when needed

# Brain tumor transforms
brain_transforms = Compose([
    LoadImaged(keys=["vol"]),
    EnsureChannelFirstd(keys=["vol"]),
    Spacingd(keys=["vol"], pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
    Orientationd(keys=["vol"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["vol"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=['vol'], source_key='vol'),
    Resized(keys=["vol"], spatial_size=[128, 128, 64]),
    ToTensord(keys=["vol"]),
])

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/brain')
def brain():
    return render_template('brain.html')

@app.route('/skin')
def skin():
    return render_template('skin.html')

@app.route('/chest')
def chest():
    return render_template('chest.html')

@app.route('/fracture')
def fracture():
    return render_template('fracture.html')

@app.route('/predict/brain', methods=['POST'])
def predict_brain():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"File saved to: {filepath}")
            
            # Process with brain tumor model
            test_sample = {"vol": filepath}
            test_ds = Dataset(data=[test_sample], transform=brain_transforms)
            sample = test_ds[0]
            t_volume = sample['vol'].unsqueeze(0).to(device)
            
            if t_volume.shape[1] != 1:
                t_volume = torch.mean(t_volume, dim=1, keepdim=True)
            
            print(f"Volume shape: {t_volume.shape}")
                
            with torch.no_grad():
                test_outputs = sliding_window_inference(t_volume, (128, 128, 64), 4, brain_model)
                test_outputs = torch.sigmoid(test_outputs)
                test_outputs = test_outputs > 0.9
            
            print("Inference complete, generating images")    
                
            # Get all slices for visualization
            orig_vol = t_volume.cpu().numpy()[0, 0]
            pred_mask = test_outputs.cpu().numpy()[0, 1]
            
            print(f"Original volume shape: {orig_vol.shape}")
            print(f"Prediction mask shape: {pred_mask.shape}")
            
            # Process all slices with tumor
            slice_scores = [np.sum(pred_mask[:,:,i]) for i in range(pred_mask.shape[2])]
            valid_slices = [i for i, score in enumerate(slice_scores) if score > 0]
            
            print(f"Number of slices with tumor: {len(valid_slices)}")
            
            result_images = []

            colors = [(0, 0, 0, 0), (246/255, 242/255, 102/255, 1)]  # RGBA format
            tumor_cmap = LinearSegmentedColormap.from_list('tumor_yellow', colors, N=256)

            for slice_idx in range(pred_mask.shape[2]):
                # Create overlay image
                plt.figure(figsize=(10, 8))
                plt.imshow(orig_vol[:,:,slice_idx], cmap='gray')
                
                    # Only overlay tumor if detected in this slice
                if slice_scores[slice_idx] > 0:
                    plt.imshow(pred_mask[:,:,slice_idx], alpha=0.5, cmap=tumor_cmap)
                plt.axis('off')
                plt.title(f"Slice {slice_idx+1}/{pred_mask.shape[2]}", fontsize=14)
                
                
                # Save to bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                
                # Convert to base64 for embedding in HTML
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                result_images.append(img_str)
            
            print(f"Generated {len(result_images)} result images")
            
            # Return all slices and info about which ones have tumors
            return jsonify({
                'status': 'success',
                'message': 'Brain scan processed successfully.',
                'images': result_images,
                'tumorSlices': valid_slices,
                'totalSlices': pred_mask.shape[2]
            })
            
        except Exception as e:
            print(f"Error processing brain scan: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': 'Invalid file format. Please upload a valid medical image.'
    })

@app.route('/predict/skin', methods=['POST'])
def predict_skin():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process with skin cancer model
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(300, 300))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            prediction = skin_model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = float(prediction[0][predicted_class]) * 100
            
            class_names = ['Benign', 'Malignant']
            result = {
                'status': 'success',
                'class': class_names[predicted_class],
                'confidence': confidence,
                'image_path': f'/static/uploads/{filename}'
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': 'Invalid file format. Please upload a valid image.'
    })

@app.route('/predict/chest', methods=['POST'])
def predict_chest():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process with chest X-ray model
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            pred_probs = chest_model.predict(img_array)[0]
            predicted_class = np.argmax(pred_probs)
            confidence = float(pred_probs[predicted_class]) * 100
            
            class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
            
            result = {
                'status': 'success',
                'class': class_labels[predicted_class],
                'confidence': confidence,
                'image_path': f'/static/uploads/{filename}'
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': 'Invalid file format. Please upload a valid X-ray image.'
    })

@app.route('/predict/fracture', methods=['POST'])
def predict_fracture():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read image
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Unable to read image file. Please upload a valid image.'
                })
            
            # Save original image as base64 before processing
            original_img = img.copy()
            _, buffer_original = cv2.imencode('.jpg', original_img)
            original_img_str = base64.b64encode(buffer_original).decode('utf-8')
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Run inference
            results = fracture_model(img)
            
            # Process results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                # Save the image with bounding boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw rectangle with the specified yellow color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(102, 242, 246), thickness=2)  # OpenCV uses BGR format (246, 242, 102)
                    
                    # Get confidence score and class name
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{fracture_model.names[cls]} {conf:.2f}"
                    
                    # Draw label with yellow background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (102, 242, 246), -1)  # Filled rectangle
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Store detection data
                    detections.append({
                        'class': fracture_model.names[cls],
                        'confidence': float(conf),
                        'box': [x1, y1, x2, y2]
                    })
            
            # Save the annotated image
            output_filename = f"result_{filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, img)
            
            # Convert processed image to base64 for display
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Create message based on detections
            if len(detections) > 0:
                message = f"Found {len(detections)} potential fracture(s)."
            else:
                message = "No fractures detected in this image."
            
            # Return results with both original and processed images
            return jsonify({
                'status': 'success',
                'message': message,
                'original_image': original_img_str,  # Add the original image
                'image': img_str,                    # Processed image with fractures
                'detections': detections,
                'dimensions': {
                    'width': width,
                    'height': height
                }
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': 'Invalid file format. Please upload a valid image file.'
    })
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)