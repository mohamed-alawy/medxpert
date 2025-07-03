import os
import numpy as np
import io
import base64
import traceback
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from functools import wraps
import torch
import tensorflow as tf
import cv2
from ultralytics import YOLO
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, ToTensord
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import Dataset

print("Starting application...")

# Flask application setup
app = Flask(__name__)
app.secret_key = 'medxpert_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login manager configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    email = db.Column(db.String(120), unique=True)
    full_name = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
        
    def is_admin(self):
        return self.role == 'admin'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Database migration helper function
def migrate_database():
    # Check if we need to add the role column
    inspector = inspect(db.engine)
    columns = [column['name'] for column in inspector.get_columns('user')]
    
    if 'role' not in columns:
        print("Migrating database: Adding 'role' column to User table")
        with db.engine.begin() as conn:
            # Using begin() to automatically handle transaction
            conn.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(20) DEFAULT 'user'"))
        print("Database migration complete")
    else:
        print("Database schema is up to date")
    
# Create database tables and default admin
with app.app_context():
    try:
        db.create_all()  # Only create tables if they don't exist
        
        # Migrate database if needed
        migrate_database()
        
        # Check if we need to create a default admin user
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                email='admin@medxpert.com',
                full_name='System Administrator',
                role='admin'
            )
            admin.set_password('admin123')  # Default password - should be changed immediately
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created")
        
        print("Database tables checked/created successfully")
    except Exception as e:
        print(f"Error creating/migrating database tables: {str(e)}")

# Configure upload folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'nii.gz'}

def allowed_file(filename):
    return '.' in filename and (
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS or
        filename.endswith('.nii.gz')  # Special handling for .nii.gz files
    )

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loading functions
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
    state_dict = torch.load("models/best_metric_model.pth", weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
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

# Initialize model variables as None for lazy loading
print("Initializing application...")
brain_model = None
skin_model = None
chest_model = None
fracture_model = None

def get_brain_model():
    global brain_model
    if brain_model is None:
        print("Loading brain model...")
        brain_model = load_brain_model()
    return brain_model

def get_skin_model():
    global skin_model
    if skin_model is None:
        print("Loading skin model...")
        skin_model = load_skin_model()
    return skin_model

def get_chest_model():
    global chest_model
    if chest_model is None:
        print("Loading chest model...")
        chest_model = load_chest_model()
    return chest_model

def get_fracture_model():
    global fracture_model
    if fracture_model is None:
        print("Loading fracture model...")
        fracture_model = load_fracture_model()
    return fracture_model

# Brain tumor transforms
brain_transforms = Compose([
    LoadImaged(keys=["vol"]),
    EnsureChannelFirstd(keys=["vol"]),
    Spacingd(keys=["vol"], pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
    Orientationd(keys=["vol"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["vol"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["vol"], source_key="vol", allow_smaller=True),
    Resized(keys=["vol"], spatial_size=[128, 128, 64]),
    ToTensord(keys=["vol"]),
])

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions for Grad-CAM visualization
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Creates a Grad-CAM heatmap using the exact implementation that works with the chest model
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on the input image with specified transparency
    """
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_img

# Function to adjust image brightness and contrast
def adjust_image(image, brightness=0, contrast=1):
    """
    Adjust the brightness and contrast of an image.
    :param image: Input image as numpy array
    :param brightness: Brightness adjustment factor (-1.0 to 1.0)
    :param contrast: Contrast adjustment factor (0.0 to 2.0)
    :return: Adjusted image
    """
    # Convert brightness from (-100 to 100) to (-1.0 to 1.0)
    brightness_factor = float(brightness) / 100.0
    
    # Convert contrast from (0 to 200) to (0.0 to 2.0)
    contrast_factor = float(contrast) / 100.0
    
    # Apply brightness adjustment
    adjusted = image + brightness_factor
    
    # Apply contrast adjustment: f(x) = (x - 0.5) * contrast + 0.5
    if contrast_factor != 1.0:
        adjusted = (adjusted - 0.5) * contrast_factor + 0.5
    
    # Clip values to valid range [0, 1]
    adjusted = np.clip(adjusted, 0, 1)
    
    return adjusted

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server error: {error}')
    return render_template('error.html', error="Internal server error occurred. Please try again."), 500

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f'Page not found: {request.url}')
    return render_template('error.html', error="Page not found."), 404

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/explore-models')
def explore_models():
    return redirect(url_for('home'))

# Keep compatibility with old route
@app.route('/information')
def information():
    return redirect(url_for('home'))

@app.route('/brain', methods=['GET', 'POST'])
def brain():
    if request.method == 'POST':
        # If it's a POST request, redirect to predict_brain
        return redirect(url_for('predict_brain'))
    # If it's a GET request, just render the template
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
        flash('No file was uploaded', 'danger')
        return render_template('brain.html')
        
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return render_template('brain.html')
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"File saved to: {filepath}")
            
            # Get model on demand
            model = get_brain_model()
            
            # Process with brain tumor model
            test_sample = {"vol": filepath}
            test_ds = Dataset(data=[test_sample], transform=brain_transforms)
            sample = test_ds[0]
            t_volume = sample['vol'].unsqueeze(0).to(device)
            
            if t_volume.shape[1] != 1:
                t_volume = torch.mean(t_volume, dim=1, keepdim=True)
            
            print(f"Volume shape: {t_volume.shape}")
                
            with torch.no_grad():
                test_outputs = sliding_window_inference(t_volume, (128, 128, 64), 4, model)
                test_outputs = torch.sigmoid(test_outputs)
                test_outputs = test_outputs > 0.9
            
            print("Inference complete, generating images")    
                
            # Get all slices for visualization
            orig_vol = t_volume.cpu().numpy()[0, 0]
            pred_mask = test_outputs.cpu().numpy()[0, 1]
            
            print(f"Original volume shape: {orig_vol.shape}")
            print(f"Prediction mask shape: {pred_mask.shape}")
            
            # Process all slices with tumor
            slice_scores = [float(np.sum(pred_mask[:,:,i])) for i in range(pred_mask.shape[2])]
            valid_slices = [i for i, score in enumerate(slice_scores) if score > 0]
            
            print(f"Number of slices with tumor: {len(valid_slices)}")
            
            result_images = []
            original_images = []  # Store original images without segmentation

            colors = [(0, 0, 0, 0), (246/255, 242/255, 102/255, 1)]  # RGBA format (yellow)
            tumor_cmap = LinearSegmentedColormap.from_list('tumor_yellow', colors, N=256)

            for slice_idx in range(pred_mask.shape[2]):
                # Create original image
                plt.figure(figsize=(10, 8))
                plt.imshow(orig_vol[:,:,slice_idx], cmap='gray')
                plt.axis('off')
                
                # Save to bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                
                # Convert to base64 for embedding in HTML
                orig_img_str = base64.b64encode(buf.read()).decode('utf-8')
                original_images.append(orig_img_str)
                
                # Create overlay image
                plt.figure(figsize=(10, 8))
                plt.imshow(orig_vol[:,:,slice_idx], cmap='gray')
                
                # Only overlay tumor if detected in this slice
                if slice_scores[slice_idx] > 0:
                    plt.imshow(pred_mask[:,:,slice_idx], alpha=0.5, cmap=tumor_cmap)
                plt.axis('off')
                
                # Save to bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close()
                buf.seek(0)
                
                # Convert to base64 for embedding in HTML
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                result_images.append(img_str)
            
            print(f"Generated {len(result_images)} result images")
            
            # Get the best slice to show first (first slice with tumor, or middle slice if no tumor)
            best_slice_idx = valid_slices[0] if valid_slices else pred_mask.shape[2] // 2
            
            # Get tumor information summary
            tumor_info = {
                'total_slices': pred_mask.shape[2],
                'slices_with_tumor': len(valid_slices),
                'tumor_slice_indices': valid_slices,
                'first_tumor_slice': valid_slices[0] if valid_slices else None
            }
            
            return render_template('brain.html', 
                                  original_img_b64=original_images[best_slice_idx],
                                  segmented_img_b64=result_images[best_slice_idx],
                                  all_original_slices=original_images,
                                  all_segmented_slices=result_images,
                                  tumor_info=tumor_info,
                                  total_slices=pred_mask.shape[2],
                                  current_slice_idx=best_slice_idx)
            
        except Exception as e:
            print(f"Error processing brain scan: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            flash(f'Error processing image: {str(e)}', 'danger')
            return render_template('brain.html')
    
    flash('Invalid file format. Please upload a valid NIFTI (.nii.gz) file.', 'warning')
    return render_template('brain.html')

@app.route('/predict/skin', methods=['POST'])
def predict_skin():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file was uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get model on demand
            model = get_skin_model()
            
            # Process with skin cancer model
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(300, 300))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = float(prediction[0][predicted_class]) * 100
            
            # Get the original image as base64
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
            class_names = ['Benign', 'Malignant']
            result = {
                'status': 'success',
                'class': class_names[predicted_class],
                'confidence': confidence,
                'image_path': f'/static/uploads/{filename}',
                'image_base64': img_base64
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
        return jsonify({'status': 'error', 'message': 'No file was uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get model on demand
            model = get_chest_model()
            
            # Process with chest X-ray model
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array_norm = np.expand_dims(img_array, axis=0) / 255.0
            
            # Make predictions
            pred_probs = model.predict(img_array_norm)[0]
            predicted_class = np.argmax(pred_probs)
            confidence = float(pred_probs[predicted_class]) * 100
            
            class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
            
            # Generate Grad-CAM visualization using the known working layer name
            last_conv_layer_name = "top_conv"
            
            try:
                # Generate the heatmap
                heatmap = make_gradcam_heatmap(img_array_norm, model, last_conv_layer_name, pred_index=predicted_class)
                
                if heatmap is not None:
                    # Convert original image to uint8 for visualization
                    img_uint8 = img_array.astype(np.uint8)
                    
                    # Create overlay visualization with the heatmap
                    heatmap_img = overlay_heatmap(img_uint8, heatmap)
                    
                    # Save Grad-CAM visualization
                    gradcam_filename = f"gradcam_{filename}"
                    gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                    cv2.imwrite(gradcam_filepath, heatmap_img)
                    
                    # Get both original and Grad-CAM images as base64
                    with open(filepath, "rb") as img_file:
                        orig_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    with open(gradcam_filepath, "rb") as img_file:
                        gradcam_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    result = {
                        'status': 'success',
                        'class': class_labels[predicted_class],
                        'confidence': confidence,
                        'image_path': f'/static/uploads/{filename}',
                        'original_base64': orig_img_base64,
                        'gradcam_base64': gradcam_base64
                    }
                else:
                    raise ValueError("Failed to generate heatmap")
            except Exception as e:
                print(f"Error generating Grad-CAM: {str(e)}")
                traceback.print_exc()
                # Fallback to original image if Grad-CAM fails
                with open(filepath, "rb") as img_file:
                    orig_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                result = {
                    'status': 'success',
                    'class': class_labels[predicted_class],
                    'confidence': confidence,
                    'image_path': f'/static/uploads/{filename}',
                    'original_base64': orig_img_base64,
                    'gradcam_error': f'Could not generate Grad-CAM: {str(e)}'
                }
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Error processing chest X-ray: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
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
        return jsonify({'status': 'error', 'message': 'No file was uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get model on demand
            model = get_fracture_model()
            
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
            results = model(img)
            
            # Process results
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                # Save the image with bounding boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw rectangle with yellow color
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(102, 242, 246), thickness=2)
                    
                    # Get confidence score and class name
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"Fracture {conf:.2f}%"
                    
                    # Draw label with yellow background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (102, 242, 246), -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # Store detection data
                    detections.append({
                        'class': model.names[cls],
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
                'original_image': original_img_str,
                'image': img_str,
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
        'message': 'Invalid file format. Please upload a valid image.'
    })

# Add security headers middleware with error handling
@app.after_request
def add_security_headers(response):
    try:
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    except Exception as e:
        logging.error(f"Error adding security headers: {str(e)}")
    return response

# Add context processor to provide current year to all templates
@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}

# Create error template if it doesn't exist
def create_error_template():
    error_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .error-container { text-align: center; }
            .error-message { color: #721c24; background-color: #f8d7da; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="error-container">
            <h1>Error</h1>
            <div class="error-message">
                {{ error }}
            </div>
            <p><a href="{{ url_for('home') }}">Return to Home</a></p>
        </div>
    </body>
    </html>
    """
    try:
        os.makedirs('templates', exist_ok=True)
        with open('templates/error.html', 'w') as f:
            f.write(error_template)
    except Exception as e:
        logging.error(f"Error creating error template: {str(e)}")

# Create error template on startup
create_error_template()

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # Update user information
        current_user.email = request.form.get('email')
        current_user.full_name = request.form.get('full_name')
        current_user.phone = request.form.get('phone')
        
        # Update password if provided
        new_password = request.form.get('new_password')
        if new_password:
            current_user.set_password(new_password)
            
        db.session.commit()
        flash('Profile updated successfully')
        return redirect(url_for('profile'))
        
    return render_template('profile.html')

# Add a dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            user.last_login = datetime.utcnow()  # Update last login time
            db.session.commit()
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('home')
            return redirect(next_page)
        else:
            # Simple error message
            return render_template('login.html', error="Invalid username or password.")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            # Instead of using flash, display error directly in the template
            return render_template('register.html', error="Username already exists")
            
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        # Success message directly to the login template
        return render_template('login.html', success_message="Registration successful! You can now log in")
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Admin decorator for routes that require admin privileges
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# Admin routes
@app.route('/admin')
@login_required
@admin_required
def admin():
    users = User.query.all()
    return render_template('admin.html', users=users)

@app.route('/admin/create_user', methods=['POST'])
@login_required
@admin_required
def create_user():
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')
    full_name = request.form.get('full_name')
    role = request.form.get('role')
    
    # Check if username already exists
    if User.query.filter_by(username=username).first():
        return render_template('admin.html', 
                              users=User.query.all(),
                              message="Username already exists", 
                              message_type="danger")
    
    # Create new user
    user = User(username=username, email=email, full_name=full_name, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    
    return render_template('admin.html', 
                          users=User.query.all(),
                          message="User created successfully", 
                          message_type="success")

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        username = request.form.get('username')
        
        # Check if username changed and is already taken
        if username != user.username and User.query.filter_by(username=username).first():
            return render_template('edit_user.html', user=user, error="Username already exists")
        
        # Update user information
        user.username = username
        user.email = request.form.get('email')
        user.full_name = request.form.get('full_name')
        user.phone = request.form.get('phone')
        user.role = request.form.get('role')
        
        # Update password if provided
        new_password = request.form.get('new_password')
        if new_password:
            user.set_password(new_password)
        
        db.session.commit()
        return redirect(url_for('admin'))
    
    return render_template('edit_user.html', user=user)

@app.route('/admin/delete_user/<int:user_id>')
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    # Prevent deleting your own account
    if user.id == current_user.id:
        return render_template('admin.html', 
                              users=User.query.all(),
                              message="You cannot delete your own account", 
                              message_type="danger")
    
    # Delete the user
    db.session.delete(user)
    db.session.commit()
    
    return render_template('admin.html', 
                          users=User.query.all(),
                          message="User deleted successfully", 
                          message_type="success")

@app.route('/brain-example')
def brain_example():
    return render_template('brain_example.html')

@app.route('/chest-example')
def chest_example():
    return render_template('chest_example.html')

@app.route('/skin-example')
def skin_example():
    return render_template('skin_example.html')

@app.route('/fracture-example')
def fracture_example():
    return render_template('fracture_example.html')

# If this file is being run directly
if __name__ == "__main__":
    try:
        # Try default Flask port 5000
        port = int(os.environ.get("PORT", 5000))
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        app.run(host="0.0.0.0", port=port, debug=True)
    except Exception as e:
        print(f"Error starting Flask application: {str(e)}")
        print("Detailed error:")
        traceback.print_exc()