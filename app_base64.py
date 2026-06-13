################# ANEMIA DETECTION API - OPTIMIZED VERSION #################
# Run `python app_base64.py` to start the Flask server on port 2000
# For external access: run `ngrok http 2000` and `python app_base64.py` in parallel
# Use the ngrok URL in Postman for testing
##############################################################################

from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
import base64
import io
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter
import os

app = Flask(__name__)

# ========== MODEL CONFIGURATION ==========
YOLO_PATH = "./models/yolov8_nail_best.pt"
ANEMIA_MODEL_PATH = "./models/anemia_cnn_model2.pth"

# Load YOLO nail detector - for detecting nails in hand images
nail_detector = YOLO(YOLO_PATH)

# ========== ANEMIA CNN MODEL DEFINITION ==========
class SimpleCNN(nn.Module):
    """
    Custom CNN model for anemia detection from nail images
    Architecture: 2 Conv layers + 2 FC layers
    Input: 224x224 RGB nail image
    Output: 2 classes (0=Non-Anemic, 1=Anemic)
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 3->16 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce spatial dimensions by half
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 16->32 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Further reduce spatial dimensions
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten feature maps to 1D
            nn.Linear(32 * 56 * 56, 128),  # 32 channels * 56*56 spatial size -> 128 features
            nn.ReLU(),
            nn.Linear(128, num_classes)  # 128 -> 2 classes (Non-Anemic/Anemic)
        )

    def forward(self, x):
        x = self.features(x)      # Extract features
        x = self.classifier(x)    # Classify
        return x

# Load pre-trained anemia detection model
anemia_model = SimpleCNN()
anemia_model.load_state_dict(torch.load(ANEMIA_MODEL_PATH, map_location=torch.device('cpu')))
anemia_model.eval()  # Set to evaluation mode

# Image preprocessing pipeline for anemia model
anemia_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size
    transforms.ToTensor(),          # Convert PIL to tensor and normalize to [0,1]
])

# ========== CORE PROCESSING FUNCTIONS ==========

def crop_nails_from_hand(image):
    """
    Detect and crop individual nails from a hand image using YOLO
    
    Args:
        image (PIL.Image): Hand image in RGB format
        
    Returns:
        list: List of tuples containing (cropped_nail_image, bounding_box_coordinates)
    """
    results = nail_detector(image)
    nail_crops = []
    
    for result in results:
        # Extract bounding box coordinates
        boxes = result.boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Crop the nail region from original image
            cropped_nail = image.crop((x1, y1, x2, y2))
            nail_crops.append((cropped_nail, (x1, y1, x2, y2)))
    
    return nail_crops

def predict_anemia_from_nail(nail_image):
    """
    Predict anemia status from a single nail image using CNN
    
    Args:
        nail_image (PIL.Image): Cropped nail image
        
    Returns:
        tuple: (predicted_class, probability_array)
            - predicted_class: 0 for Non-Anemic, 1 for Anemic
            - probability_array: [prob_non_anemic, prob_anemic]
    """
    # Preprocess the nail image
    nail_tensor = anemia_transform(nail_image)
    nail_tensor = nail_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = anemia_model(nail_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
    
    return predicted_class, probabilities.detach().numpy()[0]

def aggregate_multiple_predictions(predictions, method='mean_prob'):
    """
    Combine predictions from multiple nails to get final diagnosis
    
    Args:
        predictions (list): List of (class, probability_array) tuples
        method (str): Aggregation method - 'majority' or 'mean_prob'
        
    Returns:
        tuple: (final_prediction, average_anemia_probability)
    """
    classes = [pred[0] for pred in predictions]
    anemia_probabilities = [pred[1][1] for pred in predictions]  # Extract anemia probabilities
    
    if method == 'majority':
        # Use majority voting
        class_counts = Counter(classes)
        final_prediction = class_counts.most_common(1)[0][0]
    elif method == 'mean_prob':
        # Use average probability threshold
        mean_anemia_prob = np.mean(anemia_probabilities)
        final_prediction = 1 if mean_anemia_prob > 0.5 else 0
    else:
        # Default to majority voting
        class_counts = Counter(classes)
        final_prediction = class_counts.most_common(1)[0][0]
    
    return final_prediction, np.mean(anemia_probabilities) if anemia_probabilities else 0.0

def convert_image_to_base64(pil_image, format='PNG'):
    """
    Convert PIL Image to base64 string
    
    Args:
        pil_image (PIL.Image): Image to convert
        format (str): Output format (PNG, JPEG, etc.)
        
    Returns:
        str: Base64 encoded image string
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def process_hand_for_anemia_detection(image_pil, aggregation_method='mean_prob', include_cropped_images=True):
    """
    Complete pipeline for anemia detection from hand image
    
    Args:
        image_pil (PIL.Image): Hand image in RGB format
        aggregation_method (str): Method to combine multiple nail predictions
        include_cropped_images (bool): Whether to include base64 encoded cropped images
        
    Returns:
        dict: Comprehensive results including diagnosis, confidence, nail details and base64 images
    """
    try:
        # Step 1: Detect and crop nails from hand image
        nail_crops = crop_nails_from_hand(image_pil)
        
        if len(nail_crops) == 0:
            return {
                "success": False,
                "message": "No nails detected in the hand image",
                "nails_detected": 0
            }

        # Step 2: Process each detected nail for anemia prediction
        nail_results = []
        anemia_predictions = []

        for i, (nail_img, bounding_box) in enumerate(nail_crops):
            # Predict anemia for this nail
            anemia_class, anemia_probabilities = predict_anemia_from_nail(nail_img)
            anemia_predictions.append((anemia_class, anemia_probabilities))
            
            # Convert cropped nail image to base64 (PNG format for best quality)
            nail_base64 = convert_image_to_base64(nail_img, format='PNG') if include_cropped_images else None
            
            # Store detailed results for this nail
            nail_info = {
                "nail_id": i + 1,
                "bounding_box": bounding_box,
                "anemia_prediction": "Anemic" if anemia_class == 1 else "Non-Anemic",
                "anemia_class": int(anemia_class),  # 0 = Non-Anemic, 1 = Anemic (for training)
                "anemia_probability": float(anemia_probabilities[1]),     # P(Anemic)
                "non_anemia_probability": float(anemia_probabilities[0]), # P(Non-Anemic)
                "confidence_score": float(max(anemia_probabilities)),     # Highest probability
                "cropped_image_base64": nail_base64,  # Base64 encoded cropped nail image
                "image_format": "PNG" if include_cropped_images else None
            }
            nail_results.append(nail_info)

        # Step 3: Aggregate predictions from all nails
        final_prediction, average_anemia_probability = aggregate_multiple_predictions(
            anemia_predictions, method=aggregation_method
        )
        
        # Step 4: Return comprehensive results
        return {
            "success": True,
            "final_diagnosis": "Anemic" if final_prediction == 1 else "Non-Anemic",
            "final_class": int(final_prediction),  # 0 = Non-Anemic, 1 = Anemic
            "confidence": float(average_anemia_probability),
            "aggregation_method": aggregation_method,
            "nails_detected": len(nail_crops),
            "nail_details": nail_results,
            "summary": {
                "total_nails": len(nail_crops),
                "anemic_nails": sum(1 for pred in anemia_predictions if pred[0] == 1),
                "non_anemic_nails": sum(1 for pred in anemia_predictions if pred[0] == 0)
            },
            "training_data_info": {
                "description": "Each nail_detail contains cropped image and label for future training",
                "image_format": "PNG (base64 encoded)",
                "label_field": "anemia_class",
                "label_mapping": {"0": "Non-Anemic", "1": "Anemic"}
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }

# ========== API ENDPOINTS ==========

@app.route('/', methods=['GET'])
def home():
    """
    Health check and API information endpoint
    """
    return jsonify({
        "message": "Anemia Detection API - Optimized for PNG images",
        "version": "2.0",
        "preferred_format": "PNG",
        "endpoints": {
            "POST /predict": "Upload hand image for anemia detection with cropped nail images",
            "POST /predict_base64": "Send base64 encoded hand image",
            "GET /health": "Check API and model status"
        },
        "parameters": {
            "aggregation_method": "majority or mean_prob (default: mean_prob)",
            "include_cropped_images": "true/false - include base64 cropped nail images (default: true)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Check if all models are loaded and API is ready
    """
    return jsonify({
        "status": "healthy",
        "api_version": "2.0",
        "models_loaded": {
            "yolo_nail_detector": nail_detector is not None,
            "anemia_cnn_model": anemia_model is not None
        },
        "preferred_image_format": "PNG",
        "supported_formats": ["PNG", "JPG", "JPEG"]
    })

@app.route('/predict', methods=['POST'])
def predict_anemia_endpoint():
    """
    Main endpoint for anemia prediction from uploaded image file
    Preferred format: PNG
    """
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No image file provided. Please upload an image with key 'image'",
                "preferred_format": "PNG"
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400
        
        # Get aggregation method (optional parameter)
        aggregation_method = request.form.get('aggregation_method', 'mean_prob')
        if aggregation_method not in ['majority', 'mean_prob']:
            aggregation_method = 'mean_prob'
        
        # Get option to include cropped images (default: True)
        include_images = request.form.get('include_cropped_images', 'true').lower() == 'true'
        
        # Process the uploaded image
        image_bytes = file.read()
        
        # Convert to PIL Image (handles PNG, JPG, etc.)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # If not PNG, recommend PNG for better quality
        file_format = file.filename.split('.')[-1].upper() if '.' in file.filename else 'UNKNOWN'
        
        # Process the image for anemia detection
        result = process_hand_for_anemia_detection(
            image_pil, 
            aggregation_method=aggregation_method,
            include_cropped_images=include_images
        )
        
        # Add format recommendation
        if file_format != 'PNG':
            result["format_recommendation"] = "For best results, use PNG format images"
        
        # Return appropriate response
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_anemia_base64_endpoint():
    """
    Alternative endpoint for base64 encoded images
    Preferred format: PNG encoded as base64
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "message": "No base64 image data provided. Send JSON with 'image' key containing base64 encoded image",
                "preferred_format": "PNG (base64 encoded)"
            }), 400
        
        # Get aggregation method and image inclusion option
        aggregation_method = data.get('aggregation_method', 'mean_prob')
        if aggregation_method not in ['majority', 'mean_prob']:
            aggregation_method = 'mean_prob'
        
        include_images = data.get('include_cropped_images', True)
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Invalid base64 image data: {str(e)}"
            }), 400
        
        # Process the image for anemia detection
        result = process_hand_for_anemia_detection(
            image_pil, 
            aggregation_method=aggregation_method,
            include_cropped_images=include_images
        )
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/predict_and_save_training_data', methods=['POST'])
def predict_and_save_training_data():
    """
    Special endpoint for prediction + saving training data
    Saves each cropped nail with its prediction for future model training
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No image file provided"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400
        
        # Get parameters
        aggregation_method = request.form.get('aggregation_method', 'mean_prob')
        save_directory = request.form.get('save_directory', './training_data')
        
        # Process image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get predictions with cropped images
        result = process_hand_for_anemia_detection(
            image_pil, 
            aggregation_method=aggregation_method,
            include_cropped_images=True
        )
        
        if result["success"]:
            # Create directories for saving
            os.makedirs(f"{save_directory}/anemic", exist_ok=True)
            os.makedirs(f"{save_directory}/non_anemic", exist_ok=True)
            
            saved_files = []
            
            # Save each cropped nail image
            for nail_detail in result["nail_details"]:
                nail_id = nail_detail["nail_id"]
                anemia_class = nail_detail["anemia_class"]
                base64_image = nail_detail["cropped_image_base64"]
                
                # Decode base64 and save
                image_data = base64.b64decode(base64_image)
                
                # Determine save path based on prediction
                folder = "anemic" if anemia_class == 1 else "non_anemic"
                filename = f"nail_{nail_id}_{anemia_class}_{nail_detail['confidence_score']:.3f}.png"
                filepath = f"{save_directory}/{folder}/{filename}"
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                saved_files.append({
                    "nail_id": nail_id,
                    "label": anemia_class,
                    "confidence": nail_detail['confidence_score'],
                    "saved_path": filepath
                })
            
            result["saved_training_data"] = {
                "total_saved": len(saved_files),
                "save_directory": save_directory,
                "files": saved_files
            }
        
        return jsonify(result), 200 if result["success"] else 400
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

# ========== APPLICATION STARTUP ==========
if __name__ == '__main__':
    print("=" * 60)
    print("🔬 ANEMIA DETECTION API - OPTIMIZED VERSION")
    print("=" * 60)
    print("Loading models...")
    print("✅ YOLO nail detector loaded successfully!")
    print("✅ Anemia CNN model loaded successfully!")
    print("📋 Preferred image format: PNG")
    print("🌐 Starting Flask server on port 2000...")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=2000)