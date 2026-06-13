
#############in terminal run `python app.py` to start the Flask server#############

################# run `ngrok http 2000` and `python app.py` parallel to expose the server to the internet #################
################# paste ngrok url in post man ##############


from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
import base64
import io
from ultralytics import YOLO
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter
import os

app = Flask(__name__)

# ========== Load Models ==========
YOLO_PATH = "./models/yolov8_nail_best.pt"
CLASSIFIER_PATH = "./models/nail_cnn_classifier_model.h5"
ANEMIA_MODEL_PATH = "./models/anemia_cnn_model2.pth"

# Load YOLO nail detector
nail_detector = YOLO(YOLO_PATH)

# Load CNN nail polish classifier (Keras)
nail_classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# Load Anemia CNN model (PyTorch)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

anemia_model = SimpleCNN()
anemia_model.load_state_dict(torch.load(ANEMIA_MODEL_PATH, map_location=torch.device('cpu')))
anemia_model.eval()

# PyTorch transform
anemia_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========== Helper Functions ==========
def crop_nails(image):
    """Detect and crop nails from hand image using YOLO"""
    results = nail_detector(image)
    crops = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = image.crop((x1, y1, x2, y2))
            crops.append((cropped, (x1, y1, x2, y2)))
    return crops

def is_polish_free(nail_img):
    """Check if nail is polish-free using CNN classifier"""
    img = nail_img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = nail_classifier.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    return class_idx == 0, confidence  # True = plain, confidence

def predict_anemia(nail_img):
    """Predict anemia from nail image using custom CNN"""
    nail_tensor = anemia_transform(nail_img)
    nail_tensor = nail_tensor.unsqueeze(0)
    output = anemia_model(nail_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    class_idx = torch.argmax(probs).item()
    return class_idx, probs.detach().numpy()[0]

def aggregate_predictions(predictions, method='majority'):
    """Aggregate multiple nail predictions"""
    preds = [p[0] for p in predictions]
    probs = [p[1][1] for p in predictions]

    if method == 'majority':
        counts = Counter(preds)
        final_pred = counts.most_common(1)[0][0]
    elif method == 'mean_prob':
        mean_prob = np.mean(probs)
        final_pred = 1 if mean_prob > 0.5 else 0
    else:
        final_pred = counts.most_common(1)[0][0]  # default to majority
    
    return final_pred, np.mean(probs) if probs else 0.0

def process_hand_image_api(image_pil, agg_method='mean_prob'):
    """Main pipeline for processing hand image - API version"""
    try:
        # Step 1: Detect and crop nails
        nail_crops = crop_nails(image_pil)
        
        if len(nail_crops) == 0:
            return {
                "success": False,
                "message": "No nails detected in the image",
                "nails_detected": 0
            }

        # Step 2: Process each nail
        nail_results = []
        anemia_preds = []
        plain_nails_count = 0
        polished_nails_count = 0

        for i, (nail_img, box) in enumerate(nail_crops):
            # Check if nail is polish-free
            is_plain, polish_conf = is_polish_free(nail_img)
            
            nail_info = {
                "nail_id": i + 1,
                "bounding_box": box,
                "is_plain": bool(is_plain),
                "polish_confidence": float(polish_conf)
            }
            
            if is_plain:
                plain_nails_count += 1
                # Predict anemia
                anemia_class, anemia_prob = predict_anemia(nail_img)
                anemia_preds.append((anemia_class, anemia_prob))
                
                nail_info.update({
                    "anemia_prediction": "Anemic" if anemia_class == 1 else "Non-Anemic",
                    "anemia_probability": float(anemia_prob[1]),  # probability of being anemic
                    "non_anemia_probability": float(anemia_prob[0])
                })
            else:
                polished_nails_count += 1
                nail_info.update({
                    "anemia_prediction": "Skipped (Polished)",
                    "anemia_probability": None,
                    "non_anemia_probability": None
                })
            
            nail_results.append(nail_info)

        # Step 3: Aggregate predictions
        if len(anemia_preds) == 0:
            return {
                "success": False,
                "message": "No valid nails found for anemia detection (all nails have polish)",
                "nails_detected": len(nail_crops),
                "plain_nails": plain_nails_count,
                "polished_nails": polished_nails_count,
                "nail_details": nail_results
            }

        final_prediction, avg_probability = aggregate_predictions(anemia_preds, method=agg_method)
        
        # Step 4: Return comprehensive results
        return {
            "success": True,
            "final_diagnosis": "Anemic" if final_prediction == 1 else "Non-Anemic",
            "confidence": float(avg_probability),
            "aggregation_method": agg_method,
            "nails_detected": len(nail_crops),
            "plain_nails": plain_nails_count,
            "polished_nails": polished_nails_count,
            "nail_details": nail_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing image: {str(e)}"
        }

# ========== API Routes ==========

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "Anemia Detection API is running",
        "endpoints": {
            "POST /predict": "Upload hand image for anemia detection",
            "GET /health": "Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "yolo_detector": nail_detector is not None,
            "nail_classifier": nail_classifier is not None,
            "anemia_model": anemia_model is not None
        }
    })

@app.route('/predict', methods=['POST'])
def predict_anemia_api():
    """Main prediction endpoint"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No image file provided. Please upload an image with key 'image'"
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400
        
        # Get aggregation method from request (optional)
        agg_method = request.form.get('aggregation_method', 'mean_prob')
        if agg_method not in ['majority', 'mean_prob']:
            agg_method = 'mean_prob'
        
        # Read and process image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process the image
        result = process_hand_image_api(image_pil, agg_method=agg_method)
        
        # Return appropriate status code
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
def predict_anemia_base64():
    """Alternative endpoint for base64 encoded images"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "message": "No base64 image data provided. Please send JSON with 'image' key containing base64 encoded image"
            }), 400
        
        # Get aggregation method
        agg_method = data.get('aggregation_method', 'mean_prob')
        if agg_method not in ['majority', 'mean_prob']:
            agg_method = 'mean_prob'
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Invalid base64 image data: {str(e)}"
            }), 400
        
        # Process the image
        result = process_hand_image_api(image_pil, agg_method=agg_method)
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("Loading models...")
    print("✅ All models loaded successfully!")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=2000)