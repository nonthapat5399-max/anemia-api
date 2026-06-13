

############################################### ANEMIA DETECTION PIPELINE ###############################################

# ========== Install dependencies ==================================
# pip install ultralytics tensorflow torch torchvision matplotlib pillow opencv-python
#pip install ultralytics
# pip install tensorflow
# pip install torch torchvision
# pip install matplotlib
# pip install pillow
# pip install opencv-python

#========================= folder structure =========================

# your_project/
# ├── anemia_pipeline.py  # your main python file
# ├── models/
# │   ├── yolov8_nail_best.pt
# │   ├── nail_cnn_classifier_model.h5
# │   └── anemia_cnn_model2.pth
# ├── test_images/
# │   └── hand1.jpg
# │   └── hand2.jpg

#========================= cmd to run in terminal =========================
# python anemia_pipeline.py

######################################### ANEMIA DETECTION PIPELINE #######################################################




import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ========== MODEL PATHS ==========
YOLO_PATH = "./models/yolov8_nail_best.pt"
CLASSIFIER_PATH = "./models/nail_cnn_classifier_model.h5"
ANEMIA_MODEL_PATH = "./models/anemia_cnn_model2.pth"

# ========== CHECK IF FILES EXIST ==========
def check_model_files():
    files = [YOLO_PATH, CLASSIFIER_PATH, ANEMIA_MODEL_PATH]
    for file in files:
        if not os.path.exists(file):
            print(f"❌ Model file not found: {file}")
            return False
    print("✅ All model files found!")
    return True

# ========== LOAD MODELS ==========
def load_models():
    print("🔄 Loading models...")
    
    # Load YOLO nail detector
    nail_detector = YOLO(YOLO_PATH)
    print("✅ YOLO model loaded")
    
    # Load CNN nail polish classifier (Keras)
    nail_classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    print("✅ Nail classifier model loaded")
    
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
    print("✅ Anemia model loaded")
    
    return nail_detector, nail_classifier, anemia_model

# ========== PREPROCESSING ==========
anemia_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========== HELPER FUNCTIONS ==========

def crop_nails_from_image(image_pil, nail_detector):
    """Crop nails from hand image using YOLO"""
    results = nail_detector(image_pil)
    crops = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped = image_pil.crop((x1, y1, x2, y2))
                crops.append((cropped, (x1, y1, x2, y2)))
    
    return crops

def check_nail_polish(nail_img, nail_classifier):
    """Check if nail has polish using CNN classifier"""
    img = nail_img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = nail_classifier.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    # Assuming: 0 = Plain, 1 = Polished
    return class_idx == 0, confidence

def predict_anemia_from_nail(nail_img, anemia_model):
    """Predict anemia from nail using PyTorch model"""
    nail_tensor = anemia_transform(nail_img)
    nail_tensor = nail_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = anemia_model(nail_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        class_idx = torch.argmax(probs).item()
        confidence = probs[0][class_idx].item()
    
    return class_idx, confidence

def display_cropped_nails(nail_results, filename):
    """Display cropped nails with their classifications"""
    if not nail_results:
        print("No nails detected!")
        return
    
    cols = min(5, len(nail_results))
    rows = int(np.ceil(len(nail_results) / cols))
    
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle(f"Cropped Nails from {filename}", fontsize=16, fontweight='bold')
    
    for i, (nail_img, is_plain, polish_conf, anemia_result) in enumerate(nail_results):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(nail_img)
        plt.axis('off')
        
        polish_status = "Plain" if is_plain else "Polished"
        
        if is_plain and anemia_result:
            anemia_class, anemia_conf = anemia_result
            anemia_status = "Anemic" if anemia_class == 1 else "Non-Anemic"
            title = f"Nail {i+1}\n{polish_status}\n{anemia_status}"
        else:
            title = f"Nail {i+1}\n{polish_status}\n(Skipped)"
        
        plt.title(title, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def process_single_image(image_path, nail_detector, nail_classifier, anemia_model):
    """Process a single hand image"""
    filename = os.path.basename(image_path)
    print(f"\n🖼️ Processing: {filename}")
    
    # Load image
    image_pil = Image.open(image_path).convert('RGB')
    
    # Step 1: Crop nails using YOLO
    print("🔄 Cropping nails...")
    nail_crops = crop_nails_from_image(image_pil, nail_detector)
    print(f"✅ Detected {len(nail_crops)} nails")
    
    if len(nail_crops) == 0:
        print("❌ No nails detected in image!")
        return "No nails detected"
    
    # Step 2: Check nail polish and predict anemia
    nail_results = []
    anemia_predictions = []
    plain_nails_count = 0
    
    for i, (nail_img, box) in enumerate(nail_crops):
        print(f"🔄 Processing nail {i+1}...")
        
        # Check if nail has polish
        is_plain, polish_conf = check_nail_polish(nail_img, nail_classifier)
        
        anemia_result = None
        if is_plain:
            plain_nails_count += 1
            # Predict anemia only for plain nails
            anemia_class, anemia_conf = predict_anemia_from_nail(nail_img, anemia_model)
            anemia_result = (anemia_class, anemia_conf)
            anemia_predictions.append(anemia_class)
            print(f"   ✅ Plain nail - Anemia prediction: {'Anemic' if anemia_class == 1 else 'Non-Anemic'}")
        else:
            print(f"   ⚠️ Polished nail - Skipped for anemia detection")
        
        nail_results.append((nail_img, is_plain, polish_conf, anemia_result))
    
    # Display cropped nails
    display_cropped_nails(nail_results, filename)
    
    # Step 3: Final decision
    if len(anemia_predictions) == 0:
        print("❌ No plain nails found for anemia detection!")
        return "Insufficient data - All nails have polish"
    
    print(f"📊 Plain nails analyzed: {plain_nails_count}")
    print(f"📊 Anemia predictions: {anemia_predictions}")
    
    # Use majority voting for final decision
    anemic_count = sum(anemia_predictions)
    non_anemic_count = len(anemia_predictions) - anemic_count
    
    if anemic_count > non_anemic_count:
        final_result = "ANEMIC"
    else:
        final_result = "NON-ANEMIC"
    
    print(f"📊 Final vote: {anemic_count} Anemic, {non_anemic_count} Non-Anemic")
    
    return final_result

def run_anemia_detection():
    """Main function to run anemia detection on all test images"""
    print("🚀 Starting Anemia Detection Pipeline...")
    
    # Check if model files exist
    if not check_model_files():
        print("❌ Please ensure all model files are in the ./models/ directory")
        return
    
    # Load models
    try:
        nail_detector, nail_classifier, anemia_model = load_models()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Check test images directory
    test_images_dir = "./test_images"
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(test_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    
    if not image_files:
        print("❌ No image files found in test_images directory!")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Process each image
    results = {}
    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        try:
            result = process_single_image(image_path, nail_detector, nail_classifier, anemia_model)
            results[image_file] = result
            print(f"🏥 FINAL DIAGNOSIS for {image_file}: {result}")
        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")
            results[image_file] = "Error"
    
    # Summary
    print("\n" + "="*50)
    print("📋 FINAL RESULTS SUMMARY")
    print("="*50)
    for image_file, result in results.items():
        print(f"{image_file}: {result}")
    print("="*50)

if __name__ == "__main__":
    run_anemia_detection()