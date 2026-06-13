# Anemia Detection via Nail Bed Analysis

> **Note:** This project is under active development. The nail polish classifier is not fully reliable yet. Do not consider current predictions for clinical use.

An AI-powered system that predicts **anemia** from nail bed images using a three-stage deep learning pipeline. It combines **YOLOv8 for nail detection**, **CNN-based polish classification**, and a **custom PyTorch model** for anemia prediction. Easily test via CLI or integrate using a Flask API.

---

## Features

* **Smart Nail Detection** – YOLOv8 model auto-detects and crops nails from hand images
* **Polish Classification** – CNN classifier filters out polished nails for clean predictions
* **Anemia Prediction** – PyTorch model infers anemia status from nail bed color and texture
* **Batch Inference** – CLI support for folder-based image prediction
* **API Support** – RESTful Flask API for web & mobile integration
* **Visual Feedback** – Displays predictions with confidence and overlays

---

## Architecture

```
Hand Image
   ↓
YOLOv8 → Cropped Nails
   ↓
CNN → Polish-Free Nail Filter
   ↓
Anemia CNN → Diagnosis (Anemic / Non-Anemic)
```

---

## Project Structure

```
anemia-nail-detection/
├── anemia_pipeline.py         # CLI interface for local testing
├── app.py                     # Flask API server
├── app_base64.py              # Flask API server (cropped images data in base64)
├── models/                    # Model weights
│   ├── yolov8_nail_best.pt
│   ├── nail_cnn_classifier_model.h5
│   └── anemia_cnn_model2.pth
├── test_images/               # Sample hand images for inference
│   ├── hand1.jpg
│   └── hand2.jpg
└── requirements.txt
```

---

## Quick Start

### Prerequisites

* Python 3.8+
* `pip` (Python package manager)

### Installation

```bash
git clone https://github.com/Ramprasad4121/anemia-nail-detection.git
cd anemia-nail-detection

# Create and activate a virtual environment
python -m venv anemia_env
source anemia_env/bin/activate  # Windows: anemia_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Ensure model files are in the `models/` directory.

---

## Usage

### Method 1: Command Line (Local Testing)

```bash
python anemia_pipeline.py
```

* Processes all images in `test_images/`
* Crops nails and filters out polished ones
* Predicts and displays anemia diagnosis
* Supports: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

### Method 2: REST API (Flask + Postman)

```bash
python app.py       ## without base64
```
```bash
python app_base64.py       ## with base64
```

* Server runs at: `http://localhost:2000`

#### Endpoints

| Method | Endpoint   | Description                 |
| ------ | ---------- | --------------------------- |
| GET    | `/health`  | Health check                |
| POST   | `/predict` | Upload image (form-data)    |
| POST   | `/predict` | Base64 encoded image (JSON) |

---

## Public API via Ngrok

Open two terminals:

**Terminal 1:**

```bash
python app.py
```

**Terminal 2:**

```bash
ngrok http 2000
```

Use the provided ngrok URL in Postman:

```http
POST https://your-ngrok-id.ngrok.io/predict
```

---

## Sample Output

```
Starting Anemia Detection Pipeline...
Models Loaded

Processing: hand1.jpg
Detected 5 nails
2 polished nails skipped
Final Diagnosis: NON-ANEMIC
```

---

## Module Performance

| Module                  | Accuracy |
| ----------------------- | -------- |
| Nail Detection (YOLOv8) | \~95%    |
| Polish Classifier (CNN) | \~92%    |
| Anemia Detector (CNN)   | \~88%    |
| Avg. Time per Image     | \~2–3s   |

---

## Requirements

```
ultralytics==8.0.196
tensorflow==2.13.0
torch==2.0.1
torchvision==0.15.2
matplotlib==3.7.2
pillow==10.0.0
opencv-python==4.8.0.76
numpy==1.24.3
flask==2.3.2
```

---

## Pro Tips

* Use high-resolution, well-lit hand images
* Ensure nails are free of polish for accurate detection
* Frame the entire hand with visible nail beds

---

## Contributing

```bash
git checkout -b feature/your-feature
git commit -m "Add feature"
git push origin feature/your-feature
```

Then open a Pull Request

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

* YOLOv8
* TensorFlow
* PyTorch
* Medical research on non-invasive anemia screening

---

## Support

* Email: [ramprasadgoud34@gmail.com](mailto:ramprasadgoud34@gmail.com)
* Or create an issue on GitHub

---

## Medical Disclaimer

This system is still in **development stage**. It is **not a substitute** for professional medical advice or diagnosis. Always consult qualified healthcare providers.
