# 👁 Diabetic Retinopathy Prediction System

An AI-powered web application that detects **Diabetic Retinopathy (DR)** from retinal fundus images using a Convolutional Neural Network (CNN). Built with Python, TensorFlow/Keras, Scikit-learn, OpenCV, SQLite, Flask, and pure HTML/CSS.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [CLI Prediction](#cli-prediction)
- [Features](#features)
- [DR Classification Stages](#dr-classification-stages)
- [Database Schema](#database-schema)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## 🎯 Project Overview

Diabetic Retinopathy is a diabetes complication that affects the eyes and is a leading cause of blindness worldwide. Early detection is critical. This system automates screening using deep learning to classify retinal images into five severity stages.

**Key capabilities:**
- Upload retinal fundus images for instant AI analysis
- Classify into 5 DR stages with confidence percentage
- Store patient records and prediction history in SQLite
- Generate downloadable PDF medical reports
- Search and manage patient database
- Visualize model performance (accuracy, loss, confusion matrix)

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.x / Keras (MobileNetV2) |
| ML Utilities | Scikit-learn (metrics, reports) |
| Data Processing | Pandas, NumPy |
| Image Processing | OpenCV, Pillow |
| Database | SQLite3 (via Python `sqlite3`) |
| Web Framework | Flask (lightweight) |
| PDF Generation | ReportLab |
| Visualization | Matplotlib, Seaborn |
| Frontend | Pure HTML5 + CSS3 + Minimal JS |

---

## 📁 Project Structure

```
diabetic_retinopathy_system/
│
├── main.py                  # Flask web application (routes)
├── train_model.py           # CNN model training script
├── predict.py               # Prediction / inference module
├── database.py              # SQLite database operations
├── setup.py                 # One-click project setup
├── requirements.txt         # Python dependencies
│
├── utils/
│   ├── __init__.py
│   ├── image_processing.py  # OpenCV preprocessing pipeline
│   ├── report_generator.py  # ReportLab PDF generator
│   ├── visualizations.py    # Matplotlib chart generator
│   └── demo_data.py         # Database seeder for demo
│
├── templates/               # Jinja2 HTML templates
│   ├── base.html            # Base layout with sidebar
│   ├── home.html            # Dashboard / landing page
│   ├── predict.html         # Upload & prediction form
│   ├── result.html          # Prediction results page
│   ├── patients.html        # Patient records listing
│   ├── patient_detail.html  # Individual patient profile
│   ├── history.html         # Prediction history
│   ├── about.html           # About page
│   ├── contact.html         # Contact page
│   └── 404.html             # Error page
│
├── static/
│   ├── css/style.css        # All styles (dark medical theme)
│   ├── js/app.js            # Minimal JavaScript
│   └── images/              # Generated charts (auto-created)
│
├── uploads/                 # Uploaded retinal images
├── reports/                 # Generated PDF reports
├── trained_model/           # Saved .h5 model
├── dataset/                 # Training/validation images
│   ├── train/
│   │   ├── No_DR/
│   │   ├── Mild/
│   │   ├── Moderate/
│   │   ├── Severe/
│   │   └── Proliferative_DR/
│   └── val/
│       └── (same structure)
└── sql/
    ├── schema.sql           # SQL schema reference
    └── retinopathy.db       # SQLite database (auto-created)
```

---

## ⚙ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- VS Code (recommended)
- 4 GB RAM minimum (8 GB recommended for training)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/diabetic-retinopathy-system.git
cd diabetic-retinopathy-system
```

### Step 2 — Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠ If you encounter issues with TensorFlow on Apple Silicon (M1/M2):
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### Step 4 — Run One-Click Setup

```bash
python setup.py
```

This will:
- Create all required directories
- Initialize the SQLite database
- Seed demo patient data
- Generate placeholder dashboard charts

### Step 5 — Launch the Application

```bash
python main.py
```

Open your browser and navigate to: **http://localhost:5000**

---

## 🗂 Dataset Setup

### Option A — APTOS 2019 (Recommended)

1. Download from [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
2. Extract images and CSV files
3. Run the organizer script:

```bash
python utils/organize_dataset.py --csv train.csv --img_dir train_images/ --out dataset/
```

### Option B — EyePACS Dataset

1. Download from [Kaggle EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection)
2. Follow the same organization steps above

### Option C — Demo Synthetic Dataset (No Download Needed)

The training script auto-generates a synthetic dataset for testing:

```bash
python train_model.py
# When prompted about missing dataset, it will create synthetic images
```

### Expected Dataset Structure

```
dataset/
├── train/
│   ├── No_DR/          (label 0 — ~1800 images)
│   ├── Mild/           (label 1 — ~370 images)
│   ├── Moderate/       (label 2 — ~999 images)
│   ├── Severe/         (label 3 — ~193 images)
│   └── Proliferative_DR/ (label 4 — ~295 images)
└── val/
    └── (same 5 folders, 20% split)
```

> 💡 **Tip:** Use the APTOS 2019 `train.csv` diagnosis column:
> - 0 = No_DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative_DR

---

## 🧠 Training the Model

```bash
python train_model.py
```

**What happens during training:**

1. **Phase 1** (Epochs 1–15): Train only the classification head while the MobileNetV2 base is frozen
2. **Phase 2** (Epochs 16–30): Fine-tune the top 30 layers of MobileNetV2 with a lower learning rate
3. Best model is saved automatically to `trained_model/dr_model.h5`
4. Training plots, confusion matrix, and classification report are generated

**Training Configuration** (edit `train_model.py` to change):

```python
IMG_SIZE   = (224, 224)   # Input image dimensions
BATCH_SIZE = 32           # Batch size
EPOCHS     = 30           # Total epochs
LR         = 1e-4         # Initial learning rate
```

**Expected Training Time:**
- GPU (NVIDIA): ~15–30 minutes
- CPU only: ~2–5 hours (reduce EPOCHS to 10 for faster testing)

---

## 🚀 Running the Application

```bash
# Activate virtual environment first
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Start the server
python main.py
```

Open: **http://localhost:5000**

**Available Pages:**

| URL | Page |
|-----|------|
| `/` | Dashboard with stats and charts |
| `/predict` | Upload image and get prediction |
| `/patients` | Patient records with search |
| `/patients/<id>` | Individual patient profile |
| `/history` | All prediction history |
| `/report/<pred_id>` | Download PDF report |
| `/about` | About the system |
| `/contact` | Contact page |

---

## 💻 CLI Prediction

Predict directly from the command line without the web UI:

```bash
python predict.py path/to/retinal_image.jpg
```

**Example output:**
```
Running prediction on: sample_retina.jpg

==================================================
  Predicted Class : Moderate
  Confidence      : 82.34%
  Severity        : Moderate Non-Proliferative DR
  Risk Level      : Moderate

  Advice:
  Moderate retinopathy detected. Consult ophthalmologist within 3-6 months.
  Strict glycemic control is critical...

  All Probabilities:
    No DR                 4.21%  █
    Mild                 10.58%  ██
    Moderate             82.34%  ████████████████
    Severe                2.11%
    Proliferative DR      0.76%
==================================================
```

---

## ✨ Features

### AI / Machine Learning
- ✅ MobileNetV2 CNN with transfer learning
- ✅ CLAHE retinal image enhancement
- ✅ Data augmentation (flip, rotate, brightness)
- ✅ Two-phase training (freeze → fine-tune)
- ✅ EarlyStopping + ReduceLROnPlateau callbacks
- ✅ Classification report + confusion matrix
- ✅ Model saved as `.h5`

### Web Application
- ✅ Image upload with drag-and-drop preview
- ✅ Prediction with confidence % and risk level
- ✅ Patient registration and management
- ✅ Search patients by name/email/phone
- ✅ Full prediction history
- ✅ PDF report generation & download
- ✅ Dashboard with live statistics
- ✅ Responsive dark medical UI

### Database
- ✅ SQLite with 4 normalized tables
- ✅ CRUD operations for patients
- ✅ Prediction history with foreign keys
- ✅ Demo data seeder

---

## 🔬 DR Classification Stages

| Stage | Label | Description | Action |
|-------|-------|-------------|--------|
| 0 | No DR | No signs of retinopathy | Annual screening |
| 1 | Mild | Microaneurysms only | 9-month follow-up |
| 2 | Moderate | More than just microaneurysms | 6-month follow-up |
| 3 | Severe | Any severe NPDR sign | 1-month urgent referral |
| 4 | Proliferative | New vessel formation | Immediate treatment |

---

## 🗄 Database Schema

```sql
patients            (patient_id, full_name, age, gender, email, phone, address)
uploaded_images     (image_id, patient_id → FK, filename, filepath)
prediction_history  (prediction_id, patient_id → FK, image_id → FK,
                     predicted_class, confidence, severity_level, advice)
reports             (report_id, patient_id → FK, prediction_id → FK,
                     report_filename, report_path)
```

See `sql/schema.sql` for full DDL with indexes and sample queries.

---

## 🔮 Future Improvements

1. **Grad-CAM Visualization** — Heatmap overlay showing which retinal regions activated the model
2. **Multi-image Analysis** — Batch upload and compare both eyes
3. **REST API** — Full JSON API for mobile app integration
4. **User Authentication** — Doctor login with role-based access
5. **Cloud Storage** — AWS S3 / Google Cloud for image storage
6. **DICOM Support** — Accept medical imaging standard files
7. **Ensemble Model** — Combine EfficientNet + ResNet for higher accuracy
8. **Email Notifications** — Alert doctors for high-risk patients
9. **Export to Excel** — Bulk patient record export
10. **Docker Deployment** — Containerized for easy cloud hosting

---

## ⚠ Disclaimer

This system is developed for **educational and research purposes only**. It is **NOT** a certified medical device and should **NOT** be used as the sole basis for clinical diagnosis or treatment decisions.

Always consult a qualified ophthalmologist for proper eye examination and medical advice.

---

## 👨‍💻 Author

Built as a final year AI/ML project.  
Technologies: Python • TensorFlow • Scikit-learn • OpenCV • Flask • SQLite • HTML/CSS

**GitHub:** [github.com/yourusername/dr-predict](https://github.com/Poorvichauhan07/dr-predict)

---

*Made with ❤ for early disease detection and better healthcare outcomes.*
