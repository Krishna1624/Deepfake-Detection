# Video Link: 
https://youtu.be/geppqdT9Bhk?si=24Y2aIB31PmqGI64

<div align="center">
### Deepfake Media Detection using Hybrid CNN–RNN and Particle Swarm Optimization (PSO)

AI-Powered Deepfake Detection System for Spatial and Temporal Video Forensics

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Deep Learning](https://img.shields.io/badge/AI-DeepfakeDetection-0A66C2?style=flat-square)

</div>

---

## 📌 Overview

Secure Lens AI is a deep learning-based forensic framework developed for detecting manipulated and AI-generated video content (deepfakes).

The system combines:
- **Convolutional Neural Networks (CNNs)** for spatial artifact detection
- **Bidirectional LSTM (Bi-LSTM)** networks for temporal sequence analysis
- **Particle Swarm Optimization (PSO)** for intelligent hyperparameter tuning

The framework analyzes both:
- frame-level inconsistencies
- motion-based anomalies

to identify forged facial media with improved reliability.

The project was developed using TensorFlow, OpenCV, and Streamlit, with support for automated face localization, feature extraction, and real-time forensic inference.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎭 Deepfake Detection | Detects manipulated facial videos using deep learning |
| 🧠 Hybrid CNN–RNN Architecture | Combines Xception CNN with Bi-LSTM sequence modeling |
| 👁️ Face Localization | MTCNN-based face detection and preprocessing |
| 🎞️ Temporal Analysis | Detects frame-to-frame inconsistencies and motion artifacts |
| ⚡ PSO Optimization | Automated hyperparameter tuning using Particle Swarm Optimization |
| 📊 Frame-Level Analysis | Confidence scoring for individual video frames |
| 🌐 Streamlit Dashboard | Interactive forensic web interface for video uploads |
| 📈 Performance Evaluation | Accuracy, Recall, Specificity, and F1-score analysis |

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────┐
│                    Input Video                      │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│          Video Preprocessing & Frame Extraction     │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│              MTCNN Face Detection Layer             │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│         Xception CNN Feature Extraction             │
│        Spatial Artifact Representation              │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│           Bi-LSTM Temporal Analysis Layer           │
│          Sequence & Motion Inconsistency            │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│      Particle Swarm Optimization (PSO) Layer        │
│         Hyperparameter Optimization                 │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│               Deepfake Classification               │
│                 REAL / FAKE Verdict                 │
└──────────────────────────────────────────────────────┘
```

---

## 🔄 System Workflow

```text
1. VIDEO INPUT
   └─▶ Upload input video through Streamlit interface

2. FRAME EXTRACTION
   └─▶ Extract temporal frame sequences from video

3. FACE LOCALIZATION
   └─▶ Detect and crop faces using MTCNN

4. SPATIAL FEATURE EXTRACTION
   └─▶ Extract deep spatial features using Xception CNN

5. TEMPORAL ANALYSIS
   └─▶ Analyze frame sequences using Bi-LSTM

6. PSO OPTIMIZATION
   └─▶ Optimize hyperparameters for improved detection

7. FINAL CLASSIFICATION
   └─▶ Predict whether media is REAL or FAKE

8. RESULT VISUALIZATION
   └─▶ Display confidence scores and frame-level analysis
```

---

## 🛠️ Technology Stack

### Deep Learning & AI

| Technology | Role |
|---|---|
| TensorFlow / Keras | Deep learning framework |
| Xception CNN | Spatial feature extraction |
| Bi-LSTM | Temporal sequence modeling |
| PSO | Hyperparameter optimization |
| MTCNN | Face detection |

### Computer Vision

| Technology | Role |
|---|---|
| OpenCV | Video processing and frame extraction |
| NumPy | Numerical operations |
| Matplotlib | Visualization and analysis |

### Web Interface

| Technology | Role |
|---|---|
| Streamlit | Interactive forensic dashboard |
| Python | Backend logic and orchestration |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 76.25% |
| Sensitivity (Recall) | 65.00% |
| Specificity | 87.50% |
| F1 Score | 73.24% |

The model achieved strong specificity performance, reducing false-positive classifications for genuine media.

---

## 🗂️ Project Structure

```text
projectfinal/
├── adapters/
│   ├── CLAUDE.md
│   ├── GEMINI.md
│   └── GPT_OSS.md
│
├── assets/
│   ├── pso.png
│   ├── pso_fitness_evolution.png
│   ├── pso_iter_00.png
│   ├── pso_iter_01.png
│   ├── pso_iter_02.png
│   ├── pso_iter_03.png
│   ├── pso_iter_04.png
│   ├── pso_iter_05.png
│   ├── pso_iter_06.png
│   ├── pso_iter_07.png
│   ├── pso_iter_08.png
│   ├── pso_iter_09.png
│   ├── pso_iter_10.png
│   ├── pso_iter_11.png
│   ├── pso_iter_12.png
│   ├── pso_iter_13.png
│   ├── pso_iter_14.png
│   ├── smote.png
│   └── smote_visualization.png
│
├── dataset
│
├── docs/
│   ├── model-selection-playbook.md
│   ├── runbook.md
│   └── token-optimization-guide.md
│
├── feature_engineering/
│   ├── __pycache__/
│   ├── balancing.py
│   ├── cnn_features.py
│   └── optimization.py
│
├── logs/
│   └── train/
│       └── events.out.tfevents.*
│
├── models/
│   ├── __pycache__/
│   └── rnn_model.py
│
├── preprocessing/
│   ├── __pycache__/
│   └── preprocessing.py
│
├── scripts/
│   ├── search_repo.ps1
│   ├── search_repo.sh
│   ├── setup_search.ps1
│   ├── setup_search.sh
│   ├── validate-all.ps1
│   ├── validate-all.sh
│   ├── validate-skills.ps1
│   ├── validate-skills.sh
│   ├── validate-templates.ps1
│   ├── validate-templates.sh
│   ├── validate-workflows.ps1
│   └── validate-workflows.sh
│
├── utils/
│   ├── __pycache__/
│   ├── face_detection.py
│   └── video_processing.py
│
├── Detection of Deepfake.pdf
├── PROJECT_RULES.md
├── README.md
├── app.py
├── comparative_analysis_report.md
├── debug_model.py
├── evaluate_model.py
├── fine_tune.py
├── metrics.txt
├── metrics_out.txt
├── model_capabilities.yaml
├── pipeline.py
├── project_report.md
├── requirements.txt
├── technology_stack_report.md
├── temp_video.mp4
├── test_pipeline.py
├── train.py
├── visualize_pso.py
├── visualize_smote.py
└── yolov8n.pt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Streamlit
- 
## ▶️ Running the Application

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## 🔮 Future Enhancements

- Audio-visual deepfake detection
- Vision Transformer (ViT)-based architecture
- Real-time live-stream analysis
- Multi-face simultaneous detection
- Lightweight mobile deployment
- GAN fingerprint analysis
- Cloud-based inference pipeline

---

## 📂 Dataset Information

The project was trained and evaluated using publicly available deepfake research datasets including:

- FaceForensics++
- Deepfake benchmark datasets

Datasets are not included in this repository due to storage limitations and licensing considerations.

---

## 🏛️ Research Inspiration

This project was inspired by recent research in:
- CNN-RNN hybrid architectures
- Deepfake temporal forensics
- Particle Swarm Optimization for deep learning

---

<div align="center">

Built with 🎭 AI for digital media forensics.

</div>



