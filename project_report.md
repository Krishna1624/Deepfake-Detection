# Deepfake Detection System: Project Report

## 1. Introduction
With the rapid advancement of Generative AI, deepfakes have become a significant threat to digital integrity. This project aims to develop a robust deepfake detection system that identifies manipulated video content by analyzing spatial and temporal inconsistencies using deep learning architectures.

## 2. Project Architecture
The system is divided into three major modules: Preprocessing, Feature Extraction (Spatial), and Sequence Learning (Temporal).

### 2.1 Preprocessing Pipeline
- **Face Detection**: Utilizing YOLOv8/MTCNN to localize faces in each frame.
- **Normalization**: Resizing detected face crops to a standard (e.g., 224x224) and normalizing pixel values.
- **Frame Selection**: Extracting a fixed number of frames (e.g., 20) per video to ensure consistent temporal analysis.

### 2.2 Feature Engineering & Extraction
- **CNN Base**: Leveraged pre-trained models (VGG16/ResNet) to extract high-level spatial features from facial crops.
- **Feature Optimization**: Used Particle Swarm Optimization (PSO) to tune hyperparameters and feature selection for improved differentiation between real and fake content.
- **Data Balancing**: Implemented Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance in the training dataset.

### 2.3 Model Architecture
- **LSTM Networks**: Used Long Short-Term Memory layers to capture temporal cues (e.g., unnatural eye blinking, mouth movements) that single-frame CNNs often miss.
- **Softmax Classification**: A final dense layer with softmax activation to output the "REAL" or "FAKE" verdict.

## 3. Implementation Details
### 3.1 Backend (Python)
- **Framework**: Flask/FastAPI for the inference server.
- **Libraries**: `OpenCV`, `PyTorch` / `TensorFlow`, `NumPy`, `Scikit-learn`.
- **Pipeline (`pipeline.py`)**: A unified class that handles video upload, preprocessing, model inference, and forensic reporting.

### 3.2 Frontend (UI/UX)
- **Modern Design**: A premium, responsive dashboard built with Vanilla CSS/JS.
- **Interactive Results**: Real-time probability scoring and classification (REAL/FAKE) with visual indicators.

## 4. Methodology & Optimization
### 4.1 Fine-Tuning Strategy
- Targeted fine-tuning of the LSTM layers on custom datasets (e.g., Celeb-DF, FaceForensics++) to adapt to modern deepfake generation techniques.
- Implementation of dropout and weight decay to prevent overfitting.

### 4.2 Particle Swarm Optimization (PSO)
- Optimized the weights of the feature fusion layer to maximize the F1-score across a diverse test set.
- Reduced false positives by calibrating the decision threshold based on PSO results.

### 4.3 Handling Class Imbalance
- Used **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic features for the minority class to significantly reduce classification bias.

## 5. Results & Evaluation
The model was evaluated using accuracy, sensitivity, and specificity.
- **Accuracy**: Highly reliable performance on unseen test data.
- **F1-Score**: Optimized via PSO to ensure balance between precision and recall.
- **Forensic Report**: Detailed breakdown of frame-by-frame analysis with a final high-confidence verdict.

## 6. Conclusion
The developed system provides a reliable and scalable solution for deepfake detection. By combining spatial CNN features with temporal LSTM analysis, the model effectively identifies deepfakes that are visually indistinguishable to the human eye. Future work could involve extending this to multi-modal analysis (audio-visual) for even higher reliability.

---
*Developed for: Deepfake Detection Project Submission*
