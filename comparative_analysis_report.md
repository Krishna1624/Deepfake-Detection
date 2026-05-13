# Comparative Analysis: Alternative Technologies & Exclusions

A critical part of engineering is choosing the *right* tools, not just any tools. Below is a report on technologies we **considered but ultimately rejected**, along with the strategic reasoning for those decisions.

---

## 1. Frontend & Dashboard
### **Rejected: Django / Flask + React**
*   **Why?**: While extremely powerful for enterprise scale, they require **extensive boilerplate** (API routing, state management, separate frontend/backend builds).
*   **The Trade-off**: For this forensic application, the priority was **real-time data visualization** and **rapid iteration**. 
*   **Verdict**: **Streamlit** was chosen because it allows us to focus 100% on the AI logic while still providing a premium, interactive user experience with minimal overhead.

---

## 2. Spatial Feature Extraction (CNN)
### **Rejected: VGG16 / VGG19**
*   **Why?**: These are older architectures that are **computationally heavy** and have a very large number of parameters (~138M).
*   **The Trade-off**: VGG models tend to overfit on smaller datasets and are not as "sharp" at detecting the fine-grained texture artifacts that Xception captures.
*   **Verdict**: **Xception** was chosen for its **Depthwise Separable Convolutions**, which are more efficient and better suited for detecting subtle GAN-generated noise.

### **Rejected: Vision Transformers (ViT)**
*   **Why?**: ViTs are the cutting edge of AI, but they require **massive amounts of data** (millions of images) to beat a CNN.
*   **The Trade-off**: Without a massive GPU cluster and a multi-petabyte dataset, a ViT would perform worse than our optimized CNN.
*   **Verdict**: Not suitable for this project's scale.

---

## 3. Temporal Modeling (RNN)
### **Rejected: GRU (Gated Recurrent Units)**
*   **Why?**: GRUs are a simpler version of LSTMs. They use fewer gates and are faster to train.
*   **The Trade-off**: However, GRUs can struggle with the **very long-term dependencies** required to track "micro-flickers" across 50+ frames.
*   **Verdict**: **LSTM** was chosen because its separate "Cell State" provides a more robust memory for detecting subtle temporal inconsistencies in video.

---

## 4. Optimization Algorithms
### **Rejected: Genetic Algorithms (GA)**
*   **Why?**: GAs are powerful but can be **stochastic and slow to converge**. They "mutate" and "evolve" solutions over many generations.
*   **The Trade-off**: PSO (Particle Swarm Optimization) mimics social behavior (flocking) and typically **converges much faster** to a global minimum for hyperparameter tuning.
*   **Verdict**: **PSO** was chosen for its efficiency in high-dimensional search spaces like neural network thresholds.

---

## 5. Face Detection
### **Rejected: Haar Cascades (Default OpenCV)**
*   **Why?**: This is a very old technology (2001). It is fast but has **high false-positive rates** and fails if the face is tilted or in low light.
*   **The Trade-off**: MTCNN uses deep learning to detect not just the face, but also five landmarks (eyes, nose, mouth), making it far more reliable for forensic analysis.
*   **Verdict**: **MTCNN** was chosen for its modern accuracy and robustness against occlusion.

---

## 6. Data Realism
### **Rejected: Random Over-sampling**
*   **Why?**: Simply duplicating existing images to balance the dataset leads to **heavy overfitting**—the model just memorizes the specific images we copied.
*   **The Trade-off**: SMOTE (Synthetic Minority Over-sampling Technique) creates **interpolated, synthetic data points** in the feature space, which forces the model to learn the *general rule* rather than just memorizing images.
*   **Verdict**: **SMOTE** was chosen for better generalization.

---
*Analysis Compiled: Secure Lens AI Engineering Team*
