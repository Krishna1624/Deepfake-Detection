import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pipeline import detect_deepfake

def debug_evaluation():
    model_path = "saved_models/deepfake_model.h5"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Check some files from FF++
    dataset_path = r"C:\Users\Karthik\Desktop\FF++"
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return

    classes = {'real': 0, 'fake': 1}
    results = []

    for class_name, true_label in classes.items():
        v_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(v_dir):
            continue
            
        print(f"\nEvaluating {class_name} class...")
        v_files = [f for f in os.listdir(v_dir) if f.endswith(('.mp4', '.avi', '.mov'))][:5]
        
        for f in v_files:
            v_path = os.path.join(v_dir, f)
            print(f"Testing {f}...")
            try:
                out = detect_deepfake(v_path)
                print(f"Result: {out['prediction']} (Score: {out.get('raw_score', 'N/A')}, Confidence: {out['confidence']}%)")
                results.append((class_name, out['prediction'], out.get('raw_score', 0)))
            except Exception as e:
                print(f"Error testing {f}: {e}")

    if results:
        fake_scores = [r[2] for r in results if r[0] == 'fake']
        real_scores = [r[2] for r in results if r[0] == 'real']
        print("\nSummary:")
        print(f"Average score for Real: {np.mean(real_scores) if real_scores else 'N/A'}")
        print(f"Average score for Fake: {np.mean(fake_scores) if fake_scores else 'N/A'}")

if __name__ == "__main__":
    debug_evaluation()
