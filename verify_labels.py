import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from models.rnn_model import AttentionLayer, focal_loss

def verify():
    model_paths = ["saved_models/deepfake_model_best.h5", "saved_models/deepfake_model.h5"]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"\n[!] Skipping {model_path} (not found)")
            continue

        print(f"\n" + "="*50)
        print(f" TESTING MODEL: {model_path}")
        print("="*50)
        
        try:
            model = load_model(model_path, custom_objects={
                'AttentionLayer': AttentionLayer,
                'focal_loss': focal_loss(),
                'loss_fn': focal_loss()
            })
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        # We must re-import pipeline objects to ensure they use the newly loaded model if needed
        # Or even better, we check the raw score ourselves since pipeline uses a singleton global model
        import pipeline
        pipeline.model = model # Force the pipeline to use the model we just loaded

        # Test sample sets
        test_cases = {
            'real': os.path.join('FF++', 'real'),
            'fake': os.path.join('FF++', 'fake')
        }

        for class_name, path in test_cases.items():
            print(f"\n--- Class: {class_name.upper()} ---")
            if not os.path.exists(path):
                print(f"Directory {path} not found.")
                continue
                
            files = sorted([f for f in os.listdir(path) if f.endswith('.mp4')])[:5]
            for f in files:
                v_path = os.path.join(path, f)
                try:
                    res = pipeline.detect_deepfake(v_path)
                    print(f" {f:40} -> Result: {res['prediction']} (UI Score: {res['raw_score']:.3f})")
                except Exception as e:
                    print(f" Error on {f}: {e}")

if __name__ == "__main__":
    verify()
