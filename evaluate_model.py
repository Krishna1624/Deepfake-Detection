import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from feature_engineering.balancing import balance_dataset
from train import load_cached_features, CACHE_DIR

def main():
    print("=" * 60)
    print("   DEEPFAKE DETECTION - MODEL EVALUATION")
    print("=" * 60)

    # 1. Load cached features
    print("\nLoading cached features from disk...")
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
    if not cache_files:
        print(f"ERROR: No cached features found in '{CACHE_DIR}'.")
        print("Please run `train.py` first to generate feature caches.")
        return

    X, y = load_cached_features(cache_files)
    print(f"  Total samples loaded: {X.shape[0]}")
    
    # 2. Balance dataset (must match train.py to get the exact same test set)
    # SMOTE random state is fixed at 42 inside balance_dataset
    print("\nApplying SMOTE-DLPB to match training shape...")
    X_bal, y_bal = balance_dataset(X, y)

    # 3. Train/Test split (must match train.py exactly)
    # This ensures we are testing on the unseen test set that was used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    print(f"  Test set size: {len(X_test)}")

    # 4. Load the model
    model_path = "saved_models/deepfake_model_best.h5"
    if not os.path.exists(model_path):
        model_path = "saved_models/deepfake_model.h5"
        if not os.path.exists(model_path):
            print("ERROR: Could not find trained model in 'saved_models/'")
            return

    from models.rnn_model import AttentionLayer, focal_loss
    custom_objects = {
        'AttentionLayer': AttentionLayer,
        'loss_fn': focal_loss(),
        'focal_loss': focal_loss(),
        'AUC': tf.keras.metrics.AUC,
        'Precision': tf.keras.metrics.Precision,
        'Recall': tf.keras.metrics.Recall
    }
    
    print(f"\nLoading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # 5. Evaluate and Predict
    print("\nEvaluating model...")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 6. Calculate Metrics
    acc_val = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred) # Sensitivity is the same as Recall
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    with open("metrics_out.txt", "w", encoding="utf-8") as f:
        f.write("  -- Detailed Metrics ------------------------\n")
        f.write(f"  Accuracy    : {acc_val*100:.2f}%\n")
        f.write(f"  Sensitivity : {sensitivity*100:.2f}%  (True Positive Rate)\n")
        f.write(f"  Specificity : {specificity*100:.2f}%  (True Negative Rate)\n")
        f.write(f"  F1 Score    : {f1*100:.2f}%\n")
        f.write(f"  Confusion Matrix: T.Pos={tp}, T.Neg={tn}, F.Pos={fp}, F.Neg={fn}\n")
    
    print("Metrics written to metrics_out.txt")

if __name__ == "__main__":
    main()
