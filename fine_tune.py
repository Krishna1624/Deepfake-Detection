import os
import gc
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.video_processing import extract_frames
from preprocessing.preprocessing import preprocess_frame
from feature_engineering.cnn_features import extract_features
from feature_engineering.balancing import balance_dataset

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ⚙️  CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW_DATASET_ROOT = r"C:\Users\Karthik\Desktop\New_Videos"  # Path to your NEW real/ and fake/ videos
CACHE_DIR        = "feature_cache_new"                     # Separate temp folder so we don't mess up FF++ cache
MODEL_PATH       = "saved_models/deepfake_model_best.h5"   # Your already-trained model
NEW_SAVE_PATH    = "saved_models/deepfake_model_finetuned.h5"

BATCH_SIZE     = 10
MAX_FRAMES     = 50
MIN_FRAMES     = 10
SEQUENCE_LEN   = 50
FEATURE_DIM    = 2048
EPOCHS         = 15      # Fewer epochs needed since it's already trained!
LEARNING_RATE  = 1e-5    # VERY LOW learning rate to avoid forgetting the old FF++ data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def process_video(v_path):
    # (Same as train.py)
    try:
        frames = extract_frames(v_path, max_frames=MAX_FRAMES)
        if not frames or len(frames) < MIN_FRAMES: return None
        processed = [preprocess_frame(f) for f in frames]
        features  = extract_features(np.array(processed))
        if features.shape[0] < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - features.shape[0], FEATURE_DIM))
            features = np.vstack([features, pad])
        else:
            features = features[:SEQUENCE_LEN]
        return features
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return None


def process_directory(class_dir, label, class_name):
    # Extracts features just for the NEW videos
    video_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    total = len(video_files)
    if total == 0: return []
    
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"\n  [{class_name.upper()}] {total} NEW videos → {n_batches} batches")
    
    saved_files = []
    for batch_idx in range(n_batches):
        batch = video_files[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        cache_file = os.path.join(CACHE_DIR, f"{class_name}_new_batch_{batch_idx:04d}.npz")
        
        if os.path.exists(cache_file):
            print(f"    Batch {batch_idx+1}/{n_batches} — CACHED, skipping")
            saved_files.append(cache_file)
            continue
            
        print(f"    Batch {batch_idx+1}/{n_batches} — processing {len(batch)} videos …", end="", flush=True)
        batch_X, batch_y = [], []
        for v_file in batch:
            v_path = os.path.join(class_dir, v_file)
            feat = process_video(v_path)
            if feat is not None:
                batch_X.append(feat)
                batch_y.append(label)
                
        if batch_X:
            np.savez_compressed(cache_file, X=np.array(batch_X, dtype=np.float32), y=np.array(batch_y, dtype=np.int8))
            saved_files.append(cache_file)
            print(f" ✓  [{len(batch_X)}/{len(batch)} ok]")
        
        del batch_X, batch_y
        gc.collect()
        
    return saved_files


def main():
    print("=" * 60)
    print("   DEEPFAKE DETECTION — FINE-TUNING ON NEW DATA")
    print("=" * 60)
    
    real_dir = os.path.join(NEW_DATASET_ROOT, "real")
    fake_dir = os.path.join(NEW_DATASET_ROOT, "fake")
    if not os.path.exists(real_dir) and not os.path.exists(fake_dir):
        print(f"ERROR: Could not find '{NEW_DATASET_ROOT}'. Make sure you set your path right above!")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("STEP 1/3 ─ Feature Extraction for NEW videos")
    real_files = process_directory(real_dir, 0, "real") if os.path.exists(real_dir) else []
    fake_files = process_directory(fake_dir, 1, "fake") if os.path.exists(fake_dir) else []
    
    all_cache_files = real_files + fake_files
    if not all_cache_files:
        print("No new videos processed. Exiting.")
        return

    print("\nSTEP 2/3 ─ Loading cached new features")
    all_X, all_y = [], []
    for path in all_cache_files:
        data = np.load(path)
        all_X.append(data['X'])
        all_y.append(data['y'])
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Balance new data
    X_bal, y_bal = balance_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.1, random_state=42, stratify=y_bal)

    print(f"\nSTEP 3/3 ─ Loading existing model ({MODEL_PATH})")
    
    from models.rnn_model import AttentionLayer, focal_loss
    custom_objects = {
        'AttentionLayer': AttentionLayer,
        'loss_fn': focal_loss(),
        'focal_loss': focal_loss(),
        'AUC': tf.keras.metrics.AUC,
        'Precision': tf.keras.metrics.Precision,
        'Recall': tf.keras.metrics.Recall
    }
    # Load with custom_objects
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    
    # Very important: Recompile with a very low learning rate for fine-tuning!
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(), # Or "binary_crossentropy" depending on what you used
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("Beginning fine-tuning... (This adds new knowledge to the old model!)")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(NEW_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    print(f"✅ Finished! Fine-tuned model saved as: {NEW_SAVE_PATH}")

if __name__ == "__main__":
    main()
