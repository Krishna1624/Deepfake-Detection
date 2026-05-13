"""
train.py  ─  Batch-safe Deepfake Detection Training
=====================================================
Processes the FF++ dataset in small video batches so that
RAM never exceeds a comfortable level even on a laptop/desktop.

Strategy
--------
1.  Split the video file lists into BATCH_SIZE chunks.
2.  For each chunk:  extract frames → preprocess → CNN features
    → save the resulting numpy arrays to disk as 'feature_cache/'.
3.  After all chunks are done, load the cached features from disk,
    run SMOTE-DLPB balancing, PSO optimisation, and final LSTM training.
4.  Save the trained model.

Tune the two constants at the top to fit your GPU memory.
"""

import os
import gc
import math
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Project modules ──────────────────────────────────────────────────────────
from utils.video_processing import extract_frames
from preprocessing.preprocessing import preprocess_frame
from feature_engineering.cnn_features import extract_features
from feature_engineering.balancing import balance_dataset
from feature_engineering.optimization import run_pso_optimization
from models.rnn_model import build_rnn

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ⚙️  CONFIGURATION  ─  tweak these as needed
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_ROOT   = r"C:\Users\Karthik\Desktop\FF++"   # path to real/ and fake/
CACHE_DIR      = "feature_cache"                     # temp folder for batch features
BATCH_SIZE     = 10                                  # videos per batch (10–20 recommended)
MAX_FRAMES     = 50                                  # ↑ more frames = better temporal coverage
MIN_FRAMES     = 10                                  # skip if fewer than this
SEQUENCE_LEN   = 50                                  # must match MAX_FRAMES
FEATURE_DIM    = 2048                                # Xception output dim
EPOCHS         = 50                                  # ↑ more epochs (EarlyStopping will stop it early)
SAVE_PATH      = "saved_models/deepfake_model.h5"
BEST_PATH      = "saved_models/deepfake_model_best.h5"  # best checkpoint during training
USE_FOCAL_LOSS = False                               # set True if classes are still imbalanced after SMOTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ─────────────────────────────────────────────────────────────────────────────
#  GPU MEMORY GROWTH  (prevents TF from grabbing all VRAM at startup)
# ─────────────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"  GPU detected: {[g.name for g in gpus]} — memory growth enabled")
else:
    print("  No GPU found — running on CPU")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def chunked(lst, size):
    """Yield successive chunks of `size` from list `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def process_video(v_path):
    """
    Extract frames → preprocess → CNN feature sequence.
    Returns a (SEQUENCE_LEN, FEATURE_DIM) array or None on failure.
    """
    try:
        frames = extract_frames(v_path, max_frames=MAX_FRAMES)
        if not frames or len(frames) < MIN_FRAMES:
            return None

        processed = [preprocess_frame(f) for f in frames]
        features  = extract_features(np.array(processed))   # (N, FEATURE_DIM)

        # Pad or trim to fixed sequence length
        if features.shape[0] < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - features.shape[0], FEATURE_DIM))
            features = np.vstack([features, pad])
        else:
            features = features[:SEQUENCE_LEN]

        return features                                      # (30, 2048)
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return None


def process_class_in_batches(class_dir, label, class_name):
    """
    Walk through all videos in `class_dir`, batch by BATCH_SIZE,
    extract features, and save each batch to CACHE_DIR.
    Returns a list of saved .npz file paths.
    """
    video_files = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ])

    total = len(video_files)
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"\n  [{class_name.upper()}] {total} videos → {n_batches} batches of ≤{BATCH_SIZE}")

    saved_files = []
    for batch_idx, batch in enumerate(chunked(video_files, BATCH_SIZE)):
        cache_file = os.path.join(
            CACHE_DIR, f"{class_name}_batch_{batch_idx:04d}.npz"
        )

        # Skip if already cached (allows resume after crash)
        if os.path.exists(cache_file):
            print(f"    Batch {batch_idx+1}/{n_batches} — CACHED, skipping")
            saved_files.append(cache_file)
            continue

        print(f"    Batch {batch_idx+1}/{n_batches} — processing {len(batch)} videos …", end="", flush=True)
        t0 = time.time()

        batch_X, batch_y = [], []
        for v_file in batch:
            v_path = os.path.join(class_dir, v_file)
            feat = process_video(v_path)
            if feat is not None:
                batch_X.append(feat)
                batch_y.append(label)

        if batch_X:
            np.savez_compressed(
                cache_file,
                X=np.array(batch_X, dtype=np.float32),
                y=np.array(batch_y, dtype=np.int8)
            )
            saved_files.append(cache_file)
            elapsed = time.time() - t0
            print(f" ✓  [{len(batch_X)}/{len(batch)} ok] in {elapsed:.1f}s")
        else:
            print(" ✗  no valid videos in this batch — skipped")

        # Explicit garbage collection between batches
        del batch_X, batch_y
        gc.collect()

    return saved_files


def load_cached_features(cache_files):
    """Load & concatenate all saved .npz batch files."""
    all_X, all_y = [], []
    for path in cache_files:
        data = np.load(path)
        all_X.append(data['X'])
        all_y.append(data['y'])
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   DEEPFAKE DETECTION — BATCH TRAINING PIPELINE")
    print("=" * 60)
    print(f"  Dataset  : {DATASET_ROOT}")
    print(f"  Batch sz : {BATCH_SIZE} videos/batch")
    print(f"  Frames   : {MAX_FRAMES} per video  (seq len = {SEQUENCE_LEN})")
    print()

    # ── 1. Verify dataset paths ──────────────────────────────────────────────
    real_dir = os.path.join(DATASET_ROOT, "real")
    fake_dir = os.path.join(DATASET_ROOT, "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        print("ERROR: Could not find 'real' and 'fake' subdirectories.")
        print(f"  Expected: {real_dir}")
        print(f"            {fake_dir}")
        return

    # ── 2. Create cache directory ────────────────────────────────────────────
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── 3. Process each class in batches (features saved to disk) ───────────
    print("STEP 1/4 ─ Feature Extraction (batched)")
    real_files = process_class_in_batches(real_dir, label=0, class_name="real")
    fake_files = process_class_in_batches(fake_dir, label=1, class_name="fake")

    all_cache_files = real_files + fake_files
    if not all_cache_files:
        print("\nERROR: No features were extracted. Check your dataset.")
        return

    # ── 4. Load all cached features into memory ──────────────────────────────
    print("\nSTEP 2/4 ─ Loading cached features …")
    X, y = load_cached_features(all_cache_files)
    print(f"  Total samples : {X.shape[0]}  shape={X.shape}")
    print(f"  Class balance : real={np.sum(y==0)}, fake={np.sum(y==1)}")

    # ── 5. SMOTE-DLPB Balancing ───────────────────────────────────────────────
    print("\nSTEP 3/4 ─ SMOTE-DLPB Balancing …")
    X_bal, y_bal = balance_dataset(X, y)
    print(f"  After balancing: {X_bal.shape[0]} samples")

    # ── 6. PSO Optimisation ───────────────────────────────────────────────────
    print("\nSTEP 4/4 ─ PSO Hyperparameter Optimisation …")
    # Bounds: [lstm_units, learning_rate, batch_size]
    # Units between 32 and 256, LR between 0.00001 and 0.01, Batch between 4 and 32
    lb = [32, 1e-5, 4]
    ub = [256, 1e-2, 32]
    best_params  = run_pso_optimization(X_bal, y_bal, lb=lb, ub=ub)
    lstm_units   = int(best_params[0])
    learning_rate = best_params[1]
    batch_size   = int(best_params[2])
    print(f"  Best params → LSTM units={lstm_units}, LR={learning_rate:.5f}, batch={batch_size}")

    # ── 7. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    print(f"  Train={len(X_train)}, Test={len(X_test)}")

    # ── 8. Build & train model ────────────────────────────────────────────────
    print(f"\n  Training LSTM model — epochs={EPOCHS}, batch={batch_size} …")
    model = build_rnn(lstm_units=lstm_units, sequence_len=SEQUENCE_LEN, feature_dim=FEATURE_DIM)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Choose loss function
    if USE_FOCAL_LOSS:
        from models.rnn_model import focal_loss
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        print("  Using Focal Loss (good for imbalanced data)")
    else:
        loss_fn = 'binary_crossentropy'

    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=8,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            BEST_PATH, monitor='val_auc', mode='max',
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # ── 9. Evaluate ───────────────────────────────────────────────────────────
    # Use verbose=1 so all metrics are printed live by Keras itself
    print("\n  ── Final Evaluation ──────────────────────────")
    results = model.evaluate(X_test, y_test, verbose=1)
    
    # Calculate additional metrics using sklearn
    from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc_val = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred) # Sensitivity is the same as Recall
    f1 = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n  ── Detailed Metrics ────────────────────────")
    print(f"  Accuracy    : {acc_val*100:.2f}%")
    print(f"  Sensitivity : {sensitivity*100:.2f}%  (True Positive Rate)")
    print(f"  Specificity : {specificity*100:.2f}%  (True Negative Rate)")
    print(f"  F1 Score    : {f1*100:.2f}%")
    print(f"  Confusion Matrix: T.Pos={tp}, T.Neg={tn}, F.Pos={fp}, F.Neg={fn}")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model.save(SAVE_PATH)
    print(f"\n  ✅  Model saved → {SAVE_PATH}")
    print("\n  TIP: Feature cache saved in './{CACHE_DIR}/' — delete it to force re-extraction.")


if __name__ == "__main__":
    main()