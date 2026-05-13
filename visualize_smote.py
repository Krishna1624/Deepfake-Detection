import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Import project modules
from train import load_cached_features, CACHE_DIR

def main():
    print("=" * 60)
    print("   SMOTE (Synthetic Minority Over-sampling) VISUALIZATION")
    print("=" * 60)

    # 1. Load Data
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
    if not cache_files:
        print(f"ERROR: No cached features found in '{CACHE_DIR}'.")
        return
    
    X, y = load_cached_features(cache_files)
    X_flat = X.reshape(X.shape[0], -1)

    # 2. Artificial Imbalance Check
    real_count = sum(y == 0)
    fake_count = sum(y == 1)
    print(f"Original Balance: Real={real_count}, Fake={fake_count}")

    # If already balanced, let's create an imbalance to "show" SMOTE in action
    if abs(real_count - fake_count) < 10:
        print("\nDataset is already balanced. Creating artificial imbalance to show SMOTE...")
        # Keep all Real (0), remove 80% of Fake (1)
        real_indices = np.where(y == 0)[0]
        fake_indices = np.where(y == 1)[0]
        
        # Take only 20 samples of Fake to make it the minority
        keep_fake = fake_indices[:120]
        
        subset_indices = np.concatenate([real_indices, keep_fake])
        X_imbalanced = X_flat[subset_indices]
        y_imbalanced = y[subset_indices]
        
        print(f"Imbalanced State: Real={len(real_indices)}, Fake={len(keep_fake)}")
    else:
        X_imbalanced = X_flat
        y_imbalanced = y

    # 3. PCA to 2D (before SMOTE)
    pca = PCA(n_components=2)
    X_pca_orig = pca.fit_transform(X_imbalanced)

    # 4. Apply SMOTE
    print("\nApplying SMOTE to generate synthetic 'Fake' samples...")
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_imbalanced, y_imbalanced)
    
    # 5. Identify Synthetic points
    # Resampled includes original points at the start
    n_orig = len(y_imbalanced)
    X_synthetic = X_resampled[n_orig:]
    y_synthetic = y_resampled[n_orig:]
    
    X_pca_resampled = pca.transform(X_resampled)
    X_pca_synthetic = X_pca_resampled[n_orig:]

    # 6. Visualization
    plt.figure(figsize=(10, 7))
    
    # Plot Real
    plt.scatter(X_pca_orig[y_imbalanced == 0, 0], X_pca_orig[y_imbalanced == 0, 1], 
                alpha=0.6, label='Original Real', color='C0', marker='o')
    
    # Plot Original Fake (Minority)
    plt.scatter(X_pca_orig[y_imbalanced == 1, 0], X_pca_orig[y_imbalanced == 1, 1], 
                alpha=0.8, label='Original Fake (Minority)', color='C1', marker='o', s=100, edgecolors='black')
    
    # Plot Synthetic Fake (SMOTE results)
    plt.scatter(X_pca_synthetic[:, 0], X_pca_synthetic[:, 1], 
                alpha=0.4, label='Synthetic Fake (SMOTE)', color='salmon', marker='x', s=40)

    plt.title("SMOTE Visualization: Generating Synthetic Deepfake Data\nFilling the gap in Minority Class (Fake)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/smote_visualization.png")
    plt.show() # If running locally with GUI
    
    print(f"\nSuccess! SMOTE generated {len(y_synthetic)} synthetic 'Fake' samples.")
    print("Plot saved to 'assets/smote_visualization.png'")

if __name__ == "__main__":
    main()
