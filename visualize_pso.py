import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Import project modules
from train import load_cached_features, CACHE_DIR
from feature_engineering.balancing import balance_dataset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_function(weights, X, y):
    """
    Evaluates a linear classifier defined by weights [w1, w2, bias].
    weights: [w1, w2, bias]
    """
    w = weights[:2]
    b = weights[2]
    
    # Linear projection
    z = np.dot(X, w) + b
    predictions = (sigmoid(z) > 0.5).astype(int)
    
    # PSO minimizes, so return 1.0 - accuracy
    return 1.0 - accuracy_score(y, predictions)

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_pos = np.copy(self.position)
        self.best_score = float('inf')

def run_visual_pso(X, y, swarm_size=30, max_iter=20):
    dim = 3 # w1, w2, bias
    bounds = [(-5, 5), (-5, 5), (-2, 2)]
    
    swarm = [Particle(dim, bounds) for _ in range(swarm_size)]
    global_best_pos = np.zeros(dim)
    global_best_score = float('inf')
    
    # For visualization
    history_pos = []
    
    print(f"Starting PSO Optimization on 2D PCA data...")
    print(f"Swarm Size: {swarm_size}, Iterations: {max_iter}")
    
    for i in range(max_iter):
        current_positions = []
        for p in swarm:
            score = fitness_function(p.position, X, y)
            
            if score < p.best_score:
                p.best_score = score
                p.best_pos = np.copy(p.position)
                
            if score < global_best_score:
                global_best_score = score
                global_best_pos = np.copy(p.position)
            
            current_positions.append(np.copy(p.position))
        
        history_pos.append(current_positions)
        
        # Update swarm
        w, c1, c2 = 0.5, 1.5, 1.5
        for p in swarm:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            p.velocity = (w * p.velocity + 
                          c1 * r1 * (p.best_pos - p.position) + 
                          c2 * r2 * (global_best_pos - p.position))
            p.position += p.velocity
        
        print(f"Iteration {i+1}/{max_iter} - Best Accuracy: {1.0 - global_best_score:.4f}")

    return history_pos, global_best_pos

def main():
    print("=" * 60)
    print("   PSO SWARM VISUALIZATION ON TRAINED DATA")
    print("=" * 60)

    # 1. Load Data
    cache_files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
    if not cache_files:
        print(f"ERROR: No cached features found in '{CACHE_DIR}'.")
        return
    
    X, y = load_cached_features(cache_files)
    X_bal, y_bal = balance_dataset(X, y)
    
    # 2. PCA to 2D
    print("\nReducing features to 2D using PCA...")
    # Flatten features for PCA: (N, 50, 2048) -> (N, 50*2048)
    X_flat = X_bal.reshape(X_bal.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    
    # 3. Run PSO
    history, best_w = run_visual_pso(X_pca, y_bal, swarm_size=20, max_iter=15)
    
    # 4. Visualization
    print("\nGenerating swarm visualization...")
    os.makedirs("assets", exist_ok=True)
    
    for t, step_positions in enumerate(history):
        plt.figure(figsize=(10, 7))
        
        # Plot Data Points
        plt.scatter(X_pca[y_bal == 0, 0], X_pca[y_bal == 0, 1], alpha=0.4, label='Real (SMOTE)', color='C0')
        plt.scatter(X_pca[y_bal == 1, 0], X_pca[y_bal == 1, 1], alpha=0.4, label='Fake (SMOTE)', color='C1')
        
        # Plot Particles
        # Since particles are in [w1, w2, bias] space (3D), 
        # we can visualize them as candidate decision boundaries?
        # Or we can project their "effect" on the 2D space.
        
        # To "show the swarm" visually in the PCA space:
        # Let's map each particle's weights to a point in PCA space that it "targets"
        # or just visualize the decision boundaries.
        
        # Drawing decision boundaries for each particle
        x_min, x_max = X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1
        xx = np.linspace(x_min, x_max, 10)
        
        for p_idx, pos in enumerate(step_positions):
            w1, w2, b = pos
            # w1*x + w2*y + b = 0 => y = (-w1*x - b) / w2
            if abs(w2) > 0.001:
                yy = (-w1 * xx - b) / w2
                plt.plot(xx, yy, color='gray', alpha=0.1)
            
            # Also plot a dot for each particle represent its current 'focus'
            # (Just for visual swarm effect, let's plot them as points)
            plt.scatter(pos[0]/5, pos[1]/5, color='red', s=30, edgecolors='black', zorder=5)

        # Plot Best Boundary
        bw1, bw2, bb = best_w
        if abs(bw2) > 0.001:
            byy = (-bw1 * xx - bb) / bw2
            plt.plot(xx, byy, color='green', linewidth=3, label='Best Boundary (PSO)')

        plt.title(f"PSO Swarm Optimization - Iteration {t+1}\nOptimizing Linear Separator in PCA Space")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Limit axes to keep focus
        plt.xlim(x_min, x_max)
        plt.ylim(X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1)
        
        save_path = f"assets/pso_iter_{t:02d}.png"
        plt.savefig(save_path)
        plt.close()
        
    print(f"\nVisualization frames saved to 'assets/pso_iter_XX.png'")
    print("You can combine these into a GIF or video to show the swarm movement!")

if __name__ == "__main__":
    main()
