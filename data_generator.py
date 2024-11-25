import numpy as np
import random
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_laplacian_matrix(N, bandwidth):
    """
    Generate a Laplacian matrix for a locally connected undirected graph.
    The graph is banded with bandwidth 'bandwidth', ensuring local connectivity.
    Returns:
        L (ndarray): The Laplacian matrix of size N x N.
    """
    # Initialize adjacency matrix
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        # Define potential connections within bandwidth
        possible_connections = list(range(max(0, i - bandwidth), i)) + \
                               list(range(i + 1, min(N, i + bandwidth + 1)))
        possible_connections = list(set(possible_connections) - {i})
        # At least one connection
        num_connections = random.randint(1, len(possible_connections))
        connected_nodes = random.sample(possible_connections, num_connections)
        for j in connected_nodes:
            A[i, j] = 1
            A[j, i] = 1  # Symmetry
    # Construct Laplacian matrix
    degrees = np.sum(A, axis=1)
    L = np.diag(degrees) - A
    # Verify properties
    assert np.all(np.diag(L) > 0), "Main diagonal entries should be positive integers."
    off_diagonal = L - np.diag(np.diag(L))
    assert np.all(np.isin(off_diagonal, [-1, 0])), "Off-diagonal elements should be -1 or 0."
    assert np.all(L.sum(axis=1) == 0), "Row sums should be zero."
    return L

def visualize_random_matrices(data_dir, num_samples=5):
    """
    Load and visualize random Laplacian matrices from the generated dataset.
    
    Args:
        data_dir (str): Directory containing the generated matrices
        num_samples (int): Number of random matrices to visualize
    """
    # Get list of all generated matrices
    matrix_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if len(matrix_files) < num_samples:
        raise ValueError(f"Not enough matrices in {data_dir}. Found {len(matrix_files)}, requested {num_samples}")
    
    # Select random matrices
    selected_files = random.sample(matrix_files, num_samples)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    # Custom colormap for better visualization
    cmap = plt.cm.RdYlBu_r
    
    for idx, (ax, matrix_file) in enumerate(zip(axes, selected_files)):
        # Load matrix
        matrix = np.load(os.path.join(data_dir, matrix_file))
        
        # Plot matrix
        im = ax.imshow(matrix, cmap=cmap, aspect='equal')
        ax.set_title(f'Matrix {idx+1}')
        ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes, location='right', shrink=0.8, label='Value')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def save_matrices(output_dir, num_matrices, N, bandwidth):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Starting data generation: {num_matrices} matrices of size {N}x{N}, bandwidth {bandwidth}")
    for idx in tqdm(range(num_matrices), desc="Generating matrices", unit="matrix", ncols=100):
        L = generate_laplacian_matrix(N, bandwidth)
        np.save(os.path.join(output_dir, f"laplacian_{idx}.npy"), L)
    print("Data generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate Laplacian matrices for training.")
    parser.add_argument("--num_matrices", type=int, default=1000, help="Number of matrices to generate.")
    parser.add_argument("--size", type=int, default=128, help="Size of the matrices (N).")
    parser.add_argument("--bandwidth", type=int, default=5, help="Bandwidth (b) of the matrices.")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for the matrices.")
    parser.add_argument("--visualize", action="store_true", help="Visualize 5 random matrices after generation.")
    args = parser.parse_args()
    
    save_matrices(args.output_dir, args.num_matrices, args.size, args.bandwidth)
    
    if args.visualize:
        print("\nVisualizing 5 random matrices from the generated dataset...")
        visualize_random_matrices(args.output_dir)

if __name__ == "__main__":
    main()
