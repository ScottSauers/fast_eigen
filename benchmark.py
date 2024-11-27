import os
import argparse
import numpy as np
import time
from colorama import init, Fore, Back, Style
from scipy.linalg import eig_banded
import scipy.linalg
import pickle
from data_generator import GraphParams, GraphType

init(autoreset=True)

class LaplacianDataset:
    def __init__(self, data_dir):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load from pickle file
        import pickle
        with open(self.file_list[idx], 'rb') as f:
            L, _ = pickle.load(f)  # The second value is params which we don't need
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        return L, eigenvalues, eigenvectors

def extract_bands(L):
    """Convert full matrix to LAPACK banded storage format (lower)"""
    N = L.shape[0]
    # Find bandwidth (assuming symmetric/hermitian)
    bandwidth = 0
    for i in range(N):
        for j in range(i+1, N):
            if abs(L[i,j]) > 1e-10:
                bandwidth = max(bandwidth, j-i)
    
    # Create banded storage (lower triangular)
    bands = np.zeros((bandwidth+1, N))
    for i in range(bandwidth+1):
        bands[i,:N-i] = np.diag(L, -i)
    
    return bands

def ascii_bar(value, max_value, width=50):
    filled_len = int(round(width * value / float(max_value)))
    bar = ''
    for i in range(width):
        if i < filled_len:
            color = Fore.RED if i > width * 0.8 else Fore.YELLOW if i > width * 0.5 else Fore.GREEN
            bar += color + '██' + Style.RESET_ALL
        else:
            bar += Style.DIM + '·' + Style.RESET_ALL
    return bar

def ascii_heatmap(matrix, title=""):
    min_val, max_val = matrix.min(), matrix.max()
    result = f"\n{title}\n" if title else "\n"
    result += "    " + "".join(f"{i:^3}" for i in range(matrix.shape[1])) + "\n"
    
    for i, row in enumerate(matrix):
        result += f"{i:2d} |"
        for val in row:
            normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
            if normalized < 0.2:
                color = Back.BLUE + Fore.WHITE
            elif normalized < 0.4:
                color = Back.CYAN + Fore.BLACK
            elif normalized < 0.6:
                color = Back.GREEN + Fore.BLACK
            elif normalized < 0.8:
                color = Back.YELLOW + Fore.BLACK
            else:
                color = Back.RED + Fore.WHITE
            result += color + '█' + Style.RESET_ALL
        result += "|\n"
    return result

def ascii_bar_chart(values, max_value, width=50):
    bars = ''
    for i, val in enumerate(values):
        bar = ascii_bar(val, max_value, width)
        bars += f"{i:>3}: |{bar}| {val:.4f}\n"
    return bars

def run_benchmarks(L_np):
    print(f"{Fore.YELLOW}Running performance comparison of eigensolvers...{Style.RESET_ALL}")
    
    solvers = {
        'numpy.linalg.eigh': lambda x: np.linalg.eigh(x),
        'scipy.linalg.eigh': lambda x: scipy.linalg.eigh(x),
        'scipy.linalg.eig_banded': lambda x: eig_banded(extract_bands(x), lower=True)
    }
    
    for name, solver in solvers.items():
        times = []
        for _ in range(5):  # 5 runs each
            start = time.perf_counter()
            _ = solver(L_np)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        color = Fore.GREEN if avg_time == min(avg_time for avg_time in times) else Fore.YELLOW
        print(f"{color}{name:25s}: {avg_time:8.2f} ms{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description="Run eigendecomposition using LAPACK banded solvers.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to analyze.")
    parser.add_argument("--no_benchmark", action="store_true", help="Skip solver benchmarks")
    args = parser.parse_args()

    dataset = LaplacianDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("No data found in the specified data directory.")
        return

    total_samples = 0
    eigenvalue_errors = []
    eigenvector_similarities = []

    for idx in range(min(args.num_samples, len(dataset))):
        L, eigenvalues_true, eigenvectors_true = dataset[idx]
        
        # Convert to banded format and solve
        bands = extract_bands(L)
        start_time = time.time()
        eigenvalues_pred, eigenvectors_pred = eig_banded(bands, lower=True)
        end_time = time.time()
        inference_time = end_time - start_time

        # Compute errors
        eigenvalue_error = np.mean(np.abs(np.sort(eigenvalues_pred) - np.sort(eigenvalues_true)))
        eigenvalue_errors.append(eigenvalue_error)
        
        # Compute cosine similarities for eigenvectors
        cosine_similarities = []
        for i in range(L.shape[0]):
            v_pred = eigenvectors_pred[:, i]
            v_true = eigenvectors_true[:, i]
            cos_sim = np.abs(np.dot(v_pred, v_true)) / (np.linalg.norm(v_pred) * np.linalg.norm(v_true))
            cosine_similarities.append(cos_sim)
        mean_cosine_similarity = np.mean(cosine_similarities)
        eigenvector_similarities.append(mean_cosine_similarity)

        if total_samples < 3:
            print(f"\nSample {total_samples+1} analysis:")
            print(f"Inference Time: {inference_time * 1000:.2f} ms")

            max_eigenvalue = max(np.max(eigenvalues_true), np.max(eigenvalues_pred))
            print(f"{Fore.CYAN}=== True Eigenvalues ==={Style.RESET_ALL}")
            print(ascii_bar_chart(np.sort(eigenvalues_true), max_eigenvalue))
            print(f"{Fore.CYAN}=== LAPACK Eigenvalues ==={Style.RESET_ALL}")
            print(ascii_bar_chart(np.sort(eigenvalues_pred), max_eigenvalue))
            
            if not args.no_benchmark:
                run_benchmarks(L)

            print("\nTrue Eigenvectors (heatmap):")
            print(ascii_heatmap(eigenvectors_true.T))
            print("\nLAPACK Eigenvectors (heatmap):")
            print(ascii_heatmap(eigenvectors_pred.T))

        total_samples += 1

    # Overall statistics
    print("\n=== Overall Statistics ===")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Average Eigenvalue Absolute Error: {np.mean(eigenvalue_errors):.6f}")
    print(f"Average Eigenvector Cosine Similarity: {np.mean(eigenvector_similarities) * 100:.2f}%")

if __name__ == "__main__":
    main()
