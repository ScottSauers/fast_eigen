import os
import argparse
import numpy as np
import time
from colorama import init, Fore, Back, Style
from scipy.linalg import eig_banded, eigh_tridiagonal
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
        try:
            filename = self.file_list[idx]
            with open(filename, 'rb') as f:
                try:
                    data = pickle.load(f)
                    if isinstance(data, tuple) and len(data) == 2:
                        L, _ = data
                    else:
                        raise ValueError(f"Unexpected pickle format in {filename}")
                    return L
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    if idx + 1 < len(self.file_list):
                        return self.__getitem__(idx + 1)
                    else:
                        raise
        except Exception as e:
            print(f"Fatal error processing file: {str(e)}")
            raise

def extract_bands(L):
    """Convert full matrix to LAPACK banded storage format (lower)"""
    N = L.shape[0]
    bandwidth = 0
    for i in range(N):
        for j in range(i+1, N):
            if abs(L[i,j]) > 1e-10:
                bandwidth = max(bandwidth, j-i)
    
    bands = np.zeros((bandwidth+1, N))
    for i in range(bandwidth+1):
        bands[i,:N-i] = np.diag(L, -i)
    
    return bands

def extract_tridiagonal(L):
    """Extract main diagonal and off-diagonal for tridiagonal matrices"""
    d = np.diag(L)
    e = np.diag(L, -1)
    return d, e

def verify_results(results):
    """Verify all methods produce the same results within tolerance"""
    TOLERANCE = 1e-10
    reference_vals, reference_vecs = results[list(results.keys())[0]]
    
    for method, (vals, vecs) in results.items():
        # Sort eigenvalues and corresponding eigenvectors
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Compare with reference (accounting for possible sign flips)
        val_diff = np.max(np.abs(vals - reference_vals))
        vec_diff = min(
            np.max(np.abs(vecs - reference_vecs)),
            np.max(np.abs(vecs + reference_vecs))
        )
        
        if val_diff > TOLERANCE or vec_diff > TOLERANCE:
            print(f"{Fore.RED}Warning: {method} results differ from reference:")
            print(f"Max eigenvalue difference: {val_diff}")
            print(f"Max eigenvector difference: {vec_diff}{Style.RESET_ALL}")

def run_benchmarks(L):
    print(f"{Fore.YELLOW}Running performance comparison of eigensolvers...{Style.RESET_ALL}")
    
    # Define solvers with their setup and computation steps
    solvers = {
        'numpy.linalg.eigh': {
            'setup': lambda x: x,
            'solve': lambda x: np.linalg.eigh(x)
        },
        'scipy.linalg.eigh': {
            'setup': lambda x: x,
            'solve': lambda x: scipy.linalg.eigh(x)
        },
        'scipy.linalg.eigh(lower=True)': {
            'setup': lambda x: x,
            'solve': lambda x: scipy.linalg.eigh(x, lower=True)
        },
        'scipy.linalg.eig_banded': {
            'setup': extract_bands,
            'solve': lambda x: eig_banded(x, lower=True)
        },
        'scipy.linalg.eigh_tridiagonal': {
            'setup': extract_tridiagonal,
            'solve': lambda x: eigh_tridiagonal(*x)
        },
        'numpy.linalg.eigvals': {
            'setup': lambda x: x,
            'solve': lambda x: (np.linalg.eigvals(x), None)
        },
        'scipy.linalg.eigvals': {
            'setup': lambda x: x,
            'solve': lambda x: (scipy.linalg.eigvals(x), None)
        },
        'scipy.linalg.eigvalsh': {
            'setup': lambda x: x,
            'solve': lambda x: (scipy.linalg.eigvalsh(x), None)
        }
    }
    
    results = {}
    timing_stats = {}
    
    for name, solver in solvers.items():
        # Setup phase
        setup_times = []
        solve_times = []
        total_times = []
        
        data = solver['setup'](L)  # Initial setup to warm up
        
        for _ in range(10):  # 10 runs each
            # Time setup
            start = time.perf_counter()
            data = solver['setup'](L)
            setup_end = time.perf_counter()
            
            # Time solve
            vals_vecs = solver['solve'](data)
            solve_end = time.perf_counter()
            
            setup_times.append(setup_end - start)
            solve_times.append(solve_end - setup_end)
            total_times.append(solve_end - start)
            
            if vals_vecs[1] is not None:  # Store only if eigenvectors were computed
                results[name] = vals_vecs
        
        # Calculate statistics
        timing_stats[name] = {
            'setup': np.mean(setup_times) * 1000,  # Convert to ms
            'solve': np.mean(solve_times) * 1000,
            'total': np.mean(total_times) * 1000,
            'std': np.std(total_times) * 1000
        }
    
    # Print results
    print("\nTiming Results (milliseconds):")
    print(f"{'Method':30s} {'Setup':>10s} {'Solve':>10s} {'Total':>10s} {'Std':>10s}")
    print("-" * 70)
    
    for name, stats in timing_stats.items():
        color = Fore.GREEN if stats['total'] == min(s['total'] for s in timing_stats.values()) else Fore.WHITE
        print(f"{color}{name:30s} {stats['setup']:10.2f} {stats['solve']:10.2f} {stats['total']:10.2f} {stats['std']:10.2f}{Style.RESET_ALL}")
    
    # Verify results
    if len(results) > 1:
        print("\nVerifying solver consistency...")
        verify_results(results)

def main():
    parser = argparse.ArgumentParser(description="Benchmark eigensolvers performance.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to analyze.")
    args = parser.parse_args()

    dataset = LaplacianDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("No data found in the specified directory.")
        return

    for idx in range(min(args.num_samples, len(dataset))):
        L = dataset[idx]
        print(f"\n{Fore.CYAN}=== Sample {idx+1} (Matrix size: {L.shape[0]}x{L.shape[1]}) ==={Style.RESET_ALL}")
        run_benchmarks(L)

if __name__ == "__main__":
    main()
