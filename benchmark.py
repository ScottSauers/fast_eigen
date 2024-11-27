import os
import argparse
import numpy as np
import time
from colorama import init, Fore, Back, Style
from scipy.linalg import (eig_banded, eigh_tridiagonal, eigvalsh, 
                         eigvals_banded, eigh, cholesky_banded,
                         solve_banded, get_lapack_funcs)
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

def extract_bands(L, check_symmetric=True):
    """Convert full matrix to LAPACK banded storage format (lower)"""
    if check_symmetric:
        # Verify near-symmetry
        if not np.allclose(L, L.T, rtol=1e-10):
            print("Warning: Matrix is not symmetric")
    
    N = L.shape[0]
    bandwidth = 0
    for i in range(N):
        for j in range(i+1, N):
            if abs(L[i,j]) > 1e-10:
                bandwidth = max(bandwidth, j-i)
    
    bands = np.zeros((bandwidth+1, N))
    for i in range(bandwidth+1):
        bands[i,:N-i] = np.diag(L, -i)
    
    return bands, bandwidth

def verify_results(results):
    """Verify all methods produce the same results within tolerance"""
    EIGENVAL_TOLERANCE = 1e-8  # Relaxed tolerance for different methods
    reference_method = list(results.keys())[0]
    reference_vals = results[reference_method][0]
    
    # Sort reference eigenvalues
    reference_vals = np.sort(reference_vals)
    
    errors = {}
    for method, (vals, _) in results.items():
        if method == reference_method:
            continue
            
        # Sort current eigenvalues
        vals = np.sort(vals)
        
        # Compare eigenvalues
        val_diff = np.max(np.abs(vals - reference_vals))
        
        errors[method] = val_diff
        
        if val_diff > EIGENVAL_TOLERANCE:
            print(f"{Fore.RED}Warning: {method} eigenvalues differ from {reference_method}:")
            print(f"Max eigenvalue difference: {val_diff}{Style.RESET_ALL}")

def run_benchmarks(L):
    print(f"{Fore.YELLOW}Running performance comparison of eigensolvers...{Style.RESET_ALL}")
    
    # Extract band information once
    bands, bandwidth = extract_bands(L)
    is_positive_definite = np.all(np.linalg.eigvals(L) > -1e-10)
    
    # Define solvers with their setup and computation steps
    solvers = {
        'numpy.linalg.eigh': {
            'setup': lambda x: x,
            'solve': lambda x: np.linalg.eigh(x)
        },
        'scipy.linalg.eigh(lower,overwrite)': {
            'setup': lambda x: x,
            'solve': lambda x: scipy.linalg.eigh(x, lower=True, overwrite_a=True)
        },
        'scipy.linalg.eigvalsh(lower,overwrite)': {
            'setup': lambda x: x,
            'solve': lambda x: (scipy.linalg.eigvalsh(x, lower=True, overwrite_a=True), None)
        },
        'scipy.linalg.eigvalsh(subset_by_value)': {
            'setup': lambda x: x,
            'solve': lambda x: (scipy.linalg.eigvalsh(x, subset_by_value=[-np.inf, np.inf], driver='evr'), None)
        },
        'scipy.linalg.eig_banded': {
            'setup': lambda x: extract_bands(x)[0],
            'solve': lambda x: eig_banded(x, lower=True)
        },
        'scipy.linalg.eigvals_banded': {
            'setup': lambda x: extract_bands(x)[0],
            'solve': lambda x: (eigvals_banded(x, lower=True), None)
        },
        'scipy.linalg.eigh(driver=evx)': {
            'setup': lambda x: x,
            'solve': lambda x: scipy.linalg.eigh(x, driver='evx')
        },
        'scipy.linalg.eigh(driver=evr)': {
            'setup': lambda x: x,
            'solve': lambda x: scipy.linalg.eigh(x, driver='evr')
        }
    }
    
    if is_positive_definite:
        # Add Cholesky-based methods for positive definite matrices
        solvers.update({
            'cholesky+eigh': {
                'setup': lambda x: scipy.linalg.cholesky(x, lower=True),
                'solve': lambda x: (np.linalg.eigvalsh(x @ x.T), None)
            },
        })
    
    results = {}
    timing_stats = {}
    
    for name, solver in solvers.items():
        setup_times = []
        solve_times = []
        total_times = []
        
        try:
            # Warmup run
            data = solver['setup'](L.copy())  # Always use a copy to ensure fair comparison
            vals_vecs = solver['solve'](data)
            
            for _ in range(10):
                start = time.perf_counter()
                data = solver['setup'](L.copy())
                setup_end = time.perf_counter()
                
                vals_vecs = solver['solve'](data)
                solve_end = time.perf_counter()
                
                setup_times.append(setup_end - start)
                solve_times.append(solve_end - setup_end)
                total_times.append(solve_end - start)
                
                if isinstance(vals_vecs, tuple):
                    vals, vecs = vals_vecs
                else:
                    vals = vals_vecs
                    vecs = None
                
                results[name] = (vals, vecs)
            
            timing_stats[name] = {
                'setup': np.mean(setup_times) * 1000,
                'solve': np.mean(solve_times) * 1000,
                'total': np.mean(total_times) * 1000,
                'std': np.std(total_times) * 1000
            }
            
        except Exception as e:
            print(f"{Fore.RED}Error in {name}: {str(e)}{Style.RESET_ALL}")
            continue
    
    # Print results
    print(f"\nTiming Results for {L.shape[0]}x{L.shape[0]} matrix (bandwidth={bandwidth}):")
    print(f"{'Method':35s} {'Setup':>10s} {'Solve':>10s} {'Total':>10s} {'Std':>10s}")
    print("-" * 80)
    
    # Sort by total time
    sorted_methods = sorted(timing_stats.items(), key=lambda x: x[1]['total'])
    
    for name, stats in sorted_methods:
        color = Fore.GREEN if name == sorted_methods[0][0] else Fore.WHITE
        print(f"{color}{name:35s} {stats['setup']:10.2f} {stats['solve']:10.2f} {stats['total']:10.2f} {stats['std']:10.2f}{Style.RESET_ALL}")
    
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
