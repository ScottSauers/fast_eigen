import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import time
from contextlib import contextmanager
from data_generator import GraphParams, GraphType
from colorama import init, Fore, Back, Style
import scipy.linalg

init(autoreset=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LaplacianDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load L and params from file
        L = np.load(self.file_list[idx])
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        L = torch.from_numpy(L).float()
        eigenvalues = torch.from_numpy(eigenvalues).float()
        eigenvectors = torch.from_numpy(eigenvectors).float()
        return L, eigenvalues, eigenvectors

def extract_diagonals(L):
    diagonals = []
    N = L.size(1)
    for k in range(N):
        diag_k = torch.stack([torch.diagonal(L[i], offset=k) for i in range(L.size(0))])
        if torch.all(diag_k == 0):
            break
        diagonals.append(diag_k)
    return diagonals

# Model definitions (copy from your training code)
class BandLimitedEncoder(torch.nn.Module):
    def __init__(self, N, max_channels=64):
        super(BandLimitedEncoder, self).__init__()
        self.N = N
        self.max_channels = max_channels
        self.conv = torch.nn.Conv1d(1, max_channels, kernel_size=5, padding=2)
                
    def forward(self, diagonals):
        features = []
        for k, diag in enumerate(diagonals):
            diag = diag.unsqueeze(1)  # Add channel dimension
            feat = torch.relu(self.conv(diag))
            if k > 0:
                pad_size = diagonals[0].size(-1) - diag.size(-1)
                feat = torch.nn.functional.pad(feat, (0, pad_size))
            features.append(feat)
        return features

class FusionNetwork(torch.nn.Module):
    def __init__(self, N, hidden_dim=64):
        super(FusionNetwork, self).__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
            
    def forward(self, features):
        batch_size = features[0].size(0)
        projected_features = []
        for feat in features:
            feat = feat.permute(0, 2, 1)  # [B, N, C]
            projected_features.append(self.projection(feat))

        combined = torch.stack(projected_features, dim=1)
        combined = combined.view(-1, combined.size(-2), combined.size(-1))
        attn_output, _ = self.attention(combined, combined, combined)
        output = attn_output.view(batch_size, len(features), self.N, -1)
        output = output.mean(dim=1)  # [B, N, hidden_dim]
        return output.permute(0, 2, 1)  # [B, hidden_dim, N]

class EigenDecompositionNetwork(torch.nn.Module):
    def __init__(self, N):
        super(EigenDecompositionNetwork, self).__init__()
        self.N = N
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(N*64, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, N + N*N)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        out = self.decoder(x)
        eigenvalues = out[:, :self.N]
        eigenvectors = out[:, self.N:].reshape(batch_size, self.N, self.N)
        eigenvalues = torch.relu(eigenvalues)
        eigenvectors = torch.nn.functional.normalize(eigenvectors, dim=2)
        return eigenvalues, eigenvectors

class EigensolverModel(torch.nn.Module):
    def __init__(self, N):
        super(EigensolverModel, self).__init__()
        self.N = N
        self.version = "v2"
        self.encoder = BandLimitedEncoder(N)
        self.fusion = FusionNetwork(N)
        self.decoder = EigenDecompositionNetwork(N)

    def forward(self, diagonals):
        features = self.encoder(diagonals)
        fused = self.fusion(features)
        fused = fused.contiguous()
        eigenvalues, eigenvectors = self.decoder(fused)
        return eigenvalues, eigenvectors

def load_model(checkpoint_dir, N):
    model = EigensolverModel(N).to(device)
    
    # Try loading best model first
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Try latest checkpoint if best model not found
    if not os.path.exists(best_model_path):
        print("Warning: best model not found.")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}'")
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    try:
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'version' in checkpoint and checkpoint['version'] != model.version:
                raise ValueError(f"Version mismatch: checkpoint={checkpoint['version']}, model={model.version}")
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model from '{best_model_path}'")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")
        
    model.eval()
    return model

def compare_eigendecompositions(eigenvalues_pred, eigenvectors_pred, eigenvalues_true, eigenvectors_true):
    eigenvalues_pred_sorted, indices_pred = torch.sort(eigenvalues_pred)
    eigenvalues_true_sorted, indices_true = torch.sort(eigenvalues_true)
    eigenvectors_pred_sorted = eigenvectors_pred[:, indices_pred]
    eigenvectors_true_sorted = eigenvectors_true[:, indices_true]

    # Adjust signs of eigenvectors for comparison
    for i in range(eigenvectors_true_sorted.shape[1]):
        dot_product = torch.dot(eigenvectors_pred_sorted[:, i], eigenvectors_true_sorted[:, i])
        if dot_product < 0:
            eigenvectors_pred_sorted[:, i] *= -1

    eigenvalues_diff = eigenvalues_pred_sorted - eigenvalues_true_sorted
    eigenvectors_diff = eigenvectors_pred_sorted - eigenvectors_true_sorted
    return eigenvalues_pred_sorted, eigenvectors_pred_sorted, eigenvalues_true_sorted, eigenvectors_true_sorted, eigenvalues_diff, eigenvectors_diff

def ascii_bar(value, max_value, width=50):
    filled_len = int(round(width * value / float(max_value)))
    intensity = int(255 * (value / max_value))
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

@contextmanager
def no_grad():
    with torch.no_grad():
        yield

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained Eigensolver Neural Network.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing saved model checkpoint.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run inference on.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--no_benchmark", action="store_true", help="Skip solver benchmarks")
    args = parser.parse_args()

    # Load dataset
    dataset = LaplacianDataset(args.data_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Get a sample to determine N
    if len(dataset) == 0:
        print("No data found in the specified data directory.")
        return

    sample_L, _, _ = dataset[0]
    N = sample_L.size(0)

    # Load the model
    try:
        model = load_model(args.checkpoint_dir, N)
    except FileNotFoundError as e:
        print(str(e))
        return

    # Run inference on data
    total_time = 0.0
    total_samples = 0
    eigenvalue_errors = []
    eigenvector_similarities = []

    for idx, (L_batch, eigenvalues_batch, eigenvectors_batch) in enumerate(data_loader):
        if total_samples >= args.num_samples:
            break

        L_batch = L_batch.to(device)
        diagonals = extract_diagonals(L_batch)
        diagonals = [d.to(device) for d in diagonals]

        start_time = time.time()
        with no_grad():
            eigenvalues_pred, eigenvectors_pred = model(diagonals)
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        # Compute true eigenvalues and eigenvectors
        L_np = L_batch.cpu().numpy()[0]
        eigenvalues_true_np, eigenvectors_true_np = np.linalg.eigh(L_np)
        eigenvalues_true = torch.from_numpy(eigenvalues_true_np).float().to(device)
        eigenvectors_true = torch.from_numpy(eigenvectors_true_np).float().to(device)

        # Compare eigendecompositions
        eigenvalues_pred_sorted, eigenvectors_pred_sorted, eigenvalues_true_sorted, eigenvectors_true_sorted, eigenvalues_diff, eigenvectors_diff = compare_eigendecompositions(
            eigenvalues_pred[0], eigenvectors_pred[0], eigenvalues_true, eigenvectors_true)

        # Compute errors for overall statistics
        eigenvalue_error = torch.mean(torch.abs(eigenvalues_diff)).item()
        eigenvalue_errors.append(eigenvalue_error)
        cosine_similarities = []
        for i in range(N):
            v_pred = eigenvectors_pred_sorted[:, i]
            v_true = eigenvectors_true_sorted[:, i]
            cos_sim = torch.abs(torch.dot(v_pred, v_true) / (torch.norm(v_pred) * torch.norm(v_true) + 1e-8)).item()
            cosine_similarities.append(cos_sim)
        mean_cosine_similarity = np.mean(cosine_similarities)
        eigenvector_similarities.append(mean_cosine_similarity)

        # Limit detailed output to first few samples
        if total_samples < 3:
            # Print visualization
            print(f"Sample {total_samples+1} analysis:")
            print(f"Inference Time: {inference_time * 1000:.2f} ms")

            # Visualize eigenvalues
            max_eigenvalue = max(eigenvalues_true_sorted.max().item(), eigenvalues_pred_sorted.max().item())
            print(f"{Fore.CYAN}=== True Eigenvalues ==={Style.RESET_ALL}")
            bar_true = ascii_bar_chart(eigenvalues_true_sorted.cpu().numpy(), max_eigenvalue)
            print(bar_true)
            print(f"{Fore.CYAN}=== Predicted Eigenvalues ==={Style.RESET_ALL}")
            bar_pred = ascii_bar_chart(eigenvalues_pred_sorted.cpu().numpy(), max_eigenvalue)
            print(bar_pred)
            
            if not args.no_benchmark:
                print(f"\n{Fore.CYAN}=== Benchmarking ==={Style.RESET_ALL}")
                run_benchmarks(L_np)

            # Visualize eigenvectors
            print("\nTrue Eigenvectors (heatmap):")
            heatmap_true = ascii_heatmap(eigenvectors_true_sorted.cpu().numpy().T)
            print(heatmap_true)
            print("Predicted Eigenvectors (heatmap):")
            heatmap_pred = ascii_heatmap(eigenvectors_pred_sorted.cpu().numpy().T)
            print(heatmap_pred)

        total_samples += 1

    # Overall statistics
    avg_inference_time = total_time / total_samples
    avg_eigenvalue_error = np.mean(eigenvalue_errors)
    avg_eigenvector_similarity = np.mean(eigenvector_similarities)

    print("\n--- Overall Statistics ---")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Average Inference Time per Sample: {avg_inference_time * 1000:.2f} ms")
    print(f"Average Eigenvalue Absolute Error: {avg_eigenvalue_error:.6f}")
    print(f"Average Eigenvector Cosine Similarity: {avg_eigenvector_similarity * 100:.2f}%")

def run_benchmarks(L_np):
    results = {}
    
    # Standard eigensolvers
    solvers = {
        'numpy.linalg.eigh': lambda x: np.linalg.eigh(x),
        'scipy.linalg.eigh': lambda x: scipy.linalg.eigh(x),
        'scipy.linalg.eig_banded': lambda x: scipy.linalg.eig_banded(
            np.array([np.diag(L_np, k) for k in range(L_np.shape[0])])
        ),
        'neural_network': lambda x: model(extract_diagonals(torch.from_numpy(x).unsqueeze(0).to(device)))
    }
    
    for name, solver in solvers.items():
        times = []
        for _ in range(5):  # 5 runs each
            start = time.perf_counter()
            _ = solver(L_np)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        color = Fore.GREEN if avg_time == min(times) else Fore.YELLOW
        print(f"{color}{name:25s}: {avg_time:8.2f} ms{Style.RESET_ALL}")

def ascii_bar_chart(values, max_value, width=50):
    """
    Generates an ASCII bar chart for the given array of values.
    """
    bars = ''
    for i, val in enumerate(values):
        bar = ascii_bar(val, max_value, width)
        bars += f"{i:>3}: |{bar}| {val:.4f}\n"
    return bars

if __name__ == "__main__":
    main()
