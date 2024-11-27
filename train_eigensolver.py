import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)

# Mixed precision training
use_amp = True if torch.cuda.is_available() else False
if use_amp:
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
else:
    scaler = None

def save_checkpoint(model, state, checkpoint_dir, epoch):
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    state['version'] = model.version  # version to checkpoint
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {epoch} (version {model.version})")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        print("No checkpoint found. Starting training from scratch.")
        return 0

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Version check
        if 'version' not in checkpoint:
            print("Legacy checkpoint detected. Starting fresh training.")
            return 0
            
        if checkpoint['version'] != model.version:
            print(f"Version mismatch: checkpoint={checkpoint['version']}, model={model.version}")
            print("Starting fresh training with new architecture.")
            return 0

        # Load state dicts if versions match
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
        
    except (RuntimeError, KeyError) as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Starting fresh training.")
        return 0

class LaplacianDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        L = np.load(self.file_list[idx])
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        L = torch.from_numpy(L).float()
        eigenvalues = torch.from_numpy(eigenvalues).float()
        eigenvectors = torch.from_numpy(eigenvectors).float()
        return L, eigenvalues, eigenvectors

class BandLimitedEncoder(nn.Module):
    def __init__(self, N, max_channels=64):
        super(BandLimitedEncoder, self).__init__()
        self.N = N
        self.max_channels = max_channels
        
        # Create prototype convolutions - will be reused for all diagonals
        self.conv_main = nn.Conv1d(1, max_channels, kernel_size=5, padding=2)
        self.conv_near = nn.Conv1d(1, max_channels//2, kernel_size=5, padding=2)
        self.conv_far = nn.Conv1d(1, max_channels//4, kernel_size=5, padding=2)
            
    def forward(self, diagonals):
        features = []
        for k, diag in enumerate(diagonals):
            # Process each diagonal without padding to its natural length N-k
            diag = diag.unsqueeze(1)  # Add channel dimension
            
            # Select appropriate convolution based on diagonal position
            if k == 0:
                conv = self.conv_main  # Main diagonal gets full channels
            elif k <= 3:
                conv = self.conv_near  # Near diagonals get half channels
            else:
                conv = self.conv_far   # Far diagonals get quarter channels
                
            feat = torch.relu(conv(diag))
            
            # Handle different diagonal lengths naturally
            if k > 0:
                pad_size = diagonals[0].size(-1) - diag.size(-1)
                feat = torch.nn.functional.pad(feat, (0, pad_size))
            
            features.append(feat)
        return features

class FusionNetwork(nn.Module):
    def __init__(self, N, hidden_dim=64):
        super(FusionNetwork, self).__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        
        # Single projection that can handle any channel count
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Multi-head attention that can handle variable sequence lengths
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, features):
        batch_size = features[0].size(0)
        
        # Project and prepare features
        projected_features = []
        for feat in features:
            # Handle different channel counts
            if feat.size(1) != self.hidden_dim:
                feat = nn.functional.interpolate(
                    feat, 
                    size=(self.hidden_dim, feat.size(-1)),
                    mode='bilinear',
                    align_corners=False
                )
            feat = feat.permute(0, 2, 1)  # [B, N, C]
            projected_features.append(self.projection(feat))
            
        # Stack features for attention
        # Shape: [B, num_diagonals, N, hidden_dim]
        combined = torch.stack(projected_features, dim=1)
        
        # Reshape for attention
        combined = combined.view(-1, combined.size(-2), combined.size(-1))
        
        # Apply attention - handles variable sequence lengths automatically
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Reshape back
        output = attn_output.view(batch_size, len(features), self.N, -1)
        
        # Pool across diagonals
        output = output.mean(dim=1)  # [B, N, hidden_dim]
        
        return output.permute(0, 2, 1)  # [B, hidden_dim, N]

class EigenDecompositionNetwork(nn.Module):
    def __init__(self, N):
        super(EigenDecompositionNetwork, self).__init__()
        self.N = N
        self.version = "v2"  # version tracking
        self.decoder = nn.Sequential(
            nn.Linear(N*64, 512),
            nn.ReLU(),
            nn.Linear(512, N + N*N)
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

class EigensolverModel(nn.Module):
    def __init__(self, N):
        super(EigensolverModel, self).__init__()
        self.N = N
        self.encoder = BandLimitedEncoder(N)
        self.fusion = FusionNetwork(N)
        self.decoder = EigenDecompositionNetwork(N)

    def forward(self, diagonals):
        # Encode the diagonals
        features = self.encoder(diagonals)
        
        # Fuse the features
        fused = self.fusion(features)  # Shape: [B, 64, N]
        
        # Ensure proper tensor layout before decoding
        fused = fused.contiguous()
        
        # Decode to get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self.decoder(fused)
        
        return eigenvalues, eigenvectors

def extract_diagonals(L):
    diagonals = []
    N = L.size(1)
    for k in range(N):
        diag_k = torch.stack([torch.diagonal(L[i], offset=k) for i in range(L.size(0))])
        if torch.all(diag_k == 0):
            break
        diagonals.append(diag_k)
    return diagonals

def compute_loss(eigenvalues_pred, eigenvectors_pred, eigenvalues_true, eigenvectors_true, L_true, N, device):
    loss_eigenvalues = torch.mean((eigenvalues_pred - eigenvalues_true) ** 2)
    trace_pred = torch.sum(eigenvalues_pred, dim=1)
    trace_true = torch.sum(eigenvalues_true, dim=1)
    loss_trace = torch.mean((trace_pred - trace_true) ** 2)
    zero_eigenvalue_loss = torch.mean((eigenvalues_pred[:,0]) ** 2)

    identity = torch.eye(N, device=device).unsqueeze(0).expand(eigenvectors_pred.size(0), N, N)
    eigenvectors_pred_t = eigenvectors_pred.transpose(1,2)
    orthogonality = torch.bmm(eigenvectors_pred, eigenvectors_pred_t)
    loss_orthogonality = torch.mean((orthogonality - identity) ** 2)

    L_pred = torch.bmm(eigenvectors_pred_t, torch.bmm(torch.diag_embed(eigenvalues_pred), eigenvectors_pred))
    loss_structural = torch.mean((L_pred - L_true) ** 2)

    total_loss = 0.35 * (loss_eigenvalues + zero_eigenvalue_loss + loss_trace) \
                 + 0.35 * loss_orthogonality \
                 + 0.3 * loss_structural

    losses = {
        'eigenvalue_loss': loss_eigenvalues.item(),
        'trace_loss': loss_trace.item(),
        'zero_eigenvalue_loss': zero_eigenvalue_loss.item(),
        'orthogonality_loss': loss_orthogonality.item(),
        'structural_loss': loss_structural.item(),
        'total_loss': total_loss.item(),
    }

    return total_loss, losses

def train(model, train_loader, val_loader, optimizer, scheduler, start_epoch, checkpoint_dir, loss_history_file):
    best_val_loss = float('inf')
    loss_history = []

    epoch = start_epoch
    while True:
        print(f"\nEpoch {epoch+1}")
        model.train()
        running_loss = 0.0
        running_metrics = {}
        num_batches = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc="Training", ncols=100)
        for i, (L_batch, eigenvalues_batch, eigenvectors_batch) in progress_bar:
            L_batch = L_batch.to(device)
            eigenvalues_batch = eigenvalues_batch.to(device)
            eigenvectors_batch = eigenvectors_batch.to(device)
            optimizer.zero_grad()
            diagonals = extract_diagonals(L_batch)
            diagonals = [d.to(device) for d in diagonals]

            if use_amp:
                with autocast():
                    eigenvalues_pred, eigenvectors_pred = model(diagonals)
                    total_loss, losses = compute_loss(eigenvalues_pred, eigenvectors_pred, eigenvalues_batch, eigenvectors_batch, L_batch, model.decoder.N, device)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                eigenvalues_pred, eigenvectors_pred = model(diagonals)
                total_loss, losses = compute_loss(eigenvalues_pred, eigenvectors_pred, eigenvalues_batch, eigenvectors_batch, L_batch, model.decoder.N, device)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            running_loss += total_loss.item()
            for key in losses:
                running_metrics[key] = running_metrics.get(key, 0.0) + losses[key]

            if i % max(1, num_batches // 10) == 0 or i == num_batches -1:
                avg_loss = running_loss / (i+1)
                avg_metrics = {k: running_metrics[k]/(i+1) for k in running_metrics}
                progress_desc = f"Epoch [{epoch+1}] Batch [{i+1}/{num_batches}]"
                progress_desc += f" Loss: {avg_loss:.6f}"
                for k, v in avg_metrics.items():
                    progress_desc += f", {k}: {v:.6f}"
                progress_bar.set_description(progress_desc)

        val_loss, val_metrics = validate(model, val_loader)
        print(f"\nValidation Loss: {val_loss:.6f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.6f}")

        epoch_loss = {
            'epoch': epoch+1,
            'train_loss': running_loss / num_batches,
            'val_loss': val_loss,
        }
        epoch_loss.update({f'train_{k}': running_metrics[k] / num_batches for k in running_metrics})
        epoch_loss.update({f'val_{k}': val_metrics[k] for k in val_metrics})
        loss_history.append(epoch_loss)

        df = pd.DataFrame(loss_history)
        df.to_csv(loss_history_file, index=False)

        save_checkpoint(model, {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_dir, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved.")

def validate(model, val_loader):
    model.eval()
    running_loss = 0.0
    running_metrics = {}
    num_batches = len(val_loader)
    with torch.no_grad():
        for i, (L_batch, eigenvalues_batch, eigenvectors_batch) in enumerate(val_loader):
            L_batch = L_batch.to(device)
            eigenvalues_batch = eigenvalues_batch.to(device)
            eigenvectors_batch = eigenvectors_batch.to(device)
            diagonals = extract_diagonals(L_batch)
            diagonals = [d.to(device) for d in diagonals]
            eigenvalues_pred, eigenvectors_pred = model(diagonals)
            total_loss, losses = compute_loss(eigenvalues_pred, eigenvectors_pred, eigenvalues_batch, eigenvectors_batch, L_batch, model.decoder.N, device)
            running_loss += total_loss.item()
            for key in losses:
                running_metrics[key] = running_metrics.get(key, 0.0) + losses[key]

    avg_loss = running_loss / num_batches
    avg_metrics = {k: running_metrics[k]/num_batches for k in running_metrics}
    return avg_loss, avg_metrics




def main():
    parser = argparse.ArgumentParser(description="Train Eigensolver Neural Network.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing training data.")    
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--loss_history_file", type=str, default="loss_history.csv", help="File to save loss history.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    full_dataset = LaplacianDataset(args.data_dir)
    
    # Get a sample first to determine shapes and sizes
    sample_L, _, _ = full_dataset[0]
    N = sample_L.size(0)
    
    # Now we can use sample_L.shape in find_batch_size
    def find_batch_size(sample_shape, max_batch=32):
        batch_size = max_batch
        while batch_size > 1:
            try:
                torch.cuda.empty_cache()
                dummy = torch.randn(batch_size, *sample_shape, device=device)
                return batch_size
            except RuntimeError:
                batch_size //= 2
        return 1
    
    batch_size = find_batch_size(sample_L.shape)
    
    # Split dataset and create loaders
    total_size = len(full_dataset)
    val_size = total_size // 10
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = EigensolverModel(N).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs)

    start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint_dir)
    train(model, train_loader, val_loader, optimizer, scheduler, start_epoch, args.checkpoint_dir, args.loss_history_file)

if __name__ == "__main__":
    main()
