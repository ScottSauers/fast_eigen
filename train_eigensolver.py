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

# Mixed precision training
use_amp = True if torch.cuda.is_available() else False
if use_amp:
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
else:
    scaler = None

def save_checkpoint(state, checkpoint_dir, epoch):
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {epoch}.")

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        print("No checkpoint found. Starting training from scratch.")
        return 0  # Start from epoch 0

    # Find latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. Resuming from epoch {start_epoch}.")
    return start_epoch

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
    def __init__(self, N, b):
        super(BandLimitedEncoder, self).__init__()
        self.N = N
        self.b = b

        self.convs = nn.ModuleList()
        diag_channels = [64] + [32]*3 + [16]*(b-3)
        kernel_size = min(b,5)
        for k in range(b+1):
            in_channels = 1
            if k == 0:
                out_channels = 64
            elif k < 4:
                out_channels = 32
            else:
                out_channels = 16
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.convs.append(conv)

    def forward(self, diagonals):
        features = []
        for k, diag in enumerate(diagonals):
            diag = diag.unsqueeze(1)
            conv = self.convs[k]
            feat = torch.relu(conv(diag))
            features.append(feat)
        return features

class FusionNetwork(nn.Module):
    def __init__(self, N, b):
        super(FusionNetwork, self).__init__()
        self.N = N
        self.b = b

        self.attention_heads = nn.ModuleList()
        num_heads = 4
        for _ in range(num_heads):
            attn_head = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.1)
            self.attention_heads.append(attn_head)
        self.linear = nn.Linear(64*num_heads, 64)

    def forward(self, features):
        combined = torch.cat(features, dim=2)
        combined = combined.permute(2, 0, 1)

        attn_outputs = []
        for attn_head in self.attention_heads:
            attn_output, _ = attn_head(combined, combined, combined)
            attn_outputs.append(attn_output)

        attn_outputs_concat = torch.cat(attn_outputs, dim=2)
        output = torch.relu(self.linear(attn_outputs_concat))
        return output.permute(1, 2, 0)

class EigenDecompositionNetwork(nn.Module):
    def __init__(self, N):
        super(EigenDecompositionNetwork, self).__init__()
        self.N = N
        self.fc = nn.Sequential(
            nn.Linear(N*64, 512),
            nn.ReLU(),
            nn.Linear(512, N + N*N)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.fc(x)
        eigenvalues = out[:, :self.N]
        eigenvectors = out[:, self.N:].view(batch_size, self.N, self.N)
        eigenvalues = torch.relu(eigenvalues)
        eigenvectors = torch.nn.functional.normalize(eigenvectors, dim=2)
        return eigenvalues, eigenvectors

class EigensolverModel(nn.Module):
    def __init__(self, N, b):
        super(EigensolverModel, self).__init__()
        self.encoder = BandLimitedEncoder(N, b)
        self.fusion = FusionNetwork(N, b)
        self.decoder = EigenDecompositionNetwork(N)

    def forward(self, diagonals):
        features = self.encoder(diagonals)
        fused = self.fusion(features)
        eigenvalues, eigenvectors = self.decoder(fused)
        return eigenvalues, eigenvectors

def extract_diagonals(L, b):
    diagonals = []
    N = L.size(1)
    for k in range(b+1):
        diag_k = torch.stack([torch.diagonal(L[i], offset=k) for i in range(L.size(0))])
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

def train(model, train_loader, val_loader, optimizer, scheduler, start_epoch, num_epochs, checkpoint_dir, loss_history_file):
    best_val_loss = float('inf')
    loss_history = []

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
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
            diagonals = extract_diagonals(L_batch, model.encoder.b)
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
                progress_desc = f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{num_batches}]"
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

        save_checkpoint({
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
            diagonals = extract_diagonals(L_batch, model.encoder.b)
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
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--bandwidth", type=int, default=5, help="Bandwidth (b) of the matrices.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--loss_history_file", type=str, default="loss_history.csv", help="File to save loss history.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    full_dataset = LaplacianDataset(args.data_dir)
    total_size = len(full_dataset)
    val_size = total_size // 10
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    sample_L, _, _ = full_dataset[0]
    N = sample_L.size(0)

    model = EigensolverModel(N, args.bandwidth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*args.epochs)

    start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint_dir)
    train(model, train_loader, val_loader, optimizer, scheduler, start_epoch, args.epochs, args.checkpoint_dir, args.loss_history_file)

if __name__ == "__main__":
    main()