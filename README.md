# Fast Eigensolver: High-Speed Eigendecomposition for Banded Laplacian Matrices

## Input Specifications
- Matrix Type: Symmetric Laplacian matrices from undirected, locally connected graphs
- Values: Integer-valued entries
- Size: N×N
- Bandwidth: b (matrix has b non-zero diagonals above main diagonal)
- Sparsity: Maximum of 2bN + N non-zero elements
- Properties:
  * Main diagonal contains positive integers
  * Off-diagonals contain -1 or 0 only
  * Row sums are zero (Laplacian property)

## Matrix Representation
- Store only upper triangular part (due to symmetry)
- Each diagonal stored at its natural length:
  * Main diagonal: length N
  * First off-diagonal: length N-1
  * k-th off-diagonal: length N-k
- No padding or normalization (preserve integer structure)
- Total storage: (b+1) vectors of decreasing length

## Neural Architecture

### 1. Diagonal Encoder Network
- Separate 1D CNN for each diagonal
- Input sizes decrease with diagonal number:
  * Main diagonal: 1 × N
  * k-th diagonal: 1 × (N-k)
- Architecture per diagonal:
  * Conv1D(in=1, out=64, kernel=3) → LeakyReLU
  * Conv1D(in=64, out=128, kernel=3) → LeakyReLU
  * Conv1D(in=128, out=256, kernel=3) → LeakyReLU
- No padding used (preserve length reduction)
- No batch normalization (integer inputs don't need it)
- All diagonals processed in parallel on GPU

### 2. Diagonal Fusion Network
- Input: b+1 feature maps of shape (256 × varying_length)
- Local fusion operation:
  * Group features from corresponding positions across diagonals
  * 1x1 convolution merges diagonal features at each position
  * Output channels: 512 at each position
- No global operations (maintain locality)
- Output: 512 × N feature matrix

### 3. Decoder Network
Two parallel branches:

Eigenvalue Branch:
- Input: 512 × N feature matrix
- Global average pooling to 512 features
- Linear(512 → N) outputs eigenvalues
- Natural ordering by magnitude (no explicit sorting needed)

Eigenvector Branch:
- Input: 512 × N feature matrix
- Progressive upsampling to N × N:
  * Linear(512 → 1024) per position
  * Reshape and expand to N × N matrix
- No explicit orthogonalization (learned through loss)

## Training Protocol

### Data Generation
- Sample random connected graphs with bandwidth ≤ b
- Convert to integer-valued Laplacian matrices
- Compute ground truth using standard eigendecomposition
- Generate batches of increasing size N during training

### Loss Functions
Combined loss with three terms:

1. Eigenvalue MSE Loss (weight 0.4):
   * Direct MSE between predicted and true eigenvalues
   * No magnitude weighting (integers are already well-scaled)

2. Eigenvector Alignment Loss (weight 0.4):
   * Subspace alignment between predicted and true eigenvectors
   * Implicit orthogonality through reconstruction

3. Matrix Reconstruction Loss (weight 0.2):
   * MSE between original matrix and V∙Λ∙V^T
   * Check sparsity pattern preservation

### Training Details
- Optimizer: AdamW with lr=1e-4
- Mixed precision training (FP16)
- Gradient clipping at 1.0
- Early stopping on validation loss
- Progressive N scaling during training

## Inference Optimizations

Speed Optimizations:
- Integer-optimized input processing
- Parallel diagonal processing
- Fused CUDA kernels for key operations
- Batch processing for multiple matrices

Memory Optimizations:
- In-place operations where possible
- Shared memory usage for diagonal processing
- Progressive memory allocation
- Mixed precision inference

No post-processing steps (everything learned end-to-end)

## Performance Targets
- Time Complexity: O(nb) encoding + O(n²) decoding
- Memory Usage: O(n²) peak memory
- Accuracy: 1e-3 relative error on eigenvalues
- Throughput: Scales with available GPU memory
- Batch Processing: Yes, with dynamic batching

## Implementation Requirements
- Framework: PyTorch with CUDA
- GPU: CUDA-capable device
- Memory: Scales with largest N needed
