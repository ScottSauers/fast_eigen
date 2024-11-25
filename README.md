# Fast Eigensolver: High-Speed Approximate Eigendecomposition for Banded Laplacians

## Problem Specification
- Input: Banded Laplacian matrices from sparse, locally connected graphs
- Output: All eigenvalues and eigenvectors
- Constraint: Inference must be faster than traditional algorithms
- Trade-off: Approximate solutions accepted for speed

## Matrix Properties
- Size: N×N matrix
- Bandwidth: b (matrix is (2b+1)-diagonal)
- Symmetry: Symmetric (Laplacian property)
- Sparsity: O(bN) non-zero elements
- Structure: Banded, locally connected

## Input Representation
1. Matrix Encoding:
   - Split into 2b+1 diagonals
   - Each diagonal stored as vector length N
   - Zero-pad shorter diagonals to length N
   - Normalize values to [-1,1] range

2. Positional Information:
   - Each diagonal indexed [-b, ..., 0, ..., b]
   - Position within diagonal: normalized [0,1]
   - Combined positional encoding using sin/cos

## Architecture Details

### 1. Parallel Diagonal Encoder
- Input: 2b+1 channels, each N length
- Architecture per diagonal:
   * 3 convolutional layers
   * Channel progression: 1 → 64 → 128 → 256
   * Kernel size: 3
   * Padding: Same
   * Activation: LeakyReLU(0.2)
   * Batch Normalization after each conv

### 2. Fusion Module
- Input: (2b+1) × 256 × N features
- Operations:
   * 1×1 convolution to merge diagonal features
   * Channel reduction: (2b+1)×256 → 512
   * Global feature extraction
   * Output: 512×N feature matrix

### 3. Parallel Decoder
- Input: 512×N fused features
- Architecture:
   * Two-branch design for values/vectors
   * Eigenvalue branch:
     - 2 linear layers: 512 → 256 → N
     - Outputs N eigenvalues
   * Eigenvector branch:
     - Linear expansion: 512 → N×N
     - Reshape to N×N matrix
     - Orthogonalization layer

## Training Protocol

### Data Generation
- Source: Random graphs with specified connectivity
- Size range: Various N within GPU memory
- Bandwidth range: Various b values
- Ground truth: Exact eigendecomposition

### Loss Functions
1. Eigenvalue Loss:
   - MSE between predicted and true eigenvalues
   - Weighted by eigenvalue magnitude
   - Weight: 0.4

2. Eigenvector Loss:
   - Subspace alignment error
   - Orthogonality penalty
   - Weight: 0.4

3. Reconstruction Loss:
   - Original matrix reconstruction error
   - Weight: 0.2

### Training Parameters
- Optimizer: AdamW
- Learning rate: 1e-4 with cosine decay
- Batch size: Maximum fitting GPU
- Epochs: 100
- Mixed precision: FP16
- Gradient clipping: 1.0

## Inference Optimization
1. Hardware Utilization:
   - Full GPU parallelization
   - Tensor core optimization
   - Batch processing when possible

2. Memory Management:
   - Streaming large matrices
   - In-place operations
   - Mixed precision inference

3. Post-processing:
   - Eigenvalue sorting
   - Eigenvector orthogonalization
   - Optional refinement step

## Performance Specifications
- Time complexity: O(nb) processing + O(n²) generation
- Memory complexity: O(n²)
- Accuracy target: 1e-3 relative error
- Batch throughput: Scale with GPU memory

## Implementation Requirements
- Framework: PyTorch
- GPU: CUDA support required
- Dependencies: torch, numpy, scipy
