# Fast Eigensolver: Band-Limited Neural Architecture for Laplacian Matrices

## Input Specifications

The eigensolver operates on a specialized class of matrices derived from graph theory. These matrices must be symmetric Laplacian matrices that originate from undirected, locally connected graphs. The system processes matrices of dimension N×N. The matrix bandwidth (count of non-zero diagonals above the main diagonal) is determined dynamically from the input matrix structure, allowing the system to handle varying bandwidths without predefined limits.

Key matrix properties:
- Matrix entries must be integer-valued
- Main diagonal contains positive integers representing vertex degrees
- Off-diagonal elements are constrained to {-1, 0}
- Row sums equal zero (Laplacian property)
- Spectrum is real and non-negative (positive semi-definite)

## Matrix Structure Exploitation

The system employs an efficient storage scheme that capitalizes on matrix symmetry. Rather than storing the full N×N matrix, we maintain only the upper triangular portion through a series of vectors with naturally decreasing lengths. This approach eliminates redundancy while preserving the integer structure of the input.

Storage organization:
- Main diagonal: Full length N vector
- First off-diagonal: Length N-1 vector
- k-th off-diagonal: Length N-k vector
- Total storage: Dynamically determined (bandedness)

## Neural Architecture

### Band-Limited Encoder Network

The encoder implements a parallel processing approach optimized for GPU computation. Each diagonal receives specialized processing through dedicated convolutional paths with dimensions scaled to preserve necessary information capacity.

Channel architecture:
- Main diagonal: 1→64 channels (primary information carrier)
- First three off-diagonals: 1→32 channels
- Remaining diagonals: 1→16 channels

This channel structure ensures sufficient capacity for eigenspace information while maintaining computational efficiency. All convolutional kernels employ a fixed size of min(b,5), ensuring operations remain strictly within the banded structure.

Input processing:
- Main diagonal: Processes full length N
- k-th diagonal: Natural length N-k
- No padding or artificial extension
- Parallel processing of all diagonals on GPU

Structural preservation:
- ReLU-based activation functions preserving non-negativity
- Skip connections maintaining direct pathways
- Integer relationship preservation through specialized layers


### Fusion Network

The fusion network integrates features from the band-limited encoder using a lightweight transformer architecture. Input consists of b+1 feature maps (one per diagonal) from the CNN encoder, where b is the bandwidth of the input matrix.

Input structure:
- Main diagonal: length N with 64 channels
- First three off-diagonals: length N with 32 channels
- Remaining off-diagonals: length N with 16 channels
- Total of b+1 feature maps, where b varies by input matrix

Feature projection:
- All diagonal feature maps are projected to fixed hidden dimension (64)
- Projection necessary for consistent attention operations
- Preserves length N for each feature map
- Position encoding added to indicate diagonal index (0 to b)

Transformer architecture:
- Single-layer transformer with 4 attention heads
- Input: (b+1) × N × 64 feature tensor
- Self-attention computes relationships between diagonal features
- Output maintains shape: (b+1) × N × 64
- Same weights used regardless of bandwidth b

Attention mechanism:
- Q,K,V projections from 64-dimensional features
- Attention matrix size adapts to number of diagonals
- For bandwidth b: computes (b+1) × (b+1) attention weights
- Each head learns different diagonal relationship patterns

Output processing:
- Maintains length N and channel dimension 64
- Preserves feature relationships across diagonals
- Output features feed directly into eigendecomposition network
- No dimension reduction or expansion

The fusion network learns to process diagonal relationships that aid eigendecomposition, while automatically handling varying bandwidths through attention's natural handling of different sequence lengths. The fixed hidden dimension (64) enables consistent weight matrices while the flexible attention mechanism adapts to different numbers of diagonals.

### Direct Eigendecomposition Network

The system implements parallel eigendecomposition while maintaining all required constraints. This network processes the feature matrices to produce all N eigenvalues and N eigenvectors simultaneously.

Structural constraints:
- Non-negativity of eigenvalues
- Orthogonality of eigenvectors enforcement

## Training Protocol

Multi-objective optimization targeting both accuracy and structural preservation:

Loss function weighting:
1. Eigenvalue Accuracy (0.35)
   - Direct MSE with structure preservation
   - Trace consistency enforcement
   - Known eigenvalue bound satisfaction
   - Zero eigenvalue preservation
   - Multiplicity preservation

2. Eigenvector Quality (0.35)
   - Orthogonality
   - Accuracy

3. Structural Integrity (0.30)
   - Laplacian property preservation
   - Integer relationship preservation
   - Component-wise accuracy
   - Trace preservation

Implementation:
- AdamW optimizer with cosine schedule
- Mixed precision (FP16/BF16) training
- Gradient clipping preserving constraints
- Early stopping on validation metrics
- Batch processing optimization

## Performance Specifications

Computational complexity (for N×N matrix, bandwidth b):
- Single matrix processing: O(N²) operations (output bound)
- Batch processing K matrices: O(N²) operations total
- Memory utilization: O(N²) peak (output bound)
- Storage requirement: O(bN) per input matrix

Accuracy targets:
- Eigenvalue relative error: 1e-6
- Eigenvector orthogonality: 1e-6
- Zero eigenvalue preservation: Exact
- Integer relationship preservation: Exact
- Trace preservation: Exact

Performance advantages:
1. Batch Processing
   - Process K matrices in parallel
   - Amortized cost O(N²/K) per matrix
   - Optimal GPU tensor core utilization

2. Precision Targeting
   - Consistent 10⁻⁶ accuracy
   - Avoid unnecessary precision overhead
   - Maintain essential properties exactly

3. Hardware Optimization
   - Full GPU utilization
   - Mixed precision acceleration
   - Optimized memory access patterns
   - Parallel eigenvector computation

Hardware requirements:
- CUDA-capable GPU
- Memory scaling with max(O(bN), O(N²))
- High-bandwidth memory preferred
