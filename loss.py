import torch
from torch import Tensor
import torch.nn.functional as F



def alignment_loss(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    """
    Computes the alignment loss between two sets of embeddings x and y.
    This measures how close the positive pairs are in the embedding space.
    
    Args:
        x: First set of embeddings (shape: [N, D])
        y: Second set of embeddings (shape: [N, D])
        alpha: Exponent for the distance calculation
        
    Returns:
        Alignment loss value
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()



def uniformity_loss(x: Tensor, t: float) -> Tensor:
    """
    Computes the uniformity loss for a set of embeddings x.
    This measures how uniformly the embeddings are distributed on the hypersphere
    using a Gaussian potential kernel.
    
    Args:
        x: Set of embeddings (shape: [N, D])
        t: Temperature parameter
        
    Returns:
        Uniformity loss value
    """
    x = F.normalize(x, p=2, dim=1)
    sq_dists = torch.pdist(x, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()



def quadratic_wasserstein_loss(x: Tensor) -> Tensor:
    """
    Computes the Quadratic Wasserstein Loss (approximated W2 distance) between
    the empirical distribution of the embeddings and a target uniform distribution.

    This loss utilizes the statistical moments (mean and covariance) of the
    embeddings to encourage feature decorrelation and prevent dimensional collapse.

    Args:
        x: Input embedding tensor (shape: [N, D])
           N: Batch size, D: Dimension size

    Returns:
        The calculated Wasserstein loss value (Scalar)
    """
    N, D = x.shape
    x = F.normalize(x, p=2, dim=1)

    mu = x.mean(dim=0)
    x_centered = x - mu
    cov = (x_centered.T @ x_centered) / (N - 1)
    # In original code, they use N instead of (N - 1).
    # Waring: If we use small batch size (N), then its difference quite large.
    # But theoretically, it should be (N - 1) for the unbiased estimated covariance.

    trace_cov = cov.diagonal().sum()
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.clamp(min=1e-12)
    trace_sqrt_cov = eigvals.sqrt().sum()

    w2_sq = mu.norm().pow(2) + 1 + trace_cov - (2 / (D ** 0.5)) * trace_sqrt_cov
    return F.relu(w2_sq)
    # In original code, they return torch.sqrt(w2_sq) (in fact, W_2). 
    # But we use w2_sq (in fact, W_2^2) for the gradient stability.
    # ReLU is also not in original, but we use to avoid negative values as a inevitable computational error of eigenvalsh.