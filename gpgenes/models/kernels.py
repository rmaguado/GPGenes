from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.linalg import expm


def graph_to_weighted_adjacency(
    G: nx.DiGraph, n: int, use_abs: bool = True
) -> np.ndarray:
    """
    Convert a directed graph into an adjacency matrix A (n x n).

    A[i, j] = weight of edge i -> j
    If use_abs is True, edge weights are converted to absolute values.
    (ignores activation vs repression sign).
    """
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if use_abs:
            w = abs(w)
        A[int(u), int(v)] += w
    return A


def compute_stationary_distribution(P: np.ndarray, tol: float = 1e-9, max_iter: int = 10000) -> np.ndarray:
    """
    Compute the stationary distribution (phi) of the transistion matrix P using power iteration.
    phi @ P = phi
    """
    n = P.shape[0]
    phi = np.ones(n) / n
    
    for _ in range (max_iter):
        phi_new = phi @ P
        if np.linalg.norm(phi_new - phi, 1) < tol:
            return phi_new
        phi = phi_new

    print("Warning: Stationary distribution did not converge.")
    return phi

def directed_diffusion_kernel(
        A: np.ndarray,
        beta: float = 1.0,
        teleport_prob: float = 0.01,
        jitter: float = 1e-8
) -> np.ndarray:
    """
    construct a diffusion kernel that respects the directionality of the GRN
    
    this uses a random walk laplacian dervied from the directed graph.

    args:
        A: (n, n) directed adjacency matrix (usually abs values of weights)
        beta: diffusion parameter (time scale)
        teleport_prob: probability of teleportation. Crucial for GRNs to handle sinks / disconnected nodes.
        jitter: small value added to diagonal for numerical stability

    returns:
        K_gene: (n, n) positive-definite kernel matrix.
    """
    n = A.shape[0]
    A = np.asarray(A, dtype=float)

    # row-normalise A to get transition matrix P
    out_degrees = A.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = A / out_degrees[:, None]
    P[np.isnan(P)] = 0.0 # fix row where out_degrees was 0

    # add teleportation to ensure ergodicity
    if teleport_prob > 0:
        P = (1 - teleport_prob) * P + (teleport_prob / n) * np.ones((n, n))

    # compute stationary distribution
    phi = compute_stationary_distribution(P)
    phi = np.maximum(phi, 1e-12)  # avoid zeros

    Phi_sqrt = np.diag(np.sqrt(phi))
    Phi_inv_sqrt = np.diag(1.0 / np.sqrt(phi))

    S = Phi_sqrt @ P @ Phi_inv_sqrt

    S_sym = (S + S.T) / 2.0

    L = np.eye(n) - S_sym

    K = expm(-beta * L)
    K = (K + K.T) / 2.0  # ensure symmetry
    K += np.eye(n) * jitter  # add jitter for numerical stability
    return K


def symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Make adjacency matrix symmetric:
    A_sym = (A + A^T) / 2

    Used so that downstream kernels are symmetric and positive semidefinite
    """
    return (A + A.T) / 2.0


def diffusion_node_kernel(
    A_sym: np.ndarray, beta: float = 1.0, jitter: float = 1e-8
) -> np.ndarray:
    """
    Construct a diffusion kernel over genes. (this is the biological prior)

    Steps:
       1. Compute graph Laplacian L = D - A_sym.
       2. Compute diffusion kernel: K = exp(-beta * L)

    Interpretations:
       1. Genes close in the regulatory network are more similar.
       2. Information diffuses along regulatory paths

    Returns:
        K_gene: (n_genes, n_genes) positive-definite kernel matrix.
    """
    A_sym = np.asarray(A_sym, dtype=float)
    D = np.diag(A_sym.sum(axis=1))
    L = D - A_sym

    # matrix exponential of the laplaian
    K = expm(-beta * L)

    # TODO: normalise K or normalise L (before computing K), this will improve numerical stability and make a1, a2, a3 easier to interpret

    # ensure symmetry
    K = (K + K.T) / 2.0

    # add jitter for numerical stability on the diagonal
    K += np.eye(K.shape[0]) * jitter
    return K


def _k1_set_linear(Xa: np.ndarray, Xb: np.ndarray, K_gene: np.ndarray) -> np.ndarray:
    """
    Linear set kernel between perturbations.

    Each row of X is a multi-hot vector of knocked-out genes

    k1(Xa, Xb) where rows are perturbations:
      k1(x, x') = x^T K_gene x'

    Interpretation:
        Two perturbations are similar if they knock out related genes.

    Shapes:
        Xa: (na, n_genes)
        Xb: (nb, n_genes)
        K_gene: (n_genes, n_genes)

    Returns:
        K: (na, nb) kernel matrix
    """
    return Xa @ K_gene @ Xb.T


def _rbf_from_Z(Za: np.ndarray, Zb: np.ndarray, length_scale: float) -> np.ndarray:
    """
    RBF kernel between two feature matrices Za (na,d), Zb (nb,d).

    Interpretation:
        Smooth nonlinear similarity between perturbations in the feature space

    k(z, z') = exp(-||z - z'||^2 / (2 * l^2))
    """
    Za2 = np.sum(Za**2, axis=1, keepdims=True)
    Zb2 = np.sum(Zb**2, axis=1, keepdims=True).T
    d2 = Za2 + Zb2 - 2.0 * (Za @ Zb.T)
    return np.exp(-0.5 * d2 / (length_scale**2 + 1e-12))


def kernel_components(
    Xa: np.ndarray,
    Xb: np.ndarray,
    K_gene: np.ndarray,
    length_scale: float = 1.0,
):
    """
    Compute individual kernel components between perturbations.

    Returns (k1, k2, krbf) as (na, nb) arrays:
      k1 = x^T K_gene x'
      k2 = (k1)^2
      krbf = RBF( K_gene x , K_gene x' )
    """
    k1 = _k1_set_linear(Xa, Xb, K_gene)
    k2 = (
        k1**2
    )  # quadratic interaction kernel (captures synergistic / epistatic effects)

    # TODO: k1 can be large, which would make k2 explode and kernel can then become dominated by the interaction term.
    #       consider normalising k1 before squaring (important if we see model instability during training)

    Za = (K_gene @ Xa.T).T
    Zb = (K_gene @ Xb.T).T
    kr = _rbf_from_Z(Za, Zb, length_scale)
    return k1, k2, kr


def combined_kernel(
    Xa: np.ndarray,
    Xb: np.ndarray,
    K_gene: np.ndarray,
    a1: float = 1.0,
    a2: float = 0.5,
    a3: float = 0.2,
    length_scale: float = 1.0,
):
    """
    Final perturbation kernel as weighted sum of components:

    K = a1^2*k1 + a2^2*k2 + a3^2*krbf

    a1, a2, a3 control relative importance of:
    - linear effects (k1)
    - interaction effects (k2)
    - smooth nonlinear similarity (kr)

    Note: not added the diffusion kernel over genes here because it's pre-multiplied into X already.
    """
    k1, k2, kr = kernel_components(Xa, Xb, K_gene, length_scale=length_scale)
    K = (a1**2) * k1 + (a2**2) * k2 + (a3**2) * kr
    return K


def combined_kernel_diag(
    X: np.ndarray,
    K_gene: np.ndarray,
    a1: float = 1.0,
    a2: float = 0.5,
    a3: float = 0.2,
):
    """
    Compute diag(K(X,X)) without building full matrix.

    Used for GP predictive variance

    For components:
      diag(k1) = x^T K_gene x
      diag(k2) = diag(k1)^2
      diag(krbf) = 1
    """
    XK = X @ K_gene
    d1 = np.sum(XK * X, axis=1)
    d2 = d1**2
    dr = np.ones_like(d1)
    return (a1**2) * d1 + (a2**2) * d2 + (a3**2) * dr
