from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.linalg import expm
from enum import Enum, auto


class GeneKernelMode(Enum):
    ABSOLUTE = auto()
    SIGNED = auto()
    MIXED = auto()


def build_gene_kernel(
    A_signed: np.ndarray,
    mode: GeneKernelMode,
    beta: float,
    teleport_prob: float = 0.05,
    jitter: float = 1e-8,
    w_abs: float | None = None,
    w_pos: float | None = None,
    w_neg: float | None = None,
):
    if mode == GeneKernelMode.ABSOLUTE:
        return directed_diffusion_kernel(
            np.abs(A_signed),
            beta=beta,
            teleport_prob=teleport_prob,
            jitter=jitter,
        )

    elif mode == GeneKernelMode.SIGNED:
        return signed_directed_diffusion_kernel(
            A_signed,
            beta=beta,
            teleport_prob=teleport_prob,
            jitter=jitter,
        )

    elif mode == GeneKernelMode.MIXED:
        return mixed_signed_directed_diffusion_kernel(
            A_signed,
            beta=beta,
            teleport_prob=teleport_prob,
            jitter=jitter,
            w_abs=1.0 if w_abs is None else w_abs,
            w_pos=1.0 if w_pos is None else w_pos,
            w_neg=1.0 if w_neg is None else w_neg,
        )

    else:
        raise ValueError(f"Unknown GeneKernelMode: {mode}")


def graph_to_weighted_adjacency(
    G: nx.DiGraph, n: int, use_abs: bool = True
) -> np.ndarray:
    """
    Convert a directed graph into an adjacency matrix A (n x n).

    A[i, j] = weight of edge i -> j

    If use_abs is True, edge weights are converted to absolute values.
    If use_abs is False, sign is preserved (activation vs repression sign).
    """
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if use_abs:
            w = abs(w)
        A[int(u), int(v)] += w
    return A


def compute_stationary_distribution(
    P: np.ndarray, tol: float = 1e-9, max_iter: int = 10000
) -> np.ndarray:
    """
    Compute the stationary distribution (phi) of the transition matrix P using power iteration.
    phi @ P = phi
    """
    n = P.shape[0]
    phi = np.ones(n) / n

    for _ in range(max_iter):
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
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    Construct a diffusion kernel that respects the directionality of the GRN.

    NOTE: This expects A >= 0 (nonnegative), because we interpret it as weights for a Markov chain.

    args:
        A: (n, n) directed adjacency matrix (nonnegative weights)
        beta: diffusion parameter (time scale)
        teleport_prob: probability of teleportation (handles sinks / disconnected nodes)
        jitter: small value added to diagonal for numerical stability

    returns:
        K_gene: (n, n) positive-definite kernel matrix.
    """
    n = A.shape[0]
    A = np.asarray(A, dtype=float)

    # row-normalise A to get transition matrix P
    out_degrees = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = A / out_degrees[:, None]
    P[np.isnan(P)] = 0.0  # fix rows where out_degrees was 0

    # add teleportation to ensure ergodicity
    if teleport_prob > 0:
        P = (1 - teleport_prob) * P + (teleport_prob / n) * np.ones((n, n))

    # compute stationary distribution
    phi = compute_stationary_distribution(P)
    phi = np.maximum(phi, 1e-12)  # avoid zeros

    Phi_sqrt = np.diag(np.sqrt(phi))
    Phi_inv_sqrt = np.diag(1.0 / np.sqrt(phi))

    # symmetrize the random walk in the stationary measure
    S = Phi_sqrt @ P @ Phi_inv_sqrt
    S_sym = (S + S.T) / 2.0

    # random-walk style Laplacian
    L = np.eye(n) - S_sym

    # matrix exponential of the Laplacian
    K = expm(-beta * L)

    # ensure symmetry + add jitter for numerical stability
    K = (K + K.T) / 2.0
    K += np.eye(n) * jitter
    return K


def signed_directed_diffusion_kernel(
    A_signed: np.ndarray,
    beta: float = 1.0,
    teleport_prob: float = 0.01,
    jitter: float = 1e-8,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
) -> np.ndarray:
    """
    Construct a signed directed diffusion kernel while staying PSD.

    Idea:
      - Split signed adjacency into positive + negative magnitudes:
          A_pos = max(A, 0)
          A_neg = max(-A, 0)
        Both are nonnegative, so each can be fed into directed_diffusion_kernel.

      - Combine with nonnegative weights:
          K_gene = w_pos^2 * K_pos + w_neg^2 * K_neg

    This distinguishes activation vs repression patterns without risking indefiniteness.
    """
    A_signed = np.asarray(A_signed, dtype=float)

    A_pos = np.maximum(A_signed, 0.0)
    A_neg = np.maximum(-A_signed, 0.0)

    K_pos = directed_diffusion_kernel(
        A_pos, beta=beta, teleport_prob=teleport_prob, jitter=jitter
    )
    K_neg = directed_diffusion_kernel(
        A_neg, beta=beta, teleport_prob=teleport_prob, jitter=jitter
    )

    K = (w_pos**2) * K_pos + (w_neg**2) * K_neg

    return K


def mixed_signed_directed_diffusion_kernel(
    A_signed: np.ndarray,
    beta: float = 1.0,
    teleport_prob: float = 0.01,
    jitter: float = 1e-8,
    w_abs: float = 0.5,
    w_pos: float = 1.0,
    w_neg: float = 1.0,
) -> np.ndarray:
    """
    Combines 'absolute' connectivity (pathways) with 'signed' specificity (mechanisms).

    Weights:
        - w_abs: weight for absolute connectivity kernel (restores mixed-sign paths )
        - w_pos: weight for positive (activation) kernel
        - w_neg: weight for negative (repression) kernel
    """
    A_signed = np.asarray(A_signed, dtype=float)

    # 1. Absolute connectivity kernel
    # ensures A -> B -> C paths are captured regardless of sign
    A_abs = np.abs(A_signed)
    K_abs = directed_diffusion_kernel(
        A_abs, beta=beta, teleport_prob=teleport_prob, jitter=jitter
    )

    # 2. Positive layer (activation only)
    A_pos = np.maximum(A_signed, 0.0)
    K_pos = directed_diffusion_kernel(
        A_pos, beta=beta, teleport_prob=teleport_prob, jitter=jitter
    )

    # 3. Negative layer (repression only)
    A_neg = np.maximum(-A_signed, 0.0)
    K_neg = directed_diffusion_kernel(
        A_neg, beta=beta, teleport_prob=teleport_prob, jitter=jitter
    )

    # 4. Combine kernels with weights
    # note, we square the weights to ensure PSD is preserved easily
    K = (w_abs**2) * K_abs + (w_pos**2) * K_pos + (w_neg**2) * K_neg

    return K


def _k1_set_linear(Xa: np.ndarray, Xb: np.ndarray, K_gene: np.ndarray) -> np.ndarray:
    """
    Linear set kernel between perturbations.

    Each row of X is a multi-hot vector of knocked-out genes.

    k1(x, x') = x^T K_gene x'

    Interpretation:
        Two perturbations are similar if they knock out related genes.
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


def _pairwise_similarity_mask_from_gene_kernel_embedding(
    K_gene: np.ndarray, rank: int = 16, eps: float = 1e-12
) -> np.ndarray:
    """
    Build a smooth motif-coupling mask using a low-rank embedding of K_gene.

    Steps:
      1) Eigen-decompose K_gene and keep top components (rank).
      2) Treat those components as an embedding z_i for each gene i.
      3) Define:
            M_ij = <z_i, z_j>^2   (always >= 0)

    Intuition:
      - genes that sit in similar "network contexts" (under K_gene) get high M_ij
      - encourages pairwise KO interactions along latent motifs / circuits
    """
    K = np.asarray(K_gene, dtype=float)
    K = (K + K.T) / 2.0  # safety

    evals, evecs = np.linalg.eigh(K)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # keep only positive eigenvalues
    pos = evals > eps
    evals = evals[pos]
    evecs = evecs[:, pos]

    r = min(rank, evecs.shape[1])
    if r == 0:
        return np.zeros_like(K)

    # Embedding: Z = V sqrt(Lambda)
    Z = evecs[:, :r] * np.sqrt(evals[:r])[None, :]  # (n, r)

    # Similarity in embedding, squared to be nonnegative
    S = Z @ Z.T
    M = S**2

    # remove diagonal and normalise for scale stability
    np.fill_diagonal(M, 0.0)
    mx = np.max(M) if np.max(M) > 0 else 1.0
    return M / mx


def _k_pairwise_interaction(
    Xa: np.ndarray, Xb: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """
    Embedding-based pairwise interaction kernel between perturbations.

    For each perturbation x, define pair-features:
        f_ij(x) = x_i x_j * M_ij   for i<j

    Then:
        k2(x, x') = <f(x), f(x')>

    This is PSD (linear kernel on explicit features) and targets motif interactions.
    """
    Xa = np.asarray(Xa, float)
    Xb = np.asarray(Xb, float)
    M = np.asarray(M, float)

    n = Xa.shape[1]
    iu = np.triu_indices(n, k=1)  # i<j

    Wa = Xa[:, :, None] * Xa[:, None, :]  # (na, n, n)
    Wb = Xb[:, :, None] * Xb[:, None, :]  # (nb, n, n)

    Fa = (Wa * M)[..., iu[0], iu[1]]
    Fb = (Wb * M)[..., iu[0], iu[1]]

    return Fa @ Fb.T


def kernel_components(
    Xa: np.ndarray,
    Xb: np.ndarray,
    K_gene: np.ndarray,
    length_scale: float = 1.0,
    embed_rank: int = 16,
):
    """
    Compute individual kernel components between perturbations.

    Returns (k1, k2, krbf) as (na, nb) arrays:
      k1 = x^T K_gene x'
      k2 = embedding-based pairwise interaction kernel (motif / circuit effects)
      krbf = RBF( K_gene x , K_gene x' )
    """
    k1 = _k1_set_linear(Xa, Xb, K_gene)

    M = _pairwise_similarity_mask_from_gene_kernel_embedding(K_gene, rank=embed_rank)
    k2 = _k_pairwise_interaction(Xa, Xb, M)

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
    embed_rank: int = 16,
):
    """
    Final perturbation kernel as weighted sum of components:

    K = a1^2*k1 + a2^2*k2 + a3^2*krbf

    a1, a2, a3 control relative importance of:
    - linear effects (k1)
    - interaction effects (k2)
    - smooth nonlinear similarity (kr)
    """
    k1, k2, kr = kernel_components(
        Xa, Xb, K_gene, length_scale=length_scale, embed_rank=embed_rank
    )
    return (a1**2) * k1 + (a2**2) * k2 + (a3**2) * kr


def combined_kernel_diag(
    X: np.ndarray,
    K_gene: np.ndarray,
    a1: float = 1.0,
    a2: float = 0.5,
    a3: float = 0.2,
    embed_rank: int = 16,
):
    """
    Compute diag(K(X,X)) without building full matrix.

    Used for GP predictive variance

    For components:
      diag(k1) = x^T K_gene x
      diag(k2) = ||f(x)||^2   where f are embedding-weighted pair features
      diag(krbf) = 1
    """
    X = np.asarray(X, float)

    XK = X @ K_gene
    d1 = np.sum(XK * X, axis=1)

    M = _pairwise_similarity_mask_from_gene_kernel_embedding(K_gene, rank=embed_rank)
    n = X.shape[1]
    iu = np.triu_indices(n, k=1)
    W = X[:, :, None] * X[:, None, :]
    F = (W * M)[..., iu[0], iu[1]]
    d2 = np.sum(F * F, axis=1)

    dr = np.ones_like(d1)
    return (a1**2) * d1 + (a2**2) * d2 + (a3**2) * dr
