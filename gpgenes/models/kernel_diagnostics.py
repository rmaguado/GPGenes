import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt


def kernel_diagnostics(
    K: np.ndarray,
    name: str = "",
    tol_psd: float = 1e-10,
    max_eigs: int | None = None,
) -> Dict[str, Any]:
    """
    Compute sanity checks for a kernel / Gram matrix.

    Returns a dict of diagnostic metrics (easy to log or store).
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]

    # --- Symmetry check ---
    sym_err = np.linalg.norm(K - K.T, ord="fro") / (
        np.linalg.norm(K, ord="fro") + 1e-12
    )

    # --- Eigenvalues ---
    eigvals = np.linalg.eigvalsh(K)

    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))

    n_neg = int(np.sum(eigvals < -tol_psd))

    # --- Condition number ---
    if min_eig > 0:
        cond = max_eig / min_eig
    else:
        cond = np.inf

    # --- Effective rank ---
    eigvals_clip = np.clip(eigvals, 0.0, None)
    trace = np.sum(eigvals_clip)
    if trace > 0:
        p = eigvals_clip / trace
        entropy = -np.sum(p * np.log(p + 1e-12))
        erank = float(np.exp(entropy))
    else:
        erank = 0.0

    return {
        "name": name,
        "size": n,
        "symmetry_error": sym_err,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "n_negative_eigenvalues": n_neg,
        "condition_number": cond,
        "effective_rank": erank,
    }


def plot_eigen_spectrum(K: np.ndarray, title: str = "", logy: bool = True):
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(eigvals)[::-1]  # descending order

    plt.figure()
    plt.plot(eigvals, marker=".")
    if logy:
        plt.yscale("log")
    plt.xlabel("Eigenvalue index (Sorted)")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    plt.grid(True)
    plt.show()
