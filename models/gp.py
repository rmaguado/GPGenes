from __future__ import annotations
import numpy as np


def _jittered_cholesky(K: np.ndarray, jitter: float = 1e-8, max_tries: int = 8) -> np.ndarray:
    """
    Robust Cholesky. Adds increasing jitter if needed.
    Returns L such that (K + jitter*I) = L L^T.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    I = np.eye(n, dtype=float)

    jitter_i = float(jitter)
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(K + jitter_i * I)
        except np.linalg.LinAlgError:
            jitter_i *= 10.0

    # Last attempt: force sym and try once more
    Ksym = (K + K.T) / 2.0
    return np.linalg.cholesky(Ksym + jitter_i * I)


def _solve_chol(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve (L L^T) X = B for X, where L is lower-triangular Cholesky.
    Uses np.linalg.solve twice.

    This computes X = (L L^T)^-1 B efficiently.
    """
    Y = np.linalg.solve(L, B)
    X = np.linalg.solve(L.T, Y)
    return X


class GaussianProcessRegressor:
    """
    Zero-mean GP regressor with fixed (pre-computed) kernel Gram matrices.

    Assumes:
    - kernel matrices are computed externally
    - one-dimensional outputs (one GP per gene) 

    You provide kernel functions that compute:
      K_train = K(X_train, X_train)
      K_cross = K(X_test, X_train)
      K_test_diag = diag(K(X_test, X_test))  (optional; else computed)

    Noise handled via: K_train + (noise_variance)*I
    """

    def __init__(self, noise_variance: float = 1e-4, jitter: float = 1e-8, normalize_y: bool = True):
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)
        self.normalize_y = bool(normalize_y)

        # stored after fitting
        self.X_train = None
        self.y_mean = 0.0
        self.y_std = 1.0

        self.L = None          # Cholesky factor of (K + noise*I)
        self.alpha = None      # (K + noise*I)^-1 y

    def fit_from_gram(self, K_train: np.ndarray, y_train: np.ndarray):
        """
        Fit GP using a precomputed training Gram matrix. 

        Inputs:
            K_train: (n_train, n_train)
            y_train: (n_train,)
        """
        y = np.asarray(y_train, dtype=float).reshape(-1)
        K = np.asarray(K_train, dtype=float)

        if K.shape[0] != K.shape[1]:
            raise ValueError("K_train must be square.")
        if K.shape[0] != y.shape[0]:
            raise ValueError("K_train and y_train size mismatch.")

        # optional normalisation of targets
        if self.normalize_y:
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y) + 1e-12)
            y_use = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            y_use = y

        # add observational noise
        K_noise = K + (self.noise_variance * np.eye(K.shape[0], dtype=float))
        K_noise = (K_noise + K_noise.T) / 2.0  # help numerical symmetry

        # Cholesky factorisation
        self.L = _jittered_cholesky(K_noise, jitter=self.jitter)

        # precompute alpha = (K + noise*I)^-1 y
        self.alpha = _solve_chol(self.L, y_use[:, None]).reshape(-1)

        return self

    def predict_from_gram(
        self,
        K_cross: np.ndarray,
        K_test_diag: np.ndarray | None = None,
        return_std: bool = False,
        return_var: bool = False,
        include_noise: bool = False,
    ):
        """
        Predict GP mean and optionally variance.

        If include_noise=True, returns predictive observation variance Var(y*).
        Otherwise returns latent function variance Var(f*).

        Predict using:
          mean = K_cross @ alpha
          var  = diag(K_test) - diag(v^T v), where v solves L v = K_cross^T

        K_cross shape: (n_test, n_train)
        """
        if self.L is None or self.alpha is None:
            raise RuntimeError("Call fit_from_gram first.")

        Kx = np.asarray(K_cross, dtype=float)

        # predictive mean (normalised space)
        mean_norm = Kx @ self.alpha

        # Compute predictive variance
        # v = solve(L, Kx^T) -> shape (n_train, n_test)
        v = np.linalg.solve(self.L, Kx.T)
        var_norm = None
        if return_std or return_var:
            if K_test_diag is None:
                raise ValueError("K_test_diag is required for variance/std outputs.")
            Kdd = np.asarray(K_test_diag, dtype=float).reshape(-1)
            # diag(v^T v) = sum(v^2, axis=0)
            var_norm = Kdd - np.sum(v * v, axis=0)
            var_norm = np.maximum(var_norm, 0.0)  # guard tiny negatives

            if include_noise:
                var_norm += self.noise_variance # add observation noise to predictive variance in normalised space

        # Unnormalize
        mean = mean_norm * self.y_std + self.y_mean
        if not (return_std or return_var):
            return mean

        var = var_norm * (self.y_std ** 2)
        if return_var:
            return mean, var
        std = np.sqrt(var + 1e-12)
        return mean, std
