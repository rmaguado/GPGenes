from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

# prepare simulation output for GP training and testing

def parse_perturbation(s: str) -> Tuple[int, ...]:
    if s == "co":
        return tuple()
    parts = s.split("+")
    return tuple(sorted(int(p) for p in parts))


def encode_multihot(perturbations: List[Tuple[int, ...]], n_genes: int) -> np.ndarray:
    X = np.zeros((len(perturbations), n_genes), dtype=float)
    for i, pert in enumerate(perturbations):
        for gid in pert:
            X[i, gid] = 1.0
    return X


def build_xy_from_df(df: pd.DataFrame, n_genes: int):
    """
    Convert simulation output df perturbation labels and expression columns 
    into multi-hot inputs X and expression targets Y for GP regression.
    """
    pert_sets = [parse_perturbation(s) for s in df["perturbation"].tolist()]
    X = encode_multihot(pert_sets, n_genes) # multi-hot vector of length n_genes for each perturbation
    Y = df[[f"g{i:02d}" for i in range(n_genes)]].to_numpy(dtype=float) # raw expression levels per gene
    return X, Y, pert_sets


def compute_control_baseline(df: pd.DataFrame, n_genes: int) -> np.ndarray:
    ctrl = df[df["perturbation"] == "co"]
    if len(ctrl) == 0:
        raise ValueError("No control rows found (perturbation == 'co').")
    mu = ctrl[[f"g{i:02d}" for i in range(n_genes)]].mean(axis=0).to_numpy(dtype=float)
    return mu


def residualize(Y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return Y - mu[None, :] # GP learns effects of perturbations, not baseline expression


def split_by_perturbation(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """
    Split so that the same perturbation label doesn't appear in both train and test.
    Replicates stay together.
    """
    rng = np.random.default_rng(seed)
    perts = df["perturbation"].unique().tolist()
    perts = [p for p in perts if p != "co"]  # keep controls in train by default

    rng.shuffle(perts)
    n_test = max(1, int(len(perts) * test_frac))
    test_perts = set(perts[:n_test])

    train_mask = ~df["perturbation"].isin(test_perts)
    test_mask = df["perturbation"].isin(test_perts)

    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
