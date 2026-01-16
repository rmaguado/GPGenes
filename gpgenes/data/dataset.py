from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


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
    X = encode_multihot(
        pert_sets, n_genes
    )  # multi-hot vector of length n_genes for each perturbation
    Y = df[[f"g{i:02d}" for i in range(n_genes)]].to_numpy(
        dtype=float
    )  # raw expression levels per gene
    return X, Y, pert_sets


def compute_control_baseline(df: pd.DataFrame, n_genes: int) -> np.ndarray:
    ctrl = df[df["perturbation"] == "co"]
    if len(ctrl) == 0:
        raise ValueError("No control rows found (perturbation == 'co').")
    mu = ctrl[[f"g{i:02d}" for i in range(n_genes)]].mean(axis=0).to_numpy(dtype=float)
    return mu


def residualize(Y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (
        Y - mu[None, :]
    )  # GP learns effects of perturbations, not baseline expression


def split_by_perturbation(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    train_single_pert: bool = True,
    train_double_pert: bool = True,
    test_single_pert: bool = False,
    test_double_pert: bool = True,
    seed: int = 0,
):
    """
    Split so that the same perturbation label doesn't appear in both train and test.
    Replicates stay together.
    """
    rng = np.random.default_rng(seed)
    perts = df["perturbation"].unique().tolist()
    rng.shuffle(perts)

    controls = [p for p in perts if p == "co"]  # keep controls in train by default
    perts = [p for p in perts if p not in controls]
    single_perts = [p for p in perts if "+" not in p]
    double_perts = [p for p in perts if "+" in p]

    train_perts = []
    train_perts += controls

    share_pool = []
    if train_single_pert and not test_single_pert:
        train_perts += single_perts
    if train_single_pert and test_single_pert:
        share_pool += single_perts
    if train_double_pert and not test_double_pert:
        train_perts += double_perts
    if train_double_pert and test_double_pert:
        share_pool += double_perts

    rng.shuffle(share_pool)

    included_genes = set()
    for pert in train_perts:
        genes = pert.split("+")
        included_genes.update(genes)

    train_quota = int(train_frac * len(perts))
    for pert in share_pool:
        if len(train_perts) >= train_quota:
            break

        if any(g not in included_genes for g in pert.split("+")):
            train_perts.append(pert)
            share_pool.remove(pert)

    for pert in share_pool:
        if len(train_perts) >= train_quota:
            break
        train_perts.append(pert)

    train_mask = df["perturbation"].isin(train_perts)
    test_mask = ~df["perturbation"].isin(train_perts)

    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
