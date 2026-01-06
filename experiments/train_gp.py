import numpy as np
import pandas as pd

# simulation utilities
from data.simulate import (
    create_genes, 
    genes_to_digraph, 
    make_perturbation_list, 
    simulate_dataset
)

# kernel construction
from models.kernels import (
    graph_to_weighted_adjacency,
    symmetrize,
    diffusion_node_kernel,
    combined_kernel,
    combined_kernel_diag,
)

# dataset utilities
from data.dataset import (
    build_xy_from_df,
    compute_control_baseline,
    residualize,
    split_by_perturbation,
)

# GP model
from models.gp import GaussianProcessRegressor


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    # ---------------------------------------------------------
    # 1) Simulate gene regulatory network and perturbation data
    # ---------------------------------------------------------
    genes = create_genes(n_genes=30, tf_fraction=0.3, n_modules=3, seed=1)
    n_genes = len(genes)

    perturbations = make_perturbation_list(
        n_genes=n_genes,
        include_singles=True,
        include_doubles=True,
        n_doubles=80,
        seed=0,
    )

    rows = simulate_dataset(
        genes,
        perturbations=perturbations,
        n_reps=5,
        steps=1500,
        delta=0.01,
        tail_steps=200,
        seed=42,
    )
    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # 2) Train/test split by perturbation (keeps replicates together)
    # ---------------------------------------------------------
    df_train, df_test = split_by_perturbation(df, test_frac=0.25, seed=0)

    # ---------------------------------------------------------
    # 3) Compute baseline from train control samples only
    # ---------------------------------------------------------
    mu = compute_control_baseline(df_train, n_genes=n_genes)

    # ---------------------------------------------------------
    # 4) Build X/Y (GP inputs) and residual targets
    # ---------------------------------------------------------
    Xtr, Ytr, _ = build_xy_from_df(df_train, n_genes=n_genes)
    Xte, Yte, _ = build_xy_from_df(df_test, n_genes=n_genes)

    Rtr = residualize(Ytr, mu)
    Rte = residualize(Yte, mu)

    # ---------------------------------------------------------
    # 5) Build GRN-derived gene-level diffusion kernel K_gene
    # ---------------------------------------------------------
    G = genes_to_digraph(genes)
    A = graph_to_weighted_adjacency(G, n=n_genes, use_abs=True)
    A_sym = symmetrize(A)
    K_gene = diffusion_node_kernel(A_sym, beta=1.0, jitter=1e-8)

    # ---------------------------------------------------------
    # 6) Kernel hyperparameters (fixed for minimal version)
    # ---------------------------------------------------------
    a1, a2, a3 = 1.0, 0.5, 0.2
    length_scale = 1.0

    # Precompute Gram matrices once (same X for all genes)
    Ktr = combined_kernel(Xtr, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale)
    Kte_tr = combined_kernel(Xte, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale)
    Kte_diag = combined_kernel_diag(Xte, K_gene, a1=a1, a2=a2, a3=a3)

    # ---------------------------------------------------------
    # 7) Fit per-gene GP on residuals
    # ---------------------------------------------------------
    rmses = []
    for g in range(n_genes):
        ytr = Rtr[:, g]
        yte = Rte[:, g]

        gp = GaussianProcessRegressor(
            noise_variance=1e-4,   # increase if you add more observation noise/replicate variation
            jitter=1e-8,
            normalize_y=True,
        )

        gp.fit_from_gram(Ktr, ytr)
        pred = gp.predict_from_gram(Kte_tr, K_test_diag=Kte_diag, return_std=False, include_noise=False)

        rmses.append(rmse(yte, pred))

    # ---------------------------------------------------------
    # 8) Report performance and diagnostics
    # ---------------------------------------------------------
    print(f"Mean RMSE across genes: {np.mean(rmses):.4f}")
    print(f"Median RMSE across genes: {np.median(rmses):.4f}")
    print("Example gene 0 RMSE:", rmses[0])

    print("Total samples:", len(df))
    print("Unique perturbations:", df["perturbation"].nunique())
    print("Train/Test samples:", len(df_train), len(df_test))
    print("Train/Test perturbations:", df_train["perturbation"].nunique(), df_test["perturbation"].nunique())
    print("Controls in train:", (df_train["perturbation"] == "co").sum())
    print("K_gene shape:", K_gene.shape)
    print("Ktr shape:", Ktr.shape)

if __name__ == "__main__":
    main()
