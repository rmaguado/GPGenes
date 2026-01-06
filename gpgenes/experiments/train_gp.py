import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from typing import List

from gpgenes import data
from gpgenes.models import kernels, GaussianProcessRegressor


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gp_full(genes, n_genes, Xtr, Xte, Rtr, Rte):
    G = data.genes_to_digraph(genes)
    A = kernels.graph_to_weighted_adjacency(G, n=n_genes, use_abs=True)
    A_sym = kernels.symmetrize(A)
    K_gene = kernels.diffusion_node_kernel(A_sym, beta=1.0, jitter=1e-8)

    # ---------------------------------------------------------
    # 6) Kernel hyperparameters
    # ---------------------------------------------------------
    a1, a2, a3 = 1.0, 0.5, 0.2
    # TODO: optimise a1, a2, a3 via grid search / marginal likelihood
    length_scale = 1.0

    # Precompute Gram matrices once (same X for all genes)
    Ktr = kernels.combined_kernel(
        Xtr, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_tr = kernels.combined_kernel(
        Xte, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_diag = kernels.combined_kernel_diag(Xte, K_gene, a1=a1, a2=a2, a3=a3)

    # TODO: add kernel sanity checks (e.g. PSD, symmetry, condition number, eigen spectrum. Good for report later.)

    # ---------------------------------------------------------
    # 7) Fit per-gene GP on residuals
    # ---------------------------------------------------------
    rmses = []
    for g in range(n_genes):
        ytr = Rtr[:, g]
        yte = Rte[:, g]

        gp = GaussianProcessRegressor(
            noise_variance=1e-4,  # increase if you add more observation noise/replicate variation
            jitter=1e-8,
            normalize_y=True,
        )

        gp.fit_from_gram(Ktr, ytr)
        pred = gp.predict_from_gram(
            Kte_tr, K_test_diag=Kte_diag, include_noise=False
        ).mean

        rmses.append(rmse(yte, pred))

    return np.array(rmses), K_gene, Ktr


def gp_k1(n_genes, Xtr, Xte, Rtr, Rte, length_scale=1.0):
    I_gene = np.eye(n_genes, dtype=float)

    Ktr_k1 = kernels.combined_kernel(
        Xtr, Xtr, I_gene, a1=1.0, a2=0.0, a3=0.0, length_scale=length_scale
    )
    Kte_tr_k1 = kernels.combined_kernel(
        Xte, Xtr, I_gene, a1=1.0, a2=0.0, a3=0.0, length_scale=length_scale
    )
    Kte_diag_k1 = kernels.combined_kernel_diag(Xte, I_gene, a1=1.0, a2=0.0, a3=0.0)

    k1_rmses = []
    for g in range(n_genes):
        ytr_k1 = Rtr[:, g]
        yte_k1 = Rte[:, g]

        gp_k1 = GaussianProcessRegressor(
            noise_variance=1e-4,
            jitter=1e-8,
            normalize_y=True,
        )

        gp_k1.fit_from_gram(Ktr_k1, ytr_k1)
        pred_k1 = gp_k1.predict_from_gram(
            Kte_tr_k1, K_test_diag=Kte_diag_k1, include_noise=False
        ).mean

        k1_rmses.append(rmse(yte_k1, pred_k1))
    return np.array(k1_rmses)


def gp_identity(n_genes, Xtr, Xte, Rtr, Rte, length_scale=1.0):
    I_gene = np.eye(n_genes, dtype=float)
    a1, a2, a3 = 1.0, 0.5, 0.2

    Ktr_id = kernels.combined_kernel(
        Xtr, Xtr, I_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_tr_id = kernels.combined_kernel(
        Xte, Xtr, I_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_diag_id = kernels.combined_kernel_diag(Xte, I_gene, a1=a1, a2=a2, a3=a3)

    id_rmses = []
    for g in range(n_genes):
        ytr_id = Rtr[:, g]
        yte_id = Rte[:, g]

        gp_id = GaussianProcessRegressor(
            noise_variance=1e-4,
            jitter=1e-8,
            normalize_y=True,
        )

        gp_id.fit_from_gram(Ktr_id, ytr_id)
        pred_id = gp_id.predict_from_gram(
            Kte_tr_id, K_test_diag=Kte_diag_id, include_noise=False
        ).mean

        id_rmses.append(rmse(yte_id, pred_id))
    return np.array(id_rmses)


def linear_regression(n_genes, Xtr, Xte, Rtr, Rte):
    linear_rmses = []

    for g in range(n_genes):
        ytr_lin = Rtr[:, g]
        yte_lin = Rte[:, g]

        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xtr, ytr_lin)

        pred_lin = lr.predict(Xte)
        linear_rmses.append(rmse(yte_lin, pred_lin))

    return np.array(linear_rmses)


def main():
    results = {}

    # ---------------------------------------------------------
    # 1) Simulate gene regulatory network and perturbation data
    # ---------------------------------------------------------
    genes = data.create_genes(n_genes=30, tf_fraction=0.3, n_modules=3, seed=1)
    n_genes = len(genes)

    perturbations = data.make_perturbation_list(
        n_genes=n_genes,
        include_singles=True,
        include_doubles=True,
        n_doubles=80,
        seed=0,
    )

    rows = data.simulate_dataset(
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
    df_train, df_test = data.split_by_perturbation(df, test_frac=0.25, seed=0)

    # ---------------------------------------------------------
    # 3) Compute baseline from train control samples only
    # ---------------------------------------------------------
    mu = data.compute_control_baseline(df_train, n_genes=n_genes)

    # ---------------------------------------------------------
    # 4) Build X/Y (GP inputs) and residual targets
    # ---------------------------------------------------------
    Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
    Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

    Rtr = data.residualize(Ytr, mu)
    Rte = data.residualize(Yte, mu)

    # ---------------------------------------------------------
    # 4.1) Baseline: linear regression on X (no kernel, no GRN)
    # ---------------------------------------------------------
    linear_rmses = linear_regression(n_genes, Xtr, Xte, Rtr, Rte)
    results["linear"] = linear_rmses

    print(f"[Linear] Mean RMSE across genes: {np.mean(linear_rmses):.4f}")
    print(f"[Linear] Median RMSE across genes: {np.median(linear_rmses):.4f}")

    # ---------------------------------------------------------
    # 4.2) Baseline: GP with identity K_gene
    # ---------------------------------------------------------
    id_rmses = gp_identity(n_genes, Xtr, Xte, Rtr, Rte)
    results["gp_identity"] = id_rmses

    print(f"[Identity Kernel] Mean RMSE across genes: {np.mean(id_rmses):.4f}")
    print(f"[Identity Kernel] Median RMSE across genes: {np.median(id_rmses):.4f}")

    # ---------------------------------------------------------
    # 4.3) Baseline: GP with k1 only
    # ---------------------------------------------------------
    k1_rmses = gp_k1(n_genes, Xtr, Xte, Rtr, Rte)

    results["gp_k1"] = k1_rmses

    print(f"[K1 Kernel] Mean RMSE across genes: {np.mean(k1_rmses):.4f}")
    print(f"[K1 Kernel] Median RMSE across genes: {np.median(k1_rmses):.4f}")

    # ---------------------------------------------------------
    # 5) Build GRN-derived gene-level diffusion kernel K_gene
    # ---------------------------------------------------------
    rmses_full, K_gene, Ktr = gp_full(genes, n_genes, Xtr, Xte, Rtr, Rte)

    results["gp_full"] = rmses_full

    # ---------------------------------------------------------
    # 8) Report performance and diagnostics
    # ---------------------------------------------------------
    print(f"Mean RMSE across genes: {np.mean(rmses_full):.4f}")
    print(f"Median RMSE across genes: {np.median(rmses_full):.4f}")
    print("Example gene 0 RMSE:", rmses_full[0])

    print("Total samples:", len(df))
    print("Unique perturbations:", df["perturbation"].nunique())
    print("Train/Test samples:", len(df_train), len(df_test))
    print(
        "Train/Test perturbations:",
        df_train["perturbation"].nunique(),
        df_test["perturbation"].nunique(),
    )
    print("Controls in train:", (df_train["perturbation"] == "co").sum())
    print("K_gene shape:", K_gene.shape)
    print("Ktr shape:", Ktr.shape)

    # ---------------------------------------------------------
    # 9) plots (optional)
    # ---------------------------------------------------------

    plt.boxplot(
        [
            results["linear"],
            results["gp_identity"],
            results["gp_k1"],
            results["gp_full"],
        ],
        tick_labels=[
            "Linear",
            "GP Identity",
            "GP k1",
            "GP Full",
        ],
    )
    plt.show()


def experiment_sample_size(n_perturbs: List[int], solver_fnc, n_repetitions=10):
    avg_rmses = []

    genes = data.create_genes(n_genes=30, tf_fraction=0.3, n_modules=3, seed=0)
    n_genes = len(genes)

    for n_p in n_perturbs:

        rmses_reps = []

        for rep in range(n_repetitions):
            perturbations_full = data.make_perturbation_list(
                n_genes=n_genes,
                include_singles=True,
                include_doubles=True,
                n_doubles=n_p,
                seed=0,
            )

            rows_full = data.simulate_dataset(
                genes,
                perturbations=perturbations_full,
                n_reps=5,
                steps=100,
                delta=0.01,
                tail_steps=100,
                seed=0,
            )
            df_full = pd.DataFrame(rows_full)

            df_train, df_test = data.split_by_perturbation(
                df_full, test_frac=0.25, seed=42 + rep
            )

            mu = data.compute_control_baseline(df_train, n_genes=n_genes)

            Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
            Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

            Rtr = data.residualize(Ytr, mu)
            Rte = data.residualize(Yte, mu)

            rmses = solver_fnc(n_genes, Xtr, Xte, Rtr, Rte)
            rmses_reps.append(np.mean(rmses))

        avg_rmses.append(np.mean(rmses_reps))

    plt.plot(n_perturbs, avg_rmses, marker="o")
    plt.xlabel("N perturbation experiments")
    plt.ylabel("RMSE")
    plt.title("Effect of increasing sample size on RMSE")
    plt.grid(True)
    plt.show()

    return avg_rmses


if __name__ == "__main__":
    # experiment_sample_size(n_perturbs=[200, 250, 300, 350, 400, 450], solver_fnc=linear_regression)
    main()
