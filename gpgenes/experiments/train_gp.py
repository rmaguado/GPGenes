import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from gpgenes import data
from gpgenes.models import kernels, GaussianProcessRegressor


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def optimise_hyperparameters(kernel_builder, Xtr, Rtr, n_genes):
    """
    Grid-search hyperparameters by maximising summed log marginal likelihood across genes (on training data only)
    """
    best_lml = -np.inf
    best_params = {}

    for params in kernel_builder.param_grid():
        try:
            Ktr = kernel_builder.build_kernel(Xtr, params)
        except np.linalg.LinAlgError:
            continue

        lml_total = 0.0
        valid = True

        for g in range(n_genes):
            ytr = Rtr[:, g]

            gp = GaussianProcessRegressor(
                noise_variance=params["noise"],
                jitter=1e-8,
                normalize_y=True,
            )

            try:
                gp.fit_from_gram(Ktr, ytr)
                lml_total += gp.log_marginal_likelihood(ytr)
            except np.linalg.LinAlgError:
                valid = False
                break

        if valid and lml_total > best_lml:
            best_lml = lml_total
            best_params = params

    return best_params, best_lml


class FullGPKernelBuilder:
    def __init__(self, genes, n_genes, betas, length_scales, a_vals, noise_vals):
        G = data.genes_to_digraph(genes)
        self.A = kernels.graph_to_weighted_adjacency(G, n=n_genes, use_abs=True)
        # self.A_sym = kernels.symmetrize(A)
        self.betas = betas
        self.length_scales = length_scales
        self.a_vals = a_vals
        self.noise_vals = noise_vals

    def param_grid(self):
        items = product(
            self.betas,
            self.length_scales,
            self.a_vals,
            self.a_vals,
            self.a_vals,
            self.noise_vals,
        )

        for beta, lp, a1, a2, a3, noise in items:
            if a1 + a2 + a3 < 1e-8:
                continue

            yield {
                "beta": beta,
                "length_scale": lp,
                "a1": a1,
                "a2": a2,
                "a3": a3,
                "noise": noise,
            }

    def build_kernel(self, Xtr, p):
        K_gene = kernels.directed_diffusion_kernel(
            self.A, beta=p["beta"], teleport_prob=0.05, jitter=1e-8
        )

        return kernels.combined_kernel(
            Xtr,
            Xtr,
            K_gene,
            a1=p["a1"],
            a2=p["a2"],
            a3=p["a3"],
            length_scale=p["length_scale"],
        )


class K1KernelBuilder:
    def __init__(self, n_genes, length_scales, noise_vals):
        self.I_gene = np.eye(n_genes)
        self.length_scales = length_scales
        self.noise_vals = noise_vals

    def param_grid(self):
        for length_scale in self.length_scales:
            for noise in self.noise_vals:
                yield {
                    "length_scale": length_scale,
                    "noise": noise,
                }

    def build_kernel(self, Xtr, p):
        return kernels.combined_kernel(
            Xtr,
            Xtr,
            self.I_gene,
            a1=1.0,
            a2=0.0,
            a3=0.0,
            length_scale=p["length_scale"],
        )


class IdentityKernelBuilder:
    def __init__(self, n_genes, length_scales, a_vals, noise_vals):
        self.I_gene = np.eye(n_genes)
        self.length_scales = length_scales
        self.a_vals = a_vals
        self.noise_vals = noise_vals

    def param_grid(self):
        for length_scale in self.length_scales:
            for a1 in self.a_vals:
                for a2 in self.a_vals:
                    for a3 in self.a_vals:
                        if a1 + a2 + a3 < 1e-8:
                            continue
                        for noise in self.noise_vals:
                            yield {
                                "length_scale": length_scale,
                                "a1": a1,
                                "a2": a2,
                                "a3": a3,
                                "noise": noise,
                            }

    def build_kernel(self, Xtr, p):
        return kernels.combined_kernel(
            Xtr,
            Xtr,
            self.I_gene,
            a1=p["a1"],
            a2=p["a2"],
            a3=p["a3"],
            length_scale=p["length_scale"],
        )


def gp_full(genes, n_genes, Xtr, Xte, Rtr, Rte, params):
    G = data.genes_to_digraph(genes)
    A = kernels.graph_to_weighted_adjacency(G, n=n_genes, use_abs=True)
    A_sym = kernels.symmetrize(A)

    # 6) Kernel hyperparameters
    beta = params["beta"]
    length_scale = params["length_scale"]
    a1, a2, a3 = params["a1"], params["a2"], params["a3"]
    noise = params["noise"]

    print("best hyperparameters (GP full):")
    print(f"   beta: {beta}")
    print(f"   length_scale: {length_scale}")
    print(f"   a1: {a1}, a2: {a2}, a3: {a3}")
    print(f"   noise: {noise}")

    K_gene = kernels.diffusion_node_kernel(A_sym, beta=beta, jitter=1e-8)

    # Precompute Gram matrices once (same X for all genes)
    Ktr = kernels.combined_kernel(
        Xtr, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_tr = kernels.combined_kernel(
        Xte, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_diag = kernels.combined_kernel_diag(Xte, K_gene, a1=a1, a2=a2, a3=a3)

    # TODO: add kernel sanity checks (e.g. PSD, symmetry, condition number, eigen spectrum. Good for report later.)

    # 7) Fit per-gene GP on residuals
    rmses = []
    for g in range(n_genes):
        ytr = Rtr[:, g]
        yte = Rte[:, g]

        gp = GaussianProcessRegressor(
            noise_variance=noise,  # increase if you add more observation noise/replicate variation
            jitter=1e-8,
            normalize_y=True,
        )

        gp.fit_from_gram(Ktr, ytr)
        pred = gp.predict_from_gram(
            Kte_tr, K_test_diag=Kte_diag, include_noise=False
        ).mean

        rmses.append(rmse(yte, pred))

    return np.array(rmses), K_gene, Ktr


def gp_k1(n_genes, Xtr, Xte, Rtr, Rte, length_scale, noise):
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
            noise_variance=noise,
            jitter=1e-8,
            normalize_y=True,
        )

        gp_k1.fit_from_gram(Ktr_k1, ytr_k1)
        pred_k1 = gp_k1.predict_from_gram(
            Kte_tr_k1, K_test_diag=Kte_diag_k1, include_noise=False
        ).mean

        k1_rmses.append(rmse(yte_k1, pred_k1))
    return np.array(k1_rmses)


def gp_identity(n_genes, Xtr, Xte, Rtr, Rte, params):
    I_gene = np.eye(n_genes, dtype=float)

    a1, a2, a3 = params["a1"], params["a2"], params["a3"]
    length_scale = params["length_scale"]
    noise = params["noise"]

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
            noise_variance=noise,
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

    # 1) Simulate gene regulatory network and perturbation data
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

    # 2) Train/test split by perturbation (keeps replicates together)
    df_train, df_test = data.split_by_perturbation(df, test_frac=0.25, seed=0)

    # 3) Compute baseline from train control samples only
    mu = data.compute_control_baseline(df_train, n_genes=n_genes)

    # 4) Build X/Y (GP inputs) and residual targets
    Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
    Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

    Rtr = data.residualize(Ytr, mu)
    Rte = data.residualize(Yte, mu)

    # 5) Baseline: linear regression on X (no kernel, no GRN)
    linear_rmses = linear_regression(n_genes, Xtr, Xte, Rtr, Rte)
    results["linear"] = linear_rmses

    print(f"[Linear] Mean RMSE across genes: {np.mean(linear_rmses):.4f}")
    print(f"[Linear] Median RMSE across genes: {np.median(linear_rmses):.4f}")

    # 6) Hyperparameter grids
    length_scales = [0.7, 1.0, 1.3]
    noise_vals = [5e-4, 1e-3, 2e-3]

    # 7) GP with identity K_gene (optimised)
    print("\nOptimising GP identity kernel...")

    id_builder = IdentityKernelBuilder(
        n_genes=n_genes,
        length_scales=length_scales,
        a_vals=[0.25, 0.5, 1.0],
        noise_vals=noise_vals,
    )

    best_id, lml_id = optimise_hyperparameters(
        id_builder,
        Xtr,
        Rtr,
        n_genes,
    )

    print("Best identity params:", best_id)
    print(f"Best LML: {lml_id:.2f}")

    id_rmses = gp_identity(n_genes, Xtr, Xte, Rtr, Rte, params=best_id)
    results["gp_identity"] = id_rmses

    print(f"[Identity Kernel] Mean RMSE across genes: {np.mean(id_rmses):.4f}")
    print(f"[Identity Kernel] Median RMSE across genes: {np.median(id_rmses):.4f}")

    # 8) GP with k1 only
    print("\nOptimising GP k1-only...")

    k1_builder = K1KernelBuilder(
        n_genes=n_genes,
        length_scales=length_scales,
        noise_vals=noise_vals,
    )

    best_k1, lml_k1 = optimise_hyperparameters(k1_builder, Xtr, Rtr, n_genes)

    print("Best k1 params:", best_k1)
    print(f"Best LML: {lml_k1:.2f}")

    k1_rmses = gp_k1(
        n_genes,
        Xtr,
        Xte,
        Rtr,
        Rte,
        length_scale=best_k1["length_scale"],
        noise=best_k1["noise"],
    )

    results["gp_k1"] = k1_rmses

    print(f"[K1 Kernel] Mean RMSE across genes: {np.mean(k1_rmses):.4f}")
    print(f"[K1 Kernel] Median RMSE across genes: {np.median(k1_rmses):.4f}")

    # 9) GP full model (optimised)
    print("\nOptimising GP full model...")

    full_builder = FullGPKernelBuilder(
        genes=genes,
        n_genes=n_genes,
        betas=[0.3, 0.5, 0.7],
        length_scales=length_scales,
        a_vals=[0.0, 0.25, 0.5, 0.75, 1.0],
        noise_vals=noise_vals,
    )

    best_full, lml_full = optimise_hyperparameters(full_builder, Xtr, Rtr, n_genes)

    print("Best full params:", best_full)
    print(f"Best LML: {lml_full:.2f}")

    rmses_full, K_gene, Ktr = gp_full(genes, n_genes, Xtr, Xte, Rtr, Rte, best_full)

    results["gp_full"] = rmses_full

    print(f"[GP full] Mean RMSE across genes: {np.mean(rmses_full):.4f}")
    print(f"[GP full] Median RMSE across genes: {np.median(rmses_full):.4f}")

    # 8) Report performance and diagnostics
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

    # 9) plots (optional)
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
    for i, key in enumerate(results):
        y = results[key]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.5)

    plt.ylabel("RMSE")
    plt.title("Model comparison (each optimised independently)")
    plt.show()


if __name__ == "__main__":
    main()
