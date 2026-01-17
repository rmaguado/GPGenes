import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression

from gpgenes import data
from gpgenes.models import kernels, GaussianProcessRegressor
from gpgenes.models.kernel_diagnostics import kernel_diagnostics, plot_eigen_spectrum


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
    def __init__(
        self, 
        genes, 
        n_genes, 
        betas, 
        length_scales, 
        a_vals, 
        noise_vals,
        gene_kernel_mode,
    ):
        G = data.genes_to_digraph(genes)
        self.A = kernels.graph_to_weighted_adjacency(G, n=n_genes, use_abs=False)
        self.betas = betas
        self.length_scales = length_scales
        self.a_vals = a_vals
        self.noise_vals = noise_vals
        self.gene_kernel_mode = gene_kernel_mode

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
        K_gene = kernels.build_gene_kernel(
            self.A,
            mode=self.gene_kernel_mode,
            beta=p["beta"],
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


class RBFKernelBuilder:
    def __init__(self, length_scales, noise_vals):
        self.length_scales = length_scales
        self.noise_vals = noise_vals

    def param_grid(self):
        for ls in self.length_scales:
            for noise in self.noise_vals:
                yield {"length_scale": ls, "noise": noise}

    def build_kernel(self, Xtr, p):
        return kernels._rbf_from_Z(Xtr, Xtr, p["length_scale"])


def gp_full(genes, n_genes, Xtr, Xte, Rtr, Rte, params, plots=False):
    G = data.genes_to_digraph(genes)
    A = kernels.graph_to_weighted_adjacency(G, n=n_genes, use_abs=False)

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

    K_gene = kernels.mixed_signed_directed_diffusion_kernel(
            A, 
            beta=beta, 
            teleport_prob=0.05, 
            jitter=1e-8, 
            w_abs=0.5, 
            w_pos=1.0, 
            w_neg=1.0
            )

    # --- kernel sanity checks ---
    diag_gene = kernel_diagnostics(K_gene, name="Gene diffusion kernel")

    print("\n[Gene Kernel Diagnostics]")
    for k, v in diag_gene.items():
        print(f"{k:>25}: {v}")

    if plots:
        plot_eigen_spectrum(K_gene, title="Gene diffusion kernel spectrum")

    # Precompute Gram matrices once (same X for all genes)
    Ktr = kernels.combined_kernel(
        Xtr, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_tr = kernels.combined_kernel(
        Xte, Xtr, K_gene, a1=a1, a2=a2, a3=a3, length_scale=length_scale
    )
    Kte_diag = kernels.combined_kernel_diag(Xte, K_gene, a1=a1, a2=a2, a3=a3)

    # --- kernel sanity checks ---
    diag = kernel_diagnostics(Ktr, name="GP Full - Train Kernel")

    print("\n[Kernel Diagnostics]")
    for k, v in diag.items():
        print(f"{k:>25}: {v}")

    if plots:
        plot_eigen_spectrum(Ktr, title="GP Full - Ktr Eigen Spectrum")

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


def gp_rbf(n_genes, Xtr, Xte, Rtr, Rte, length_scale, noise):
    Ktr = kernels._rbf_from_Z(Xtr, Xtr, length_scale)
    Kte_tr = kernels._rbf_from_Z(Xte, Xtr, length_scale)
    Kte_diag = np.ones(Xte.shape[0])  # RBF(x,x)=1

    rmses = []
    for g in range(n_genes):
        ytr = Rtr[:, g]
        yte = Rte[:, g]

        gp = GaussianProcessRegressor(
            noise_variance=noise,
            jitter=1e-8,
            normalize_y=True,
        )
        gp.fit_from_gram(Ktr, ytr)
        pred = gp.predict_from_gram(
            Kte_tr, K_test_diag=Kte_diag, include_noise=False
        ).mean
        rmses.append(rmse(yte, pred))

    return np.array(rmses)


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


def solver_linear(genes, n_genes, Xtr, Rtr):
    models = []
    for g in range(n_genes):
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xtr, Rtr[:, g])
        models.append(lr)

    def solver(Xte, Rte):
        rmses = []
        for g, lr in enumerate(models):
            pred = lr.predict(Xte)
            rmses.append(rmse(Rte[:, g], pred))
        return np.array(rmses)

    return solver


def solver_rbf(genes, n_genes, Xtr, Rtr):
    builder = RBFKernelBuilder(
        length_scales=[0.7, 1.0, 1.3],
        noise_vals=[5e-4, 1e-3, 2e-3],
    )
    best_params, _ = optimise_hyperparameters(builder, Xtr, Rtr, n_genes)

    def solver(Xte, Rte):
        return gp_rbf(
            n_genes,
            Xtr,
            Xte,
            Rtr,
            Rte,
            length_scale=best_params["length_scale"],
            noise=best_params["noise"],
        )

    return solver


def solver_identity(genes, n_genes, Xtr, Rtr):
    builder = IdentityKernelBuilder(
        n_genes=n_genes,
        length_scales=[0.7, 1.0, 1.3],
        a_vals=[0.25, 0.5, 1.0],
        noise_vals=[5e-4, 1e-3, 2e-3],
    )

    best_params, _ = optimise_hyperparameters(builder, Xtr, Rtr, n_genes)

    def solver(Xte, Rte):
        return gp_identity(n_genes, Xtr, Xte, Rtr, Rte, params=best_params)

    return solver


def solver_k1(genes, n_genes, Xtr, Rtr):
    builder = K1KernelBuilder(
        n_genes=n_genes,
        length_scales=[0.7, 1.0, 1.3],
        noise_vals=[5e-4, 1e-3, 2e-3],
    )

    best_params, _ = optimise_hyperparameters(builder, Xtr, Rtr, n_genes)

    def solver(Xte, Rte):
        return gp_k1(
            n_genes,
            Xtr,
            Xte,
            Rtr,
            Rte,
            length_scale=best_params["length_scale"],
            noise=best_params["noise"],
        )

    return solver


def solver_full(genes, n_genes, Xtr, Rtr):
    builder = FullGPKernelBuilder(
        genes=genes,
        n_genes=n_genes,
        betas=[0.3, 0.5, 0.7],
        length_scales=[0.7, 1.0, 1.3],
        a_vals=[0.0, 0.25, 0.5, 0.75, 1.0],
        noise_vals=[5e-4, 1e-3, 2e-3],
    )

    best_params, _ = optimise_hyperparameters(builder, Xtr, Rtr, n_genes)

    def solver(Xte, Rte):
        rmses, _, _ = gp_full(genes, n_genes, Xtr, Xte, Rtr, Rte, best_params)
        return rmses

    return solver
