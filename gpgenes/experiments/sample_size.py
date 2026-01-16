from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from .train_gp import *


def experiment_sample_size(
    n_perturbs: List[int],
    solver_fnc,
    n_repetitions: int = 10,
    base_seed: int = 0,
):
    avg_rmses = []

    genes = data.create_genes(n_genes=5, n_sparse=2, n_motif=3, seed=base_seed)
    n_genes = len(genes)

    for n_p in n_perturbs:
        rmses_reps = []

        for rep in range(n_repetitions):
            seed = base_seed + 1000 * rep + n_p

            perturbations = data.make_perturbation_list(
                n_genes=n_genes,
                include_singles=True,
                include_doubles=True,
                n_doubles=n_p,
                seed=seed,
            )

            rows = data.simulate_dataset(
                genes,
                perturbations=perturbations,
                n_reps=1,
                steps=100,
                delta=0.01,
                tail_steps=10,
                seed=seed,
            )
            df = pd.DataFrame(rows)

            df_train, df_test = data.split_by_perturbation(
                df, test_frac=0.25, seed=seed
            )

            mu = data.compute_control_baseline(df_train, n_genes=n_genes)

            Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
            Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

            Rtr = data.residualize(Ytr, mu)
            Rte = data.residualize(Yte, mu)

            solver = solver_fnc(genes, n_genes, Xtr, Rtr)

            rmses = solver(Xte, Rte)
            rmses_reps.append(np.mean(rmses))

        avg_rmses.append(np.mean(rmses_reps))

    return avg_rmses


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


if __name__ == "__main__":
    n_perturbs = [100, 200, 300, 400, 600, 800]
    result_lr = experiment_sample_size(n_perturbs=n_perturbs, solver_fnc=solver_linear)
    result_i = experiment_sample_size(n_perturbs=n_perturbs, solver_fnc=solver_identity)
    result_k1 = experiment_sample_size(n_perturbs=n_perturbs, solver_fnc=solver_k1)
    result_full = experiment_sample_size(n_perturbs=n_perturbs, solver_fnc=solver_full)

    plt.plot(n_perturbs, result_lr, label="Linear Regression")
    plt.plot(n_perturbs, result_i, label="GP Identity")
    plt.plot(n_perturbs, result_k1, label="GP k1")
    plt.plot(n_perturbs, result_full, label="GP full")
    plt.xlabel("Sample Size")
    plt.ylabel("Mean RMSE")
    plt.title(f"Sample size vs performance")
    plt.grid(True)
    plt.legend()
    plt.show()
