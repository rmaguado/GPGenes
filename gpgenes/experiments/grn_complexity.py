import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.models.train import *


def test_grn_complexity(solver_fnc, n_genes, n_motif, n_sparse):
    genes = data.create_genes(
        n_genes=n_genes, n_motif=n_motif, n_sparse=n_sparse, seed=1
    )

    perturbations = data.make_perturbation_list(
        n_genes=n_genes,
        include_singles=True,
        include_doubles=True,
        n_doubles=40,
        seed=0,
    )

    rows = data.simulate_dataset(
        genes,
        perturbations=perturbations,
        n_reps=5,
        seed=42,
    )
    df = pd.DataFrame(rows)

    df_train, df_test = data.split_by_perturbation(
        df,
        train_frac=0.8,
        train_single_pert=True,
        train_double_pert=True,
        test_single_pert=False,
        test_double_pert=True,
        seed=0,
    )

    mu = data.compute_control_baseline(df_train, n_genes=n_genes)

    Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
    Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

    Rtr = data.residualize(Ytr, mu)
    Rte = data.residualize(Yte, mu)

    solver = solver_fnc(genes, n_genes, Xtr, Rtr)
    rmses = solver(Xte, Rte)

    return np.mean(rmses), np.std(rmses)


if __name__ == "__main__":
    methods = ["Linear", "Identity", "K1", "RBF", "Full GP"]
    conditions = ["Simple", "Complex"]

    n_methods = len(methods)

    simple_params = {
        "n_genes": 30,
        "n_motif": 0,
        "n_sparse": 30,
    }
    complex_params = {
        "n_genes": 30,
        "n_motif": 30,
        "n_sparse": 0,
    }

    result_lr = [
        test_grn_complexity(solver_fnc=solver_linear, **simple_params),
        test_grn_complexity(solver_fnc=solver_linear, **complex_params),
    ]
    results_id = [
        test_grn_complexity(solver_fnc=solver_identity, **simple_params),
        test_grn_complexity(solver_fnc=solver_identity, **complex_params),
    ]
    results_k1 = [
        test_grn_complexity(solver_fnc=solver_k1, **simple_params),
        test_grn_complexity(solver_fnc=solver_k1, **complex_params),
    ]
    results_rbf = [
        test_grn_complexity(solver_fnc=solver_rbf, **simple_params),
        test_grn_complexity(solver_fnc=solver_rbf, **complex_params),
    ]
    result_full = [
        test_grn_complexity(solver_fnc=solver_full, **simple_params),
        test_grn_complexity(solver_fnc=solver_full, **complex_params),
    ]
    all_results = np.array(
        [result_lr, results_id, results_k1, results_rbf, result_full]
    )

    means = np.array([[m for m, s in r] for r in all_results])
    stds = np.array([[s for m, s in r] for r in all_results])

    n = 5
    ci95 = 1.96 * stds / np.sqrt(n)

    x = np.arange(n_methods)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        means[:, 0],
        width,
        yerr=ci95[:, 0],
        capsize=5,
        color="lightgrey",
        label="Simple",
    )

    ax.bar(
        x + width / 2,
        means[:, 1],
        width,
        yerr=ci95[:, 1],
        capsize=5,
        color="lightblue",
        label="Complex",
    )

    ax.set_title("GRN Complexity vs Model Performance")
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Mean RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.legend(frameon=False)

    plt.grid(axis="y", zorder=0)
    plt.show()
