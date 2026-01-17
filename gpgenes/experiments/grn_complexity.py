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

    mean_rmse = np.mean(rmses)

    return mean_rmse


if __name__ == "__main__":
    simple_params = {
        "n_genes": 10,
        "n_motif": 0,
        "n_sparse": 8,
    }
    complex_params = {
        "n_genes": 10,
        "n_motif": 8,
        "n_sparse": 2,
    }

    result_lr = [
        test_grn_complexity(solver_fnc=solver_linear, **simple_params),
        test_grn_complexity(solver_fnc=solver_linear, **complex_params),
    ]
    result_full = [
        test_grn_complexity(solver_fnc=solver_full, **simple_params),
        test_grn_complexity(solver_fnc=solver_full, **complex_params),
    ]
    results_all = np.array([result_lr, result_full])

    vmin = results_all.min()
    vmax = results_all.max()
    thresh = (vmax - vmin) / 2 + vmin

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(results_all, vmin=vmin, vmax=vmax)

    ax.set_title("GRN Complexity vs Model Performance")

    ax.set_xlabel("GRN Complexity")
    ax.set_ylabel("Model Type")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Simple", "Complex"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Linear", "Full GP"])

    for i in range(results_all.shape[0]):
        for j in range(results_all.shape[1]):
            value = results_all[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < thresh else "black",
            )

    fig.colorbar(im, ax=ax, label="Mean RMSE")
    plt.show()
