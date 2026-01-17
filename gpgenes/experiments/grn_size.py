import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.models.train import *

if __name__ == "__main__":
    n_genes_list = [5, 10, 20, 30]
    n_repetitions = 5

    results_lr = []
    results_full = []
    errors_lr = []
    errors_full = []

    for n_genes in n_genes_list:
        print(f"Starting experiment with {n_genes} genes.")
        n_sparse = int(n_genes * 0.33)
        n_motif = n_genes - n_sparse

        rmses_lr = []
        rmses_full = []

        for seed in range(n_repetitions):
            genes = data.create_genes(
                n_genes=n_genes, n_sparse=n_sparse, n_motif=n_motif, seed=seed
            )

            perturbations = data.make_perturbation_list(
                n_genes=n_genes,
                include_singles=True,
                include_doubles=True,
                n_doubles=int(n_genes**2 / 4),
                seed=seed,
            )

            rows = data.simulate_dataset(
                genes,
                perturbations=perturbations,
                n_reps=1,
                seed=seed,
            )
            df = pd.DataFrame(rows)

            df_train, df_test = data.split_by_perturbation(
                df, train_frac=0.8, seed=seed
            )

            mu = data.compute_control_baseline(df_train, n_genes=n_genes)

            Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
            Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

            Rtr = data.residualize(Ytr, mu)
            Rte = data.residualize(Yte, mu)

            solver = solver_linear(genes, n_genes, Xtr, Rtr)
            rmses_lr.append(np.mean(solver(Xte, Rte)))

            solver = solver_full(genes, n_genes, Xtr, Rtr)
            rmses_full.append(np.mean(solver(Xte, Rte)))

        results_lr.append(np.mean(rmses_lr))
        results_full.append(np.mean(rmses_full))
        errors_lr.append(np.std(rmses_lr))
        errors_full.append(np.std(rmses_full))

    plt.errorbar(
        n_genes_list,
        results_lr,
        yerr=errors_lr,
        label="Linear Regression",
        marker="o",
        capsize=5,
    )
    plt.errorbar(
        n_genes_list,
        results_full,
        yerr=errors_full,
        label="GP full",
        marker="o",
        capsize=5,
    )
    plt.xlabel("GNR Size (n_genes)")
    plt.ylabel("Mean RMSE")
    plt.title("GNR size vs performance with error bars")
    plt.grid(True)
    plt.legend()
    plt.show()
