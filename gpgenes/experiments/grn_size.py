import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.experiments.preset_solvers import preset_solver_linear, preset_solver_gp


def test_gnr_size():
    n_genes_list = [10, 20, 30, 40, 50]
    n_repetitions = 25

    results_lr = []
    results_gp = []
    errors_lr = []
    errors_gp = []

    for n_genes in n_genes_list:
        print(f"Starting experiment with {n_genes} genes.")

        n_sparse = int(n_genes * 0.33)
        n_motif = n_genes - n_sparse

        rmses_lr = []
        rmses_gp = []

        for seed in range(n_repetitions):
            genes = data.create_genes(
                n_genes=n_genes,
                n_sparse=n_sparse,
                n_motif=n_motif,
                seed=seed,
            )

            perturbations = data.make_perturbation_list(
                n_genes=n_genes,
                include_singles=True,
                include_doubles=True,
                n_doubles=None,
                seed=seed,
            )

            rows = data.simulate_dataset(
                genes,
                perturbations=perturbations,
                n_reps=3,
                seed=seed,
            )
            df = pd.DataFrame(rows)

            df_train, df_test = data.split_by_perturbation(
                df,
                train_frac=0.2,
                seed=seed,
            )

            mu = data.compute_control_baseline(df_train, n_genes=n_genes)

            Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
            Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

            Rtr = data.residualize(Ytr, mu)
            Rte = data.residualize(Yte, mu)

            linear_rmses = preset_solver_linear(n_genes, Xtr, Xte, Rtr, Rte)
            rmses_lr.append(np.mean(linear_rmses))

            gp_rmses = preset_solver_gp(genes, n_genes, Xtr, Xte, Rtr, Rte)
            rmses_gp.append(np.mean(gp_rmses))

        results_lr.append(np.mean(rmses_lr))
        results_gp.append(np.mean(rmses_gp))

        errors_lr.append(np.std(rmses_lr))
        errors_gp.append(np.std(rmses_gp))

    results_lr = np.array(results_lr)
    results_gp = np.array(results_gp)
    errors_lr = np.array(errors_lr)
    errors_gp = np.array(errors_gp)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(n_genes_list, results_lr, label="Linear Regression")
    ax.fill_between(
        n_genes_list,
        results_lr - errors_lr,
        results_lr + errors_lr,
        alpha=0.2,
    )

    ax.plot(n_genes_list, results_gp, label="Full GP (mixed)")
    ax.fill_between(
        n_genes_list,
        results_gp - errors_gp,
        results_gp + errors_gp,
        alpha=0.2,
    )

    ax.set_xlabel("n Genes")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("GNR size vs RMSE")
    ax.set_xlim(n_genes_list[0], n_genes_list[-1])
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
    )
    fig.text(0.01, 0.99, "(b)", ha="left", va="top", fontsize=12, color="blue")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("figures/sub_b.png", dpi=300)


if __name__ == "__main__":
    test_gnr_size()
