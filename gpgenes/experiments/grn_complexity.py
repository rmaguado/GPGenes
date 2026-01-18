import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.experiments.preset_solvers import preset_solver_linear, preset_solver_gp


def run_complexity_experiment(
    n_genes, n_motif, n_sparse, n_repetitions=10, base_seed=0
):
    linear_results = []
    gp_results = []

    for rep in range(n_repetitions):
        seed = base_seed + 1000 * rep

        genes = data.create_genes(
            n_genes=n_genes, n_motif=n_motif, n_sparse=n_sparse, seed=1
        )

        perturbations = data.make_perturbation_list(
            n_genes=n_genes,
            include_singles=True,
            include_doubles=True,
            n_doubles=100,
            seed=0,
        )

        rows = data.simulate_dataset(
            genes,
            perturbations=perturbations,
            n_reps=3,
            seed=42,
        )
        df = pd.DataFrame(rows)

        df_train, df_test = data.split_by_perturbation(
            df,
            train_frac=0.8,
            train_single_pert=False,
            train_double_pert=True,
            test_single_pert=False,
            test_double_pert=True,
            seed=seed,
        )

        mu = data.compute_control_baseline(df_train, n_genes=n_genes)

        Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
        Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

        Rtr = data.residualize(Ytr, mu)
        Rte = data.residualize(Yte, mu)

        linear_rmses = preset_solver_linear(n_genes, Xtr, Xte, Rtr, Rte)
        gp_rmses = preset_solver_gp(genes, n_genes, Xtr, Xte, Rtr, Rte)

        linear_results.append(np.mean(linear_rmses))
        gp_results.append(np.mean(gp_rmses))

    linear_results = np.array(linear_results)
    gp_results = np.array(gp_results)

    def mean_ci(arr):
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        ci = 1.96 * se
        return mean, ci

    lr_mean, lr_ci = mean_ci(linear_results)
    gp_mean, gp_ci = mean_ci(gp_results)

    return (lr_mean, lr_ci), (gp_mean, gp_ci)


if __name__ == "__main__":

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

    simple_stats = run_complexity_experiment(**simple_params)
    complex_stats = run_complexity_experiment(**complex_params)

    methods = ["Linear", "Full GP (mixed)"]

    means = np.array(
        [
            [simple_stats[0][0], complex_stats[0][0]],
            [simple_stats[1][0], complex_stats[1][0]],
        ]
    )

    cis = np.array(
        [
            [simple_stats[0][1], complex_stats[0][1]],
            [simple_stats[1][1], complex_stats[1][1]],
        ]
    )

    x = np.arange(len(methods)) * 0.7
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 2.5))

    ax.bar(
        x - width / 2,
        means[:, 0],
        width,
        yerr=cis[:, 0],
        capsize=5,
        label="Simple",
        color="lightgrey",
        zorder=3,
    )

    ax.bar(
        x + width / 2,
        means[:, 1],
        width,
        yerr=cis[:, 1],
        capsize=5,
        label="Complex",
        color="lightblue",
        zorder=3,
    )

    ax.set_title("GRN Complexity vs RMSE")
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Mean RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
    )
    fig.text(0.01, 0.99, "(d)", ha="left", va="top", fontsize=12, color="blue")
    plt.tight_layout()
    plt.grid(axis="y", zorder=0)
    plt.savefig("figures/sub_d.png", dpi=300)
