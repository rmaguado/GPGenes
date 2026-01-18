import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.experiments.preset_solvers import preset_solver_linear, preset_solver_gp


_experiments = [
    {"train": "S+D", "test": "S+D"},
    {"train": "S+D", "test": "S"},
    {"train": "S+D", "test": "D"},
    {"train": "S", "test": "S+D"},
    {"train": "S", "test": "S"},
    {"train": "S", "test": "D"},
    {"train": "D", "test": "S+D"},
    {"train": "D", "test": "S"},
    {"train": "D", "test": "D"},
]

_plot_exp_order = ["S", "D", "S+D"]


def test_split_groups():
    n_genes = 30
    n_motifs = 20
    n_sparse = 10

    genes = data.create_genes(
        n_genes=n_genes, n_motif=n_motifs, n_sparse=n_sparse, seed=1
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
        n_reps=5,
        seed=42,
    )
    df = pd.DataFrame(rows)

    results_lr = np.ndarray((3, 3), dtype=np.float32)
    results_gp = np.ndarray((3, 3), dtype=np.float32)

    for cfg in _experiments:
        results_i = _plot_exp_order.index(cfg["train"])
        results_j = _plot_exp_order.index(cfg["test"])

        print(f"Running experiment: {cfg}")
        df_train, df_test = data.split_by_perturbation(
            df,
            train_frac=0.8,
            train_single_pert="S" in cfg["train"],
            train_double_pert="D" in cfg["train"],
            test_single_pert="S" in cfg["test"],
            test_double_pert="D" in cfg["test"],
            seed=0,
        )

        mu = data.compute_control_baseline(df_train, n_genes=n_genes)

        Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
        Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

        Rtr = data.residualize(Ytr, mu)
        Rte = data.residualize(Yte, mu)

        linear_rmses = preset_solver_linear(n_genes, Xtr, Xte, Rtr, Rte)
        results_lr[results_i][results_j] = np.mean(linear_rmses)

        gp_rmses = preset_solver_gp(genes, n_genes, Xtr, Xte, Rtr, Rte)
        results_gp[results_i][results_j] = np.mean(gp_rmses)

    results_colate = [
        (results_lr, "Linear"),
        (results_gp, "Full GP (mixed)"),
    ]

    vmin = 0
    vmax = max(results_lr.max(), results_gp.max())
    thresh = (vmax - vmin) / 2 + vmin

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), constrained_layout=True)
    # fig.subplots_adjust(right=0.85)

    images = []
    for idx, (results, title) in enumerate(results_colate):
        ax = axes[idx]
        im = ax.imshow(results, vmin=vmin, vmax=vmax, cmap="gray")
        images.append(im)

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(_plot_exp_order)
        ax.set_yticklabels(_plot_exp_order)

        ax.set_xlabel("Test")
        ax.set_ylabel("Train")
        ax.set_title(title)

        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                value = results[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < thresh else "black",
                )

    fig.colorbar(images[0], ax=axes, label="Mean RMSE")
    fig.text(0.01, 0.99, "(c)", ha="left", va="top", fontsize=12, color="blue")
    plt.savefig("figures/sub_c.png", dpi=300)


if __name__ == "__main__":
    test_split_groups()
