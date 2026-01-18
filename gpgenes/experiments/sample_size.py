import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.experiments.preset_solvers import preset_solver_linear, preset_solver_gp


def test_sample_size():
    base_seed = 0
    n_repetitions = 15

    n_fracs = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
    genes = data.create_genes(n_genes=30, n_sparse=10, n_motif=20, seed=0)

    n_genes = len(genes)

    data.plot_graph(genes)

    perturbations = data.make_perturbation_list(
        n_genes=n_genes,
        include_singles=True,
        include_doubles=True,
        seed=0,
    )
    rows = data.simulate_dataset(
        genes,
        perturbations=perturbations,
        n_reps=3,
        seed=0,
    )
    df = pd.DataFrame(rows)

    result_lr_mean = []
    result_lr_low = []
    result_lr_high = []

    result_gp_mean = []
    result_gp_low = []
    result_gp_high = []

    for train_frac in tqdm(n_fracs, position=0):
        reps_rmses_linear = []
        reps_rmses_gp = []

        for rep in tqdm(range(n_repetitions), position=1, leave=False):
            seed = base_seed + 1000 * rep

            df_train, df_test = data.split_by_perturbation(
                df,
                train_frac=train_frac,
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
            reps_rmses_linear.append(np.mean(linear_rmses))

            gp_rmses = preset_solver_gp(genes, n_genes, Xtr, Xte, Rtr, Rte)
            reps_rmses_gp.append(np.mean(gp_rmses))

        lr = np.array(reps_rmses_linear)
        gp = np.array(reps_rmses_gp)

        lr_mean = lr.mean()
        lr_se = lr.std(ddof=1) / np.sqrt(len(lr))
        lr_ci = 1.96 * lr_se

        gp_mean = gp.mean()
        gp_se = gp.std(ddof=1) / np.sqrt(len(gp))
        gp_ci = 1.96 * gp_se

        result_lr_mean.append(lr_mean)
        result_lr_low.append(lr_mean - lr_ci)
        result_lr_high.append(lr_mean + lr_ci)

        result_gp_mean.append(gp_mean)
        result_gp_low.append(gp_mean - gp_ci)
        result_gp_high.append(gp_mean + gp_ci)

    fig, ax = plt.subplots(figsize=(6, 2.5))

    ax.plot(n_fracs, result_lr_mean, label="Linear Regression")
    ax.fill_between(n_fracs, result_lr_low, result_lr_high, alpha=0.2)

    ax.plot(n_fracs, result_gp_mean, label="Full GP (mixed)")
    ax.fill_between(n_fracs, result_gp_low, result_gp_high, alpha=0.2)

    ax.set_title(f"Sample size vs RMSE")
    ax.set_xlabel("train fraction")
    ax.set_ylabel("Mean RMSE")
    ax.set_xlim(n_fracs[0], n_fracs[-1])
    fig.text(0.01, 0.99, "(a)", ha="left", va="top", fontsize=12, color="blue")
    plt.tight_layout()

    plt.grid(True)
    plt.savefig("figures/sub_a.png", dpi=300)


if __name__ == "__main__":

    test_sample_size()
