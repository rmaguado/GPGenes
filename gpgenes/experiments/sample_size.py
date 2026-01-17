from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.models.train import *


def experiment_sample_size(
    genes,
    n_perturbs: List[int],
    solver_fnc,
    n_repetitions: int = 10,
    base_seed: int = 0,
):
    avg_rmses = []

    n_genes = len(genes)

    for n_p in tqdm(n_perturbs, position=0):
        rmses_reps = []

        for rep in tqdm(range(n_repetitions), position=1, leave=False):
            seed = base_seed + 1000 * rep

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

            solver = solver_fnc(genes, n_genes, Xtr, Rtr)

            rmses = solver(Xte, Rte)
            rmses_reps.append(np.mean(rmses))

        avg_rmses.append(np.mean(rmses_reps))

    return avg_rmses


if __name__ == "__main__":
    genes = data.create_genes(n_genes=10, n_sparse=0, n_motif=10, seed=0)
    data.plot_graph(genes)
    n_perturbs = [5, 10, 15, 20, 25]
    result_lr = experiment_sample_size(
        genes, n_perturbs=n_perturbs, solver_fnc=solver_linear
    )
    result_full = experiment_sample_size(
        genes, n_perturbs=n_perturbs, solver_fnc=solver_full
    )

    plt.plot(n_perturbs, result_lr, label="Linear Regression")
    plt.plot(n_perturbs, result_full, label="GP full")
    plt.xlabel("Sample Size")
    plt.ylabel("Mean RMSE")
    plt.title(f"Sample size vs performance")
    plt.grid(True)
    plt.legend()
    plt.show()
