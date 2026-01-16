import pandas as pd
import matplotlib.pyplot as plt

from gpgenes.models.train import *


def main():
    n_genes = 10
    n_motifs = 8
    n_sparse = 2

    results = {}

    # 1) Simulate gene regulatory network and perturbation data
    genes = data.create_genes(
        n_genes=n_genes, n_motif=n_motifs, n_sparse=n_sparse, seed=1
    )

    perturbations = data.make_perturbation_list(
        n_genes=n_genes,
        include_singles=True,
        include_doubles=True,
        n_doubles=80,
        seed=0,
    )

    rows = data.simulate_dataset(
        genes,
        perturbations=perturbations,
        n_reps=5,
        steps=1500,
        delta=0.01,
        tail_steps=200,
        seed=42,
    )
    df = pd.DataFrame(rows)

    # 2) Train/test split by perturbation (keeps replicates together)
    df_train, df_test = data.split_by_perturbation(df, train_frac=0.8, seed=0)

    # 3) Compute baseline from train control samples only
    mu = data.compute_control_baseline(df_train, n_genes=n_genes)

    # 4) Build X/Y (GP inputs) and residual targets
    Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
    Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

    Rtr = data.residualize(Ytr, mu)
    Rte = data.residualize(Yte, mu)

    # 5) Baseline: linear regression on X (no kernel, no GRN)
    solver = solver_linear(genes, n_genes, Xtr, Rtr)
    linear_rmses = solver(Xte, Rte)
    results["linear"] = linear_rmses

    print(f"[Linear] Mean RMSE across genes: {np.mean(linear_rmses):.4f}")
    print(f"[Linear] Median RMSE across genes: {np.median(linear_rmses):.4f}")

    # 6) GP with identity K_gene (optimised)
    print("\nOptimising GP identity kernel...")

    solver = solver_identity(genes, n_genes, Xtr, Rtr)
    id_rmses = solver(Xte, Rte)
    results["gp_identity"] = id_rmses

    print(f"[Identity Kernel] Mean RMSE across genes: {np.mean(id_rmses):.4f}")
    print(f"[Identity Kernel] Median RMSE across genes: {np.median(id_rmses):.4f}")

    # 7) GP with k1 only
    print("\nOptimising GP k1-only...")

    solver = solver_k1(genes, n_genes, Xtr, Rtr)
    k1_rmses = solver(Xte, Rte)
    results["gp_k1"] = k1_rmses

    print(f"[K1 Kernel] Mean RMSE across genes: {np.mean(k1_rmses):.4f}")
    print(f"[K1 Kernel] Median RMSE across genes: {np.median(k1_rmses):.4f}")

    # 8) GP full model (optimised)
    print("\nOptimising GP full model...")

    solver = solver_full(genes, n_genes, Xtr, Rtr)
    rmses_full = solver(Xte, Rte)
    results["gp_full"] = rmses_full

    print(f"[GP full] Mean RMSE across genes: {np.mean(rmses_full):.4f}")
    print(f"[GP full] Median RMSE across genes: {np.median(rmses_full):.4f}")

    # 9) Report performance and diagnostics
    print("Example gene 0 RMSE:", rmses_full[0])

    print("Total samples:", len(df))
    print("Unique perturbations:", df["perturbation"].nunique())
    print("Train/Test samples:", len(df_train), len(df_test))
    print(
        "Train/Test perturbations:",
        df_train["perturbation"].nunique(),
        df_test["perturbation"].nunique(),
    )
    print("Controls in train:", (df_train["perturbation"] == "co").sum())

    plt.boxplot(
        [
            results["linear"],
            results["gp_identity"],
            results["gp_k1"],
            results["gp_full"],
        ],
        tick_labels=[
            "Linear",
            "GP Identity",
            "GP k1",
            "GP Full",
        ],
    )
    for i, key in enumerate(results):
        y = results[key]
        x = np.random.normal(i + 1, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.5)

    plt.ylabel("RMSE")
    plt.title("Model comparison (each optimised independently)")
    plt.show()


if __name__ == "__main__":
    main()
