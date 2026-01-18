from gpgenes.models.kernels import GeneKernelMode
from gpgenes.models.train import *


def preset_solver_linear(n_genes, Xtr, Xte, Rtr, Rte):
    models = []
    for g in range(n_genes):
        lr = LinearRegression(fit_intercept=True)
        lr.fit(Xtr, Rtr[:, g])
        models.append(lr)

    rmses = []
    for g, lr in enumerate(models):
        pred = lr.predict(Xte)
        rmses.append(rmse(Rte[:, g], pred))
    return np.array(rmses)


def preset_solver_gp(genes, n_genes, Xtr, Xte, Rtr, Rte):

    rmses, _, _ = gp_full(
        genes,
        n_genes,
        Xtr,
        Xte,
        Rtr,
        Rte,
        {
            "beta": 0.7,
            "length_scale": 1.8,
            "a1": 1.0,
            "a2": 0.75,
            "a3": 0.25,
            "noise": 0.0005,
            "w_abs": 0.05,
            "w_pos": 0.1,
            "w_neg": 0.8,
        },
        gene_kernel_mode=GeneKernelMode.MIXED,
    )

    return rmses
