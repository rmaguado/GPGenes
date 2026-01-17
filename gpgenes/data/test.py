from . import create_genes, plot_graph

if __name__ == "__main__":

    genes = create_genes(
        n_genes=10,
        n_sparse=2,
        n_motif=8,
        seed=1,
    )

    plot_graph(genes)
