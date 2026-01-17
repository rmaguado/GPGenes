from . import create_genes, plot_graph

if __name__ == "__main__":

    genes = create_genes(
        n_genes=20,
        n_sparse=6,
        n_motif=14,
        seed=0,
    )

    plot_graph(genes)
