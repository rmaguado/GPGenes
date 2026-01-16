from __future__ import annotations
import csv
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import networkx as nx
import numpy as np

from typing import List, Tuple
from itertools import combinations
from typing import Optional

from .primitives import Gene, Regulation
from .motifs import MOTIFS


def create_genes(
    n_genes: int,
    n_sparse: int,
    n_motif: int,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    assert n_sparse + n_motif <= n_genes

    genes = []

    for i in range(n_genes):
        gene = Gene(
            gid=i,
            base=random.uniform(0.2, 2.0),
            decay=random.uniform(0.05, 0.2),
        )
        genes.append(gene)

    n_tf = n_sparse + n_motif
    tf_genes = genes[:n_tf]

    motif_pool = set(random.sample(tf_genes, n_motif))
    sparse_pool = set(tf_genes) - motif_pool

    while True:
        possible = [(fn, k) for fn, k in MOTIFS if len(motif_pool) >= k]

        if not possible:
            break

        fn, k = random.choice(possible)

        selected = random.sample(list(motif_pool), k)
        fn(*selected)

        for g in selected:
            motif_pool.remove(g)

    for _ in range(n_sparse):
        if len(sparse_pool) < 2:
            break

        source, target = random.sample(list(sparse_pool), 2)

        strength = random.normalvariate(0, 0.7)
        target.regulators.append(Regulation(source=source, strength=strength))

        sparse_pool.remove(source)
        sparse_pool.remove(target)

    return genes


def clone_genes(genes):
    cloned = copy.deepcopy(genes)
    id_map = {g.id: g for g in cloned}

    for g in cloned:
        for r in g.regulators:
            r.source = id_map[r.source.id]

    return cloned


def genes_to_digraph(genes: List[Gene]) -> nx.DiGraph:
    G = nx.DiGraph()
    for g in genes:
        G.add_node(g.id)
    for tgt in genes:
        for r in tgt.regulators:
            G.add_edge(r.source.id, tgt.id, weight=float(r.strength))
    return G


def run(genes, steps=100, delta=0.01):
    for g in genes:
        g.reset()

    for _ in range(steps):
        for g in genes:
            g.sync()
        inputs = [g.compute_input() for g in genes]
        for g, inp in zip(genes, inputs):
            g.step(delta, inp)


def run_with_knockout(genes, ko_gene_id=None, steps=1000, delta=0.01):
    genes = clone_genes(genes)

    if ko_gene_id is not None:
        if isinstance(ko_gene_id, (list, tuple, set)):
            ko_ids = list(ko_gene_id)
        else:
            ko_ids = [int(ko_gene_id)]

        for gid in ko_ids:
            genes[gid].knock_out()

    run(genes, steps=steps, delta=delta)
    return genes


def summarize_run(genes, n=100):
    summary = {}
    for g in genes:
        summary[g.id] = float(np.mean(g.history[-n:]))
    return summary


def make_perturbation_list(
    n_genes: int,
    include_singles: bool = True,
    include_doubles: bool = False,
    n_doubles: int = 50,
    seed: int = 0,
) -> List[Tuple[int, ...]]:
    """
    Returns perturbations as tuples of KO gene IDs:
      () is control, (i,) is single KO, (i,j) is double KO.
    """
    rng = random.Random(seed)
    perts: List[Tuple[int, ...]] = [tuple()]  # control

    if include_singles:
        perts.extend([(i,) for i in range(n_genes)])

    if include_doubles:
        pairs = list(combinations(range(n_genes), 2))
        rng.shuffle(pairs)
        perts.extend(pairs[: min(n_doubles, len(pairs))])

    return perts


# I've added n_reps and seed parameters to allow random replicates for each deletion
# thought then we can tune noise/get more meaningful uncertainty estimates
# can remove or set n_reps=1 if not needed.
def simulate_dataset(
    genes: List[Gene],
    perturbations: List[Tuple[int, ...]],
    n_reps: int = 1,
    steps: int = 1500,
    delta: float = 0.01,
    tail_steps: int = 200,
    seed: int = 0,
    csv_path: Optional[str] = None,
) -> list[dict]:
    """
    Simulate a dataset in the long format expected by dataset.py:

      columns:
        - perturbation: "co" or "03" or "03+17"
        - replicate: int
        - seed: int
        - g00..gNN: float expression summaries

    Returns:
      rows: list of dicts
    """
    rows: list[dict] = []
    n_genes = len(genes)
    rng = random.Random(seed)

    for pert in perturbations:
        pert_label = "co" if len(pert) == 0 else "+".join(f"{i:02d}" for i in pert)

        for rep in range(n_reps):
            run_seed = rng.randint(0, 2**31 - 1)
            random.seed(run_seed)

            sim_genes = run_with_knockout(
                genes,
                ko_gene_id=list(pert),
                steps=steps,
                delta=delta,
            )
            summary = summarize_run(sim_genes, n=tail_steps)

            row = {"perturbation": pert_label, "replicate": rep, "seed": run_seed}
            for gid in range(n_genes):
                row[f"g{gid:02d}"] = float(summary[gid])
            rows.append(row)

    if csv_path is not None:
        fieldnames = ["perturbation", "replicate", "seed"] + [
            f"g{i:02d}" for i in range(n_genes)
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def plot_graph(genes):
    G = nx.DiGraph()

    for g in genes:
        G.add_node(g.id)

    for g in genes:
        for r in g.regulators:
            G.add_edge(
                r.source.id,
                g.id,
                weight=r.strength,
            )

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    values = np.array([g.value for g in genes])
    vmax = float(np.percentile(values, 95))
    threshold = vmax * 0.5

    node_sizes = [300 for g in genes]

    edges = G.edges(data=True)
    widths = [min(3.0, abs(d["weight"]) * 1.5) for (_, _, d) in edges]
    edge_colors = ["green" if d["weight"] > 0 else "red" for (_, _, d) in edges]

    fig, ax = plt.subplots(figsize=(10, 7))

    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=values,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        edge_color=edge_colors,
        width=widths,
        with_labels=False,
        arrows=True,
    )

    for g in genes:
        color = "white" if g.value < threshold else "black"
        nx.draw_networkx_labels(
            G, pos, labels={g.id: g.id}, font_color=color, font_size=8, ax=ax
        )

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=mcl.Normalize(vmin=0, vmax=vmax),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    plt.show()


def plot_trajectories(genes):
    trajectories = np.array([g.history for g in genes])
    n_genes, n_steps = trajectories.shape

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        trajectories,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    ax.set_ylabel("Gene ID")
    ax.set_title("Gene Expression History")
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels([g.id for g in genes])

    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    genes = create_genes(
        n_genes=5,
        n_sparse=1,
        n_motif=3,
        seed=1,
    )
    run(genes, steps=1000)
    plot_graph(genes)
    plot_trajectories(genes)

    perts = make_perturbation_list(
        n_genes=len(genes), include_singles=True, include_doubles=False, seed=0
    )
    simulate_dataset(
        genes,
        perturbations=perts,
        steps=1000,
        delta=0.01,
        tail_steps=100,
        n_reps=3,
        seed=0,
        csv_path="data/knockout_data.csv",
    )
