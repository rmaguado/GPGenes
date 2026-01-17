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
from .motifs import MOTIFS, PRESETS


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
        base = random.normalvariate(1.0, 0.5)
        if random.random() < 0.05:
            base = 0.0
        base = max(0.0, base)
        gene = Gene(gid=i, base=base)
        genes.append(gene)

    n_tf = n_sparse + n_motif
    tf_genes = genes[:n_tf]

    motif_pool = set(random.sample(tf_genes, n_motif))
    sparse_pool = set(tf_genes) - motif_pool

    while True:
        possible = [(fn, k) for fn, k in PRESETS if len(motif_pool) >= k]

        if not possible:
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

        sign = random.choice([-1, 1])
        strength = sign * random.normalvariate(1.0, 0.2)
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


def run(genes, steps, delta):
    for g in genes:
        g.reset()

    for _ in range(steps):
        for g in genes:
            g.sync()
        inputs = [g.compute_input() for g in genes]
        for g, inp in zip(genes, inputs):
            g.step(delta, inp)


def run_with_knockout(genes, ko_gene_id=None, steps=100, delta=1.0):
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
    n_reps: int = 3,
    steps: int = 100,
    delta: float = 0.1,
    noise: float = 0.001,
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
            observations = {}
            for g in sim_genes:
                observations[g.id] = float(g.history[-1] + rng.gauss(0.0, noise))

            row = {"perturbation": pert_label, "replicate": rep, "seed": run_seed}
            for gid in range(n_genes):
                row[f"g{gid:02d}"] = float(observations[gid])
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


def plot_trajectories(genes):
    trajectories = np.array([g.history for g in genes])
    n_genes, n_steps = trajectories.shape

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        trajectories,
        cmap="viridis",
        origin="lower",
        aspect=n_steps / n_genes,
        interpolation="nearest",
    )
    ax.set_ylabel("Gene ID")
    ax.set_title("Gene Expression History")
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels([g.id for g in genes])

    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_graph(genes, steps=100, delta=0.1):
    genes = clone_genes(genes)

    G = genes_to_digraph(genes)
    pos = nx.spring_layout(G, k=40.0, method="energy", iterations=1000)

    active = {g.id: True for g in genes}

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Click nodes to toggle genes")

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=mcl.Normalize(vmin=0, vmax=1.0),
    )
    sm.set_array([])

    def run_and_update():
        for g in genes:
            if active[g.id]:
                g.knocked_out = False
            else:
                g.knock_out()

        run(genes, steps=steps, delta=delta)

    def redraw():
        ax.clear()

        edges = G.edges(data=True)
        widths = [abs(d["weight"]) * genes[s].value * 2.0 for (s, _, d) in edges]
        widths = [min(max(w, 1.0), 3.0) for w in widths]
        edge_colors = ["green" if d["weight"] > 0 else "red" for (_, _, d) in edges]

        cmap = plt.get_cmap("viridis")
        node_colors = []
        for g in genes:
            color = cmap(g.value)
            if not active[g.id]:
                color = (0.2, 0.2, 0.2)
            node_colors.append(color)

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            linewidths=0,
            node_size=800,
        )
        connection_styles = ["arc3" for e in edges]  # rad=0.2

        nodes.set_picker(True)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=widths,
            edge_color=edge_colors,
            arrowsize=12,
            arrows=True,
            node_size=800,
            arrowstyle="-|>,head_length=0.4,head_width=0.2",
            connectionstyle=connection_styles,
        )

        for g in genes:
            font_color = "white" if g.value < 0.5 else "black"
            nx.draw_networkx_labels(
                G,
                pos,
                labels={g.id: f"{max(round(g.value,3), 0):0.3}"},
                font_color=font_color,
                font_size=8,
                ax=ax,
            )

        fig.canvas.draw_idle()
        return nodes

    run_and_update()
    nodes = redraw()

    def on_pick(event):
        ind = event.ind[0]
        node_id = list(G.nodes)[ind]
        active[node_id] = not active[node_id]
        run_and_update()
        redraw()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.colorbar(sm, ax=ax)

    plt.show()
