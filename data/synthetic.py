import csv
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class Regulation:
    source: "Gene"
    strength: float
    delay: int


class Gene:
    def __init__(
        self,
        gid: int,
        is_tf: bool = False,
        base: float = 0.0,
        limit: float = 10.0,
        decay: float = 0.1,
        noise_sigma: float = 0.02,
    ):
        self.id = gid
        self.is_tf = is_tf
        self.base = base
        self.limit = limit
        self.decay = decay
        self.noise_sigma = noise_sigma

        self.value = 0.0
        self.history = []
        self.regulators: list[Regulation] = []
        self.delay_buffer = deque(maxlen=5)

        self.knocked_out = False

    def knock_out(self):
        self.knocked_out = True
        self.value = 0.0
        self.delay_buffer.clear()
        self.delay_buffer.append(0.0)

    def reset(self):
        self.value = self.base + random.uniform(-0.1, 0.1)
        if self.knocked_out:
            self.value = 0.0
        self.history.clear()
        self.delay_buffer.clear()
        self.delay_buffer.append(self.value)

    @staticmethod
    def hill(x, K=1.0, n=2):
        return x**n / (K**n + x**n)

    def compute_input(self):
        if self.knocked_out:
            return 0.0

        total = 0.0
        for reg in self.regulators:
            if len(reg.source.delay_buffer) > reg.delay:
                x = reg.source.delay_buffer[-reg.delay - 1]
                h = self.hill(x)
                total += reg.strength * h
        return total

    def step(self, delta, input_signal):
        if self.knocked_out:
            self.value = 0.0
            self.delay_buffer.append(0.0)
            self.history.append(0.0)
            return

        noise = random.gauss(0.0, self.noise_sigma)
        dv = delta * (self.base + input_signal - self.decay * self.value + noise)
        self.value = max(0.0, min(self.limit, self.value + dv))
        self.delay_buffer.append(self.value)
        self.history.append(self.value)


def create_genes(
    n_genes=10,
    tf_fraction=0.3,
    n_modules=3,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    genes = []
    n_tf = int(n_genes * tf_fraction)

    modules = {i: random.randint(0, n_modules - 1) for i in range(n_genes)}

    for i in range(n_genes):
        gene = Gene(
            gid=i,
            is_tf=i < n_tf,
            base=random.uniform(0.2, 2.0),
            decay=random.uniform(0.05, 0.2),
        )
        genes.append(gene)

    tf_out_degree = defaultdict(int)

    for target in genes:
        for source in genes[:n_tf]:
            if source is target:
                continue

            same_module = modules[source.id] == modules[target.id]
            p = 0.4 if same_module else 0.05

            if source.is_tf and target.is_tf:
                p *= 0.3

            p *= (tf_out_degree[source.id] + 1) / 3.0

            if random.random() < p:
                strength = random.normalvariate(0, 0.7)
                delay = random.randint(0, 2)
                target.regulators.append(
                    Regulation(source=source, strength=strength, delay=delay)
                )
                tf_out_degree[source.id] += 1

    if n_tf >= 3:
        A, B, C = random.sample(range(n_tf), 3)
        genes[B].regulators.append(Regulation(genes[A], 1.0, 1))
        genes[C].regulators.append(Regulation(genes[B], 1.0, 1))
        genes[C].regulators.append(Regulation(genes[A], 0.5, 1))

    if n_tf >= 2:
        i, j = random.sample(range(n_tf), 2)
        genes[i].regulators.append(Regulation(genes[j], -1.0, 1))
        genes[j].regulators.append(Regulation(genes[i], -1.0, 1))

    return genes


def clone_genes(genes):
    cloned = copy.deepcopy(genes)
    id_map = {g.id: g for g in cloned}

    for g in cloned:
        for r in g.regulators:
            r.source = id_map[r.source.id]

    return cloned


def run(genes, steps=5000, delta=0.01):
    for g in genes:
        g.reset()

    for _ in range(steps):
        inputs = [g.compute_input() for g in genes]
        for g, inp in zip(genes, inputs):
            g.step(delta, inp)


def run_with_knockout(genes, ko_gene_id=None, steps=5000, delta=0.01):
    genes = clone_genes(genes)

    if ko_gene_id is not None:
        genes[ko_gene_id].knock_out()

    run(genes, steps=steps, delta=delta)
    return genes


def summarize_run(genes, n=100):
    summary = {}
    for g in genes:
        summary[g.id] = float(np.mean(g.history[-n:]))
    return summary


def simulate_all_single_knockouts(
    genes,
    steps=1000,
    delta=0.01,
    tail_steps=100,
    csv_path="knockout_data.csv",
):
    rows = []

    control_genes = run_with_knockout(genes, ko_gene_id=None, steps=steps, delta=delta)
    control_summary = summarize_run(control_genes, n=tail_steps)

    control_row = {"knockout_gene": "co"}
    for gid, val in control_summary.items():
        control_row[f"{gid:02}"] = val
    rows.append(control_row)

    for ko_id in range(len(genes)):
        ko_genes = run_with_knockout(genes, ko_gene_id=ko_id, steps=steps, delta=delta)
        ko_summary = summarize_run(ko_genes, n=tail_steps)

        row = {"knockout_gene": f"{ko_id:02}"}
        for gid, val in ko_summary.items():
            row[f"{gid:02}"] = val
        rows.append(row)

    fieldnames = ["knockout_gene"] + [f"{i:02}" for i in range(len(genes))]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import networkx as nx
import numpy as np


def plot_graph(genes):
    G = nx.DiGraph()

    for g in genes:
        G.add_node(g.id, tf=g.is_tf)

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

    node_sizes = [600 if g.is_tf else 300 for g in genes]

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
        n_genes=30,
        tf_fraction=0.3,
        n_modules=3,
        seed=1,
    )
    run(genes)
    plot_graph(genes)
    plot_trajectories(genes)

    simulate_all_single_knockouts(
        genes,
        steps=1000,
        delta=0.01,
        tail_steps=100,
        csv_path="knockout_data.csv",
    )
