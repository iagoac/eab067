import csv
import math
import random

import networkx as nx
import networkx.algorithms.community as nx_comm


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def dense_random(n: int = 100, p: float = 0.7, seed: int = 42) -> nx.Graph:
    """Erdős–Rényi random graph with high edge probability."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def powerlaw(n: int = 100, m: int = 3, seed: int = 42) -> nx.Graph:
    """Barabási–Albert preferential-attachment graph (power-law degree distribution)."""
    return nx.barabasi_albert_graph(n, m, seed=seed)


def complete(n: int = 50, seed=None) -> nx.Graph:
    """Complete graph K_n (deterministic; seed is accepted but ignored)."""
    return nx.complete_graph(n)


def unit_disk(n: int = 100, radius: float = 0.2, seed: int = 42) -> nx.Graph:
    """
    Random geometric (unit-disk) graph: nodes are placed uniformly at random
    in the unit square and edges connect nodes within *radius* of each other.
    """
    return nx.random_geometric_graph(n, radius, seed=seed)


def grid_2d(rows: int = 10, cols: int = 10, seed=None) -> nx.Graph:
    """2-D grid graph (rows × cols); deterministic; seed is accepted but ignored."""
    return nx.grid_2d_graph(rows, cols)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _largest_component(G: nx.Graph) -> nx.Graph:
    """Return the subgraph induced by the largest connected component."""
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def compute_metrics(G: nx.Graph, graph_class: str) -> dict:
    """Compute all requested metrics for graph *G*."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    max_edges = n * (n - 1) / 2 if n > 1 else 1

    density = nx.density(G)
    is_dense = density > 0.5

    # Use LCC for path-length / small-world (requires connected graph)
    G_lcc = _largest_component(G)
    n_lcc = G_lcc.number_of_nodes()

    # --- clustering coefficient (average) ---
    avg_clustering = nx.average_clustering(G)

    # --- betweenness centrality (average over nodes) ---
    bc = nx.betweenness_centrality(G, normalized=True)
    avg_betweenness = sum(bc.values()) / len(bc) if bc else 0.0

    # --- degree centrality (average over nodes) ---
    dc = nx.degree_centrality(G)
    avg_degree_centrality = sum(dc.values()) / len(dc) if dc else 0.0

    # --- average path length (on LCC) ---
    if n_lcc > 1:
        avg_path_length = nx.average_shortest_path_length(G_lcc)
    else:
        avg_path_length = float("nan")

    # --- small-world coefficient σ = (C/C_rand) / (L/L_rand)
    #     Uses a comparable Erdős–Rényi random graph as reference.
    small_world_sigma = float("nan")
    if n_lcc > 2 and G_lcc.number_of_edges() > 0:
        try:
            p_ref = nx.density(G_lcc)
            # Average clustering of equivalent random graph ≈ p
            c_rand = p_ref
            # Average path length of equivalent random graph ≈ ln(n)/ln(k)
            avg_k = 2 * G_lcc.number_of_edges() / n_lcc
            if avg_k > 1:
                l_rand = math.log(n_lcc) / math.log(avg_k)
            else:
                l_rand = float("nan")

            c_lcc = nx.average_clustering(G_lcc)
            if not math.isnan(avg_path_length) and not math.isnan(l_rand):
                if l_rand > 0 and c_rand > 0:
                    small_world_sigma = (c_lcc / c_rand) / (avg_path_length / l_rand)
        except Exception:
            pass

    # --- modularity (Louvain community detection) ---
    try:
        communities = nx_comm.louvain_communities(G, seed=42)
        modularity = nx_comm.modularity(G, communities)
    except Exception:
        modularity = float("nan")

    return {
        "graph_class": graph_class,
        "seed": None,  # filled in by caller
        "nodes": n,
        "edges": e,
        "density": round(density, 6),
        "is_dense": is_dense,
        "avg_clustering_coefficient": round(avg_clustering, 6),
        "avg_betweenness_centrality": round(avg_betweenness, 6),
        "avg_degree_centrality": round(avg_degree_centrality, 6),
        "avg_path_length": round(avg_path_length, 6) if not math.isnan(avg_path_length) else "nan",
        "small_world_sigma": round(small_world_sigma, 6) if not math.isnan(small_world_sigma) else "nan",
        "modularity": round(modularity, 6) if not math.isnan(modularity) else "nan",
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(records: list[dict], filepath: str = "graph_metrics.csv") -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Metrics exported to '{filepath}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Each entry: (generator_function, fixed_kwargs_without_seed)
CONFIGS = [
    (dense_random, {"n": 100, "p": 0.7}),
    (dense_random, {"n": 100, "p": 0.9}),
    (powerlaw,     {"n": 100, "m": 3}),
    (powerlaw,     {"n": 100, "m": 5}),
    (complete,     {"n": 50}),
    (unit_disk,    {"n": 100, "radius": 0.2}),
    (unit_disk,    {"n": 100, "radius": 0.3}),
    (grid_2d,      {"rows": 10, "cols": 10}),
    (grid_2d,      {"rows": 15, "cols": 15}),
]

TOTAL_GRAPHS = 1000


def main():
    n_configs = len(CONFIGS)
    records = []

    for i in range(TOTAL_GRAPHS):
        func, kwargs = CONFIGS[i % n_configs]
        seed = i // n_configs
        G = func(**kwargs, seed=seed)
        metrics = compute_metrics(G, func.__name__)
        metrics["seed"] = seed
        records.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{TOTAL_GRAPHS} graphs processed...")

    print(f"All {TOTAL_GRAPHS} graphs processed.")
    export_csv(records, filepath="graph_metrics.csv")


if __name__ == "__main__":
    main()
