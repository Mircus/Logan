#!/usr/bin/env python3
"""
Experiment 1: MSO-Style Property Satisfaction
See paper Section: Experimental Validation (Exp.1)

Tests whether our MSO property checkers correctly identify positive and negative examples
of graph properties (bipartite, planarity, tree, connectivity, triangle).
"""

import sys
from pathlib import Path
import argparse
import csv
import random
import networkx as nx

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from logical_gans.logic.mso import MSOPropertyLibrary
except ImportError:
    print("Warning: Could not import from logical_gans package. Using fallback.")
    from logical_gans import MSOPropertyLibrary

RNG = random.Random(1337)


def generate_positive_examples(property_name: str, n: int, count: int):
    """Generate graphs that SHOULD satisfy the property."""
    graphs = []
    for _ in range(count):
        if property_name == "bipartite":
            n1, n2 = max(1, n // 2), max(1, n - n // 2)
            p = RNG.uniform(0.2, 0.8)
            g = nx.bipartite.random_graph(n1, n2, p, seed=RNG.randint(0, 1000000))
        elif property_name == "planarity":
            # Generate planar tree
            g = nx.path_graph(n) if n > 1 else nx.empty_graph(n)
        elif property_name == "tree":
            # Generate tree (use path or star graph)
            if n > 1:
                if RNG.random() < 0.5:
                    g = nx.path_graph(n)
                else:
                    # Star graph
                    g = nx.star_graph(n - 1)
            else:
                g = nx.empty_graph(n)
        elif property_name == "connectivity":
            g = nx.erdos_renyi_graph(n, 0.3, seed=RNG.randint(0, 1000000))
            if not nx.is_connected(g) and len(g) > 0:
                # Force connectivity by connecting components
                cc = list(nx.connected_components(g))
                for i in range(len(cc) - 1):
                    u = RNG.choice(list(cc[i]))
                    v = RNG.choice(list(cc[i+1]))
                    g.add_edge(u, v)
        elif property_name == "has_triangle":
            # Create triangle plus additional edges
            g = nx.complete_graph(3)
            for i in range(3, n):
                g.add_node(i)
                u = RNG.choice(range(i))
                g.add_edge(i, u)
        else:
            g = nx.erdos_renyi_graph(n, 0.3, seed=RNG.randint(0, 1000000))
        graphs.append(g)
    return graphs


def generate_negative_examples(property_name: str, n: int, count: int):
    """Generate graphs that should NOT satisfy the property."""
    graphs = []
    for _ in range(count):
        if property_name == "bipartite":
            # Add odd cycle to ensure non-bipartite
            g = nx.cycle_graph(5)
            for i in range(5, n):
                g.add_node(i)
                u = RNG.choice(range(i))
                g.add_edge(i, u)
        elif property_name == "planarity":
            # K5 or K3,3 are non-planar
            if n >= 5:
                g = nx.complete_graph(5)
                for i in range(5, n):
                    g.add_node(i)
                    # Connect to multiple nodes to maintain non-planarity
                    for _ in range(min(3, i)):
                        u = RNG.choice(range(i))
                        g.add_edge(i, u)
            else:
                g = nx.complete_graph(n)
        elif property_name == "tree":
            # Add cycle to make it not a tree
            g = nx.cycle_graph(min(n, 4))
            for i in range(len(g), n):
                g.add_node(i)
                u = RNG.choice(range(i))
                g.add_edge(i, u)
        elif property_name == "connectivity":
            # Create disconnected graph
            n1 = max(1, n // 2)
            n2 = n - n1
            g = nx.empty_graph(n)
            # Two separate cliques
            for i in range(n1):
                for j in range(i+1, n1):
                    if RNG.random() < 0.5:
                        g.add_edge(i, j)
            for i in range(n1, n):
                for j in range(i+1, n):
                    if RNG.random() < 0.5:
                        g.add_edge(i, j)
        elif property_name == "has_triangle":
            # Create tree (no triangles)
            g = nx.path_graph(n) if n > 1 else nx.empty_graph(n)
        else:
            g = nx.erdos_renyi_graph(n, 0.1, seed=RNG.randint(0, 1000000))
        graphs.append(g)
    return graphs


def run(property_name: str, n_min: int, n_max: int, samples_per_size: int, output_csv: Path):
    """Run Experiment 1: MSO property satisfaction."""
    print(f"Running Exp.1: MSO satisfaction for property='{property_name}'")

    lib = MSOPropertyLibrary()
    checker = lib.get_property(property_name)

    results = []

    for n in range(n_min, n_max + 1):
        # Positive examples
        pos_graphs = generate_positive_examples(property_name, n, samples_per_size)
        pos_results = [checker.check(g) for g in pos_graphs]
        pos_rate = sum(pos_results) / len(pos_results) if pos_results else 0.0

        # Negative examples
        neg_graphs = generate_negative_examples(property_name, n, samples_per_size)
        neg_results = [checker.check(g) for g in neg_graphs]
        neg_reject_rate = 1.0 - (sum(neg_results) / len(neg_results) if neg_results else 0.0)

        results.append({
            "property": property_name,
            "n": n,
            "positive_pass_rate": pos_rate,
            "negative_reject_rate": neg_reject_rate,
        })

        print(f"  n={n:2d}: pos_pass={pos_rate:.2f}, neg_reject={neg_reject_rate:.2f}")

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "n", "positive_pass_rate", "negative_reject_rate"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

    # Summary
    avg_pos = sum(r["positive_pass_rate"] for r in results) / len(results)
    avg_neg = sum(r["negative_reject_rate"] for r in results) / len(results)
    print(f"Summary: avg_positive_pass={avg_pos:.2f}, avg_negative_reject={avg_neg:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp.1: MSO property satisfaction")
    parser.add_argument("--property", default="bipartite",
                        choices=["bipartite", "tree", "planarity", "connectivity", "has_triangle"])
    parser.add_argument("--n-min", type=int, default=6)
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20, help="Samples per graph size")
    parser.add_argument("--out", type=Path, default=Path("results/exp1.csv"))
    args = parser.parse_args()

    run(args.property, args.n_min, args.n_max, args.samples, args.out)
