#!/usr/bin/env python3
"""
Experiment 2: EF-Distance Prototype Classifier
See paper Section: Experimental Validation (Exp.2)

Tests EF-distance as a classifier using naive prototypes (one per class).
Expected: ~0.50 accuracy with poor prototypes, demonstrating the baseline.
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
    from logical_gans.logic.ef_games import EFGameSimulator
    from logical_gans.logic.mso import MSOPropertyLibrary
except ImportError:
    print("Warning: Could not import from logical_gans package. Using fallback.")
    from logical_gans import EFGameSimulator, MSOPropertyLibrary

RNG = random.Random(42)


def generate_test_graphs(property_name: str, n: int, count: int, positive: bool):
    """Generate test graphs for classification."""
    graphs = []
    for _ in range(count):
        if property_name == "bipartite":
            if positive:
                n1, n2 = max(1, n // 2), max(1, n - n // 2)
                p = RNG.uniform(0.3, 0.7)
                g = nx.bipartite.random_graph(n1, n2, p, seed=RNG.randint(0, 1000000))
            else:
                # Non-bipartite: odd cycle
                g = nx.cycle_graph(5)
                for i in range(5, n):
                    g.add_node(i)
                    u = RNG.choice(range(i))
                    g.add_edge(i, u)
        elif property_name == "tree":
            if positive:
                g = nx.path_graph(n) if n > 1 else nx.empty_graph(n)
            else:
                g = nx.cycle_graph(min(n, 4))
                for i in range(len(g), n):
                    g.add_node(i)
                    u = RNG.choice(range(i))
                    g.add_edge(i, u)
        else:
            g = nx.erdos_renyi_graph(n, 0.3, seed=RNG.randint(0, 1000000))
        graphs.append(g)
    return graphs


def create_naive_prototypes(property_name: str, n: int):
    """Create one naive prototype for positive and negative class."""
    if property_name == "bipartite":
        # Positive: simple bipartite
        pos_proto = nx.complete_bipartite_graph(n // 2, n - n // 2)
        # Negative: odd cycle
        neg_proto = nx.cycle_graph(5)
        for i in range(5, n):
            neg_proto.add_node(i)
            if i > 5:
                neg_proto.add_edge(i, RNG.choice(range(i)))
    elif property_name == "tree":
        # Positive: path graph
        pos_proto = nx.path_graph(n)
        # Negative: cycle
        neg_proto = nx.cycle_graph(n)
    else:
        pos_proto = nx.erdos_renyi_graph(n, 0.3, seed=42)
        neg_proto = nx.erdos_renyi_graph(n, 0.7, seed=43)

    return pos_proto, neg_proto


def classify_with_ef_distance(graph: nx.Graph, pos_proto: nx.Graph, neg_proto: nx.Graph, k: int) -> bool:
    """Classify graph based on EF-distance to prototypes.

    Returns True if closer to positive prototype, False otherwise.
    """
    # Compute EF distance to both prototypes
    sim_pos = EFGameSimulator(graph, pos_proto)
    sim_neg = EFGameSimulator(graph, neg_proto)

    dist_pos = sim_pos.ef_distance(max_rounds=k)
    dist_neg = sim_neg.ef_distance(max_rounds=k)

    # Lower distance = more similar = predicted class
    return dist_pos <= dist_neg


def accuracy_vs_k(property_name: str, n_min: int, n_max: int, samples: int,
                  k_values: list, output_csv: Path):
    """Run Experiment 2: EF-distance prototype classifier."""
    print(f"Running Exp.2: EF-distance classifier for property='{property_name}'")

    results = []

    for k in k_values:
        print(f"  Testing k={k}...")
        correct = 0
        total = 0

        for n in range(n_min, n_max + 1):
            # Create naive prototypes
            pos_proto, neg_proto = create_naive_prototypes(property_name, n)

            # Generate test samples (balanced)
            num_per_class = samples // 2
            pos_graphs = generate_test_graphs(property_name, n, num_per_class, positive=True)
            neg_graphs = generate_test_graphs(property_name, n, num_per_class, positive=False)

            # Classify positive examples
            for g in pos_graphs:
                pred = classify_with_ef_distance(g, pos_proto, neg_proto, k)
                if pred:  # Correctly predicted as positive
                    correct += 1
                total += 1

            # Classify negative examples
            for g in neg_graphs:
                pred = classify_with_ef_distance(g, pos_proto, neg_proto, k)
                if not pred:  # Correctly predicted as negative
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        results.append({"property": property_name, "k": k, "accuracy": accuracy})
        print(f"    k={k}: accuracy={accuracy:.3f} ({correct}/{total})")

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["property", "k", "accuracy"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

    # Summary
    avg_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"Summary: avg_accuracy={avg_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp.2: EF-distance prototype classifier")
    parser.add_argument("--property", default="bipartite",
                        choices=["bipartite", "tree"])
    parser.add_argument("--n-min", type=int, default=6)
    parser.add_argument("--n-max", type=int, default=12)
    parser.add_argument("--samples", type=int, default=40, help="Total samples (split 50/50 pos/neg)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5],
                        help="EF depths to test")
    parser.add_argument("--out", type=Path, default=Path("results/exp2.csv"))
    args = parser.parse_args()

    accuracy_vs_k(args.property, args.n_min, args.n_max, args.samples,
                  args.k_values, args.out)
