import argparse
import networkx as nx
from .ef_games import EFGameSimulator

def parse_graph(spec: str) -> nx.Graph:
    """
    Supported specs:
      - path:N
      - cycle:N
      - complete:N            (K_N)
      - kxy:a,b               (complete bipartite K_{a,b})
    """
    try:
        kind, args = spec.split(":", 1)
    except ValueError:
        raise SystemExit(f"Invalid graph spec: {spec}")
    kind = kind.strip().lower()
    args = args.strip()
    if kind == "path":
        return nx.path_graph(int(args))
    if kind == "cycle":
        return nx.cycle_graph(int(args))
    if kind == "complete":
        return nx.complete_graph(int(args))
    if kind in {"kxy", "kij", "kbip"}:
        a, b = (int(x) for x in args.split(",", 1))
        return nx.complete_bipartite_graph(a, b)
    raise SystemExit(f"Unknown graph kind: {kind}")

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute EF distance between two graphs.")
    p.add_argument("--graph1", required=True, help="Graph spec, e.g., path:4, cycle:6, kxy:3,3")
    p.add_argument("--graph2", required=True, help="Graph spec, e.g., cycle:4, path:4")
    p.add_argument("--rounds", "-k", type=int, default=3, help="Number of EF rounds (k).")
    args = p.parse_args(argv)

    G1 = parse_graph(args.graph1)
    G2 = parse_graph(args.graph2)

    sim = EFGameSimulator(G1, G2)
    dist = sim.ef_distance(args.rounds)
    dup  = sim.duplicator_wins(args.rounds)

    print(f"EF-distance(k={args.rounds}): {dist}")
    print(f"Duplicator wins? {dup}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
