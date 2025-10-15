"""
logical_gans.repro — run both experiments and save CSVs to results/.
"""
from pathlib import Path
import argparse
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce Logical-GANs core experiments")
    parser.add_argument("--property", default="bipartite",
                        choices=["bipartite","tree","planarity","connectivity","has_triangle"])
    parser.add_argument("--quick", action="store_true", help="Fewer samples for a quick smoke run")
    parser.add_argument("--exp", choices=["1", "2", "both"], default="both",
                        help="Which experiment to run (default: both)")
    args = parser.parse_args(argv)

    # Add experiments directory to path
    exp_dir = Path(__file__).parent.parent.parent / "experiments"
    sys.path.insert(0, str(exp_dir))

    results = Path("results")
    results.mkdir(exist_ok=True, parents=True)

    if args.exp in ["1", "both"]:
        print(f"\n=== Running Experiment 1: MSO Property Satisfaction ===")
        from exp1_mso_satisfaction import run as run_exp1
        samples = 10 if args.quick else 20
        run_exp1(args.property, 6, 16 if not args.quick else 10, samples,
                 results / f"exp1_{args.property}.csv")

    if args.exp in ["2", "both"] and args.property in ("bipartite", "tree"):
        print(f"\n=== Running Experiment 2: EF-Distance Classifier ===")
        from exp2_ef_distance_proto import accuracy_vs_k as run_exp2_acc
        run_exp2_acc(args.property, 6, 10 if args.quick else 12,
                     20 if args.quick else 40,
                     [2, 3, 4, 5], results / f"exp2_{args.property}_acc.csv")

    print(f"\nDone. CSVs saved in {results.absolute()}")
