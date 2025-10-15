"""
Experiment 3: Training Validation - Prove Logical GAN Framework Works

This experiment validates that training with logical loss actually improves
performance beyond the 50% naive baseline from Experiment 2.

Goal: Show property satisfaction rate > 0.50 after training
Expected: ~0.70-0.90 depending on property and epochs
"""

import argparse
import sys
from pathlib import Path
import csv
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logical_gans.logic.logical_gan_framework import LogicalGANTrainer


def run_training_validation(property_name: str, epochs: int, output_csv: Path):
    """
    Run training validation experiment.

    This demonstrates that with actual training (not just naive prototype matching),
    the framework can generate graphs satisfying target properties at rates > 50%.
    """

    print(f"=" * 60)
    print(f"Experiment 3: Training Validation for '{property_name}'")
    print(f"=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Property: {property_name}")
    print()

    # Configure training (small scale for validation)
    config = {
        'property': property_name,
        'latent_dim': 64,  # Smaller for faster training
        'max_nodes': 12,   # Smaller graphs for speed
        'logic_depth': 2,  # Reduced depth
        'ef_weight': 0.05,  # Lower weight for stability
        'epochs': epochs,
        'batch_size': 16,
        'theory_size': 200,  # Smaller theory set
        'gnn_type': 'GCN',
        'eval_samples': 100,
        'log_interval': max(1, epochs // 10)  # Log 10 times
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Initialize trainer
    print("Initializing Logical GAN trainer...")
    try:
        trainer = LogicalGANTrainer(config)
        print(f"Theory graphs generated: {len(trainer.theory_graphs)}")
        print()
    except Exception as e:
        print(f"ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate BEFORE training (baseline)
    print("Evaluating BEFORE training (baseline)...")
    try:
        baseline_results = trainer.evaluate()
        baseline_sat = baseline_results['property_satisfaction_rate']
        print(f"Baseline property satisfaction rate: {baseline_sat:.4f}")
        print()
    except Exception as e:
        print(f"ERROR during baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Train the model
    print("Starting training...")
    print()
    try:
        training_history = trainer.train()
        print()
        print("Training complete!")
        print()
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate AFTER training
    print("Evaluating AFTER training...")
    try:
        final_results = trainer.evaluate()
        final_sat = final_results['property_satisfaction_rate']
        print(f"Final property satisfaction rate: {final_sat:.4f}")
        print()
    except Exception as e:
        print(f"ERROR during final evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compute improvement
    improvement = final_sat - baseline_sat
    improvement_pct = (improvement / max(baseline_sat, 0.01)) * 100

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Property: {property_name}")
    print(f"Epochs: {epochs}")
    print(f"Baseline satisfaction rate: {baseline_sat:.4f}")
    print(f"Final satisfaction rate: {final_sat:.4f}")
    print(f"Absolute improvement: {improvement:+.4f}")
    print(f"Relative improvement: {improvement_pct:+.1f}%")
    print()

    # Check if we beat naive baseline
    if final_sat > 0.55:  # More than 5% above random
        print("SUCCESS: Framework improves beyond naive 50% baseline!")
    elif final_sat > baseline_sat:
        print("PARTIAL SUCCESS: Improvement shown, may need more epochs")
    else:
        print("WARNING: No improvement over baseline - may need tuning")
    print("=" * 60)
    print()

    # Save results to CSV
    output_csv.parent.mkdir(exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['property', 'epochs', 'baseline_sat', 'final_sat',
                        'improvement', 'improvement_pct'])
        writer.writerow([property_name, epochs,
                        f"{baseline_sat:.4f}", f"{final_sat:.4f}",
                        f"{improvement:+.4f}", f"{improvement_pct:+.1f}"])

    print(f"Results saved to: {output_csv}")
    print()

    # Also save training history
    history_csv = output_csv.parent / f"exp3_{property_name}_history.csv"
    with open(history_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'property_satisfaction_rate'])
        for i, sat_rate in enumerate(training_history['property_satisfaction_rates']):
            writer.writerow([i, f"{sat_rate:.4f}"])

    print(f"Training history saved to: {history_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Training Validation"
    )
    parser.add_argument(
        "--property",
        type=str,
        default="tree",
        choices=["tree", "connectivity", "bipartite"],
        help="Property to validate (default: tree)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100, use 50 for quick test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/exp3_{property}.csv)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = Path(__file__).parent.parent / "results" / f"exp3_{args.property}.csv"

    # Run experiment
    run_training_validation(args.property, args.epochs, output_csv)


if __name__ == "__main__":
    main()
