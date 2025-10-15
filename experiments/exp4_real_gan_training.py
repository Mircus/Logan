#!/usr/bin/env python3
"""
Experiment 4: Real Logical GAN Training
Trains actual neural GANs with logical loss for graph generation.
"""

import sys
from pathlib import Path
import argparse
import csv
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logical_gans.logic.logical_gan_framework import LogicalGANTrainer


def run_gan_training(property_name: str, epochs: int, output_csv: Path, quick: bool = False):
    """
    Run real GAN training with logical loss.

    Args:
        property_name: Target property (tree, bipartite, connectivity)
        epochs: Number of training epochs
        output_csv: Path to save results
        quick: If True, use reduced parameters for quick testing
    """
    print(f"\n{'='*60}")
    print(f"Experiment 4: Real Logical GAN Training")
    print(f"Property: {property_name}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}\n")

    # Configuration
    if quick:
        config = {
            'property': property_name,
            'latent_dim': 32,
            'max_nodes': 10,
            'logic_depth': 2,
            'ef_weight': 0.05,
            'max_ef_rounds': 2,
            'epochs': epochs,
            'batch_size': 8,
            'theory_size': 50,
            'eval_samples': 50,
            'log_interval': max(1, epochs // 10),
            'gnn_type': 'GCN',
            'generator_hidden_dims': [128, 256],
            'discriminator_hidden_dim': 32
        }
    else:
        config = {
            'property': property_name,
            'latent_dim': 64,
            'max_nodes': 12,
            'logic_depth': 3,
            'ef_weight': 0.05,
            'max_ef_rounds': 3,
            'epochs': epochs,
            'batch_size': 16,
            'theory_size': 200,
            'eval_samples': 100,
            'log_interval': max(1, epochs // 20),
            'gnn_type': 'GCN',
            'generator_hidden_dims': [256, 512, 1024],
            'discriminator_hidden_dim': 64
        }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Initialize trainer
    print("Initializing Logical GAN...")
    trainer = LogicalGANTrainer(config)

    # Evaluate before training
    print("\nEvaluating BEFORE training (untrained baseline)...")
    baseline_results = trainer.evaluate()

    print("Baseline Results:")
    for metric, value in baseline_results.items():
        print(f"  {metric}: {value:.4f}")

    # Train
    print(f"\nTraining Logical GAN for {epochs} epochs...")
    print("-" * 60)
    training_history = trainer.train()
    print("-" * 60)

    # Evaluate after training
    print("\nEvaluating AFTER training...")
    final_results = trainer.evaluate()

    print("\nFinal Results:")
    for metric, value in final_results.items():
        print(f"  {metric}: {value:.4f}")

    # Compute improvements
    property_sat_improvement = final_results['property_satisfaction_rate'] - baseline_results['property_satisfaction_rate']
    ef_distance_improvement = baseline_results['average_ef_distance'] - final_results['average_ef_distance']

    print("\nImprovement Summary:")
    print(f"  Property satisfaction: {baseline_results['property_satisfaction_rate']:.3f} -> {final_results['property_satisfaction_rate']:.3f} (+{property_sat_improvement:+.3f})")
    print(f"  Average EF-distance: {baseline_results['average_ef_distance']:.3f} -> {final_results['average_ef_distance']:.3f} ({ef_distance_improvement:+.3f})")

    # Save results
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'property', 'epochs', 'baseline_sat', 'final_sat', 'sat_improvement',
            'baseline_ef_dist', 'final_ef_dist', 'ef_improvement',
            'final_perfect_ef_rate', 'latent_dim', 'logic_depth', 'batch_size'
        ])
        writer.writerow([
            property_name,
            epochs,
            f"{baseline_results['property_satisfaction_rate']:.4f}",
            f"{final_results['property_satisfaction_rate']:.4f}",
            f"{property_sat_improvement:+.4f}",
            f"{baseline_results['average_ef_distance']:.4f}",
            f"{final_results['average_ef_distance']:.4f}",
            f"{ef_distance_improvement:+.4f}",
            f"{final_results['perfect_ef_distance_rate']:.4f}",
            config['latent_dim'],
            config['logic_depth'],
            config['batch_size']
        ])

    print(f"\nResults saved to: {output_csv}")

    # Verdict
    print("\n" + "="*60)
    if property_sat_improvement > 0.05 and final_results['property_satisfaction_rate'] > 0.5:
        print("PASS: GAN training improved property satisfaction significantly!")
        print(f"   Achieved {final_results['property_satisfaction_rate']*100:.1f}% satisfaction")
        verdict = "PASS"
    elif property_sat_improvement > 0.0:
        print("PARTIAL: Some improvement observed but may need more training")
        verdict = "PARTIAL"
    else:
        print("FAIL: No significant improvement - may need hyperparameter tuning")
        verdict = "FAIL"
    print("="*60)

    return {
        'baseline': baseline_results,
        'final': final_results,
        'improvement': property_sat_improvement,
        'verdict': verdict,
        'history': training_history
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Real Logical GAN Training"
    )
    parser.add_argument(
        '--property',
        default='tree',
        choices=['tree', 'bipartite', 'connectivity'],
        help='Target graph property'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with reduced parameters'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: results/exp4_{property}_gan.csv)'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = Path(__file__).parent.parent / "results" / f"exp4_{args.property}_gan.csv"

    # Run training
    results = run_gan_training(
        property_name=args.property,
        epochs=args.epochs,
        output_csv=output_csv,
        quick=args.quick
    )

    return 0 if results['verdict'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())
