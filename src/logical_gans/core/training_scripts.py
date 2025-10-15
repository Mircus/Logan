"""
Training Scripts and Configuration Files for Logical GANs
"""

# === scripts/train_model.py ===

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any

from logical_gans.core.logical_gan import LogicalGANTrainer
from logical_gans.utils.metrics import GraphMetrics, PerformanceProfiler
from logical_gans.utils.graph_utils import GraphUtils, set_random_seeds


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: Dict[str, Any]) -> LogicalGANTrainer:
    """Setup experiment based on configuration."""
    
    # Set random seeds for reproducibility
    set_random_seeds(config.get('seed', 42))
    
    # Create trainer
    trainer = LogicalGANTrainer(config)
    
    return trainer


def train_logical_gan(config_path: str, output_dir: str = "results", 
                     verbose: bool = True) -> Dict[str, Any]:
    """
    Main training function for Logical GANs.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Directory to save results  
        verbose: Whether to print progress
        
    Returns:
        Training results and metrics
    """
    
    # Load configuration
    config = load_config(config_path)
    
    if verbose:
        print(f"Training Logical GAN with configuration:")
        print(yaml.dump(config, default_flow_style=False))
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_path / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Setup experiment
    trainer = setup_experiment(config)
    
    # Setup performance profiler
    profiler = PerformanceProfiler()
    
    # Train model
    start_time = time.time()
    
    if verbose:
        print(f"Starting training for {config.get('epochs', 1000)} epochs...")
    
    training_history, training_time = profiler.time_function(
        "full_training", trainer.train
    )
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {total_time:.2f} seconds")
    
    # Evaluate trained model
    if verbose:
        print("Evaluating trained model...")
    
    evaluation_results, eval_time = profiler.time_function(
        "model_evaluation", trainer.evaluate
    )
    
    # Generate sample graphs for analysis
    generated_graphs, gen_time = profiler.time_function(
        "graph_generation", trainer.logical_gan.generate,
        config.get('eval_samples', 500)
    )
    
    # Compute comprehensive metrics
    if verbose:
        print("Computing comprehensive metrics...")
    
    diversity_metrics = GraphMetrics.structural_diversity(generated_graphs)
    ef_metrics = GraphMetrics.ef_distance_distribution(
        generated_graphs, trainer.theory_graphs, 
        max_rounds=config.get('max_ef_rounds', 3)
    )
    
    # Compile final results
    results = {
        'config': config,
        'training_history': training_history,
        'evaluation_results': evaluation_results,
        'diversity_metrics': diversity_metrics,
        'ef_metrics': ef_metrics,
        'performance_metrics': {
            'total_training_time': total_time,
            'evaluation_time': eval_time,
            'generation_time': gen_time,
            'timing_breakdown': profiler.get_timing_summary()
        },
        'generated_samples': len(generated_graphs),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    with open(output_path / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save generated graphs
    GraphUtils.save_graphs(generated_graphs[:100], 
                          str(output_path / "generated_graphs.json"))
    
    # Save model checkpoints
    torch.save(trainer.generator.state_dict(), 
              output_path / "generator_checkpoint.pth")
    torch.save(trainer.discriminator.state_dict(), 
              output_path / "discriminator_checkpoint.pth")
    
    if verbose:
        print(f"Results saved to {output_path}")
        print(f"Property satisfaction rate: {evaluation_results['property_satisfaction_rate']:.2%}")
        print(f"Average EF-distance: {ef_metrics['avg_ef_distance']:.3f}")
        print(f"Structural diversity: {diversity_metrics['diversity_score']:.3f}")
    
    return results


def main():
    """Command-line interface for training."""
    
    parser = argparse.ArgumentParser(description="Train Logical GAN")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID (CPU if not specified)')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}")
    else:
        print("Using CPU")
    
    # Run training
    results = train_logical_gan(args.config, args.output_dir, args.verbose)
    
    print("Training completed successfully!")
    return results


if __name__ == "__main__":
    main()


# === scripts/evaluate_model.py ===

import argparse
import torch
import yaml
import json
import numpy as np
from pathlib import Path

from logical_gans.core.logical_gan import LogicalGANTrainer, GraphGenerator, LogicalDiscriminator
from logical_gans.utils.metrics import GraphMetrics
from logical_gans.utils.graph_utils import GraphUtils
from logical_gans.core.mso_compiler import MSOPropertyLibrary


def load_trained_model(results_dir: str) -> Tuple[GraphGenerator, LogicalDiscriminator, Dict]:
    """Load trained model from results directory."""
    
    results_path = Path(results_dir)
    
    # Load configuration
    with open(results_path / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Recreate models
    generator = GraphGenerator(
        latent_dim=config.get('latent_dim', 128),
        max_nodes=config.get('max_nodes', 20)
    )
    
    discriminator = LogicalDiscriminator(
        logic_depth=config.get('logic_depth', 3),
        hidden_dim=config.get('discriminator_hidden_dim', 64)
    )
    
    # Load weights
    generator.load_state_dict(torch.load(results_path / "generator_checkpoint.pth"))
    discriminator.load_state_dict(torch.load(results_path / "discriminator_checkpoint.pth"))
    
    return generator, discriminator, config


def comprehensive_evaluation(generator: GraphGenerator, config: Dict, 
                           num_samples: int = 1000) -> Dict[str, Any]:
    """Perform comprehensive evaluation of trained generator."""
    
    generator.eval()
    
    # Generate samples
    with torch.no_grad():
        z = torch.randn(num_samples, generator.latent_dim)
        generated_graphs = generator.sample_graphs(z)
    
    # Setup property checker
    property_library = MSOPropertyLibrary()
    property_checker = property_library.get_property(config['property'])
    
    # Generate theory graphs for comparison
    theory_graphs = property_library.generate_theory_graphs(
        config['property'], num_graphs=500, max_nodes=config['max_nodes']
    )
    
    # Compute metrics
    results = {}
    
    # Property satisfaction
    results['property_satisfaction'] = GraphMetrics.property_satisfaction_rate(
        generated_graphs, property_checker
    )
    
    # Structural diversity
    results['diversity'] = GraphMetrics.structural_diversity(generated_graphs)
    
    # EF-distance metrics
    results['ef_metrics'] = GraphMetrics.ef_distance_distribution(
        generated_graphs, theory_graphs, max_rounds=config.get('max_ef_rounds', 3)
    )
    
    # Distribution comparison with theory
    results['distribution_comparison'] = GraphMetrics.compare_graph_distributions(
        theory_graphs, generated_graphs
    )
    
    # Graph statistics
    results['generated_stats'] = GraphUtils.compute_graph_statistics(generated_graphs)
    results['theory_stats'] = GraphUtils.compute_graph_statistics(theory_graphs)
    
    # Spectral properties
    results['spectral'] = GraphMetrics.spectral_properties(generated_graphs)
    
    return results


def evaluate_model_main():
    """Command-line interface for model evaluation."""
    
    parser = argparse.ArgumentParser(description="Evaluate trained Logical GAN")
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of graphs to generate for evaluation')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for evaluation results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_dir}")
    generator, discriminator, config = load_trained_model(args.model_dir)
    
    # Run evaluation
    print(f"Evaluating with {args.num_samples} samples...")
    eval_results = comprehensive_evaluation(generator, config, args.num_samples)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Property satisfaction rate: {eval_results['property_satisfaction']:.2%}")
    print(f"Average EF-distance: {eval_results['ef_metrics']['avg_ef_distance']:.3f}")
    print(f"Structural diversity score: {eval_results['diversity']['diversity_score']:.3f}")
    print(f"Perfect EF-distance rate: {eval_results['ef_metrics']['perfect_ef_distance_rate']:.2%}")
    
    # Save detailed results
    output_file = args.output_file or f"{args.model_dir}/detailed_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    print(f"Detailed results saved to {output_file}")
    
    # Generate visualizations if requested
    if args.visualize:
        from logical_gans.utils.graph_utils import GraphVisualization
        
        # Generate sample graphs for visualization
        with torch.no_grad():
            z_sample = torch.randn(12, generator.latent_dim)
            sample_graphs = generator.sample_graphs(z_sample)
        
        GraphVisualization.plot_graph_collection(
            sample_graphs, f"Generated {config['property'].title()} Graphs"
        )


if __name__ == "__main__":
    evaluate_model_main()


# === Configuration Files ===

# configs/tree_generation.yaml
TREE_CONFIG = """
# Configuration for tree generation experiment

# Dataset parameters
property: "tree"
max_nodes: 15
theory_size: 1000

# Model architecture
latent_dim: 128
generator_hidden_dims: [256, 512, 1024]
logic_depth: 3
discriminator_hidden_dim: 64
gnn_type: "GCN"

# Training parameters
epochs: 500
batch_size: 32
learning_rate_generator: 0.0002
learning_rate_discriminator: 0.0001
ef_weight: 0.1
max_ef_rounds: 3

# Optimization
adam_beta1: 0.5
adam_beta2: 0.999
weight_decay: 0.0001

# Evaluation
eval_samples: 500
eval_interval: 50
save_interval: 100

# Reproducibility
seed: 42
num_seeds: 5

# Output
log_interval: 50
save_graphs: true
save_checkpoints: true
"""

# configs/connectivity.yaml  
CONNECTIVITY_CONFIG = """
# Configuration for connectivity generation experiment

property: "connectivity"
max_nodes: 20
theory_size: 1500

# Model architecture  
latent_dim: 128
generator_hidden_dims: [256, 512, 768]
logic_depth: 2
discriminator_hidden_dim: 64
gnn_type: "GIN"

# Training parameters
epochs: 800
batch_size: 32
learning_rate_generator: 0.0002
learning_rate_discriminator: 0.0001
ef_weight: 0.15
max_ef_rounds: 2

# Optimization
adam_beta1: 0.5
adam_beta2: 0.999
weight_decay: 0.0001

# Evaluation
eval_samples: 500
eval_interval: 50

# Reproducibility  
seed: 42
"""

# configs/molecular.yaml
MOLECULAR_CONFIG = """
# Configuration for molecular structure generation

property: "molecular_validity"
max_nodes: 25
theory_size: 800

# Model architecture
latent_dim: 128
generator_hidden_dims: [256, 512, 1024, 512]
logic_depth: 4
discriminator_hidden_dim: 96
gnn_type: "GCN"

# Training parameters
epochs: 2000
batch_size: 64
learning_rate_generator: 0.0001
learning_rate_discriminator: 0.00005
ef_weight: 0.2
max_ef_rounds: 4

# Chemical constraints
max_valency: 4
forbidden_substructures: ["excessive_branching", "unstable_rings"]
required_connectivity: true

# Evaluation
eval_samples: 200
chemical_validity_check: true

seed: 42
"""

# configs/security_topology.yaml
SECURITY_CONFIG = """
# Configuration for security topology generation

property: "security_topology"
max_nodes: 30
theory_size: 500
security_level: "high"

# Model architecture
latent_dim: 128
generator_hidden_dims: [256, 512, 1024, 512, 256]
logic_depth: 5
discriminator_hidden_dim: 128
gnn_type: "GCN"

# Training parameters
epochs: 1500
batch_size: 16
learning_rate_generator: 0.0001
learning_rate_discriminator: 0.00008
ef_weight: 0.25
max_ef_rounds: 5

# Security constraints
redundant_paths: true
firewall_protection: true
network_segmentation: true
no_single_failure: true
max_diameter: 6

# Evaluation
eval_samples: 100
security_evaluation: true

seed: 42
"""


def save_configs():
    """Save configuration files to configs directory."""
    configs = {
        'tree_generation.yaml': TREE_CONFIG,
        'connectivity.yaml': CONNECTIVITY_CONFIG, 
        'molecular.yaml': MOLECULAR_CONFIG,
        'security_topology.yaml': SECURITY_CONFIG
    }
    
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)
    
    for filename, content in configs.items():
        with open(config_dir / filename, 'w') as f:
            f.write(content.strip())
    
    print("Configuration files saved to configs/")


# === scripts/generate_graphs.py ===

def generate_graphs_script():
    """Script for generating graphs with trained models."""
    
    script_content = '''#!/usr/bin/env python3
"""
Generate graphs using trained Logical GAN models.
"""

import argparse
import torch
import yaml
import json
from pathlib import Path

from logical_gans.core.logical_gan import GraphGenerator
from logical_gans.utils.graph_utils import GraphUtils, GraphVisualization


def generate_graphs(model_dir: str, num_graphs: int = 100, 
                   output_file: str = None, visualize: bool = False):
    """Generate graphs using trained model."""
    
    model_path = Path(model_dir)
    
    # Load configuration
    with open(model_path / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load generator
    generator = GraphGenerator(
        latent_dim=config.get('latent_dim', 128),
        max_nodes=config.get('max_nodes', 20)
    )
    
    generator.load_state_dict(torch.load(model_path / "generator_checkpoint.pth"))
    generator.eval()
    
    # Generate graphs
    print(f"Generating {num_graphs} graphs...")
    with torch.no_grad():
        z = torch.randn(num_graphs, generator.latent_dim)
        generated_graphs = generator.sample_graphs(z)
    
    print(f"Generated {len(generated_graphs)} graphs")
    
    # Compute statistics
    stats = GraphUtils.compute_graph_statistics(generated_graphs)
    print("\\nGraph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Save graphs
    if output_file:
        GraphUtils.save_graphs(generated_graphs, output_file)
        print(f"Graphs saved to {output_file}")
    
    # Visualize if requested
    if visualize:
        GraphVisualization.plot_graph_collection(
            generated_graphs[:12], 
            f"Generated {config['property'].title()} Graphs"
        )
    
    return generated_graphs


def main():
    parser = argparse.ArgumentParser(description="Generate graphs with trained Logical GAN")
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--num-graphs', type=int, default=100,
                       help='Number of graphs to generate')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for generated graphs')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize generated graphs')
    
    args = parser.parse_args()
    
    generate_graphs(args.model_dir, args.num_graphs, args.output_file, args.visualize)


if __name__ == "__main__":
    main()
'''
    
    return script_content


# === scripts/run_experiments.py ===

def run_experiments_script():
    """Main experimental runner script."""
    
    script_content = '''#!/usr/bin/env python3
"""
Run comprehensive experiments for Logical GANs paper reproduction.
"""

import argparse
import yaml
from pathlib import Path
import time

from logical_gans.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from logical_gans.applications.application_benchmark import ApplicationBenchmark


def run_paper_experiments(experiment_type: str = "all", output_dir: str = "paper_results"):
    """Run experiments to reproduce paper results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create experiment configuration
    config = ExperimentConfig(
        num_epochs=1000,
        batch_size=32,
        num_seeds=5,
        output_dir=output_dir
    )
    
    runner = ExperimentRunner(config)
    
    print(f"Running {experiment_type} experiments...")
    print(f"Results will be saved to {output_path}")
    
    start_time = time.time()
    
    if experiment_type == "all":
        results = runner.run_full_experimental_suite()
    elif experiment_type == "property_satisfaction":
        results = runner.run_property_satisfaction_comparison()
    elif experiment_type == "logic_fragments":
        results = runner.run_logic_fragment_analysis()
    elif experiment_type == "training_dynamics":
        results = runner.run_training_dynamics_analysis()
    elif experiment_type == "complexity":
        results = runner.run_computational_complexity_analysis()
    elif experiment_type == "applications":
        benchmark = ApplicationBenchmark(output_dir=str(output_path / "applications"))
        results = benchmark.run_full_application_suite()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    total_time = time.time() - start_time
    
    print(f"\\nExperiments completed in {total_time:.2f} seconds")
    print(f"Results available in {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Logical GANs experiments")
    parser.add_argument('--experiment', type=str, default="all",
                       choices=["all", "property_satisfaction", "logic_fragments", 
                               "training_dynamics", "complexity", "applications"],
                       help='Type of experiment to run')
    parser.add_argument('--output-dir', type=str, default="paper_results",
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default=None,
                       help='Custom configuration file')
    
    args = parser.parse_args()
    
    results = run_paper_experiments(args.experiment, args.output_dir)
    
    print("\\nExperiment suite completed successfully!")


if __name__ == "__main__":
    main()
'''
    
    return script_content


# === Makefile for easy usage ===

MAKEFILE_CONTENT = """
# Makefile for Logical GANs

.PHONY: install train evaluate experiments clean help

# Default target
help:
	@echo "Logical GANs - Available commands:"
	@echo "  install      - Install the package and dependencies"
	@echo "  train        - Train models with default configurations"
	@echo "  evaluate     - Evaluate trained models"
	@echo "  experiments  - Run full experimental suite"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean up generated files"
	@echo "  notebooks    - Start Jupyter notebook server"

# Installation
install:
	pip install -e .
	python -c "from scripts.train_model import save_configs; save_configs()"

# Training
train:
	@echo "Training tree generation model..."
	python scripts/train_model.py --config configs/tree_generation.yaml --output-dir results/tree
	@echo "Training connectivity model..."
	python scripts/train_model.py --config configs/connectivity.yaml --output-dir results/connectivity

# Evaluation
evaluate:
	@echo "Evaluating tree model..."
	python scripts/evaluate_model.py --model-dir results/tree --num-samples 500 --visualize
	@echo "Evaluating connectivity model..."
	python scripts/evaluate_model.py --model-dir results/connectivity --num-samples 500

# Full experiments (reproduces paper results)
experiments:
	python scripts/run_experiments.py --experiment all --output-dir paper_reproduction

# Applications
applications:
	python scripts/run_experiments.py --experiment applications --output-dir applications_results

# Testing
test:
	pytest tests/ -v

# Clean up
clean:
	rm -rf results/ paper_results/ applications_results/
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf *.egg-info/

# Jupyter notebooks
notebooks:
	jupyter notebook notebooks/

# Quick demo
demo:
	python -c "
from logical_gans.core.logical_gan import LogicalGANTrainer;
config = {'property': 'tree', 'epochs': 50, 'batch_size': 16};
trainer = LogicalGANTrainer(config);
trainer.train();
graphs = trainer.logical_gan.generate(10);
print(f'Generated {len(graphs)} trees');
print(f'All are trees: {all(nx.is_tree(g) for g in graphs)}')
"
"""


# === Docker configuration ===

DOCKERFILE_CONTENT = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    graphviz \\
    graphviz-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p results configs data

# Generate config files
RUN python -c "from scripts.train_model import save_configs; save_configs()"

# Default command
CMD ["python", "scripts/train_model.py", "--config", "configs/tree_generation.yaml", "--verbose"]
"""


def create_all_scripts_and_configs():
    """Create all scripts and configuration files."""
    
    # Create directories
    for dir_name in ['scripts', 'configs']:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Save configuration files
    save_configs()
    
    # Save scripts
    with open('scripts/generate_graphs.py', 'w') as f:
        f.write(generate_graphs_script())
    
    with open('scripts/run_experiments.py', 'w') as f:
        f.write(run_experiments_script())
    
    # Make scripts executable
    import stat
    for script in ['scripts/generate_graphs.py', 'scripts/run_experiments.py']:
        Path(script).chmod(Path(script).stat().st_mode | stat.S_IEXEC)
    
    # Save Makefile
    with open('Makefile', 'w') as f:
        f.write(MAKEFILE_CONTENT.strip())
    
    # Save Dockerfile
    with open('Dockerfile', 'w') as f:
        f.write(DOCKERFILE_CONTENT.strip())
    
    print("All scripts and configuration files created!")
    print("\\nQuick start:")
    print("  make install    # Install package and setup configs")
    print("  make demo       # Run quick demo")
    print("  make train      # Train default models")
    print("  make experiments # Reproduce paper results")


if __name__ == "__main__":
    create_all_scripts_and_configs()
