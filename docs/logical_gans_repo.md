# Logical GANs: Implementation Repository
# Based on "Logical GANs: Adversarial Learning through Ehrenfeucht-Fraïssé Games"

## Repository Structure
```
logical-gans/
├── README.md
├── requirements.txt
├── setup.py
├── logical_gans/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ef_games.py          # Ehrenfeucht-Fraïssé game simulation
│   │   ├── gnn_models.py        # Graph Neural Network architectures
│   │   ├── logical_gan.py       # Main Logical GAN framework
│   │   └── mso_compiler.py      # MSO formula compiler
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── graph_utils.py       # Graph generation and manipulation
│   │   ├── logic_utils.py       # Logic formula utilities
│   │   └── metrics.py           # Evaluation metrics
│   └── properties/
│       ├── __init__.py
│       ├── tree_property.py     # Tree generation
│       ├── connectivity.py      # Connectivity properties
│       ├── planarity.py         # Planarity checking
│       └── bipartiteness.py     # Bipartite graph properties
├── experiments/
│   ├── __init__.py
│   ├── run_experiments.py       # Main experimental runner
│   ├── baseline_comparison.py   # Comparison with baselines
│   ├── ablation_studies.py      # Ablation experiments
│   └── applications/
│       ├── network_security.py  # Security topology generation
│       ├── molecular_design.py  # Molecular structure generation
│       └── formal_verification.py # Counterexample generation
├── tests/
│   ├── test_ef_games.py
│   ├── test_gnn_models.py
│   ├── test_logical_gan.py
│   └── test_mso_compiler.py
├── notebooks/
│   ├── 01_ef_games_tutorial.ipynb
│   ├── 02_logical_gan_demo.ipynb
│   ├── 03_property_generation.ipynb
│   └── 04_applications_showcase.ipynb
├── data/
│   ├── synthetic/              # Generated datasets
│   └── benchmarks/             # Standard graph benchmarks
├── configs/
│   ├── tree_generation.yaml
│   ├── connectivity.yaml
│   ├── bipartite.yaml
│   └── molecular.yaml
└── scripts/
    ├── train_model.py
    ├── evaluate_model.py
    └── generate_graphs.py
```

## Requirements.txt
```
torch>=1.9.0
torch-geometric>=2.0.0
networkx>=2.6
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
tqdm>=4.62.0
pytest>=6.2.0
jupyter>=1.0.0
pandas>=1.3.0
scikit-learn>=1.0.0
plotly>=5.3.0
pygraphviz>=1.7
```

## Setup.py
```python
from setuptools import setup, find_packages

setup(
    name="logical-gans",
    version="0.1.0",
    description="Logical GANs: Adversarial Learning through Ehrenfeucht-Fraïssé Games",
    author="Anonymous",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "networkx>=2.6",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "jupyter>=1.0.0"],
        "viz": ["plotly>=5.3.0", "seaborn>=0.11.0"],
    },
    python_requires=">=3.8",
)
```

## README.md
```markdown
# Logical GANs

Implementation of "Logical GANs: Adversarial Learning through Ehrenfeucht-Fraïssé Games"

## Overview

Logical GANs bridge model theory and adversarial machine learning by casting generative adversarial training as Ehrenfeucht-Fraïssé (EF) games. The framework constrains discriminators to logical expressiveness bounds, creating a principled connection between neural network depth and logical quantifier depth.

## Key Features

- **Theoretical Foundation**: Rigorous connection between GNN expressiveness and first-order logic
- **EF-Distance Metrics**: Novel distance measures based on logical indistinguishability 
- **MSO Property Support**: Monadic Second-Order logic for complex graph properties
- **Convergence Guarantees**: Proven Nash equilibrium convergence
- **Multiple Applications**: Network security, molecular design, formal verification

## Quick Start

```bash
# Install the package
pip install -e .

# Run basic tree generation experiment
python experiments/run_experiments.py --property trees --epochs 100

# Generate graphs with specific properties
python scripts/generate_graphs.py --property connectivity --num_graphs 1000
```

## Core Components

### EF Games (`logical_gans/core/ef_games.py`)
- EF-distance computation
- Game simulation algorithms
- Approximation methods for scalability

### Logical GAN Framework (`logical_gans/core/logical_gan.py`)
- Generator and discriminator architectures
- Logic-constrained loss functions
- Training algorithms with convergence guarantees

### MSO Compiler (`logical_gans/core/mso_compiler.py`)
- MSO formula parsing and compilation
- Efficient property checking
- Tree automaton construction

### Graph Properties (`logical_gans/properties/`)
- Pre-defined MSO formulas for common properties
- Property-specific generators and checkers
- Validation utilities

## Experiments

Reproduce paper results:
```bash
# Table 1: Property satisfaction rates
python experiments/baseline_comparison.py

# Table 2: Logic fragment analysis  
python experiments/ablation_studies.py --study logic_fragments

# Figure 1: Training dynamics
python experiments/run_experiments.py --plot_dynamics
```

## Applications

### Network Security
```python
from logical_gans.applications import SecurityTopologyGAN

gan = SecurityTopologyGAN(
    redundant_paths=True,
    firewall_protection=True
)
secure_networks = gan.generate(num_samples=100)
```

### Molecular Design
```python
from logical_gans.applications import MolecularGAN

mol_gan = MolecularGAN(
    valency_constraints=True,
    planarity_constraints=True
)
molecules = mol_gan.generate_valid_molecules(num_samples=50)
```

## Citation

```bibtex
@article{logical_gans_2025,
  title={Logical GANs: Adversarial Learning through Ehrenfeucht-Fra{\"i}ss{\'e} Games},
  author={Anonymous},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
```
```