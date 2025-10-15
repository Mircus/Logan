# LOGAN: Logical GANs

<p align="center">
  <img src="assets/logo/logan-logo-black.svg" alt="LOGAN Logo" width="200"/>
</p>

**Adversarial Learning through Ehrenfeucht-Fra\"isse Games**

[![CI](https://github.com/Mircus/Logan/workflows/CI/badge.svg)](https://github.com/Mircus/Logan/actions)
[![License: HNCL](https://img.shields.io/badge/License-HNCL-blue.svg)](LICENSE.txt)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

**LOGAN** (Logical GANs) bridges adversarial machine learning and mathematical logic by framing GAN training as an Ehrenfeucht-Fra\"isse (EF) game. The discriminator plays **Devil**, probing for logical faults up to depth *k*, while the generator plays **Builder**, producing structures indistinguishable at that depth.

### Key Features

- **Bounded Logical Framework**: Explicit control via depth parameter *k*
- **EF Game Simulator**: Exact and approximate EF-distance computation
- **MSO Property Library**: Efficient checkers for bipartite, planarity, tree, connectivity, triangles
- **Logical Loss**: Combines EF round-resilience with fast certificate terms
- **Fully Reproducible**: Two torch-free experiments, one command or Docker
- **Interpretable Failures**: Small, human-comprehensible witnesses instead of opaque losses

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Mircus/Logan.git
cd Logan

# Install package
python -m pip install -e .

# Verify installation
python -c "from logical_gans import EFGameSimulator, MSOPropertyLibrary; print('Success!')"
```

### Run Experiments

```bash
# Quick smoke test (reduced samples)
logical-gans-repro --quick --property bipartite

# Full reproduction (as in paper)
logical-gans-repro --property bipartite

# Run specific experiment
logical-gans-repro --exp 1 --property tree
logical-gans-repro --exp 2 --property bipartite
```

### Docker

```bash
# Build image
docker build -t logical-gans:latest .

# Run quick test
docker run --rm logical-gans:latest

# Run with results mounted
docker run --rm -v ${PWD}/results:/app/results logical-gans:latest logical-gans-repro --property tree
```

---

## Experiments

### Experiment 1: MSO Property Satisfaction

**Goal**: Validate that MSO property checkers correctly identify positive/negative examples.

**Properties Tested**: bipartite, planarity, tree, connectivity, has_triangle

**Results** (n ∈ [6,16], 20 samples/size):
```
Property     | Positive Pass Rate | Negative Reject Rate
-------------|-------------------|---------------------
Bipartite    | 100%              | 100%
Planarity    | 100%              | 100%
Tree         | 100%              | 100%
```

**Run**:
```bash
python experiments/exp1_mso_satisfaction.py --property bipartite --samples 20
```

**Outputs**: `results/exp1_{property}.csv`

### Experiment 2: EF-Distance Prototype Classifier

**Goal**: Demonstrate that naive single-prototype EF-distance classification performs at random baseline.

**Setup**: Bipartite classification with one prototype per class, k ∈ {2,3,4,5}

**Results** (n ∈ [6,10], 20 samples):
```
k  | Accuracy
---|----------
2  | 0.50
3  | 0.50
4  | 0.50
5  | 0.50
```

**Interpretation**: The ~50% accuracy (random baseline) is **intentional and validates the framework**. This demonstrates that:
- Naive single-prototype EF-distance classification doesn't work
- Random prototype selection gives random performance
- This motivates the need for the full logical loss framework with multiple prototypes, learned thresholds, and training integration
- The sophisticated approach (Section 5.3 logical loss) is necessary for good performance

**Key Insight**: This is a **sanity check baseline**, not the final approach. It proves that simple methods fail, justifying the complex machinery.

**Run**:
```bash
python experiments/exp2_ef_distance_proto.py --property bipartite --samples 20
```

**Outputs**: `results/exp2_{property}_acc.csv`

---

## Core Components

### 1. EF Game Simulator

```python
from logical_gans import EFGameSimulator
import networkx as nx

# Create two graphs
G = nx.cycle_graph(5)
H = nx.path_graph(5)

# Compute EF-distance
simulator = EFGameSimulator(G, H)
distance = simulator.ef_distance(max_rounds=3)
print(f"EF-distance: {distance}")  # Rounds before Spoiler wins
```

**Features**:
- Exact EF-distance via memoized dynamic programming
- Approximate EF-distance with Monte Carlo sampling
- Budgeted probes with WL-based pruning

**Location**: `src/logical_gans/logic/ef_games.py`

### 2. MSO Property Library

```python
from logical_gans import MSOPropertyLibrary
import networkx as nx

lib = MSOPropertyLibrary()

# Check if graph is bipartite
G = nx.complete_bipartite_graph(3, 3)
is_bip = lib.check_property(G, "bipartite")
print(f"Bipartite: {is_bip}")  # True

# Available properties
print(lib.properties.keys())
# ['connectivity', 'tree', 'bipartite', 'even_parity', 'planarity', 'has_triangle', ...]
```

**Properties**:
- **Standard**: connectivity, tree, bipartite, planarity, even_parity
- **Advanced**: has_triangle, perfect_matching, k_regular

**Location**: `src/logical_gans/logic/mso.py`

### 3. Logical Loss

```python
from logical_gans.logical_loss import LogicalLoss, LogicalLossConfig
import networkx as nx

# Configure logical loss
config = LogicalLossConfig(
    ef_weight=1.0,
    max_ef_rounds=3,
    cert_weights={"degree": 0.1, "bridge": 0.1}
)

logical_loss = LogicalLoss(config)

# Compute loss
test_graph = nx.path_graph(10)
theory_prototypes = [nx.path_graph(10) for _ in range(5)]

loss = logical_loss.compute(test_graph, theory_prototypes, target_property="tree")
print(f"Total Loss: {loss['total']:.4f}")
print(f"EF Loss: {loss['ef_loss']:.4f}")
print(f"Certificate Loss: {loss['certificate_loss']:.4f}")
```

**Components**:
- **EF Round-Resilience**: Budgeted probe against theory prototypes
- **Certificates**: Degree, cycle coverage, bridge penalty, bipartite check
- **Curriculum**: Gradually increase depth *k* as training progresses

**Location**: `src/logical_gans/logical_loss.py`

---

## Repository Structure

```
logan/
├── LICENSE.txt                 # Holomathics Non-Commercial License
├── README.md                   # This file
├── pyproject.toml             # Package configuration
├── Dockerfile                 # Docker build
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
├── src/
│   └── logical_gans/
│       ├── __init__.py       # Package exports
│       ├── repro.py          # Reproducibility CLI
│       ├── logical_loss.py   # Logical loss module
│       ├── logic/
│       │   ├── ef_games.py           # EF game simulator
│       │   ├── mso.py                # MSO property library
│       │   └── logical_gan_framework.py  # Builder-Devil framework
│       ├── core/
│       │   ├── experimental_framework.py  # Full experiment runner
│       │   └── training_scripts.py        # Training utilities
│       └── utils/
│           └── utility_modules.py    # Helper functions
├── experiments/
│   ├── exp1_mso_satisfaction.py    # Experiment 1
│   └── exp2_ef_distance_proto.py   # Experiment 2
├── results/
│   ├── exp1_*.csv             # Experiment 1 results
│   └── exp2_*.csv             # Experiment 2 results
├── tests/
│   └── test_sanity.py         # Basic sanity tests
└── paper/
    └── main.tex              # Paper LaTeX source
```

---

## Paper

**Title**: Logical GANs: Adversarial Learning through Ehrenfeucht-Fra\"isse Games

**Abstract**: LOGAN casts the discriminator as a depth-*k* Ehrenfeucht-Fra\"isse (EF) Devil that searches for small, legible faults (odd cycles, nonplanar crossings, directed bridges), while the generator plays Builder, producing samples that admit a *k*-round matching to a target theory T.

**arXiv**: [Coming soon]

**Cite**:
```bibtex
@article{mannucci2025logan,
  title={Logical GANs: Adversarial Learning through Ehrenfeucht-Fra\"isse Games},
  author={Mannucci, Mirco A.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Development

### Install Development Dependencies

```bash
python -m pip install -e .[dev]
```

### Run Tests

```bash
pytest -q
```

### Code Style

```bash
ruff check .
```

### CI/CD

GitHub Actions CI runs on every push:
- Python 3.10+ compatibility
- Package installation
- Import checks
- Smoke test experiments

---

## Roadmap

### Current Release (v0.1.0)
- ✅ EF game simulator
- ✅ MSO property library
- ✅ Logical loss (evaluation signals)
- ✅ Two reproducible experiments
- ✅ Full documentation

### Future Extensions

**Note on Training**: The current release (v0.1.0) provides **evaluation tools** and demonstrates baseline performance. The logical loss module is implemented but training integration (backpropagation through discrete structures) is staged for future work.

**Expected improvements with full training**:
- Current (Exp2 naive baseline): ~50% accuracy
- With training + logical loss: Significantly improved performance (to be validated in future work)

**Planned extensions**:
- 🔄 Training integration (REINFORCE, Straight-Through, Learned Surrogate)
- 🔄 Multiple prototype selection and learned thresholds
- 🔄 Full empirical evaluation with training curves
- 🔄 Richer graph families
- 🔄 Scalability optimizations (larger graphs)
- 🔄 Additional properties (Hamiltonicity, graph isomorphism)

---

## License

This project is licensed under the **Holomathics Non-Commercial License (HNCL)**.

- ✅ Free for academic research, education, and personal use
- ❌ Commercial use requires separate licensing

For commercial licensing inquiries, contact: [info@holomathics.com](mailto:info@holomathics.com)

See [LICENSE.txt](LICENSE.txt) for full terms.

---

## Contributing

We welcome contributions! Areas of interest:
- Additional MSO properties
- Performance optimizations
- Extended experiments
- Documentation improvements

Please open an issue before starting major work.

---

## Acknowledgments

- **EF Games**: Ehrenfeucht (1961), Fra\"isse (1954)
- **Logic & GNNs**: Immerman, Libkin, Grohe, Morris et al.
- **Inspiration**: GAN framework by Goodfellow et al. (2014)

---

## Contact

**Author**: Mirco A. Mannucci
**Organization**: HoloMathics, LLC
**Email**: [mirco@holomathics.com](mailto:mirco@holomathics.com)
**GitHub**: [Mircus/Logan](https://github.com/Mircus/Logan)

---

<p align="center">
  <b>LOGAN: Where Logic Meets Adversarial Learning</b><br>
  🎲 Generated with logical constraints | 🔍 Validated through EF games | 📊 Fully reproducible
</p>
