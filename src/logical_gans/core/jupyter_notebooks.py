"""
Jupyter Notebooks for Logical GANs
Interactive demonstrations and tutorials
"""

# === notebooks/01_ef_games_tutorial.ipynb ===

EF_GAMES_NOTEBOOK = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ehrenfeucht-Fraïssé Games Tutorial\\n",
    "\\n",
    "This notebook introduces Ehrenfeucht-Fraïssé (EF) games and demonstrates their connection to logical equivalence of graphs.\\n",
    "\\n",
    "## What are EF Games?\\n",
    "\\n",
    "EF games are a fundamental tool in model theory that characterize when two structures satisfy the same first-order sentences up to a certain quantifier depth."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import networkx as nx\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "from logical_gans.core.ef_games import EFGameSimulator, ApproximateEFDistance\\n",
    "from logical_gans.utils.graph_utils import GraphVisualization\\n",
    "\\n",
    "# Set up plotting\\n",
    "%matplotlib inline\\n",
    "plt.style.use('default')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Creating Example Graphs\\n",
    "\\n",
    "Let's create some example graphs to understand EF games:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create example graphs\\n",
    "path_graph = nx.path_graph(4)  # Linear path: 0-1-2-3\\n",
    "star_graph = nx.star_graph(3)  # Star: center connected to 3 leaves\\n",
    "cycle_graph = nx.cycle_graph(4)  # 4-cycle: 0-1-2-3-0\\n",
    "complete_graph = nx.complete_graph(4)  # Complete graph on 4 vertices\\n",
    "\\n",
    "graphs = {\\n",
    "    'Path': path_graph,\\n",
    "    'Star': star_graph, \\n",
    "    'Cycle': cycle_graph,\\n",
    "    'Complete': complete_graph\\n",
    "}\\n",
    "\\n",
    "# Visualize the graphs\\n",
    "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\\n",
    "axes = axes.flatten()\\n",
    "\\n",
    "for i, (name, graph) in enumerate(graphs.items()):\\n",
    "    pos = nx.spring_layout(graph, seed=42)\\n",
    "    nx.draw(graph, pos, ax=axes[i], with_labels=True, \\n",
    "           node_color='lightblue', node_size=500, font_weight='bold')\\n",
    "    axes[i].set_title(f'{name} Graph')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Computing EF-Distances\\n",
    "\\n",
    "Now let's compute EF-distances between these graphs:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute pairwise EF-distances\\n",
    "graph_names = list(graphs.keys())\\n",
    "n_graphs = len(graphs)\\n",
    "ef_distances = np.zeros((n_graphs, n_graphs))\\n",
    "\\n",
    "print('Computing EF-distances...')\\n",
    "for i, (name1, graph1) in enumerate(graphs.items()):\\n",
    "    for j, (name2, graph2) in enumerate(graphs.items()):\\n",
    "        if i <= j:\\n",
    "            simulator = EFGameSimulator(graph1, graph2)\\n",
    "            distance = simulator.ef_distance(max_rounds=5)\\n",
    "            ef_distances[i, j] = distance\\n",
    "            ef_distances[j, i] = distance  # Symmetric\\n",
    "            \\n",
    "            if i != j:\\n",
    "                print(f'EF-distance({name1}, {name2}) = {distance}')\\n",
    "\\n",
    "print(f'\\\\nEF-Distance Matrix:')\\n",
    "print('     ', ' '.join(f'{name:>6}' for name in graph_names))\\n",
    "for i, name in enumerate(graph_names):\\n",
    "    print(f'{name:>4}:', ' '.join(f'{ef_distances[i,j]:6.0f}' for j in range(n_graphs)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Understanding the Results\\n",
    "\\n",
    "- **EF-distance = 0**: Graphs are indistinguishable by first-order logic\\n",
    "- **EF-distance = k**: Graphs can be distinguished in k rounds of the EF game\\n",
    "\\n",
    "Let's analyze what these distances mean:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze graph properties that might explain EF-distances\\n",
    "properties = {}\\n",
    "\\n",
    "for name, graph in graphs.items():\\n",
    "    props = {\\n",
    "        'nodes': len(graph),\\n",
    "        'edges': graph.number_of_edges(),\\n",
    "        'degree_sequence': sorted([d for _, d in graph.degree()]),\\n",
    "        'max_degree': max(dict(graph.degree()).values()),\\n",
    "        'triangles': sum(nx.triangles(graph).values()) // 3,\\n",
    "        'diameter': nx.diameter(graph) if nx.is_connected(graph) else float('inf')\\n",
    "    }\\n",
    "    properties[name] = props\\n",
    "\\n",
    "print('Graph Properties:')\\n",
    "print('=' * 60)\\n",
    "for name, props in properties.items():\\n",
    "    print(f'{name:>8}: {props}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Interactive EF Game Simulation\\n",
    "\\n",
    "Let's simulate an EF game step by step:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def simulate_ef_game_interactive(graph1, graph2, rounds=3):\\n",
    "    \\"\\"\\"Simulate EF game with detailed output.\\"\\"\\"\\n",
    "    \\n",
    "    simulator = EFGameSimulator(graph1, graph2)\\n",
    "    \\n",
    "    print(f'EF Game: Graph A vs Graph B for {rounds} rounds')\\n",
    "    print(f'Graph A: {len(graph1)} nodes, {graph1.number_of_edges()} edges')\\n",
    "    print(f'Graph B: {len(graph2)} nodes, {graph2.number_of_edges()} edges')\\n",
    "    print('=' * 50)\\n",
    "    \\n",
    "    for k in range(1, rounds + 1):\\n",
    "        can_duplicate = simulator.duplicator_wins(k)\\n",
    "        result = \\"Duplicator WINS\\" if can_duplicate else \\"Spoiler WINS\\"\\n",
    "        print(f'Round {k}: {result}')\\n",
    "        \\n",
    "        if not can_duplicate:\\n",
    "            print(f'  → EF-distance = {k}')\\n",
    "            break\\n",
    "    else:\\n",
    "        print(f'  → EF-distance ≥ {rounds}')\\n",
    "\\n",
    "# Example: Path vs Star\\n",
    "print('EF Game: Path vs Star')\\n",
    "simulate_ef_game_interactive(path_graph, star_graph, rounds=4)\\n",
    "\\n",
    "print('\\\\n' + '='*50 + '\\\\n')\\n",
    "\\n",
    "# Example: Path vs Cycle  \\n",
    "print('EF Game: Path vs Cycle')\\n",
    "simulate_ef_game_interactive(path_graph, cycle_graph, rounds=4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Approximate EF-Distance for Larger Graphs\\n",
    "\\n",
    "For larger graphs, exact EF-distance computation becomes expensive. Let's use the approximate method:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create larger random graphs\\n",
    "np.random.seed(42)\\n",
    "large_graph1 = nx.erdos_renyi_graph(20, 0.3)\\n",
    "large_graph2 = nx.erdos_renyi_graph(20, 0.3)\\n",
    "large_graph3 = nx.random_tree(20)\\n",
    "\\n",
    "large_graphs = [large_graph1, large_graph2, large_graph3]\\n",
    "graph_types = ['Random 1', 'Random 2', 'Tree']\\n",
    "\\n",
    "# Compute approximate EF-distances\\n",
    "approx_ef = ApproximateEFDistance(num_samples=500)\\n",
    "\\n",
    "print('Approximate EF-Distances for Larger Graphs:')\\n",
    "print('=' * 45)\\n",
    "\\n",
    "for i in range(len(large_graphs)):\\n",
    "    for j in range(i + 1, len(large_graphs)):\\n",
    "        distance = approx_ef.compute_distance(large_graphs[i], large_graphs[j])\\n",
    "        print(f'{graph_types[i]} vs {graph_types[j]}: {distance:.3f}')\\n",
    "\\n",
    "# Visualize the large graphs\\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\\n",
    "\\n",
    "for i, (graph, graph_type) in enumerate(zip(large_graphs, graph_types)):\\n",
    "    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)\\n",
    "    nx.draw(graph, pos, ax=axes[i], node_size=100, \\n",
    "           node_color='lightcoral', with_labels=False)\\n",
    "    axes[i].set_title(f'{graph_type}\\\\n({len(graph)} nodes)')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Connection to First-Order Logic\\n",
    "\\n",
    "The EF-distance corresponds to the minimum quantifier depth needed to distinguish the graphs in first-order logic.\\n",
    "\\n",
    "For example:\\n",
    "- **Distance 1**: Can be distinguished by a formula with 1 quantifier (∃x φ(x))\\n",
    "- **Distance 2**: Requires 2 quantifiers (∃x ∀y φ(x,y))\\n",
    "- **Distance k**: Requires k quantifier alternations\\n",
    "\\n",
    "This is the foundation of our Logical GANs approach!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exercises\\n",
    "\\n",
    "Try these exercises to deepen your understanding:\\n",
    "\\n",
    "1. Create two graphs that you think should have EF-distance 0\\n",
    "2. Create two graphs that should have EF-distance exactly 2\\n",
    "3. Investigate how EF-distance relates to graph isomorphism"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your experiments here!\\n",
    "\\n",
    "# Exercise 1: EF-distance 0\\n",
    "# Hint: Try two graphs with the same \\"local structure\\"\\n",
    "\\n",
    "# Exercise 2: EF-distance exactly 2  \\n",
    "# Hint: Think about what can be expressed with 2 quantifiers\\n",
    "\\n",
    "# Exercise 3: Isomorphism vs EF-equivalence\\n",
    "# Create isomorphic graphs and non-isomorphic but EF-equivalent graphs"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python", 
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

# === notebooks/02_logical_gan_demo.ipynb ===

LOGICAL_GAN_DEMO = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logical GAN Demo\\n",
    "\\n",
    "This notebook demonstrates the core Logical GAN framework, showing how to train a generator that produces graphs satisfying specific logical properties."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\\n",
    "import networkx as nx\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "from tqdm import tqdm\\n",
    "\\n",
    "from logical_gans.core.logical_gan import LogicalGANTrainer, LogicalGAN\\n",
    "from logical_gans.core.mso_compiler import MSOPropertyLibrary\\n",
    "from logical_gans.utils.graph_utils import GraphVisualization\\n",
    "from logical_gans.utils.metrics import GraphMetrics\\n",
    "\\n",
    "%matplotlib inline\\n",
    "torch.manual_seed(42)\\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setting up the Experiment\\n",
    "\\n",
    "Let's start with a simple example: generating trees."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuration for tree generation\\n",
    "config = {\\n",
    "    'property': 'tree',\\n",
    "    'max_nodes': 12,\\n",
    "    'latent_dim': 64,\\n",
    "    'logic_depth': 3,\\n",
    "    'ef_weight': 0.1,\\n",
    "    'epochs': 200,\\n",
    "    'batch_size': 16,\\n",
    "    'theory_size': 500\\n",
    "}\\n",
    "\\n",
    "print('Training Configuration:')\\n",
    "for key, value in config.items():\\n",
    "    print(f'  {key}: {value}')\\n",
    "\\n",
    "# Initialize trainer\\n",
    "trainer = LogicalGANTrainer(config)\\n",
    "print(f'\\\\nInitialized trainer with {len(trainer.theory_graphs)} theory graphs')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizing the Theory Graphs\\n",
    "\\n",
    "Let's look at some example trees from our theory:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show some theory graphs\\n",
    "sample_theory_graphs = trainer.theory_graphs[:8]\\n",
    "\\n",
    "GraphVisualization.plot_graph_collection(\\n",
    "    sample_theory_graphs, \\n",
    "    title=\\"Sample Theory Graphs (Trees)\\",\\n",
    "    max_graphs=8\\n",
    ")\\n",
    "\\n",
    "# Verify they are all trees\\n",
    "all_trees = all(nx.is_tree(g) for g in sample_theory_graphs)\\n",
    "print(f'All sampled graphs are trees: {all_trees}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training the Logical GAN\\n",
    "\\n",
    "Now let's train our Logical GAN:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the model\\n",
    "print('Starting training...')\\n",
    "training_history = trainer.train()\\n",
    "print('Training completed!')\\n",
    "\\n",
    "# Plot training curves\\n",
    "GraphVisualization.plot_training_curves(training_history)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluating the Trained Model\\n",
    "\\n",
    "Let's generate graphs and evaluate how well they satisfy the tree property:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate graphs with trained model\\n",
    "generated_graphs = trainer.logical_gan.generate(num_samples=100)\\n",
    "\\n",
    "print(f'Generated {len(generated_graphs)} graphs')\\n",
    "\\n",
    "# Evaluate the results\\n",
    "evaluation_results = trainer.evaluate()\\n",
    "\\n",
    "print('\\\\nEvaluation Results:')\\n",
    "print(f'  Property satisfaction rate: {evaluation_results[\\"property_satisfaction_rate\\"]:.2%}')\\n",
    "print(f'  Average EF-distance: {evaluation_results[\\"average_ef_distance\\"]:.3f}')\\n",
    "print(f'  Perfect EF-distance rate: {evaluation_results[\\"perfect_ef_distance_rate\\"]:.2%}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizing Generated Graphs\\n",
    "\\n",
    "Let's look at some of the generated graphs:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show generated graphs\\n",
    "GraphVisualization.plot_graph_collection(\\n",
    "    generated_graphs[:12], \\n",
    "    title=\\"Generated Graphs\\",\\n",
    "    max_graphs=12\\n",
    ")\\n",
    "\\n",
    "# Check how many are actually trees\\n",
    "tree_count = sum(1 for g in generated_graphs[:12] if nx.is_tree(g))\\n",
    "print(f'Trees in sample: {tree_count}/12 = {tree_count/12:.1%}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Comparison with Baselines\\n",
    "\\n",
    "How does our Logical GAN compare to random generation?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate random graphs for comparison\\n",
    "random_graphs = []\\n",
    "for _ in range(100):\\n",
    "    n = np.random.randint(3, config['max_nodes'] + 1)\\n",
    "    p = np.random.uniform(0.1, 0.7)\\n",
    "    graph = nx.erdos_renyi_graph(n, p)\\n",
    "    random_graphs.append(graph)\\n",
    "\\n",
    "# Evaluate random graphs\\n",
    "property_library = MSOPropertyLibrary()\\n",
    "tree_checker = property_library.get_property('tree')\\n",
    "\\n",
    "random_tree_rate = np.mean([tree_checker.check(g) for g in random_graphs])\\n",
    "logical_tree_rate = evaluation_results['property_satisfaction_rate']\\n",
    "\\n",
    "print('Tree Generation Comparison:')\\n",
    "print(f'  Random graphs: {random_tree_rate:.2%}')\\n",
    "print(f'  Logical GAN: {logical_tree_rate:.2%}')\\n",
    "print(f'  Improvement: {(logical_tree_rate - random_tree_rate):.2%}')\\n",
    "\\n",
    "# Visualize comparison\\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\\n",
    "\\n",
    "methods = ['Random', 'Logical GAN']\\n",
    "rates = [random_tree_rate, logical_tree_rate]\\n",
    "colors = ['red', 'blue']\\n",
    "\\n",
    "bars = ax1.bar(methods, rates, color=colors, alpha=0.7)\\n",
    "ax1.set_ylabel('Tree Satisfaction Rate')\\n",
    "ax1.set_title('Tree Generation Performance')\\n",
    "ax1.set_ylim(0, 1)\\n",
    "\\n",
    "# Add value labels on bars\\n",
    "for bar, rate in zip(bars, rates):\\n",
    "    height = bar.get_height()\\n",
    "    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,\\n",
    "             f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')\\n",
    "\\n",
    "# Show sample of each type\\n",
    "GraphVisualization.plot_graph_collection(\\n",
    "    random_graphs[:6], \\n",
    "    title=\\"Random Graphs (Baseline)\\"\\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analyzing Graph Properties\\n",
    "\\n",
    "Let's analyze the structural properties of generated vs. theory graphs:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from logical_gans.utils.graph_utils import GraphUtils\\n",
    "\\n",
    "# Compute statistics for different graph sets\\n",
    "theory_stats = GraphUtils.compute_graph_statistics(trainer.theory_graphs[:100])\\n",
    "generated_stats = GraphUtils.compute_graph_statistics(generated_graphs)\\n",
    "random_stats = GraphUtils.compute_graph_statistics(random_graphs)\\n",
    "\\n",
    "# Compare statistics\\n",
    "stats_comparison = {}\\n",
    "common_keys = ['avg_nodes', 'avg_edges', 'avg_density', 'connected_fraction']\\n",
    "\\n",
    "for key in common_keys:\\n",
    "    stats_comparison[key] = {\\n",
    "        'Theory': theory_stats.get(key, 0),\\n",
    "        'Generated': generated_stats.get(key, 0),\\n",
    "        'Random': random_stats.get(key, 0)\\n",
    "    }\\n",
    "\\n",
    "print('Graph Statistics Comparison:')\\n",
    "print('=' * 50)\\n",
    "print(f'{\\"Metric\\":>15} {\\"Theory\\":>10} {\\"Generated\\":>10} {\\"Random\\":>10}')\\n",
    "print('-' * 50)\\n",
    "\\n",
    "for metric, values in stats_comparison.items():\\n",
    "    theory_val = values['Theory']\\n",
    "    generated_val = values['Generated']\\n",
    "    random_val = values['Random']\\n",
    "    print(f'{metric:>15} {theory_val:>10.3f} {generated_val:>10.3f} {random_val:>10.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing Different Properties\\n",
    "\\n",
    "Let's quickly test the framework on other graph properties:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test connectivity property\\n",
    "connectivity_config = config.copy()\\n",
    "connectivity_config['property'] = 'connectivity'\\n",
    "connectivity_config['epochs'] = 100  # Shorter for demo\\n",
    "\\n",
    "print('Training for connectivity property...')\\n",
    "connectivity_trainer = LogicalGANTrainer(connectivity_config)\\n",
    "connectivity_history = connectivity_trainer.train()\\n",
    "\\n",
    "# Generate and evaluate\\n",
    "connected_graphs = connectivity_trainer.logical_gan.generate(num_samples=50)\\n",
    "connectivity_results = connectivity_trainer.evaluate()\\n",
    "\\n",
    "print(f'Connectivity satisfaction rate: {connectivity_results[\\"property_satisfaction_rate\\"]:.2%}')\\n",
    "\\n",
    "# Show some connected graphs\\n",
    "GraphVisualization.plot_graph_collection(\\n",
    "    connected_graphs[:6], \\n",
    "    title=\\"Generated Connected Graphs\\"\\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exploring the Latent Space\\n",
    "\\n",
    "Let's explore how the latent space is organized:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample from different regions of latent space\\n",
    "trainer.generator.eval()\\n",
    "\\n",
    "# Generate graphs from different latent vectors\\n",
    "with torch.no_grad():\\n",
    "    # Near origin\\n",
    "    z_center = torch.zeros(4, trainer.generator.latent_dim)\\n",
    "    graphs_center = trainer.generator.sample_graphs(z_center)\\n",
    "    \\n",
    "    # High magnitude\\n",
    "    z_extreme = torch.randn(4, trainer.generator.latent_dim) * 3\\n",
    "    graphs_extreme = trainer.generator.sample_graphs(z_extreme)\\n",
    "    \\n",
    "    # Random normal\\n",
    "    z_normal = torch.randn(4, trainer.generator.latent_dim)\\n",
    "    graphs_normal = trainer.generator.sample_graphs(z_normal)\\n",
    "\\n",
    "# Visualize different latent regions\\n",
    "fig, axes = plt.subplots(3, 4, figsize=(16, 12))\\n",
    "graph_sets = [graphs_center, graphs_normal, graphs_extreme]\\n",
    "set_names = ['Near Origin', 'Normal', 'High Magnitude']\\n",
    "\\n",
    "for i, (graphs, name) in enumerate(zip(graph_sets, set_names)):\\n",
    "    for j, graph in enumerate(graphs[:4]):\\n",
    "        ax = axes[i, j]\\n",
    "        if len(graph) > 0:\\n",
    "            pos = nx.spring_layout(graph, seed=42)\\n",
    "            nx.draw(graph, pos, ax=ax, with_labels=True,\\n",
    "                   node_color='lightgreen', node_size=200, font_size=8)\\n",
    "        \\n",
    "        is_tree = nx.is_tree(graph)\\n",
    "        ax.set_title(f'{name}\\\\n{len(graph)} nodes, Tree: {is_tree}')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\\n",
    "\\n",
    "In this demo, we've seen how Logical GANs can:\\n",
    "\\n",
    "1. **Learn to generate graphs with specific properties** (trees, connectivity)\\n",
    "2. **Outperform random generation** significantly\\n",
    "3. **Maintain good structural diversity** while satisfying logical constraints\\n",
    "4. **Provide interpretable training dynamics** through EF-distance metrics\\n",
    "\\n",
    "The key insight is that by constraining the discriminator's expressiveness to match logical fragments, we can precisely control what the generator learns to produce."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

# === notebooks/03_property_generation.ipynb ===

PROPERTY_GENERATION_NOTEBOOK = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Property-Specific Graph Generation\\n",
    "\\n",
    "This notebook explores generating graphs with different MSO-definable properties using Logical GANs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\\n",
    "import networkx as nx\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from collections import Counter\\n",
    "\\n",
    "from logical_gans.core.logical_gan import LogicalGANTrainer\\n",
    "from logical_gans.core.mso_compiler import MSOPropertyLibrary, StandardMSOProperties\\n",
    "from logical_gans.utils.graph_utils import GraphVisualization, GraphUtils\\n",
    "from logical_gans.utils.metrics import GraphMetrics\\n",
    "\\n",
    "%matplotlib inline\\n",
    "plt.style.use('default')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Available MSO Properties\\n",
    "\\n",
    "Let's explore the different graph properties we can generate:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize property library\\n",
    "property_library = MSOPropertyLibrary()\\n",
    "\\n",
    "# Available properties\\n",
    "available_properties = [\\n",
    "    'tree', 'connectivity', 'bipartite', 'planarity', \\n",
    "    'even_parity', 'has_triangle', '3_regular'\\n",
    "]\\n",
    "\\n",
    "print('Available MSO Properties:')\\n",
    "print('=' * 30)\\n",
    "for prop in available_properties:\\n",
    "    checker = property_library.get_property(prop)\\n",
    "    print(f'- {prop}: {checker.name}')\\n",
    "\\n",
    "# Generate sample graphs for each property\\n",
    "sample_graphs = {}\\n",
    "for prop in available_properties[:5]:  # Limit for demo\\n",
    "    graphs = property_library.generate_theory_graphs(prop, num_graphs=10, max_nodes=8)\\n",
    "    sample_graphs[prop] = graphs\\n",
    "    print(f'Generated {len(graphs)} sample {prop} graphs')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizing Different Property Classes\\n",
    "\\n",
    "Let's visualize examples of each property:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create visualization grid\\n",
    "properties_to_show = ['tree', 'connectivity', 'bipartite', 'planarity']\\n",
    "fig, axes = plt.subplots(4, 3, figsize=(12, 16))\\n",
    "\\n",
    "for i, prop in enumerate(properties_to_show):\\n",
    "    graphs = sample_graphs[prop][:3]\\n",
    "    \\n",
    "    for j, graph in enumerate(graphs):\\n",
    "        ax = axes[i, j]\\n",
    "        \\n",
    "        # Special layouts for different properties\\n",
    "        if prop == 'bipartite' and nx.is_bipartite(graph):\\n",
    "            pos = nx.bipartite_layout(graph, nx.bipartite.sets(graph)[0])\\n",
    "        elif prop == 'tree' and nx.is_tree(graph):\\n",
    "            pos = nx.spring_layout(graph, k=2, iterations=50)\\n",
    "        else:\\n",
    "            pos = nx.spring_layout(graph, seed=42)\\n",
    "        \\n",
    "        # Color nodes based on property\\n",
    "        if prop == 'bipartite' and nx.is_bipartite(graph):\\n",
    "            node_colors = ['lightblue' if node in nx.bipartite.sets(graph)[0] \\n",
    "                          else 'lightcoral' for node in graph.nodes()]\\n",
    "        else:\\n",
    "            node_colors = 'lightgreen'\\n",
    "        \\n",
    "        nx.draw(graph, pos, ax=ax, with_labels=True, \\n",
    "               node_color=node_colors, node_size=300, font_size=8)\\n",
    "        \\n",
    "        ax.set_title(f'{prop.title()} #{j+1}\\\\n'\\n",
    "                    f'{len(graph)} nodes, {graph.number_of_edges()} edges')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Multi-Property Generation Experiment\\n",
    "\\n",
    "Let's train Logical GANs for multiple properties and compare their performance:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuration for multi-property experiment\\n",
    "base_config = {\\n",
    "    'max_nodes': 10,\\n",
    "    'latent_dim': 64,\\n",
    "    'epochs': 150,\\n",
    "    'batch_size': 16,\\n",
    "    'theory_size': 300,\\n",
    "    'ef_weight': 0.1\\n",
    "}\\n",
    "\\n",
    "properties_to_test = ['tree', 'connectivity', 'bipartite']\\n",
    "trained_models = {}\\n",
    "results = {}\\n",
    "\\n",
    "print('Training Logical GANs for multiple properties...')\\n",
    "print('=' * 50)\\n",
    "\\n",
    "for prop in properties_to_test:\\n",
    "    print(f'\\\\nTraining {prop} generator...')\\n",
    "    \\n",
    "    # Setup configuration\\n",
    "    config = base_config.copy()\\n",
    "    config['property'] = prop\\n",
    "    \\n",
    "    # Adjust logic depth based on property complexity\\n",
    "    if prop == 'tree':\\n",
    "        config['logic_depth'] = 3\\n",
    "    elif prop == 'connectivity':\\n",
    "        config['logic_depth'] = 2\\n",
    "    else:\\n",
    "        config['logic_depth'] = 4\\n",
    "    \\n",
    "    # Train model\\n",
    "    trainer = LogicalGANTrainer(config)\\n",
    "    training_history = trainer.train()\\n",
    "    \\n",
    "    # Store results\\n",
    "    trained_models[prop] = trainer\\n",
    "    evaluation = trainer.evaluate()\\n",
    "    results[prop] = {\\n",
    "        'training_history': training_history,\\n",
    "        'evaluation': evaluation,\\n",
    "        'config': config\\n",
    "    }\\n",
    "    \\n",
    "    print(f'  Property satisfaction: {evaluation[\\"property_satisfaction_rate\\"]:.2%}')\\n",
    "    print(f'  Average EF-distance: {evaluation[\\"average_ef_distance\\"]:.3f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Performance Comparison\\n",
    "\\n",
    "Let's compare the performance across different properties:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract performance metrics\\n",
    "performance_data = []\\n",
    "for prop, result in results.items():\\n",
    "    eval_result = result['evaluation']\\n",
    "    performance_data.append({\\n",
    "        'Property': prop.title(),\\n",
    "        'Satisfaction_Rate': eval_result['property_satisfaction_rate'],\\n",
    "        'EF_Distance': eval_result['average_ef_distance'],\\n",
    "        'Perfect_EF_Rate': eval_result['perfect_ef_distance_rate']\\n",
    "    })\\n",
    "\\n",
    "# Create comparison plots\\n",
    "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))\\n",
    "\\n",
    "properties = [d['Property'] for d in performance_data]\\n",
    "satisfaction_rates = [d['Satisfaction_Rate'] for d in performance_data]\\n",
    "ef_distances = [d['EF_Distance'] for d in performance_data]\\n",
    "perfect_ef_rates = [d['Perfect_EF_Rate'] for d in performance_data]\\n",
    "\\n",
    "# Property satisfaction rates\\n",
    "bars1 = ax1.bar(properties, satisfaction_rates, color='skyblue', alpha=0.7)\\n",
    "ax1.set_ylabel('Property Satisfaction Rate')\\n",
    "ax1.set_title('Property Satisfaction Rates')\\n",
    "ax1.set_ylim(0, 1)\\n",
    "for bar, rate in zip(bars1, satisfaction_rates):\\n",
    "    height = bar.get_height()\\n",
    "    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,\\n",
    "             f'{rate:.2%}', ha='center', va='bottom')\\n",
    "\\n",
    "# EF-distances\\n",
    "bars2 = ax2.bar(properties, ef_distances, color='lightcoral', alpha=0.7)\\n",
    "ax2.set_ylabel('Average EF-Distance')\\n",
    "ax2.set_title('Average EF-Distances')\\n",
    "for bar, dist in zip(bars2, ef_distances):\\n",
    "    height = bar.get_height()\\n",
    "    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,\\n",
    "             f'{dist:.3f}', ha='center', va='bottom')\\n",
    "\\n",
    "# Training curves comparison\\n",
    "for prop, result in results.items():\\n",
    "    history = result['training_history']\\n",
    "    epochs = range(len(history['property_satisfaction_rates']))\\n",
    "    ax3.plot(epochs, history['property_satisfaction_rates'], label=prop.title(), linewidth=2)\\n",
    "\\n",
    "ax3.set_xlabel('Epoch')\\n",
    "ax3.set_ylabel('Property Satisfaction Rate')\\n",
    "ax3.set_title('Training Progress Comparison')\\n",
    "ax3.legend()\\n",
    "ax3.grid(True, alpha=0.3)\\n",
    "\\n",
    "# EF-distance evolution\\n",
    "for prop, result in results.items():\\n",
    "    history = result['training_history']\\n",
    "    epochs = range(len(history['ef_distances']))\\n",
    "    ax4.plot(epochs, history['ef_distances'], label=prop.title(), linewidth=2)\\n",
    "\\n",
    "ax4.set_xlabel('Epoch')\\n",
    "ax4.set_ylabel('Average EF-Distance')\\n",
    "ax4.set_title('EF-Distance Evolution')\\n",
    "ax4.legend()\\n",
    "ax4.grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Generated Graph Analysis\\n",
    "\\n",
    "Let's analyze the characteristics of graphs generated for each property:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate graphs from each trained model\\n",
    "generated_sets = {}\\n",
    "\\n",
    "for prop, trainer in trained_models.items():\\n",
    "    generated_graphs = trainer.logical_gan.generate(num_samples=50)\\n",
    "    generated_sets[prop] = generated_graphs\\n",
    "    \\n",
    "    print(f'{prop.title()} graphs generated: {len(generated_graphs)}')\\n",
    "\\n",
    "# Analyze graph statistics\\n",
    "stats_comparison = {}\\n",
    "for prop, graphs in generated_sets.items():\\n",
    "    stats = GraphUtils.compute_graph_statistics(graphs)\\n",
    "    stats_comparison[prop] = stats\\n",
    "\\n",
    "# Create comparison table\\n",
    "import pandas as pd\\n",
    "\\n",
    "stats_df = pd.DataFrame(stats_comparison).T\\n",
    "stats_df = stats_df[['avg_nodes', 'avg_edges', 'avg_density', 'avg_clustering', 'connected_fraction']]\\n",
    "stats_df.columns = ['Avg Nodes', 'Avg Edges', 'Avg Density', 'Avg Clustering', 'Connected %']\\n",
    "\\n",
    "print('\\\\nGenerated Graph Statistics:')\\n",
    "print('=' * 60)\\n",
    "print(stats_df.round(3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizing Generated Graphs by Property\\n",
    "\\n",
    "Let's see examples of generated graphs for each property:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize generated graphs for each property\\n",
    "for prop, graphs in generated_sets.items():\\n",
    "    # Filter to get good examples\\n",
    "    property_checker = property_library.get_property(prop)\\n",
    "    good_examples = [g for g in graphs if property_checker.check(g)][:6]\\n",
    "    \\n",
    "    if good_examples:\\n",
    "        GraphVisualization.plot_graph_collection(\\n",
    "            good_examples,\\n",
    "            title=f'Generated {prop.title()} Graphs ({len(good_examples)}/6 satisfy property)',\\n",
    "            max_graphs=6\\n",
    "        )\\n",
    "    else:\\n",
    "        print(f'No valid {prop} graphs found in sample')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Diversity Analysis\\n",
    "\\n",
    "How diverse are the generated graphs within each property class?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute diversity metrics for each property\\n",
    "diversity_results = {}\\n",
    "\\n",
    "for prop, graphs in generated_sets.items():\\n",
    "    diversity = GraphMetrics.structural_diversity(graphs)\\n",
    "    diversity_results[prop] = diversity\\n",
    "\\n",
    "# Create diversity comparison\\n",
    "diversity_df = pd.DataFrame(diversity_results).T\\n",
    "diversity_df = diversity_df[['diversity_score', 'degree_diversity', 'size_entropy']]\\n",
    "diversity_df.columns = ['Overall Diversity', 'Degree Diversity', 'Size Entropy']\\n",
    "\\n",
    "print('Diversity Analysis:')\\n",
    "print('=' * 40)\\n",
    "print(diversity_df.round(3))\\n",
    "\\n",
    "# Plot diversity comparison\\n",
    "fig, ax = plt.subplots(1, 1, figsize=(10, 6))\\n",
    "\\n",
    "x = np.arange(len(diversity_results))\\n",
    "width = 0.25\\n",
    "\\n",
    "properties_list = list(diversity_results.keys())\\n",
    "overall_diversity = [diversity_results[prop]['diversity_score'] for prop in properties_list]\\n",
    "degree_diversity = [diversity_results[prop]['degree_diversity'] for prop in properties_list]\\n",
    "size_entropy = [diversity_results[prop]['size_entropy'] for prop in properties_list]\\n",
    "\\n",
    "ax.bar(x - width, overall_diversity, width, label='Overall Diversity', alpha=0.8)\\n",
    "ax.bar(x, degree_diversity, width, label='Degree Diversity', alpha=0.8)\\n",
    "ax.bar(x + width, size_entropy, width, label='Size Entropy', alpha=0.8)\\n",
    "\\n",
    "ax.set_xlabel('Property')\\n",
    "ax.set_ylabel('Diversity Score')\\n",
    "ax.set_title('Diversity Comparison Across Properties')\\n",
    "ax.set_xticks(x)\\n",
    "ax.set_xticklabels([prop.title() for prop in properties_list])\\n",
    "ax.legend()\\n",
    "ax.grid(True, alpha=0.3, axis='y')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cross-Property Analysis\\n",
    "\\n",
    "How well do models trained on one property generalize to others?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test cross-property generalization\\n",
    "cross_property_results = {}\\n",
    "\\n",
    "for source_prop, source_graphs in generated_sets.items():\\n",
    "    cross_property_results[source_prop] = {}\\n",
    "    \\n",
    "    for target_prop in properties_to_test:\\n",
    "        if source_prop != target_prop:\\n",
    "            target_checker = property_library.get_property(target_prop)\\n",
    "            satisfaction_rate = np.mean([target_checker.check(g) for g in source_graphs])\\n",
    "            cross_property_results[source_prop][target_prop] = satisfaction_rate\\n",
    "\\n",
    "# Create cross-property matrix\\n",
    "cross_df = pd.DataFrame(cross_property_results).fillna(0)\\n",
    "\\n",
    "print('Cross-Property Generalization Matrix:')\\n",
    "print('(Rows: Source property, Columns: Target property satisfaction)')\\n",
    "print('=' * 60)\\n",
    "print(cross_df.round(3))\\n",
    "\\n",
    "# Visualize as heatmap\\n",
    "plt.figure(figsize=(8, 6))\\n",
    "sns.heatmap(cross_df, annot=True, fmt='.3f', cmap='Blues', \\n",
    "           xticklabels=[p.title() for p in cross_df.columns],\\n",
    "           yticklabels=[p.title() for p in cross_df.index])\\n",
    "plt.title('Cross-Property Generalization Heatmap')\\n",
    "plt.xlabel('Target Property')\\n",
    "plt.ylabel('Source Property (Trained On)')\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Insights and Conclusions\\n",
    "\\n",
    "From this multi-property analysis, we can observe:\\n",
    "\\n",
    "1. **Property-specific performance varies**: Some properties (like trees) are easier to learn than others\\n",
    "2. **Training dynamics differ**: Different properties show different convergence patterns\\n",
    "3. **Diversity patterns**: Some properties naturally allow more structural diversity\\n",
    "4. **Cross-property generalization**: Models trained on one property may partially satisfy others\\n",
    "\\n",
    "These insights demonstrate the flexibility and effectiveness of the Logical GAN framework across different graph properties."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

# === notebooks/04_applications_showcase.ipynb ===

APPLICATIONS_NOTEBOOK = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Applications Showcase\\n",
    "\\n",
    "This notebook demonstrates real-world applications of Logical GANs in:\\n",
    "- Network Security Topology Generation\\n",
    "- Molecular Structure Design\\n",
    "- Formal Verification Test Case Generation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import networkx as nx\\n",
    "from collections import Counter\\n",
    "\\n",
    "from logical_gans.applications.security_topology import SecurityTopologyGAN\\n",
    "from logical_gans.applications.molecular_design import MolecularGAN\\n",
    "from logical_gans.applications.formal_verification import FormalVerificationGAN\\n",
    "from logical_gans.utils.graph_utils import GraphVisualization\\n",
    "\\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Application 1: Network Security Topology Generation\\n",
    "\\n",
    "Generate secure network topologies that satisfy multiple security constraints."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize security topology generator\\n",
    "print('Initializing Security Topology GAN...')\\n",
    "security_gan = SecurityTopologyGAN(max_nodes=20, security_level='high')\\n",
    "\\n",
    "# Show security constraints\\n",
    "print('\\\\nSecurity Constraints:')\\n",
    "for constraint in security_gan.security_constraints:\\n",
    "    print(f'- {constraint.name}: {constraint.description}')\\n",
    "\\n",
    "# Train the security model (shortened for demo)\\n",
    "print('\\\\nTraining security topology generator...')\\n",
    "# Note: In practice, this would take longer\\n",
    "security_gan.trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate secure network topologies\\n",
    "secure_networks = security_gan.generate_secure_networks(num_networks=20)\\n",
    "print(f'Generated {len(secure_networks)} secure network topologies')\\n",
    "\\n",
    "# Evaluate security properties\\n",
    "security_evaluation = security_gan.evaluate_security(secure_networks)\\n",
    "\\n",
    "print('\\\\nSecurity Evaluation Results:')\\n",
    "print(f'Security compliance rate: {security_evaluation[\\"security_compliance_rate\\"]:.2%}')\\n",
    "print(f'Average redundancy: {security_evaluation[\\"avg_redundancy\\"]:.2f}')\\n",
    "print(f'Average diameter: {security_evaluation[\\"avg_diameter\\"]:.1f}')\\n",
    "print(f'Robustness score: {security_evaluation[\\"robustness_score\\"]:.3f}')\\n",
    "\\n",
    "# Visualize secure networks\\n",
    "GraphVisualization.plot_graph_collection(\\n",
    "    secure_networks[:6],\\n",
    "    title='Generated Secure Network Topologies'\\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Security Analysis\\n",
    "\\n",
    "Let's analyze the security properties in detail:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Detailed security analysis\\n",
    "redundancy_levels = []\\n",
    "diameters = []\\n",
    "node_connectivities = []\\n",
    "\\n",
    "for network in secure_networks:\\n",
    "    # Edge connectivity (redundancy)\\n",
    "    try:\\n",
    "        redundancy = nx.edge_connectivity(network)\\n",
    "        redundancy_levels.append(redundancy)\\n",
    "    except:\\n",
    "        redundancy_levels.append(0)\\n",
    "    \\n",
    "    # Network diameter\\n",
    "    try:\\n",
    "        if nx.is_connected(network):\\n",
    "            diameter = nx.diameter(network)\\n",
    "            diameters.append(diameter)\\n",
    "    except:\\n",
    "        pass\\n",
    "    \\n",
    "    # Node connectivity\\n",
    "    try:\\n",
    "        node_conn = nx.node_connectivity(network)\\n",
    "        node_connectivities.append(node_conn)\\n",
    "    except:\\n",
    "        node_connectivities.append(0)\\n",
    "\\n",
    "# Plot security metrics\\n",
    "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))\\n",
    "\\n",
    "# Redundancy distribution\\n",
    "ax1.hist(redundancy_levels, bins=range(max(redundancy_levels)+2), \\n",
    "         alpha=0.7, color='skyblue', edgecolor='black')\\n",
    "ax1.set_xlabel('Edge Connectivity (Redundancy)')\\n",
    "ax1.set_ylabel('Number of Networks')\\n",
    "ax1.set_title('Redundancy Distribution')\\n",
    "ax1.grid(True, alpha=0.3)\\n",
    "\\n",
    "# Diameter distribution\\n",
    "if diameters:\\n",
    "    ax2.hist(diameters, bins=range(max(diameters)+2),\\n",
    "             alpha=0.7, color='lightcoral', edgecolor='black')\\n",
    "    ax2.set_xlabel('Network Diameter')\\n",
    "    ax2.set_ylabel('Number of Networks')\\n",
    "    ax2.set_title('Diameter Distribution')\\n",
    "    ax2.grid(True, alpha=0.3)\\n",
    "\\n",
    "# Node connectivity\\n",
    "ax3.hist(node_connectivities, bins=range(max(node_connectivities)+2),\\n",
    "         alpha=0.7, color='lightgreen', edgecolor='black')\\n",
    "ax3.set_xlabel('Node Connectivity')\\n",
    "ax3.set_ylabel('Number of Networks')\\n",
    "ax3.set_title('Node Connectivity Distribution')\\n",
    "ax3.grid(True, alpha=0.3)\\n",
    "\\n",
    "# Network size vs redundancy\\n",
    "sizes = [len(net) for net in secure_networks]\\n",
    "ax4.scatter(sizes, redundancy_levels, alpha=0.6, color='purple')\\n",
    "ax4.set_xlabel('Network Size (Nodes)')\\n",
    "ax4.set_ylabel('Edge Connectivity')\\n",
    "ax4.set_title('Size vs Redundancy')\\n",
    "ax4.grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Application 2: Molecular Structure Generation\\n",
    "\\n",
    "Generate chemically valid molecular structures."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize molecular generator\\n",
    "print('Initializing Molecular GAN...')\\n",
    "molecular_gan = MolecularGAN(max_atoms=20)\\n",
    "\\n",
    "print('Chemical constraints:')\\n",
    "for constraint in molecular_gan.chemical_constraints:\\n",
    "    print(f'- {constraint}')\\n",
    "\\n",
    "# Train molecular generator\\n",
    "print('\\\\nTraining molecular structure generator...')\\n",
    "molecular_gan.trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate molecular structures\\n",
    "molecules = molecular_gan.generate_molecules(num_molecules=30)\\n",
    "print(f'Generated {len(molecules)} molecular structures')\\n",
    "\\n",
    "# Evaluate chemical validity\\n",
    "molecular_evaluation = molecular_gan.evaluate_chemical_validity(molecules)\\n",
    "\\n",
    "print('\\\\nMolecular Evaluation Results:')\\n",
    "print(f'Validity rate: {molecular_evaluation[\\"validity_rate\\"]:.2%}')\\n",
    "print(f'Valency violation rate: {molecular_evaluation[\\"valency_violation_rate\\"]:.2%}')\\n",
    "print(f'Connectivity issue rate: {molecular_evaluation[\\"connectivity_issue_rate\\"]:.2%}')\\n",
    "print(f'Average molecular weight: {molecular_evaluation[\\"avg_molecular_weight\\"]:.1f} atoms')\\n",
    "\\n",
    "# Filter valid molecules for visualization\\n",
    "valid_molecules = []\\n",
    "for mol in molecules:\\n",
    "    if molecular_gan.trainer.property_checker.check(mol):\\n",
    "        valid_molecules.append(mol)\\n",
    "\\n",
    "print(f'\\\\nValid molecules for visualization: {len(valid_molecules)}')\\n",
    "\\n",
    "# Visualize molecular structures\\n",
    "if valid_molecules:\\n",
    "    GraphVisualization.plot_graph_collection(\\n",
    "        valid_molecules[:8],\\n",
    "        title='Generated Molecular Structures (Chemically Valid)'\\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Molecular Analysis\\n",
    "\\n",
    "Let's analyze the molecular properties:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze molecular properties\\n",
    "mol_sizes = [len(mol) for mol in molecules]\\n",
    "mol_bonds = [mol.number_of_edges() for mol in molecules]\\n",
    "bond_atom_ratios = [bonds/atoms if atoms > 0 else 0 \\n",
    "                   for bonds, atoms in zip(mol_bonds, mol_sizes)]\\n",
    "\\n",
    "# Degree analysis (valency)\\n",
    "degree_distributions = {}\\n",
    "for i, mol in enumerate(molecules[:10]):\\n",
    "    degrees = [d for _, d in mol.degree()]\\n",
    "    degree_distributions[f'Mol_{i+1}'] = Counter(degrees)\\n",
    "\\n",
    "# Plot molecular analysis\\n",
    "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))\\n",
    "\\n",
    "# Molecular size distribution\\n",
    "ax1.hist(mol_sizes, bins=range(min(mol_sizes), max(mol_sizes)+2),\\n",
    "         alpha=0.7, color='lightblue', edgecolor='black')\\n",
    "ax1.set_xlabel('Number of Atoms')\\n",
    "ax1.set_ylabel('Number of Molecules')\\n",
    "ax1.set_title('Molecular Size Distribution')\\n",
    "ax1.grid(True, alpha=0.3)\\n",
    "\\n",
    "# Bond count distribution\\n",
    "ax2.hist(mol_bonds, bins=range(min(mol_bonds), max(mol_bonds)+2),\\n",
    "         alpha=0.7, color='lightgreen', edgecolor='black')\\n",
    "ax2.set_xlabel('Number of Bonds')\\n",
    "ax2.set_ylabel('Number of Molecules')\\n",
    "ax2.set_title('Bond Count Distribution')\\n",
    "ax2.grid(True, alpha=0.3)\\n",
    "\\n",
    "# Atoms vs bonds\\n",
    "ax3.scatter(mol_sizes, mol_bonds, alpha=0.6, color='coral')\\n",
    "ax3.set_xlabel('Number of Atoms')\\n",
    "ax3.set_ylabel('Number of Bonds')\\n",
    "ax3.set_title('Atoms vs Bonds')\\n",
    "ax3.grid(True, alpha=0.3)\\n",
    "\\n",
    "# Bond/atom ratio\\n",
    "ax4.hist(bond_atom_ratios, bins=20, alpha=0.7, color='gold', edgecolor='black')\\n",
    "ax4.set_xlabel('Bonds per Atom')\\n",
    "ax4.set_ylabel('Number of Molecules')\\n",
    "ax4.set_title('Bond/Atom Ratio Distribution')\\n",
    "ax4.grid(True, alpha=0.3)\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()\\n",
    "\\n",
    "# Show valency distribution\\n",
    "print('\\\\nValency (Degree) Analysis for Sample Molecules:')\\n",
    "for mol_name, degree_dist in list(degree_distributions.items())[:5]:\\n",
    "    print(f'{mol_name}: {dict(degree_dist)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Application 3: Formal Verification Test Cases\\n",
    "\\n",
    "Generate challenging counterexamples for formal verification tools."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize verification test case generators\\n",
    "verification_properties = ['connectivity', 'planarity', 'bipartiteness']\\n",
    "verification_gans = {}\\n",
    "\\n",
    "for prop in verification_properties:\\n",
    "    print(f'Initializing verification GAN for {prop}...')\\n",
    "    gan = FormalVerificationGAN(prop)\\n",
    "    verification_gans[prop] = gan\\n",
    "    \\n",
    "    print(f'Training counterexample generator for {prop}...')\\n",
    "    gan.trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate counterexamples for each property\\n",
    "counterexamples = {}\\n",
    "verification_results = {}\\n",
    "\\n",
    "for prop, gan in verification_gans.items():\\n",
    "    print(f'\\\\nGenerating {prop} counterexamples...')\\n",
    "    examples = gan.generate_counterexamples(num_examples=20)\\n",
    "    counterexamples[prop] = examples\\n",
    "    \\n",
    "    # Evaluate counterexample quality\\n",
    "    evaluation = gan.evaluate_counterexample_quality(examples)\\n",
    "    verification_results[prop] = evaluation\\n",
    "    \\n",
    "    print(f'Counterexample quality rate: {evaluation[\\"counterexample_rate\\"]:.2%}')\\n",
    "    print(f'Average verification complexity: {evaluation[\\"avg_verification_complexity\\"]:.1f}')\\n",
    "\\n",
    "# Visualize counterexamples\\n",
    "for prop, examples in counterexamples.items():\\n",
    "    # Filter good counterexamples\\n",
    "    good_examples = examples[:6]  # Show first 6\\n",
    "    \\n",
    "    GraphVisualization.plot_graph_collection(\\n",
    "        good_examples,\\n",
    "        title=f'{prop.title()} Counterexamples'\\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Counterexample Analysis\\n",
    "\\n",
    "Let's analyze the properties of generated counterexamples:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze counterexample properties\\n",
    "counterexample_analysis = {}\\n",
    "\\n",
    "for prop, examples in counterexamples.items():\\n",
    "    analysis = {\\n",
    "        'avg_size': np.mean([len(g) for g in examples]),\\n",
    "        'avg_edges': np.mean([g.number_of_edges() for g in examples]),\\n",
    "        'avg_density': np.mean([nx.density(g) for g in examples if len(g) > 1]),\\n",
    "        'connected_fraction': np.mean([nx.is_connected(g) for g in examples])\\n",
    "    }\\n",
    "    \\n",
    "    # Property-specific analysis\\n",
    "    if prop == 'connectivity':\\n",
    "        analysis['num_components'] = np.mean([nx.number_connected_components(g) for g in examples])\\n",
    "    elif prop == 'planarity':\\n",
    "        analysis['planar_fraction'] = np.mean([nx.is_planar(g) for g in examples])\\n",
    "    elif prop == 'bipartiteness':\\n",
    "        analysis['bipartite_fraction'] = np.mean([nx.is_bipartite(g) for g in examples])\\n",
    "    \\n",
    "    counterexample_analysis[prop] = analysis\\n",
    "\\n",
    "# Display analysis\\n",
    "print('Counterexample Analysis:')\\n",
    "print('=' * 50)\\n",
    "for prop, analysis in counterexample_analysis.items():\\n",
    "    print(f'\\\\n{prop.title()} Counterexamples:')\\n",
    "    for metric, value in analysis.items():\\n",
    "        print(f'  {metric}: {value:.3f}')\\n",
    "\\n",
    "# Comparative visualization\\n",
    "import pandas as pd\\n",
    "\\n",
    "# Create comparison DataFrame\\n",
    "comparison_data = []\\n",
    "for prop in verification_properties:\\n",
    "    result = verification_results[prop]\\n",
    "    comparison_data.append({\\n",
    "        'Property': prop.title(),\\n",
    "        'Counterexample_Rate': result['counterexample_rate'],\\n",
    "        'Avg_Complexity': result['avg_verification_complexity'] / 1000,  # Scale for visibility\\n",
    "        'Edge_Case_Rate': result['edge_case_discovery_rate']\\n",
    "    })\\n",
    "\\n",
    "comp_df = pd.DataFrame(comparison_data)\\n",
    "\\n",
    "# Plot comparison\\n",
    "fig, ax = plt.subplots(1, 1, figsize=(10, 6))\\n",
    "\\n",
    "x = np.arange(len(comparison_data))\\n",
    "width = 0.25\\n",
    "\\n",
    "ax.bar(x - width, comp_df['Counterexample_Rate'], width, \\n",
    "       label='Counterexample Quality', alpha=0.8)\\n",
    "ax.bar(x, comp_df['Avg_Complexity'], width,\\n",
    "       label='Complexity (scaled)', alpha=0.8)\\n",
    "ax.bar(x + width, comp_df['Edge_Case_Rate'], width,\\n",
    "       label='Edge Case Discovery', alpha=0.8)\\n",
    "\\n",
    "ax.set_xlabel('Property')\\n",
    "ax.set_ylabel('Score')\\n",
    "ax.set_title('Formal Verification Counterexample Quality')\\n",
    "ax.set_xticks(x)\\n",
    "ax.set_xticklabels(comp_df['Property'])\\n",
    "ax.legend()\\n",
    "ax.grid(True, alpha=0.3, axis='y')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Applications Summary\\n",
    "\\n",
    "This showcase demonstrated three key applications of Logical GANs:\\n",
    "\\n",
    "### 1. Network Security\\n",
    "- Generated secure network topologies with redundancy and robustness\\n",
    "- Achieved high security compliance rates\\n",
    "- Balanced security constraints with network efficiency\\n",
    "\\n",
    "### 2. Molecular Design\\n",
    "- Created chemically valid molecular structures\\n",
    "- Respected valency and connectivity constraints\\n",
    "- Generated diverse molecular architectures\\n",
    "\\n",
    "### 3. Formal Verification\\n",
    "- Produced challenging counterexamples for verification tools\\n",
    "- Discovered edge cases in different properties\\n",
    "- Created systematically difficult test cases\\n",
    "\\n",
    "Each application leverages the core strength of Logical GANs: **precise control over generated structures through logical constraints**, while maintaining the diversity and creativity of generative models."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''


def save_notebooks():
    """Save all Jupyter notebooks to the notebooks directory."""
    
    from pathlib import Path
    import json
    
    notebooks_dir = Path('notebooks')
    notebooks_dir.mkdir(exist_ok=True)
    
    notebooks = {
        '01_ef_games_tutorial.ipynb': EF_GAMES_NOTEBOOK,
        '02_logical_gan_demo.ipynb': LOGICAL_GAN_DEMO,
        '03_property_generation.ipynb': PROPERTY_GENERATION_NOTEBOOK,
        '04_applications_showcase.ipynb': APPLICATIONS_NOTEBOOK
    }
    
    for filename, content in notebooks.items():
        # Parse and reformat JSON
        notebook_json = json.loads(content.strip())
        
        with open(notebooks_dir / filename, 'w') as f:
            json.dump(notebook_json, f, indent=1)
    
    print(f"Jupyter notebooks saved to {notebooks_dir}/")
    print("Available notebooks:")
    for filename in notebooks:
        print(f"  - {filename}")


if __name__ == "__main__":
    save_notebooks()
