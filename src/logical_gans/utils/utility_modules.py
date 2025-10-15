"""
Utility modules for Logical GANs
Graph utilities, metrics, and helper functions
"""

# === graph_utils.py ===

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data, Batch
from scipy.sparse import csr_matrix
import json


class GraphUtils:
    """Utility functions for graph generation and manipulation."""
    
    @staticmethod
    def networkx_to_pyg(graph: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        
        # Node features (simple: just degree)
        num_nodes = len(graph)
        if num_nodes == 0:
            return Data(x=torch.zeros((0, 1)), edge_index=torch.zeros((2, 0), dtype=torch.long))
        
        node_features = torch.tensor([[graph.degree(node)] for node in graph.nodes()], 
                                   dtype=torch.float32)
        
        # Edge indices
        edge_list = list(graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index)
    
    @staticmethod
    def pyg_to_networkx(data: Data) -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX graph."""
        
        graph = nx.Graph()
        
        # Add nodes
        num_nodes = data.x.size(0) if data.x is not None else 0
        graph.add_nodes_from(range(num_nodes))
        
        # Add edges
        if data.edge_index.size(1) > 0:
            edge_list = data.edge_index.t().numpy().tolist()
            graph.add_edges_from(edge_list)
        
        return graph
    
    @staticmethod
    def adjacency_to_networkx(adj_matrix: np.ndarray, threshold: float = 0.5) -> nx.Graph:
        """Convert adjacency matrix to NetworkX graph."""
        
        # Threshold and symmetrize
        adj_binary = (adj_matrix > threshold).astype(int)
        adj_symmetric = np.maximum(adj_binary, adj_binary.T)
        
        # Remove self-loops
        np.fill_diagonal(adj_symmetric, 0)
        
        # Create graph
        graph = nx.from_numpy_array(adj_symmetric)
        
        return graph
    
    @staticmethod
    def batch_graphs_to_networkx(batch: Batch) -> List[nx.Graph]:
        """Convert a batch of PyG graphs to list of NetworkX graphs."""
        
        graphs = []
        batch_indices = batch.batch.numpy()
        
        for i in range(batch.num_graphs):
            # Extract nodes for this graph
            node_mask = (batch_indices == i)
            graph_nodes = np.where(node_mask)[0]
            
            # Extract edges for this graph
            edge_mask = np.isin(batch.edge_index[0].numpy(), graph_nodes)
            graph_edge_index = batch.edge_index[:, edge_mask]
            
            # Remap node indices to start from 0
            node_mapping = {old: new for new, old in enumerate(graph_nodes)}
            remapped_edges = [[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] 
                            for edge in graph_edge_index.t()]
            
            # Create NetworkX graph
            graph = nx.Graph()
            graph.add_nodes_from(range(len(graph_nodes)))
            graph.add_edges_from(remapped_edges)
            
            graphs.append(graph)
        
        return graphs
    
    @staticmethod
    def generate_graph_dataset(property_name: str, num_graphs: int = 1000, 
                             max_nodes: int = 20) -> List[nx.Graph]:
        """Generate dataset of graphs with specific property."""
        
        graphs = []
        attempts = 0
        max_attempts = num_graphs * 10
        
        while len(graphs) < num_graphs and attempts < max_attempts:
            attempts += 1
            n = np.random.randint(3, max_nodes + 1)
            
            if property_name == "tree":
                graph = nx.random_tree(n)
            elif property_name == "cycle":
                graph = nx.cycle_graph(n)
            elif property_name == "complete":
                graph = nx.complete_graph(n)
            elif property_name == "bipartite":
                n1, n2 = n // 2, n - n // 2
                graph = nx.complete_bipartite_graph(n1, n2)
            elif property_name == "planar":
                graph = nx.random_tree(n)
                # Add edges while maintaining planarity
                max_edges = 3 * n - 6
                current_edges = n - 1
                for _ in range(min(max_edges - current_edges, n)):
                    u, v = np.random.choice(n, 2, replace=False)
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v)
                        if not nx.is_planar(graph):
                            graph.remove_edge(u, v)
            else:
                # Random graph
                p = np.random.uniform(0.1, 0.7)
                graph = nx.erdos_renyi_graph(n, p)
            
            graphs.append(graph)
        
        return graphs
    
    @staticmethod
    def compute_graph_statistics(graphs: List[nx.Graph]) -> Dict[str, float]:
        """Compute statistical properties of a graph collection."""
        
        if not graphs:
            return {}
        
        stats = {
            'num_graphs': len(graphs),
            'avg_nodes': np.mean([len(g) for g in graphs]),
            'std_nodes': np.std([len(g) for g in graphs]),
            'avg_edges': np.mean([g.number_of_edges() for g in graphs]),
            'std_edges': np.std([g.number_of_edges() for g in graphs]),
            'avg_density': np.mean([nx.density(g) for g in graphs if len(g) > 1]),
            'avg_clustering': np.mean([nx.average_clustering(g) for g in graphs if len(g) > 2]),
            'connected_fraction': np.mean([nx.is_connected(g) for g in graphs if len(g) > 0])
        }
        
        # Degree statistics
        all_degrees = []
        for graph in graphs:
            all_degrees.extend([d for _, d in graph.degree()])
        
        if all_degrees:
            stats.update({
                'avg_degree': np.mean(all_degrees),
                'std_degree': np.std(all_degrees),
                'max_degree': np.max(all_degrees)
            })
        
        return stats
    
    @staticmethod
    def save_graphs(graphs: List[nx.Graph], filepath: str, format: str = "json"):
        """Save graphs to file in various formats."""
        
        if format == "json":
            graph_data = []
            for i, graph in enumerate(graphs):
                data = {
                    'id': i,
                    'nodes': list(graph.nodes()),
                    'edges': list(graph.edges()),
                    'num_nodes': len(graph),
                    'num_edges': graph.number_of_edges()
                }
                graph_data.append(data)
            
            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        elif format == "graphml":
            # Save as individual GraphML files
            import os
            os.makedirs(filepath, exist_ok=True)
            for i, graph in enumerate(graphs):
                nx.write_graphml(graph, f"{filepath}/graph_{i:04d}.graphml")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_graphs(filepath: str, format: str = "json") -> List[nx.Graph]:
        """Load graphs from file."""
        
        if format == "json":
            with open(filepath, 'r') as f:
                graph_data = json.load(f)
            
            graphs = []
            for data in graph_data:
                graph = nx.Graph()
                graph.add_nodes_from(data['nodes'])
                graph.add_edges_from(data['edges'])
                graphs.append(graph)
            
            return graphs
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# === metrics.py ===

import networkx as nx
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
from collections import Counter, defaultdict


class GraphMetrics:
    """Comprehensive metrics for evaluating generated graphs."""
    
    @staticmethod
    def property_satisfaction_rate(graphs: List[nx.Graph], property_checker) -> float:
        """Compute rate of property satisfaction."""
        if not graphs:
            return 0.0
        
        satisfying = sum(1 for graph in graphs if property_checker.check(graph))
        return satisfying / len(graphs)
    
    @staticmethod
    def structural_diversity(graphs: List[nx.Graph]) -> Dict[str, float]:
        """Compute structural diversity metrics."""
        
        if not graphs:
            return {'diversity_score': 0.0}
        
        # Degree sequence diversity
        degree_sequences = []
        for graph in graphs:
            deg_seq = tuple(sorted([d for _, d in graph.degree()]))
            degree_sequences.append(deg_seq)
        
        unique_degree_seqs = len(set(degree_sequences))
        degree_diversity = unique_degree_seqs / len(graphs)
        
        # Size diversity
        sizes = [len(graph) for graph in graphs]
        size_entropy = stats.entropy(list(Counter(sizes).values()))
        
        # Edge count diversity
        edge_counts = [graph.number_of_edges() for graph in graphs]
        edge_entropy = stats.entropy(list(Counter(edge_counts).values()))
        
        # Clustering coefficient diversity
        clustering_coeffs = []
        for graph in graphs:
            if len(graph) > 2:
                clustering_coeffs.append(nx.average_clustering(graph))
        
        clustering_diversity = np.std(clustering_coeffs) if clustering_coeffs else 0.0
        
        return {
            'diversity_score': (degree_diversity + size_entropy + edge_entropy) / 3,
            'degree_diversity': degree_diversity,
            'size_entropy': size_entropy,
            'edge_entropy': edge_entropy,
            'clustering_diversity': clustering_diversity,
            'unique_degree_sequences': unique_degree_seqs
        }
    
    @staticmethod
    def ef_distance_distribution(graphs: List[nx.Graph], theory_graphs: List[nx.Graph],
                               max_rounds: int = 3) -> Dict[str, float]:
        """Compute distribution of EF-distances to theory."""
        
        if not graphs or not theory_graphs:
            return {'avg_ef_distance': float('inf')}
        
        from ..core.ef_games import ef_distance_to_theory
        
        ef_distances = []
        for graph in graphs[:100]:  # Sample for efficiency
            ef_dist = ef_distance_to_theory(graph, theory_graphs[:50], max_rounds)
            ef_distances.append(ef_dist)
        
        return {
            'avg_ef_distance': np.mean(ef_distances),
            'std_ef_distance': np.std(ef_distances),
            'min_ef_distance': np.min(ef_distances),
            'max_ef_distance': np.max(ef_distances),
            'perfect_ef_distance_rate': np.mean([d == 0 for d in ef_distances])
        }
    
    @staticmethod
    def graph_isomorphism_diversity(graphs: List[nx.Graph]) -> Dict[str, float]:
        """Compute diversity based on graph isomorphism classes."""
        
        if not graphs:
            return {'isomorphism_diversity': 0.0}
        
        # Group graphs by isomorphism class (simplified)
        isomorphism_classes = []
        
        for graph in graphs:
            # Use simple graph invariants as proxy for isomorphism class
            invariant = (
                len(graph),
                graph.number_of_edges(),
                tuple(sorted([d for _, d in graph.degree()])),
                nx.number_of_triangles(graph) if len(graph) > 2 else 0
            )
            isomorphism_classes.append(invariant)
        
        unique_classes = len(set(isomorphism_classes))
        
        return {
            'isomorphism_diversity': unique_classes / len(graphs),
            'unique_classes': unique_classes,
            'total_graphs': len(graphs)
        }
    
    @staticmethod
    def spectral_properties(graphs: List[nx.Graph]) -> Dict[str, float]:
        """Compute spectral properties of graphs."""
        
        spectral_gaps = []
        largest_eigenvalues = []
        
        for graph in graphs:
            if len(graph) > 1 and graph.number_of_edges() > 0:
                try:
                    # Compute adjacency matrix eigenvalues
                    adj_matrix = nx.adjacency_matrix(graph).toarray()
                    eigenvals = np.linalg.eigvals(adj_matrix)
                    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                    
                    largest_eigenvalues.append(eigenvals[0])
                    
                    if len(eigenvals) > 1:
                        spectral_gap = eigenvals[0] - eigenvals[1]
                        spectral_gaps.append(spectral_gap)
                
                except:
                    continue
        
        return {
            'avg_spectral_gap': np.mean(spectral_gaps) if spectral_gaps else 0.0,
            'std_spectral_gap': np.std(spectral_gaps) if spectral_gaps else 0.0,
            'avg_largest_eigenvalue': np.mean(largest_eigenvalues) if largest_eigenvalues else 0.0
        }
    
    @staticmethod
    def compare_graph_distributions(real_graphs: List[nx.Graph], 
                                  generated_graphs: List[nx.Graph]) -> Dict[str, float]:
        """Compare distributions of real vs generated graphs."""
        
        def extract_features(graphs):
            features = {
                'sizes': [len(g) for g in graphs],
                'edge_counts': [g.number_of_edges() for g in graphs],
                'densities': [nx.density(g) for g in graphs if len(g) > 1],
                'clusterings': [nx.average_clustering(g) for g in graphs if len(g) > 2],
                'diameters': []
            }
            
            for g in graphs:
                try:
                    if nx.is_connected(g) and len(g) > 1:
                        features['diameters'].append(nx.diameter(g))
                except:
                    pass
            
            return features
        
        real_features = extract_features(real_graphs)
        gen_features = extract_features(generated_graphs)
        
        # Compute KS test statistics for distribution comparison
        comparison_results = {}
        
        for feature_name in real_features:
            if real_features[feature_name] and gen_features[feature_name]:
                ks_stat, p_value = stats.ks_2samp(real_features[feature_name], 
                                                gen_features[feature_name])
                comparison_results[f'{feature_name}_ks_stat'] = ks_stat
                comparison_results[f'{feature_name}_p_value'] = p_value
        
        return comparison_results


class GraphVisualization:
    """Utilities for visualizing graphs and training dynamics."""
    
    @staticmethod
    def plot_graph_collection(graphs: List[nx.Graph], title: str = "Graph Collection",
                            max_graphs: int = 12, figsize: Tuple[int, int] = (15, 10)):
        """Plot a collection of graphs in a grid."""
        
        display_graphs = graphs[:max_graphs]
        cols = 4
        rows = (len(display_graphs) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, graph in enumerate(display_graphs):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Compute layout
            if len(graph) <= 15:
                pos = nx.spring_layout(graph, seed=42)
            else:
                pos = nx.circular_layout(graph)
            
            # Draw graph
            nx.draw(graph, pos, ax=ax, with_labels=True, 
                   node_color='lightblue', node_size=200, 
                   font_size=8, font_weight='bold')
            
            ax.set_title(f"Graph {i+1}\n({len(graph)} nodes, {graph.number_of_edges()} edges)")
        
        # Hide empty subplots
        for i in range(len(display_graphs), rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_curves(training_history: Dict[str, List[float]], 
                           save_path: Optional[str] = None):
        """Plot training curves for Logical GAN."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(len(training_history['generator_losses']))
        
        # Generator and discriminator losses
        ax1.plot(epochs, training_history['generator_losses'], label='Generator', color='blue')
        ax1.plot(epochs, training_history['discriminator_losses'], label='Discriminator', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Adversarial Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # EF-distance evolution
        ax2.plot(epochs, training_history['ef_distances'], color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('EF-Distance')
        ax2.set_title('EF-Distance to Theory')
        ax2.grid(True, alpha=0.3)
        
        # Property satisfaction rate
        ax3.plot(epochs, training_history['property_satisfaction_rates'], color='purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Property Satisfaction Rate')
        ax3.set_title('Property Satisfaction Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Combined metric
        if len(training_history['ef_distances']) == len(training_history['property_satisfaction_rates']):
            combined = [1 - ef + sat for ef, sat in zip(training_history['ef_distances'], 
                                                       training_history['property_satisfaction_rates'])]
            ax4.plot(epochs, combined, color='orange', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Combined Score')
            ax4.set_title('Combined Performance Metric')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_property_comparison(results: Dict[str, Dict[str, float]], 
                               save_path: Optional[str] = None):
        """Plot comparison of different methods across properties."""
        
        properties = list(results.keys())
        methods = list(results[properties[0]].keys())
        
        # Extract data for plotting
        data = []
        for prop in properties:
            for method in methods:
                data.append({
                    'Property': prop.replace('_', ' ').title(),
                    'Method': method.replace('_', ' ').title(),
                    'Satisfaction_Rate': results[prop][method]['mean'] if isinstance(results[prop][method], dict) else results[prop][method]
                })
        
        # Create grouped bar plot
        import pandas as pd
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Pivot for easier plotting
        pivot_df = df.pivot(index='Property', columns='Method', values='Satisfaction_Rate')
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Graph Property')
        ax.set_ylabel('Property Satisfaction Rate (%)')
        ax.set_title('Method Comparison Across Graph Properties')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class PerformanceProfiler:
    """Profiler for analyzing computational performance."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
    
    def time_function(self, func_name: str, func, *args, **kwargs):
        """Time a function call and store results."""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        self.timing_data[func_name].append(elapsed_time)
        
        return result, elapsed_time
    
    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of timing data."""
        
        summary = {}
        for func_name, times in self.timing_data.items():
            summary[func_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_calls': len(times)
            }
        
        return summary
    
    def plot_timing_analysis(self, save_path: Optional[str] = None):
        """Plot timing analysis results."""
        
        timing_summary = self.get_timing_summary()
        
        if not timing_summary:
            print("No timing data available")
            return
        
        func_names = list(timing_summary.keys())
        mean_times = [timing_summary[fn]['mean_time'] for fn in func_names]
        std_times = [timing_summary[fn]['std_time'] for fn in func_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(func_names))
        bars = ax.bar(x_pos, mean_times, yerr=std_times, capsize=5, 
                     color='skyblue', alpha=0.7)
        
        ax.set_xlabel('Function')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Performance Profile: Function Execution Times')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(func_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean_time in zip(bars, mean_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(mean_times) * 0.01,
                   f'{mean_time:.3f}s', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class DatasetGenerator:
    """Generate datasets for training and evaluation."""
    
    @staticmethod
    def create_property_dataset(property_name: str, num_positive: int = 5000, 
                              num_negative: int = 5000, max_nodes: int = 20) -> Tuple[List[nx.Graph], List[int]]:
        """Create balanced dataset for a specific property."""
        
        graphs = []
        labels = []
        
        # Generate positive examples
        positive_graphs = GraphUtils.generate_graph_dataset(property_name, num_positive, max_nodes)
        graphs.extend(positive_graphs)
        labels.extend([1] * len(positive_graphs))
        
        # Generate negative examples
        negative_attempts = 0
        max_negative_attempts = num_negative * 10
        
        while len([l for l in labels if l == 0]) < num_negative and negative_attempts < max_negative_attempts:
            negative_attempts += 1
            
            # Generate random graph
            n = np.random.randint(3, max_nodes + 1)
            p = np.random.uniform(0.1, 0.8)
            graph = nx.erdos_renyi_graph(n, p)
            
            # Check if it's a negative example
            from ..core.mso_compiler import MSOPropertyLibrary
            library = MSOPropertyLibrary()
            property_checker = library.get_property(property_name)
            
            if not property_checker.check(graph):
                graphs.append(graph)
                labels.append(0)
        
        return graphs, labels
    
    @staticmethod
    def split_dataset(graphs: List[nx.Graph], labels: List[int], 
                     train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """Split dataset into train/validation/test sets."""
        
        assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1.0"
        
        n = len(graphs)
        indices = np.random.permutation(n)
        
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_graphs = [graphs[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        
        val_graphs = [graphs[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        test_graphs = [graphs[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        return (train_graphs, train_labels), (val_graphs, val_labels), (test_graphs, test_labels)


# === Training utilities ===

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# === Example usage and testing ===

if __name__ == "__main__":
    # Test graph utilities
    print("Testing graph utilities...")
    
    # Generate test graphs
    test_graphs = GraphUtils.generate_graph_dataset("tree", num_graphs=50, max_nodes=10)
    print(f"Generated {len(test_graphs)} test graphs")
    
    # Compute statistics
    stats = GraphUtils.compute_graph_statistics(test_graphs)
    print("Graph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Test metrics
    print("\nTesting metrics...")
    
    # Generate comparison graphs
    random_graphs = [nx.erdos_renyi_graph(np.random.randint(5, 15), 0.3) for _ in range(50)]
    
    # Structural diversity
    diversity = GraphMetrics.structural_diversity(test_graphs)
    print(f"Structural diversity: {diversity['diversity_score']:.3f}")
    
    # Distribution comparison
    comparison = GraphMetrics.compare_graph_distributions(test_graphs, random_graphs)
    print("Distribution comparison (KS test p-values):")
    for key, value in comparison.items():
        if 'p_value' in key:
            print(f"  {key}: {value:.4f}")
    
    # Test visualization
    print("\nGenerating visualizations...")
    GraphVisualization.plot_graph_collection(test_graphs[:8], "Sample Tree Graphs")
    
    # Test performance profiler
    profiler = PerformanceProfiler()
    
    # Profile some functions
    _, time1 = profiler.time_function("graph_generation", 
                                     GraphUtils.generate_graph_dataset, "tree", 10)
    _, time2 = profiler.time_function("statistics_computation",
                                     GraphUtils.compute_graph_statistics, test_graphs)
    
    print(f"Graph generation took: {time1:.3f}s")
    print(f"Statistics computation took: {time2:.3f}s")
    
    # Show timing summary
    timing_summary = profiler.get_timing_summary()
    print("\nTiming summary:")
    for func_name, timing_data in timing_summary.items():
        print(f"  {func_name}: {timing_data['mean_time']:.4f}s ± {timing_data['std_time']:.4f}s")
