"""
Logical GAN Framework Implementation
Core framework connecting EF games with adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod

from .ef_games import EFGameSimulator, ef_distance_to_theory


class GraphGenerator(nn.Module):
    """
    Graph generator network that produces adjacency matrices and node features.
    """
    
    def __init__(self, latent_dim: int = 128, max_nodes: int = 20, 
                 hidden_dims: List[int] = [256, 512, 1024]):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        
        # Edge generation network
        layers = []
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Output layer for adjacency matrix
        layers.append(nn.Linear(in_dim, max_nodes * max_nodes))
        self.edge_net = nn.Sequential(*layers)
        
        # Node count prediction
        self.node_count_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_nodes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate graphs from latent codes.
        Returns: (adjacency_matrix, node_count_probs)
        """
        batch_size = z.size(0)
        
        # Generate adjacency matrices
        edge_logits = self.edge_net(z)
        edge_logits = edge_logits.view(batch_size, self.max_nodes, self.max_nodes)
        
        # Make symmetric and remove self-loops
        edge_logits = (edge_logits + edge_logits.transpose(-1, -2)) / 2
        edge_logits.fill_diagonal_(0)
        
        adj_matrix = torch.sigmoid(edge_logits)
        
        # Generate node counts
        node_count_probs = self.node_count_net(z)
        
        return adj_matrix, node_count_probs
    
    def sample_graphs(self, z: torch.Tensor, threshold: float = 0.5) -> List[nx.Graph]:
        """Convert generated tensors to NetworkX graphs."""
        adj_matrix, node_count_probs = self.forward(z)
        graphs = []
        
        for i in range(z.size(0)):
            # Sample number of nodes
            num_nodes = torch.multinomial(node_count_probs[i], 1).item() + 1
            num_nodes = min(num_nodes, self.max_nodes)
            
            # Threshold adjacency matrix
            adj = adj_matrix[i, :num_nodes, :num_nodes]
            edges = (adj > threshold).nonzero(as_tuple=False)
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))
            edge_list = [(u.item(), v.item()) for u, v in edges if u != v]
            G.add_edges_from(edge_list)
            
            graphs.append(G)
        
        return graphs


class LogicalDiscriminator(nn.Module):
    """
    Logic-bounded discriminator using Graph Neural Networks.
    Architecture depth corresponds to logical quantifier depth.
    """
    
    def __init__(self, logic_depth: int = 3, hidden_dim: int = 64, 
                 gnn_type: str = "GCN"):
        super().__init__()
        self.logic_depth = logic_depth
        self.hidden_dim = hidden_dim
        
        # Build GNN layers corresponding to quantifier depth
        self.node_embedding = nn.Linear(1, hidden_dim)  # Simple node features
        
        if gnn_type == "GCN":
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(logic_depth)
            ])
        elif gnn_type == "GIN":
            self.gnn_layers = nn.ModuleList([
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )) for _ in range(logic_depth)
            ])
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Discriminate graphs based on logical properties.
        Limited expressiveness by network depth = quantifier depth.
        """
        x = self.node_embedding(batch.x.float())
        
        # Apply GNN layers (quantifier alternations)
        for layer in self.gnn_layers:
            x = F.relu(layer(x, batch.edge_index))
        
        # Global pooling and classification
        graph_embeddings = global_mean_pool(x, batch.batch)
        return self.classifier(graph_embeddings).squeeze()


class PropertyChecker(ABC):
    """Abstract base class for graph property checking."""
    
    @abstractmethod
    def check(self, graph: nx.Graph) -> bool:
        """Check if graph satisfies the property."""
        pass


class TreeProperty(PropertyChecker):
    """Check if graph is a tree (connected and acyclic)."""
    
    def check(self, graph: nx.Graph) -> bool:
        return nx.is_tree(graph)


class ConnectivityProperty(PropertyChecker):
    """Check if graph is connected."""
    
    def check(self, graph: nx.Graph) -> bool:
        return nx.is_connected(graph)


class BipartiteProperty(PropertyChecker):
    """Check if graph is bipartite."""
    
    def check(self, graph: nx.Graph) -> bool:
        return nx.is_bipartite(graph)


class LogicalGAN:
    """
    Main Logical GAN framework integrating EF games with adversarial training.
    """
    
    def __init__(self, 
                 generator: GraphGenerator,
                 discriminator: LogicalDiscriminator,
                 property_checker: PropertyChecker,
                 theory_graphs: List[nx.Graph],
                 ef_weight: float = 0.1,
                 max_ef_rounds: int = 3):
        
        self.generator = generator
        self.discriminator = discriminator
        self.property_checker = property_checker
        self.theory_graphs = theory_graphs
        self.ef_weight = ef_weight
        self.max_ef_rounds = max_ef_rounds
        
        # Optimizers
        self.gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        # Loss history
        self.gen_losses = []
        self.disc_losses = []
        self.ef_distances = []
        self.property_satisfaction_rates = []
    
    def generator_loss(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute generator loss combining adversarial and EF-distance terms.
        L_G = E[δ_k(G_θ(z), T)] + λ * E[1_{G_θ(z) ⊭ T}]
        """
        self.generator.eval()
        generated_graphs = self.generator.sample_graphs(z)
        
        # Adversarial loss
        batch_data = self._graphs_to_batch(generated_graphs)
        disc_scores = self.discriminator(batch_data)
        adversarial_loss = -torch.mean(torch.log(disc_scores + 1e-8))
        
        # EF-distance loss
        ef_distances = []
        property_violations = []
        
        for graph in generated_graphs:
            # Compute EF-distance to theory
            ef_dist = ef_distance_to_theory(graph, self.theory_graphs, self.max_ef_rounds)
            ef_distances.append(ef_dist)
            
            # Property violation indicator
            satisfies_property = self.property_checker.check(graph)
            property_violations.append(0.0 if satisfies_property else 1.0)
        
        ef_loss = torch.tensor(np.mean(ef_distances), dtype=torch.float32, requires_grad=True)
        property_loss = torch.tensor(np.mean(property_violations), dtype=torch.float32, requires_grad=True)
        
        total_loss = adversarial_loss + self.ef_weight * ef_loss + self.ef_weight * property_loss
        
        metrics = {
            'adversarial_loss': adversarial_loss.item(),
            'ef_loss': ef_loss.item(),
            'property_loss': property_loss.item(),
            'avg_ef_distance': np.mean(ef_distances),
            'property_satisfaction_rate': 1.0 - np.mean(property_violations)
        }
        
        return total_loss, metrics
    
    def discriminator_loss(self, real_batch: Batch, fake_batch: Batch) -> torch.Tensor:
        """
        Compute discriminator loss.
        L_D = -E[log D(A)] - E[log(1 - D(G(z)))]
        """
        # Real samples
        real_scores = self.discriminator(real_batch)
        real_loss = -torch.mean(torch.log(real_scores + 1e-8))
        
        # Fake samples
        fake_scores = self.discriminator(fake_batch)
        fake_loss = -torch.mean(torch.log(1 - fake_scores + 1e-8))
        
        return real_loss + fake_loss
    
    def train_step(self, real_graphs: List[nx.Graph], batch_size: int = 32) -> Dict[str, float]:
        """Single training step for both generator and discriminator."""
        
        # Sample latent codes
        z = torch.randn(batch_size, self.generator.latent_dim)
        
        # Train discriminator
        self.discriminator.train()
        self.generator.eval()
        
        generated_graphs = self.generator.sample_graphs(z)
        real_batch = self._graphs_to_batch(real_graphs[:batch_size])
        fake_batch = self._graphs_to_batch(generated_graphs)
        
        self.disc_optimizer.zero_grad()
        disc_loss = self.discriminator_loss(real_batch, fake_batch)
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train generator
        self.generator.train()
        self.discriminator.eval()
        
        self.gen_optimizer.zero_grad()
        gen_loss, gen_metrics = self.generator_loss(z)
        gen_loss.backward()
        self.gen_optimizer.step()
        
        # Update loss history
        self.gen_losses.append(gen_loss.item())
        self.disc_losses.append(disc_loss.item())
        self.ef_distances.append(gen_metrics['avg_ef_distance'])
        self.property_satisfaction_rates.append(gen_metrics['property_satisfaction_rate'])
        
        metrics = {
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item(),
            **gen_metrics
        }
        
        return metrics
    
    def train(self, theory_graphs: List[nx.Graph], epochs: int = 1000, 
              batch_size: int = 32, log_interval: int = 100) -> Dict[str, List[float]]:
        """
        Full training procedure for Logical GAN.
        """
        print(f"Training Logical GAN for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Sample real graphs from theory
            real_sample = np.random.choice(theory_graphs, size=batch_size, replace=True)
            
            metrics = self.train_step(real_sample.tolist(), batch_size)
            
            if epoch % log_interval == 0:
                print(f"Epoch {epoch:04d} | "
                      f"G_loss: {metrics['gen_loss']:.4f} | "
                      f"D_loss: {metrics['disc_loss']:.4f} | "
                      f"EF_dist: {metrics['avg_ef_distance']:.3f} | "
                      f"Prop_sat: {metrics['property_satisfaction_rate']:.3f}")
        
        training_history = {
            'generator_losses': self.gen_losses,
            'discriminator_losses': self.disc_losses,
            'ef_distances': self.ef_distances,
            'property_satisfaction_rates': self.property_satisfaction_rates
        }
        
        return training_history
    
    def generate(self, num_samples: int = 100) -> List[nx.Graph]:
        """Generate graphs using trained generator."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.latent_dim)
            generated_graphs = self.generator.sample_graphs(z)
        
        return generated_graphs
    
    def evaluate(self, generated_graphs: List[nx.Graph]) -> Dict[str, float]:
        """Evaluate generated graphs against theory."""
        
        property_satisfaction = []
        ef_distances = []
        
        for graph in generated_graphs:
            # Check property satisfaction
            satisfies = self.property_checker.check(graph)
            property_satisfaction.append(satisfies)
            
            # Compute EF-distance to theory
            ef_dist = ef_distance_to_theory(graph, self.theory_graphs, self.max_ef_rounds)
            ef_distances.append(ef_dist)
        
        metrics = {
            'property_satisfaction_rate': np.mean(property_satisfaction),
            'average_ef_distance': np.mean(ef_distances),
            'std_ef_distance': np.std(ef_distances),
            'perfect_ef_distance_rate': np.mean([d == 0 for d in ef_distances])
        }
        
        return metrics
    
    def _graphs_to_batch(self, graphs: List[nx.Graph]) -> Batch:
        """Convert list of NetworkX graphs to PyTorch Geometric batch."""
        data_list = []
        
        for graph in graphs:
            # Simple node features (just degree)
            node_features = torch.tensor([[graph.degree(node)] for node in graph.nodes()], 
                                       dtype=torch.float32)
            
            # Edge indices
            edge_list = list(graph.edges())
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            data = Data(x=node_features, edge_index=edge_index)
            data_list.append(data)
        
        return Batch.from_data_list(data_list)


class LogicalGANTrainer:
    """
    High-level trainer class for Logical GANs with different properties.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components based on config
        self.generator = GraphGenerator(
            latent_dim=config.get('latent_dim', 128),
            max_nodes=config.get('max_nodes', 20),
            hidden_dims=config.get('generator_hidden_dims', [256, 512, 1024])
        )
        
        self.discriminator = LogicalDiscriminator(
            logic_depth=config.get('logic_depth', 3),
            hidden_dim=config.get('discriminator_hidden_dim', 64),
            gnn_type=config.get('gnn_type', 'GCN')
        )
        
        # Property-specific setup
        self.property_checker = self._get_property_checker(config['property'])
        self.theory_graphs = self._generate_theory_graphs(config)
        
        self.logical_gan = LogicalGAN(
            generator=self.generator,
            discriminator=self.discriminator,
            property_checker=self.property_checker,
            theory_graphs=self.theory_graphs,
            ef_weight=config.get('ef_weight', 0.1),
            max_ef_rounds=config.get('max_ef_rounds', 3)
        )
    
    def _get_property_checker(self, property_name: str) -> PropertyChecker:
        """Get property checker based on name."""
        property_map = {
            'tree': TreeProperty(),
            'connectivity': ConnectivityProperty(),
            'bipartite': BipartiteProperty()
        }
        
        if property_name not in property_map:
            raise ValueError(f"Unknown property: {property_name}")
        
        return property_map[property_name]
    
    def _generate_theory_graphs(self, config: Dict) -> List[nx.Graph]:
        """Generate graphs satisfying the target theory."""
        property_name = config['property']
        num_graphs = config.get('theory_size', 1000)
        max_nodes = config.get('max_nodes', 20)
        
        theory_graphs = []
        
        if property_name == 'tree':
            # Generate random trees
            for _ in range(num_graphs):
                n = np.random.randint(3, max_nodes)
                tree = nx.random_tree(n)
                theory_graphs.append(tree)
                
        elif property_name == 'connectivity':
            # Generate connected graphs
            for _ in range(num_graphs):
                n = np.random.randint(3, max_nodes)
                # Ensure connectivity with higher edge probability
                p = max(2.0 / n, 0.3)  # Above percolation threshold
                while True:
                    graph = nx.erdos_renyi_graph(n, p)
                    if nx.is_connected(graph):
                        theory_graphs.append(graph)
                        break
                        
        elif property_name == 'bipartite':
            # Generate bipartite graphs
            for _ in range(num_graphs):
                n1 = np.random.randint(2, max_nodes // 2)
                n2 = np.random.randint(2, max_nodes - n1)
                p = np.random.uniform(0.1, 0.8)
                graph = nx.bipartite.random_graph(n1, n2, p)
                theory_graphs.append(graph)
        
        return theory_graphs
    
    def train(self) -> Dict[str, List[float]]:
        """Train the Logical GAN."""
        return self.logical_gan.train(
            theory_graphs=self.theory_graphs,
            epochs=self.config.get('epochs', 1000),
            batch_size=self.config.get('batch_size', 32),
            log_interval=self.config.get('log_interval', 100)
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate trained model."""
        generated_graphs = self.logical_gan.generate(
            num_samples=self.config.get('eval_samples', 500)
        )
        return self.logical_gan.evaluate(generated_graphs)


# Example usage and configuration
def create_tree_config() -> Dict:
    """Configuration for tree generation experiment."""
    return {
        'property': 'tree',
        'latent_dim': 128,
        'max_nodes': 15,
        'logic_depth': 3,
        'ef_weight': 0.1,
        'epochs': 500,
        'batch_size': 32,
        'theory_size': 1000,
        'gnn_type': 'GCN'
    }


def create_connectivity_config() -> Dict:
    """Configuration for connectivity experiment."""
    return {
        'property': 'connectivity',
        'latent_dim': 128,
        'max_nodes': 20,
        'logic_depth': 2,
        'ef_weight': 0.15,
        'epochs': 800,
        'batch_size': 32,
        'theory_size': 1500,
        'gnn_type': 'GIN'
    }


# Training example
if __name__ == "__main__":
    # Tree generation experiment
    config = create_tree_config()
    trainer = LogicalGANTrainer(config)
    
    print("Training Logical GAN for tree generation...")
    training_history = trainer.train()
    
    print("Evaluating generated trees...")
    results = trainer.evaluate()
    
    print("Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
