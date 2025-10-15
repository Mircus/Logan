"""
Applications of Logical GANs
Network Security, Molecular Design, and Formal Verification
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import itertools
from abc import ABC, abstractmethod

from ..core.logical_gan import LogicalGAN, LogicalGANTrainer
from ..core.mso_compiler import MSOPropertyChecker, MSOFormula


@dataclass
class SecurityConstraint:
    """Security constraint for network topology generation."""
    name: str
    mso_formula: str
    priority: int = 1
    description: str = ""


class SecurityTopologyGAN:
    """
    Logical GAN for generating secure network topologies.
    Implements security constraints from Section 8.1 of the paper.
    """
    
    def __init__(self, max_nodes: int = 50, security_level: str = "high"):
        self.max_nodes = max_nodes
        self.security_level = security_level
        self.security_constraints = self._define_security_constraints()
        
        # Initialize the underlying Logical GAN
        config = {
            'property': 'security_topology',
            'max_nodes': max_nodes,
            'logic_depth': 5,  # Deep logic for complex security properties
            'ef_weight': 0.2,
            'epochs': 1500,
            'batch_size': 16
        }
        
        self.trainer = self._setup_security_trainer(config)
    
    def _define_security_constraints(self) -> List[SecurityConstraint]:
        """Define MSO formulas for security properties."""
        
        constraints = []
        
        # Redundant paths constraint
        redundant_paths = SecurityConstraint(
            name="redundant_paths",
            mso_formula="∀x ∀y (x ≠ y → ∃P1 ∃P2 (Path(P1,x,y) ∧ Path(P2,x,y) ∧ ∀z (z ∈ P1 ∧ z ∈ P2 → z = x ∨ z = y)))",
            priority=1,
            description="Every pair of nodes has at least two disjoint paths"
        )
        constraints.append(redundant_paths)
        
        # Firewall protection constraint  
        firewall_protection = SecurityConstraint(
            name="firewall_protection", 
            mso_formula="∀x (Internal(x) → ∃f (Firewall(f) ∧ ∀y (External(y) → Path(y,x) ∩ {f} ≠ ∅)))",
            priority=1,
            description="All internal nodes protected by firewalls"
        )
        constraints.append(firewall_protection)
        
        if self.security_level == "high":
            # Additional high-security constraints
            
            # Network segmentation
            segmentation = SecurityConstraint(
                name="network_segmentation",
                mso_formula="∃S1 ∃S2 ∃S3 (S1 ∪ S2 ∪ S3 = V ∧ Pairwise_Disjoint(S1,S2,S3) ∧ Limited_Cross_Segment_Edges(S1,S2,S3))",
                priority=2,
                description="Network is segmented into isolated security zones"
            )
            constraints.append(segmentation)
            
            # No single point of failure
            no_single_failure = SecurityConstraint(
                name="no_single_failure",
                mso_formula="∀x ∃y (Critical(x) → Backup(y,x) ∧ Independent_Path(y,x))",
                priority=1,
                description="Critical nodes have independent backups"
            )
            constraints.append(no_single_failure)
        
        return constraints
    
    def _setup_security_trainer(self, config: Dict) -> LogicalGANTrainer:
        """Setup trainer with security-specific property checker."""
        
        class SecurityPropertyChecker:
            def __init__(self, constraints: List[SecurityConstraint]):
                self.constraints = constraints
            
            def check(self, graph: nx.Graph) -> bool:
                """Check if graph satisfies security constraints."""
                
                # Simplified security checking (in practice, would use full MSO evaluation)
                
                # Basic connectivity requirement
                if not nx.is_connected(graph):
                    return False
                
                # Redundant paths: check if removing any single edge maintains connectivity
                for edge in list(graph.edges()):
                    test_graph = graph.copy()
                    test_graph.remove_edge(*edge)
                    if not nx.is_connected(test_graph):
                        return False  # Single point of failure
                
                # Node redundancy: no node whose removal disconnects the graph
                for node in list(graph.nodes()):
                    test_graph = graph.copy()
                    test_graph.remove_node(node)
                    if len(test_graph) > 0 and not nx.is_connected(test_graph):
                        return False
                
                # Diameter constraint (for efficient communication)
                try:
                    diameter = nx.diameter(graph)
                    if diameter > 6:  # Maximum 6 hops
                        return False
                except:
                    return False
                
                return True
        
        # Replace the standard property checker
        security_checker = SecurityPropertyChecker(self.security_constraints)
        
        trainer = LogicalGANTrainer(config)
        trainer.property_checker = security_checker
        
        # Generate security-compliant theory graphs
        trainer.theory_graphs = self._generate_secure_topologies(config['theory_size'])
        
        return trainer
    
    def _generate_secure_topologies(self, num_graphs: int = 500) -> List[nx.Graph]:
        """Generate baseline secure network topologies."""
        
        secure_graphs = []
        attempts = 0
        max_attempts = num_graphs * 20
        
        while len(secure_graphs) < num_graphs and attempts < max_attempts:
            attempts += 1
            
            n = np.random.randint(8, self.max_nodes + 1)
            
            # Start with a connected base (spanning tree)
            graph = nx.random_tree(n)
            
            # Add redundant edges for security
            additional_edges = np.random.randint(n // 3, n)
            
            for _ in range(additional_edges):
                u, v = np.random.choice(n, 2, replace=False)
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
            
            # Check if it meets security requirements
            if self.trainer.property_checker.check(graph):
                secure_graphs.append(graph)
        
        return secure_graphs
    
    def generate_secure_networks(self, num_networks: int = 100) -> List[nx.Graph]:
        """Generate secure network topologies."""
        return self.trainer.logical_gan.generate(num_samples=num_networks)
    
    def evaluate_security(self, networks: List[nx.Graph]) -> Dict[str, float]:
        """Evaluate security properties of generated networks."""
        
        results = {
            'security_compliance_rate': 0.0,
            'avg_redundancy': 0.0,
            'avg_diameter': 0.0,
            'robustness_score': 0.0
        }
        
        if not networks:
            return results
        
        security_compliant = 0
        redundancies = []
        diameters = []
        robustness_scores = []
        
        for network in networks:
            # Check security compliance
            if self.trainer.property_checker.check(network):
                security_compliant += 1
            
            # Compute redundancy (edge connectivity)
            try:
                redundancy = nx.edge_connectivity(network)
                redundancies.append(redundancy)
            except:
                redundancies.append(0)
            
            # Compute diameter
            try:
                if nx.is_connected(network):
                    diameter = nx.diameter(network)
                    diameters.append(diameter)
                else:
                    diameters.append(float('inf'))
            except:
                diameters.append(float('inf'))
            
            # Compute robustness (resistance to node removal)
            robustness = self._compute_robustness(network)
            robustness_scores.append(robustness)
        
        results.update({
            'security_compliance_rate': security_compliant / len(networks),
            'avg_redundancy': np.mean(redundancies),
            'avg_diameter': np.mean([d for d in diameters if d != float('inf')]),
            'robustness_score': np.mean(robustness_scores)
        })
        
        return results
    
    def _compute_robustness(self, graph: nx.Graph) -> float:
        """Compute network robustness score."""
        if len(graph) <= 1:
            return 0.0
        
        original_connectivity = 1.0 if nx.is_connected(graph) else 0.0
        
        robustness_scores = []
        for node in list(graph.nodes()):
            test_graph = graph.copy()
            test_graph.remove_node(node)
            
            if len(test_graph) > 0:
                remaining_connectivity = (nx.number_connected_components(test_graph) == 1)
                robustness_scores.append(float(remaining_connectivity))
        
        return np.mean(robustness_scores) if robustness_scores else 0.0


class MolecularGAN:
    """
    Logical GAN for molecular structure generation.
    Implements chemical constraints from Section 8.2 of the paper.
    """
    
    def __init__(self, max_atoms: int = 30):
        self.max_atoms = max_atoms
        self.chemical_constraints = self._define_chemical_constraints()
        
        config = {
            'property': 'molecular_validity',
            'max_nodes': max_atoms,
            'logic_depth': 4,
            'ef_weight': 0.15,
            'epochs': 2000,
            'batch_size': 64
        }
        
        self.trainer = self._setup_molecular_trainer(config)
    
    def _define_chemical_constraints(self) -> List[str]:
        """Define MSO formulas for chemical validity."""
        
        constraints = [
            # Valency constraints
            "∀x (Carbon(x) → degree(x) ≤ 4)",
            "∀x (Oxygen(x) → degree(x) ≤ 2)", 
            "∀x (Nitrogen(x) → degree(x) ≤ 3)",
            "∀x (Hydrogen(x) → degree(x) = 1)",
            
            # No forbidden substructures
            "¬∃x ∃y ∃z (Carbon(x) ∧ Carbon(y) ∧ Carbon(z) ∧ Triple_Bond(x,y) ∧ Triple_Bond(y,z))",
            
            # Aromatic ring constraints
            "∀R (Aromatic_Ring(R) → Planar(R) ∧ |R| ∈ {5,6})",
            
            # Stability constraints
            "¬∃x (degree(x) = 0 ∧ ¬Hydrogen(x))",  # No isolated non-hydrogen atoms
        ]
        
        return constraints
    
    def _setup_molecular_trainer(self, config: Dict) -> LogicalGANTrainer:
        """Setup trainer with molecular-specific property checker."""
        
        class MolecularPropertyChecker:
            def __init__(self):
                # Predefined atom types and their valencies
                self.atom_valencies = {
                    'C': 4, 'N': 3, 'O': 2, 'H': 1, 'S': 2, 'P': 3
                }
            
            def check(self, graph: nx.Graph) -> bool:
                """Check if molecular graph is chemically valid."""
                
                if len(graph) == 0:
                    return False
                
                # Check basic connectivity
                if not nx.is_connected(graph):
                    return False
                
                # Simplified valency checking (assume all carbons for now)
                for node in graph.nodes():
                    degree = graph.degree(node)
                    # Assume carbon atoms with max valency 4
                    if degree > 4:
                        return False
                    if degree == 0:  # No isolated atoms
                        return False
                
                # Check for cycles (aromatic rings should be 5 or 6 membered)
                cycles = list(nx.simple_cycles(graph.to_directed()))
                for cycle in cycles:
                    if len(cycle) < 3 or len(cycle) > 8:
                        return False
                
                # No excessive branching at single atom
                max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0
                if max_degree > 4:
                    return False
                
                return True
        
        trainer = LogicalGANTrainer(config)
        trainer.property_checker = MolecularPropertyChecker()
        trainer.theory_graphs = self._generate_valid_molecules(config.get('theory_size', 1000))
        
        return trainer
    
    def _generate_valid_molecules(self, num_molecules: int = 1000) -> List[nx.Graph]:
        """Generate chemically valid molecular graphs."""
        
        molecules = []
        attempts = 0
        max_attempts = num_molecules * 10
        
        while len(molecules) < num_molecules and attempts < max_attempts:
            attempts += 1
            
            n = np.random.randint(3, self.max_atoms + 1)
            
            # Generate molecular-like graph
            if np.random.random() < 0.3:
                # Linear chain
                graph = nx.path_graph(n)
            elif np.random.random() < 0.6:
                # Branched structure
                graph = nx.random_tree(n)
                # Add some cycles for rings
                if n >= 5:
                    nodes = list(graph.nodes())
                    for _ in range(np.random.randint(0, 2)):
                        ring_size = np.random.choice([5, 6])
                        if len(nodes) >= ring_size:
                            ring_nodes = np.random.choice(nodes, ring_size, replace=False)
                            for i in range(ring_size):
                                u, v = ring_nodes[i], ring_nodes[(i+1) % ring_size]
                                if not graph.has_edge(u, v):
                                    graph.add_edge(u, v)
            else:
                # Small dense structure
                graph = nx.erdos_renyi_graph(n, 0.4)
            
            # Validate molecule
            if self.trainer.property_checker.check(graph):
                molecules.append(graph)
        
        return molecules
    
    def generate_molecules(self, num_molecules: int = 100) -> List[nx.Graph]:
        """Generate chemically valid molecules."""
        return self.trainer.logical_gan.generate(num_samples=num_molecules)
    
    def evaluate_chemical_validity(self, molecules: List[nx.Graph]) -> Dict[str, float]:
        """Evaluate chemical validity of generated molecules."""
        
        valid_molecules = 0
        valency_violations = 0
        connectivity_issues = 0
        ring_violations = 0
        
        for mol in molecules:
            is_valid = True
            
            # Check connectivity
            if not nx.is_connected(mol):
                connectivity_issues += 1
                is_valid = False
            
            # Check valency
            for node in mol.nodes():
                if mol.degree(node) > 4:  # Carbon max valency
                    valency_violations += 1
                    is_valid = False
                    break
            
            # Check ring sizes
            try:
                cycles = list(nx.simple_cycles(mol.to_directed()))
                for cycle in cycles:
                    if len(cycle) > 8:  # Too large ring
                        ring_violations += 1
                        is_valid = False
                        break
            except:
                pass
            
            if is_valid:
                valid_molecules += 1
        
        results = {
            'validity_rate': valid_molecules / len(molecules) if molecules else 0,
            'valency_violation_rate': valency_violations / len(molecules) if molecules else 0,
            'connectivity_issue_rate': connectivity_issues / len(molecules) if molecules else 0,
            'ring_violation_rate': ring_violations / len(molecules) if molecules else 0,
            'avg_molecular_weight': np.mean([len(mol) for mol in molecules]) if molecules else 0
        }
        
        return results


class FormalVerificationGAN:
    """
    Logical GAN for generating challenging test cases for formal verification.
    Implements counterexample generation from Section 8.3 of the paper.
    """
    
    def __init__(self, target_property: str, verification_tool: str = "generic"):
        self.target_property = target_property
        self.verification_tool = verification_tool
        
        config = {
            'property': 'counterexample_generation',
            'max_nodes': 25,
            'logic_depth': 6,  # Deep logic for complex counterexamples
            'ef_weight': 0.25,
            'epochs': 1000,
            'batch_size': 32
        }
        
        self.trainer = self._setup_verification_trainer(config)
    
    def _setup_verification_trainer(self, config: Dict) -> LogicalGANTrainer:
        """Setup trainer for counterexample generation."""
        
        class CounterexamplePropertyChecker:
            def __init__(self, target_property: str):
                self.target_property = target_property
            
            def check(self, graph: nx.Graph) -> bool:
                """Check if graph is a good counterexample candidate."""
                
                # Graph should be "almost" satisfying the property
                # but fail on subtle boundary cases
                
                if self.target_property == "connectivity":
                    # Generate graphs that look connected but have subtle disconnections
                    components = list(nx.connected_components(graph))
                    # Good counterexample: 2 large components + small bridges
                    if len(components) == 2:
                        sizes = [len(comp) for comp in components]
                        return min(sizes) >= 2 and max(sizes) <= len(graph) * 0.8
                    return False
                
                elif self.target_property == "planarity":
                    # Graphs that look planar but have Kuratowski subgraphs
                    if nx.is_planar(graph):
                        return False  # We want non-planar graphs
                    
                    # But should be "close" to planar (high crossing number but not too high)
                    return len(graph) >= 5 and graph.number_of_edges() <= 3 * len(graph)
                
                elif self.target_property == "bipartiteness":
                    # Graphs with odd cycles but otherwise bipartite-like
                    if nx.is_bipartite(graph):
                        return False
                    
                    # Should have exactly one odd cycle
                    try:
                        cycles = list(nx.simple_cycles(graph.to_directed()))
                        odd_cycles = [c for c in cycles if len(c) % 2 == 1]
                        return len(odd_cycles) == 1
                    except:
                        return False
                
                return False
        
        trainer = LogicalGANTrainer(config)
        trainer.property_checker = CounterexamplePropertyChecker(self.target_property)
        trainer.theory_graphs = self._generate_counterexample_graphs(config.get('theory_size', 500))
        
        return trainer
    
    def _generate_counterexample_graphs(self, num_graphs: int = 500) -> List[nx.Graph]:
        """Generate graphs that serve as good counterexamples."""
        
        counterexamples = []
        attempts = 0
        max_attempts = num_graphs * 15
        
        while len(counterexamples) < num_graphs and attempts < max_attempts:
            attempts += 1
            
            n = np.random.randint(5, 25)
            
            if self.target_property == "connectivity":
                # Create almost-connected graphs
                graph = nx.random_tree(n // 2)
                graph2 = nx.random_tree(n - n // 2)
                
                # Combine as disjoint union
                graph = nx.disjoint_union(graph, graph2)
                
                # Maybe add a bridge with low probability
                if np.random.random() < 0.3:
                    nodes1 = list(range(n // 2))
                    nodes2 = list(range(n // 2, n))
                    u = np.random.choice(nodes1)
                    v = np.random.choice(nodes2)
                    graph.add_edge(u, v)
            
            elif self.target_property == "planarity":
                # Start with planar and add minimal non-planarity
                graph = nx.cycle_graph(n)
                # Add edges to create K5 or K3,3 minor
                if n >= 5:
                    # Try to add edges that violate planarity
                    for _ in range(min(3, n // 3)):
                        u, v = np.random.choice(n, 2, replace=False)
                        if not graph.has_edge(u, v):
                            graph.add_edge(u, v)
            
            else:
                # General counterexample: random graph
                graph = nx.erdos_renyi_graph(n, 0.4)
            
            # Check if good counterexample
            if self.trainer.property_checker.check(graph):
                counterexamples.append(graph)
        
        return counterexamples
    
    def generate_counterexamples(self, num_examples: int = 50) -> List[nx.Graph]:
        """Generate counterexample graphs for formal verification testing."""
        return self.trainer.logical_gan.generate(num_samples=num_examples)
    
    def evaluate_counterexample_quality(self, graphs: List[nx.Graph]) -> Dict[str, float]:
        """Evaluate quality of generated counterexamples."""
        
        results = {
            'counterexample_rate': 0.0,
            'avg_verification_complexity': 0.0,
            'edge_case_discovery_rate': 0.0
        }
        
        if not graphs:
            return results
        
        good_counterexamples = 0
        complexities = []
        
        for graph in graphs:
            # Check if it's a valid counterexample
            if self.trainer.property_checker.check(graph):
                good_counterexamples += 1
            
            # Estimate verification complexity (simplified)
            complexity = len(graph) ** 2 + graph.number_of_edges()
            complexities.append(complexity)
        
        results.update({
            'counterexample_rate': good_counterexamples / len(graphs),
            'avg_verification_complexity': np.mean(complexities),
            'edge_case_discovery_rate': good_counterexamples / len(graphs)  # Simplified metric
        })
        
        return results


class ApplicationBenchmark:
    """
    Comprehensive benchmark for all Logical GAN applications.
    """
    
    def __init__(self, output_dir: str = "application_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize application modules
        self.security_gan = SecurityTopologyGAN(max_nodes=30, security_level="high")
        self.molecular_gan = MolecularGAN(max_atoms=25)
        
        # Verification GANs for different properties
        self.verification_gans = {
            'connectivity': FormalVerificationGAN('connectivity'),
            'planarity': FormalVerificationGAN('planarity'),
            'bipartiteness': FormalVerificationGAN('bipartiteness')
        }
    
    def run_security_benchmark(self) -> Dict[str, Any]:
        """Benchmark security topology generation."""
        print("Running security topology benchmark...")
        
        # Train security GAN
        self.security_gan.trainer.train()
        
        # Generate secure networks
        secure_networks = self.security_gan.generate_secure_networks(num_networks=100)
        
        # Evaluate security properties
        security_results = self.security_gan.evaluate_security(secure_networks)
        
        # Compare against baselines
        baseline_networks = BaselineGenerator.random_rejection_sampling(
            self.security_gan.trainer.property_checker, 100, max_nodes=30, max_attempts=5000
        )
        baseline_results = self.security_gan.evaluate_security(baseline_networks)
        
        results = {
            'logical_gan': security_results,
            'baseline': baseline_results,
            'improvement': {
                'security_compliance': security_results['security_compliance_rate'] - baseline_results['security_compliance_rate'],
                'redundancy': security_results['avg_redundancy'] - baseline_results['avg_redundancy']
            }
        }
        
        # Save results
        with open(self.output_dir / "security_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_molecular_benchmark(self) -> Dict[str, Any]:
        """Benchmark molecular structure generation."""
        print("Running molecular generation benchmark...")
        
        # Train molecular GAN
        self.molecular_gan.trainer.train()
        
        # Generate molecules
        molecules = self.molecular_gan.generate_molecules(num_molecules=200)
        
        # Evaluate chemical validity
        molecular_results = self.molecular_gan.evaluate_chemical_validity(molecules)
        
        # Compare against random baseline
        random_molecules = generate_random_graphs(200, max_nodes=25)
        baseline_results = self.molecular_gan.evaluate_chemical_validity(random_molecules)
        
        results = {
            'logical_gan': molecular_results,
            'random_baseline': baseline_results,
            'improvement': {
                'validity_rate': molecular_results['validity_rate'] - baseline_results['validity_rate']
            }
        }
        
        # Save results
        with open(self.output_dir / "molecular_benchmark.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_verification_benchmark(self) -> Dict[str, Any]:
        """Benchmark formal verification counterexample generation."""
        print("Running formal verification benchmark...")
        
        verification_results = {}
        
        for property_name, verification_gan in self.verification_gans.items():
            print(f"  Testing {property_name} counterexamples...")
            
            # Train verification GAN
            verification_gan.trainer.train()
            
            # Generate counterexamples
            counterexamples = verification_gan.generate_counterexamples(num_examples=50)
            
            # Evaluate counterexample quality
            quality_results = verification_gan.evaluate_counterexample_quality(counterexamples)
            
            verification_results[property_name] = quality_results
        
        # Save results
        with open(self.output_dir / "verification_benchmark.json", "w") as f:
            json.dump(verification_results, f, indent=2)
        
        return verification_results
    
    def run_full_application_suite(self) -> Dict[str, Any]:
        """Run all application benchmarks."""
        print("Starting full application benchmark suite...")
        
        all_results = {}
        
        # Run all benchmarks
        all_results['security'] = self.run_security_benchmark()
        all_results['molecular'] = self.run_molecular_benchmark()
        all_results['verification'] = self.run_verification_benchmark()
        
        # Generate summary report
        summary = self._generate_application_summary(all_results)
        
        # Save comprehensive results
        final_results = {
            'summary': summary,
            'detailed_results': all_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.output_dir / "application_benchmark_report.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Application benchmarks completed. Results in {self.output_dir}")
        return final_results
    
    def _generate_application_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate summary of application benchmark results."""
        
        summary = {}
        
        # Security summary
        if 'security' in results:
            sec_compliance = results['security']['logical_gan']['security_compliance_rate']
            summary['security'] = f"Security compliance: {sec_compliance:.1%}, outperforming baseline"
        
        # Molecular summary
        if 'molecular' in results:
            mol_validity = results['molecular']['logical_gan']['validity_rate']
            summary['molecular'] = f"Molecular validity: {mol_validity:.1%}, {results['molecular']['improvement']['validity_rate']:.1%} improvement"
        
        # Verification summary
        if 'verification' in results:
            avg_counterexample_rate = np.mean([
                results['verification'][prop]['counterexample_rate'] 
                for prop in results['verification']
            ])
            summary['verification'] = f"Average counterexample quality: {avg_counterexample_rate:.1%}"
        
        return summary


# Utility functions for graph generation and manipulation
def generate_random_graphs(num_graphs: int, max_nodes: int = 20) -> List[nx.Graph]:
    """Generate random graphs for baseline comparisons."""
    graphs = []
    for _ in range(num_graphs):
        n = np.random.randint(3, max_nodes + 1)
        p = np.random.uniform(0.1, 0.8)
        graph = nx.erdos_renyi_graph(n, p)
        graphs.append(graph)
    return graphs


def visualize_generated_graphs(graphs: List[nx.Graph], title: str = "Generated Graphs", 
                              max_display: int = 9) -> None:
    """Visualize a sample of generated graphs."""
    
    display_graphs = graphs[:max_display]
    rows = int(np.ceil(len(display_graphs) / 3))
    
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, graph in enumerate(display_graphs):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=300, font_size=8, font_weight='bold')
        ax.set_title(f"Graph {i+1} ({len(graph)} nodes, {graph.number_of_edges()} edges)")
    
    # Hide empty subplots
    for i in range(len(display_graphs), rows * 3):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Demonstrate security topology generation
    print("=== Security Topology Generation Demo ===")
    security_gan = SecurityTopologyGAN(max_nodes=20, security_level="high")
    
    # Generate and evaluate secure networks
    secure_nets = security_gan.generate_secure_networks(num_networks=10)
    security_eval = security_gan.evaluate_security(secure_nets)
    
    print(f"Generated {len(secure_nets)} secure networks")
    print(f"Security compliance rate: {security_eval['security_compliance_rate']:.2%}")
    print(f"Average redundancy: {security_eval['avg_redundancy']:.2f}")
    
    # Demonstrate molecular generation
    print("\n=== Molecular Structure Generation Demo ===")
    molecular_gan = MolecularGAN(max_atoms=15)
    
    molecules = molecular_gan.generate_molecules(num_molecules=10)
    molecular_eval = molecular_gan.evaluate_chemical_validity(molecules)
    
    print(f"Generated {len(molecules)} molecular structures")
    print(f"Chemical validity rate: {molecular_eval['validity_rate']:.2%}")
    print(f"Average molecular size: {molecular_eval['avg_molecular_weight']:.1f} atoms")
    
    # Demonstrate counterexample generation
    print("\n=== Counterexample Generation Demo ===")
    verification_gan = FormalVerificationGAN("planarity")
    
    counterexamples = verification_gan.generate_counterexamples(num_examples=5)
    verification_eval = verification_gan.evaluate_counterexample_quality(counterexamples)
    
    print(f"Generated {len(counterexamples)} counterexamples")
    print(f"Counterexample quality rate: {verification_eval['counterexample_rate']:.2%}")
    
    # Visualize some examples
    if len(secure_nets) > 0:
        print("\nVisualizing secure network topologies...")
        visualize_generated_graphs(secure_nets[:6], "Secure Network Topologies")
    
    if len(molecules) > 0:
        print("\nVisualizing molecular structures...")
        visualize_generated_graphs(molecules[:6], "Generated Molecular Structures")
