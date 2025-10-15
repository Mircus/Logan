"""
Ehrenfeucht-Fraïssé Games Implementation
Based on the Logical GANs paper
"""

import networkx as nx
import itertools
from typing import Dict, List, Tuple, Set, FrozenSet
import numpy as np
from functools import lru_cache


class EFGameSimulator:
    """
    Simulator for Ehrenfeucht-Fraïssé games between graph structures.
    Implements both exact and approximate EF-distance computation.
    """
    
    def __init__(self, graph_a: nx.Graph, graph_b: nx.Graph):
        self.graph_a = graph_a
        self.graph_b = graph_b
        self.memo = {}  # Memoization for dynamic programming
        
    def ef_distance(self, max_rounds: int = 10) -> int:
        """
        Compute EF-distance between graphs.
        Returns the minimum number of rounds where Spoiler wins.
        """
        # Check basic compatibility first
        if not self._initial_compatibility_check():
            return 0
            
        for k in range(1, max_rounds + 1):
            if not self.duplicator_wins(k):
                return k
        return max_rounds
    
    def duplicator_wins(self, rounds: int) -> bool:
        """Check if Duplicator has winning strategy in k rounds."""
        if rounds == 0:
            return self._initial_position_check()
        
        # Use dynamic programming with memoization
        return self._duplicator_wins_memo(frozenset(), frozenset(), rounds)
    
    @lru_cache(maxsize=10000)
    def _duplicator_wins_memo(self, 
                              chosen_a: FrozenSet[int], 
                              chosen_b: FrozenSet[int],
                              rounds_left: int) -> bool:
        """Memoized recursive check for Duplicator winning strategy."""
        
        if rounds_left == 0:
            return self._check_partial_isomorphism(chosen_a, chosen_b)
        
        # Spoiler's turn - they choose from either graph
        # Duplicator wins if they can respond to ANY Spoiler choice
        
        # Case 1: Spoiler chooses from graph A
        for node_a in self.graph_a.nodes():
            if node_a not in chosen_a:
                # Duplicator must find a response in graph B
                duplicator_can_respond = False
                for node_b in self.graph_b.nodes():
                    if node_b not in chosen_b:
                        if (self._check_extension_valid(chosen_a, chosen_b, node_a, node_b) and
                            self._duplicator_wins_memo(chosen_a | {node_a}, 
                                                      chosen_b | {node_b}, 
                                                      rounds_left - 1)):
                            duplicator_can_respond = True
                            break
                if not duplicator_can_respond:
                    return False
        
        # Case 2: Spoiler chooses from graph B  
        for node_b in self.graph_b.nodes():
            if node_b not in chosen_b:
                # Duplicator must find a response in graph A
                duplicator_can_respond = False
                for node_a in self.graph_a.nodes():
                    if node_a not in chosen_a:
                        if (self._check_extension_valid(chosen_a, chosen_b, node_a, node_b) and
                            self._duplicator_wins_memo(chosen_a | {node_a}, 
                                                      chosen_b | {node_b}, 
                                                      rounds_left - 1)):
                            duplicator_can_respond = True
                            break
                if not duplicator_can_respond:
                    return False
        
        return True
    
    def _check_extension_valid(self, 
                              chosen_a: FrozenSet[int], 
                              chosen_b: FrozenSet[int],
                              new_a: int, 
                              new_b: int) -> bool:
        """Check if adding (new_a, new_b) preserves partial isomorphism."""
        
        # Convert to lists to maintain order
        a_list = list(chosen_a) + [new_a]
        b_list = list(chosen_b) + [new_b]
        
        # Check all pairs preserve edge relationships
        for i in range(len(a_list)):
            for j in range(i + 1, len(a_list)):
                edge_a = self.graph_a.has_edge(a_list[i], a_list[j])
                edge_b = self.graph_b.has_edge(b_list[i], b_list[j])
                if edge_a != edge_b:
                    return False
        return True
    
    def _check_partial_isomorphism(self, 
                                  chosen_a: FrozenSet[int], 
                                  chosen_b: FrozenSet[int]) -> bool:
        """Check if chosen nodes form a partial isomorphism."""
        if len(chosen_a) != len(chosen_b):
            return False
            
        if len(chosen_a) == 0:
            return True
            
        a_list = list(chosen_a)
        b_list = list(chosen_b)
        
        # Check all pairs preserve edge relationships
        for i, j in itertools.combinations(range(len(a_list)), 2):
            edge_a = self.graph_a.has_edge(a_list[i], a_list[j])
            edge_b = self.graph_b.has_edge(b_list[i], b_list[j])
            if edge_a != edge_b:
                return False
        return True
    
    def _initial_position_check(self) -> bool:
        """Check basic structural compatibility."""
        # Compare degree sequences
        degree_seq_a = sorted([d for n, d in self.graph_a.degree()])
        degree_seq_b = sorted([d for n, d in self.graph_b.degree()])
        return degree_seq_a == degree_seq_b
    
    def _initial_compatibility_check(self) -> bool:
        """Quick compatibility check before expensive computation."""
        # Number of nodes must match for 0-distance
        if len(self.graph_a) != len(self.graph_b):
            return len(self.graph_a) == len(self.graph_b)
        
        # Number of edges must match
        if self.graph_a.number_of_edges() != self.graph_b.number_of_edges():
            return False
            
        return True


class ApproximateEFDistance:
    """
    Approximate EF-distance computation for scalability.
    Uses Monte Carlo sampling and heuristics.
    """
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def compute_distance(self, graph_a: nx.Graph, graph_b: nx.Graph, max_rounds: int = 5) -> float:
        """Compute approximate EF-distance using sampling."""
        
        if len(graph_a) != len(graph_b):
            return float('inf')
        
        # Sample random node pairs and check preservation
        distances = []
        
        for _ in range(self.num_samples):
            # Random sampling strategy
            sample_distance = self._sample_ef_distance(graph_a, graph_b, max_rounds)
            distances.append(sample_distance)
        
        return np.mean(distances)
    
    def _sample_ef_distance(self, graph_a: nx.Graph, graph_b: nx.Graph, max_rounds: int) -> int:
        """Sample-based EF-distance estimation."""
        
        for k in range(1, max_rounds + 1):
            # Sample random partial mappings
            if not self._sample_duplicator_strategy(graph_a, graph_b, k):
                return k
        return max_rounds
    
    def _sample_duplicator_strategy(self, graph_a: nx.Graph, graph_b: nx.Graph, rounds: int) -> bool:
        """Sample whether duplicator can win for k rounds."""
        
        num_trials = min(100, 2 ** rounds)  # Limit trials for efficiency
        
        for _ in range(num_trials):
            # Simulate random game
            chosen_a = set()
            chosen_b = set()
            nodes_a = list(graph_a.nodes())
            nodes_b = list(graph_b.nodes())
            
            for round_num in range(rounds):
                # Spoiler chooses randomly
                if np.random.random() < 0.5 and len(chosen_a) < len(nodes_a):
                    # Choose from A
                    available_a = [n for n in nodes_a if n not in chosen_a]
                    spoiler_choice = np.random.choice(available_a)
                    
                    # Duplicator responds
                    available_b = [n for n in nodes_b if n not in chosen_b]
                    if not available_b:
                        return False
                    
                    # Try to find valid response
                    found_response = False
                    for node_b in available_b:
                        if self._is_valid_extension(graph_a, graph_b, chosen_a, chosen_b, 
                                                  spoiler_choice, node_b):
                            chosen_a.add(spoiler_choice)
                            chosen_b.add(node_b)
                            found_response = True
                            break
                    
                    if not found_response:
                        return False
                        
                elif len(chosen_b) < len(nodes_b):
                    # Choose from B
                    available_b = [n for n in nodes_b if n not in chosen_b]
                    spoiler_choice = np.random.choice(available_b)
                    
                    # Duplicator responds
                    available_a = [n for n in nodes_a if n not in chosen_a]
                    if not available_a:
                        return False
                    
                    # Try to find valid response
                    found_response = False
                    for node_a in available_a:
                        if self._is_valid_extension(graph_a, graph_b, chosen_a, chosen_b,
                                                  node_a, spoiler_choice):
                            chosen_a.add(node_a)
                            chosen_b.add(spoiler_choice)
                            found_response = True
                            break
                    
                    if not found_response:
                        return False
        
        return True  # If all trials succeed, assume duplicator can win
    
    def _is_valid_extension(self, graph_a: nx.Graph, graph_b: nx.Graph,
                           chosen_a: Set[int], chosen_b: Set[int],
                           new_a: int, new_b: int) -> bool:
        """Check if extension preserves partial isomorphism."""
        
        # Check against all existing pairs
        for a_node, b_node in zip(chosen_a, chosen_b):
            edge_a = graph_a.has_edge(a_node, new_a)
            edge_b = graph_b.has_edge(b_node, new_b)
            if edge_a != edge_b:
                return False
        return True


def ef_distance_to_theory(graph: nx.Graph, theory_graphs: List[nx.Graph], 
                         max_rounds: int = 5) -> float:
    """
    Compute EF-distance from a graph to a theory (set of graphs).
    Returns minimum distance to any graph satisfying the theory.
    """
    min_distance = float('inf')
    
    for theory_graph in theory_graphs:
        simulator = EFGameSimulator(graph, theory_graph)
        distance = simulator.ef_distance(max_rounds)
        min_distance = min(min_distance, distance)
        
        if min_distance == 0:  # Early termination
            break
    
    return min_distance


def batch_ef_distances(graphs_a: List[nx.Graph], graphs_b: List[nx.Graph],
                      max_rounds: int = 5) -> np.ndarray:
    """Compute EF-distances for batches of graphs efficiently."""
    
    distances = np.zeros((len(graphs_a), len(graphs_b)))
    
    for i, graph_a in enumerate(graphs_a):
        for j, graph_b in enumerate(graphs_b):
            simulator = EFGameSimulator(graph_a, graph_b)
            distances[i, j] = simulator.ef_distance(max_rounds)
    
    return distances


# Example usage and testing
if __name__ == "__main__":
    # Create test graphs
    tree1 = nx.path_graph(4)  # Linear tree
    tree2 = nx.star_graph(3)  # Star tree
    cycle = nx.cycle_graph(4)  # 4-cycle (not a tree)
    
    # Test EF-distance computation
    simulator1 = EFGameSimulator(tree1, tree2)
    simulator2 = EFGameSimulator(tree1, cycle)
    
    print(f"EF-distance between trees: {simulator1.ef_distance()}")
    print(f"EF-distance tree vs cycle: {simulator2.ef_distance()}")
    
    # Test approximate distance
    approx = ApproximateEFDistance(num_samples=500)
    approx_dist = approx.compute_distance(tree1, cycle, max_rounds=3)
    print(f"Approximate EF-distance: {approx_dist}")
