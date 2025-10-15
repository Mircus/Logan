"""MSO compiler & property library (packaged)."""
"""
Monadic Second-Order Logic Formula Compiler
Compiles MSO formulas to efficient graph property checkers
"""

import networkx as nx
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Union, Optional
from dataclasses import dataclass
from enum import Enum
import itertools


class QuantifierType(Enum):
    EXISTENTIAL_FO = "∃"      # First-order existential
    UNIVERSAL_FO = "∀"        # First-order universal  
    EXISTENTIAL_SO = "∃_SO"   # Second-order existential
    UNIVERSAL_SO = "∀_SO"     # Second-order universal


@dataclass
class Variable:
    name: str
    is_set_variable: bool = False
    
    def __post_init__(self):
        # Convention: uppercase variables are set variables
        if self.name and self.name[0].isupper():
            self.is_set_variable = True


class MSOFormula(ABC):
    """Abstract base class for MSO formulas."""
    
    @abstractmethod
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        """Evaluate formula on structure with given variable assignment."""
        pass
    
    @abstractmethod
    def free_variables(self) -> Set[str]:
        """Return set of free variables in formula."""
        pass


class AtomicFormula(MSOFormula):
    """Atomic formulas: Edge(x,y), In(x,X), etc."""
    
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate
        self.args = args
    
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        if self.predicate == "Edge":
            u, v = self.args
            if u not in assignment or v not in assignment:
                return False
            return structure.has_edge(assignment[u], assignment[v])
        
        elif self.predicate == "In":
            elem, set_var = self.args
            if elem not in assignment or set_var not in assignment:
                return False
            return assignment[elem] in assignment[set_var]
        
        elif self.predicate == "Eq":
            x, y = self.args
            if x not in assignment or y not in assignment:
                return False
            return assignment[x] == assignment[y]
        
        else:
            raise NotImplementedError(f"Predicate {self.predicate} not implemented")
    
    def free_variables(self) -> Set[str]:
        return set(self.args)


class Conjunction(MSOFormula):
    """Logical AND."""
    
    def __init__(self, left: MSOFormula, right: MSOFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        return (self.left.evaluate(structure, assignment) and 
                self.right.evaluate(structure, assignment))
    
    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()


class Disjunction(MSOFormula):
    """Logical OR."""
    
    def __init__(self, left: MSOFormula, right: MSOFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        return (self.left.evaluate(structure, assignment) or 
                self.right.evaluate(structure, assignment))
    
    def free_variables(self) -> Set[str]:
        return self.left.free_variables() | self.right.free_variables()


class Negation(MSOFormula):
    """Logical NOT."""
    
    def __init__(self, formula: MSOFormula):
        self.formula = formula
    
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        return not self.formula.evaluate(structure, assignment)
    
    def free_variables(self) -> Set[str]:
        return self.formula.free_variables()


class ExistentialQuantifier(MSOFormula):
    """Existential quantification (both first and second-order)."""
    
    def __init__(self, variable: Variable, formula: MSOFormula):
        self.variable = variable
        self.formula = formula
    
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        if self.variable.is_set_variable:
            # Second-order: quantify over subsets of vertices
            return self._evaluate_second_order(structure, assignment)
        else:
            # First-order: quantify over vertices
            return self._evaluate_first_order(structure, assignment)
    
    def _evaluate_first_order(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        """Quantify over vertices."""
        for node in structure.nodes():
            new_assignment = assignment.copy()
            new_assignment[self.variable.name] = node
            
            if self.formula.evaluate(structure, new_assignment):
                return True
        return False
    
    def _evaluate_second_order(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        """Quantify over vertex subsets."""
        nodes = list(structure.nodes())
        
        # For small graphs, enumerate all subsets
        if len(nodes) <= 15:
            for i in range(2 ** len(nodes)):
                subset = {nodes[j] for j in range(len(nodes)) if (i >> j) & 1}
                new_assignment = assignment.copy()
                new_assignment[self.variable.name] = subset
                
                if self.formula.evaluate(structure, new_assignment):
                    return True
        else:
            # For larger graphs, use heuristic sampling
            return self._heuristic_second_order_eval(structure, assignment, nodes)
        
        return False
    
    def _heuristic_second_order_eval(self, structure: nx.Graph, 
                                   assignment: Dict[str, Any], nodes: List) -> bool:
        """Heuristic evaluation for large graphs."""
        # Sample various subset sizes and random subsets
        max_samples = 1000
        
        for _ in range(max_samples):
            subset_size = np.random.randint(0, len(nodes) + 1)
            subset = set(np.random.choice(nodes, size=subset_size, replace=False))
            
            new_assignment = assignment.copy()
            new_assignment[self.variable.name] = subset
            
            if self.formula.evaluate(structure, new_assignment):
                return True
        
        return False
    
    def free_variables(self) -> Set[str]:
        return self.formula.free_variables() - {self.variable.name}


class UniversalQuantifier(MSOFormula):
    """Universal quantification (¬∃¬)."""
    
    def __init__(self, variable: Variable, formula: MSOFormula):
        self.variable = variable
        self.formula = formula
    
    def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
        # ∀x φ(x) ≡ ¬∃x ¬φ(x)
        negated_inner = Negation(self.formula)
        existential = ExistentialQuantifier(self.variable, negated_inner)
        negated_existential = Negation(existential)
        return negated_existential.evaluate(structure, assignment)
    
    def free_variables(self) -> Set[str]:
        return self.formula.free_variables() - {self.variable.name}


class MSOCompiler:
    """
    Compiler for MSO formulas to executable property checkers.
    """
    
    def __init__(self):
        self.operators = {
            '∧': 'and',
            '∨': 'or', 
            '¬': 'not',
            '∃': 'exists',
            '∀': 'forall',
            '→': 'implies',
            '↔': 'iff'
        }
        
    def compile(self, formula_string: str) -> MSOFormula:
        """Compile MSO formula string to executable formula object."""
        tokens = self._tokenize(formula_string)
        parsed = self._parse(tokens)
        return parsed
    
    def _tokenize(self, formula: str) -> List[str]:
        """Tokenize MSO formula string."""
        # Define token patterns
        patterns = [
            r'∃[A-Z][a-zA-Z0-9]*',  # Second-order existential
            r'∀[A-Z][a-zA-Z0-9]*',  # Second-order universal
            r'∃[a-z][a-zA-Z0-9]*',  # First-order existential
            r'∀[a-z][a-zA-Z0-9]*',  # First-order universal
            r'[A-Za-z_][A-Za-z0-9_]*',  # Variables and predicates
            r'[∧∨¬→↔]',  # Logical operators
            r'[().,]',   # Punctuation
            r'=',        # Equality
        ]
        
        pattern = '|'.join(f'({p})' for p in patterns)
        tokens = re.findall(pattern, formula)
        
        # Flatten and filter empty strings
        return [token for group in tokens for token in group if token]
    
    def _parse(self, tokens: List[str]) -> MSOFormula:
        """Parse tokens into formula tree using recursive descent."""
        self.tokens = tokens
        self.pos = 0
        return self._parse_formula()
    
    def _parse_formula(self) -> MSOFormula:
        """Parse a complete formula."""
        return self._parse_implication()
    
    def _parse_implication(self) -> MSOFormula:
        """Parse implications (→, ↔)."""
        left = self._parse_disjunction()
        
        while self._current_token() in ['→', '↔']:
            op = self._consume_token()
            right = self._parse_disjunction()
            
            if op == '→':
                # A → B ≡ ¬A ∨ B
                left = Disjunction(Negation(left), right)
            elif op == '↔':
                # A ↔ B ≡ (A → B) ∧ (B → A)
                left = Conjunction(
                    Disjunction(Negation(left), right),
                    Disjunction(Negation(right), left)
                )
        
        return left
    
    def _parse_disjunction(self) -> MSOFormula:
        """Parse disjunctions (∨)."""
        left = self._parse_conjunction()
        
        while self._current_token() == '∨':
            self._consume_token()
            right = self._parse_conjunction()
            left = Disjunction(left, right)
        
        return left
    
    def _parse_conjunction(self) -> MSOFormula:
        """Parse conjunctions (∧)."""
        left = self._parse_negation()
        
        while self._current_token() == '∧':
            self._consume_token()
            right = self._parse_negation()
            left = Conjunction(left, right)
        
        return left
    
    def _parse_negation(self) -> MSOFormula:
        """Parse negations (¬)."""
        if self._current_token() == '¬':
            self._consume_token()
            formula = self._parse_negation()
            return Negation(formula)
        
        return self._parse_quantification()
    
    def _parse_quantification(self) -> MSOFormula:
        """Parse quantifiers (∃, ∀)."""
        token = self._current_token()
        
        if token and (token.startswith('∃') or token.startswith('∀')):
            quantifier_token = self._consume_token()
            var_name = quantifier_token[1:]  # Remove quantifier symbol
            variable = Variable(var_name)
            
            formula = self._parse_quantification()
            
            if quantifier_token.startswith('∃'):
                return ExistentialQuantifier(variable, formula)
            else:
                return UniversalQuantifier(variable, formula)
        
        return self._parse_atomic()
    
    def _parse_atomic(self) -> MSOFormula:
        """Parse atomic formulas and parenthesized expressions."""
        if self._current_token() == '(':
            self._consume_token()
            formula = self._parse_formula()
            if self._current_token() != ')':
                raise SyntaxError("Expected closing parenthesis")
            self._consume_token()
            return formula
        
        # Parse atomic formula
        predicate = self._consume_token()
        
        if self._current_token() == '(':
            self._consume_token()
            args = []
            
            while self._current_token() != ')':
                args.append(self._consume_token())
                if self._current_token() == ',':
                    self._consume_token()
            
            self._consume_token()  # Consume ')'
            return AtomicFormula(predicate, args)
        
        raise SyntaxError(f"Unexpected token: {predicate}")
    
    def _current_token(self) -> Optional[str]:
        """Get current token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def _consume_token(self) -> str:
        """Consume and return current token."""
        if self.pos >= len(self.tokens):
            raise SyntaxError("Unexpected end of input")
        token = self.tokens[self.pos]
        self.pos += 1
        return token


class MSOPropertyChecker:
    """
    Efficient checker for MSO-definable graph properties.
    Uses compiled formulas for fast evaluation.
    """
    
    def __init__(self, formula: Union[str, MSOFormula], name: str = ""):
        self.name = name
        if isinstance(formula, str):
            compiler = MSOCompiler()
            self.formula = compiler.compile(formula)
        else:
            self.formula = formula
    
    def check(self, graph: nx.Graph) -> bool:
        """Check if graph satisfies the MSO property."""
        # Start with empty assignment
        empty_assignment = {}
        return self.formula.evaluate(graph, empty_assignment)
    
    def batch_check(self, graphs: List[nx.Graph]) -> List[bool]:
        """Efficiently check multiple graphs."""
        return [self.check(graph) for graph in graphs]


# Pre-defined MSO properties
class StandardMSOProperties:
    """Collection of standard MSO-definable graph properties."""
    
    @staticmethod
    def connectivity() -> MSOPropertyChecker:
        """∃X (X ⊆ V ∧ Connected(X) ∧ X = V)"""
        # Simplified: just check if graph is connected
        class ConnectivityFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                return nx.is_connected(structure) if len(structure) > 0 else True
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(ConnectivityFormula(), "connectivity")
    
    @staticmethod
    def tree_property() -> MSOPropertyChecker:
        """Connected(G) ∧ |E| = |V| - 1"""
        class TreeFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                if len(structure) == 0:
                    return True
                return (nx.is_connected(structure) and 
                       structure.number_of_edges() == len(structure) - 1)
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(TreeFormula(), "tree")
    
    @staticmethod
    def bipartiteness() -> MSOPropertyChecker:
        """∃X ∃Y (X ∪ Y = V ∧ X ∩ Y = ∅ ∧ ∀e ∈ E: endpoint(e) ∉ X ∨ X ∩ Y ∨ Y)"""
        class BipartiteFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                return nx.is_bipartite(structure)
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(BipartiteFormula(), "bipartite")
    
    @staticmethod
    def even_parity() -> MSOPropertyChecker:
        """Even number of vertices."""
        class EvenParityFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                return len(structure) % 2 == 0
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(EvenParityFormula(), "even_parity")
    
    @staticmethod
    def planarity() -> MSOPropertyChecker:
        """Graph is planar (Kuratowski's theorem)."""
        class PlanarityFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                try:
                    return nx.is_planar(structure)
                except:
                    return False  # Default for edge cases
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(PlanarityFormula(), "planarity")


class AdvancedMSOProperties:
    """More complex MSO properties for advanced applications."""
    
    @staticmethod
    def has_triangle() -> MSOPropertyChecker:
        """∃x ∃y ∃z (Edge(x,y) ∧ Edge(y,z) ∧ Edge(x,z))"""
        class TriangleFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                tri = nx.triangles(structure)
                return any(v > 0 for v in tri.values()) if len(structure) >= 3 else False
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(TriangleFormula(), "has_triangle")
    @staticmethod
    def has_perfect_matching() -> MSOPropertyChecker:
        """Graph has a perfect matching."""
        class PerfectMatchingFormula(MSOFormula):
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                if len(structure) % 2 != 0:
                    return False
                try:
                    matching = nx.max_weight_matching(structure)
                    return len(matching) * 2 == len(structure)
                except:
                    return False
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(PerfectMatchingFormula(), "perfect_matching")
    
    @staticmethod
    def is_regular(k: int) -> MSOPropertyChecker:
        """All vertices have degree k."""
        class RegularFormula(MSOFormula):
            def __init__(self, degree: int):
                self.degree = degree
            
            def evaluate(self, structure: nx.Graph, assignment: Dict[str, Any]) -> bool:
                if len(structure) == 0:
                    return self.degree == 0
                degrees = [d for n, d in structure.degree()]
                return all(d == self.degree for d in degrees)
            
            def free_variables(self) -> Set[str]:
                return set()
        
        return MSOPropertyChecker(RegularFormula(k), f"{k}_regular")


class MSOPropertyLibrary:
    """Library of MSO properties for easy access."""
    
    def __init__(self):
        self.properties = {}
        self._register_standard_properties()
    
    def _register_standard_properties(self):
        """Register standard MSO properties."""
        self.properties.update({
            'connectivity': StandardMSOProperties.connectivity(),
            'tree': StandardMSOProperties.tree_property(),
            'bipartite': StandardMSOProperties.bipartiteness(),
            'even_parity': StandardMSOProperties.even_parity(),
            'planarity': StandardMSOProperties.planarity(),
            'has_triangle': AdvancedMSOProperties.has_triangle(),
            'perfect_matching': AdvancedMSOProperties.has_perfect_matching(),
        })
        
        # Add regular graph properties
        for k in range(6):
            self.properties[f'{k}_regular'] = AdvancedMSOProperties.is_regular(k)
    
    def get_property(self, name: str) -> MSOPropertyChecker:
        """Get property checker by name."""
        if name not in self.properties:
            raise ValueError(f"Unknown property: {name}. Available: {list(self.properties.keys())}")
        return self.properties[name]
    
    def register_custom_property(self, name: str, formula: Union[str, MSOFormula]):
        """Register a custom MSO property."""
        self.properties[name] = MSOPropertyChecker(formula, name)
    
    def check_property(self, graph: nx.Graph, property_name: str) -> bool:
        """Check if graph satisfies named property."""
        checker = self.get_property(property_name)
        return checker.check(graph)
    
    def generate_theory_graphs(self, property_name: str, num_graphs: int = 1000, 
                             max_nodes: int = 15) -> List[nx.Graph]:
        """Generate graphs satisfying a specific property."""
        property_checker = self.get_property(property_name)
        theory_graphs = []
        
        attempts = 0
        max_attempts = num_graphs * 10
        
        while len(theory_graphs) < num_graphs and attempts < max_attempts:
            attempts += 1
            
            # Generate random graph
            n = np.random.randint(3, max_nodes + 1)
            
            if property_name == 'tree':
                graph = nx.random_tree(n)
            elif property_name in ['connectivity', 'planarity']:
                p = np.random.uniform(0.3, 0.7)
                graph = nx.erdos_renyi_graph(n, p)
            elif property_name == 'bipartite':
                n1 = np.random.randint(1, n // 2 + 1)
                n2 = n - n1
                p = np.random.uniform(0.2, 0.8)
                graph = nx.bipartite.random_graph(n1, n2, p)
            else:
                # General case: random graph
                p = np.random.uniform(0.1, 0.9)
                graph = nx.erdos_renyi_graph(n, p)
            
            # Check if it satisfies the property
            if property_checker.check(graph):
                theory_graphs.append(graph)
        
        if len(theory_graphs) < num_graphs:
            print(f"Warning: Only generated {len(theory_graphs)} graphs "
                  f"satisfying {property_name} after {attempts} attempts")
        
        return theory_graphs


# Example usage and testing
if __name__ == "__main__":
    # Test MSO property library
    library = MSOPropertyLibrary()
    
    # Create test graphs
    tree = nx.path_graph(5)
    cycle = nx.cycle_graph(5) 
    complete_bipartite = nx.complete_bipartite_graph(3, 3)
    
    # Test properties
    test_graphs = [
        ("tree", tree),
        ("cycle", cycle),
        ("bipartite", complete_bipartite)
    ]
    
    properties_to_test = ['connectivity', 'tree', 'bipartite', 'planarity', 'even_parity']
    
    print("Property testing results:")
    print("-" * 50)
    
    for graph_name, graph in test_graphs:
        print(f"\nGraph: {graph_name}")
        for prop_name in properties_to_test:
            satisfies = library.check_property(graph, prop_name)
            print(f"  {prop_name}: {'✓' if satisfies else '✗'}")
    
    # Test theory graph generation
    print(f"\nGenerating theory graphs...")
    tree_theory = library.generate_theory_graphs('tree', num_graphs=10, max_nodes=8)
    print(f"Generated {len(tree_theory)} trees")
    
    # Verify all generated graphs satisfy the property
    tree_checker = library.get_property('tree')
    all_trees = all(tree_checker.check(g) for g in tree_theory)
    print(f"All generated graphs are trees: {all_trees}")
