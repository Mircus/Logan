import networkx as nx
from logical_gans import EFGameSimulator, MSOPropertyLibrary

def test_ef_distance_basic():
    G1 = nx.path_graph(4)
    G2 = nx.cycle_graph(4)
    sim = EFGameSimulator(G1, G2)
    assert sim.duplicator_wins(3) in (True, False)
    assert isinstance(sim.ef_distance(3), int)

def test_mso_properties():
    lib = MSOPropertyLibrary()
    Gp = nx.path_graph(4)
    Gc = nx.cycle_graph(4)
    tri = lib.get_property("has_triangle")
    assert tri.check(Gp) is False
    assert tri.check(Gc) is False
    bp = lib.get_property("bipartite")
    assert bp.check(Gp) is True
    assert bp.check(Gc) is True
