from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.theory import Theory
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.core.clauses import HornClause
from logical_gans.modelbuilder.core.atoms import RelAtom
from logical_gans.modelbuilder.core.terms import Var
from logical_gans.modelbuilder.learned.priors import (
    MockLLMPrior,
    ObligationFirstPrior,
    UniformPrior,
)
from logical_gans.modelbuilder.learned.semantic_actions import (
    SetRelation,
    legal_semantic_edits,
)

PREORDER_SIG = Signature.build(relations=[("R", 2)])


def _preorder_theory():
    x, y, z = Var("x"), Var("y"), Var("z")
    return Theory("preorder", PREORDER_SIG, [
        HornClause("reflexive", ("x",), (), RelAtom("R", (x, x))),
        HornClause("transitive", ("x", "y", "z"),
                   (RelAtom("R", (x, y)), RelAtom("R", (y, z))), RelAtom("R", (x, z))),
    ])


def _state_and_result():
    theory = _preorder_theory()
    s = PartialStructure.empty(theory.signature, 2)   # empty -> reflexive obligation on R(0,0)
    return s, run_devil(s, theory.clauses), legal_semantic_edits(s)


def test_uniform_prior_equal_scores():
    s, result, allowed = _state_and_result()
    scores = UniformPrior().score(s, result, allowed)
    assert len(set(scores.values())) == 1


def test_obligation_first_prior_boosts_matching():
    s, result, allowed = _state_and_result()
    scores = ObligationFirstPrior().score(s, result, allowed)
    matching = scores[SetRelation("R", (0, 0), Truth.TRUE)]   # the obligation cell, preferred TRUE
    non_matching = scores[SetRelation("R", (0, 1), Truth.TRUE)]
    assert matching > non_matching


def test_mock_llm_prior_validates_before_scoring():
    s, result, allowed = _state_and_result()
    scores = MockLLMPrior(strategy="first_true").score(s, result, allowed)
    assert set(scores) == set(allowed)
    boosted = max(scores.values())
    # the boosted edit is one that passed validation (i.e., a legal allowed edit)
    top = [e for e, v in scores.items() if v == boosted]
    assert all(e in set(allowed) for e in top)
    assert boosted > min(scores.values())
