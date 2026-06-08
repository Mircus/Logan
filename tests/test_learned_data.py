from pathlib import Path

from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.actions import decode_action_index
from logical_gans.modelbuilder.learned.data import (
    make_relation_training_examples,
    read_training_examples_jsonl,
    write_training_examples_jsonl,
)

THEORIES = Path(__file__).resolve().parents[1] / "examples" / "theories"


def _examples(n_samples=200):
    theory = load_theory(THEORIES / "preorder.json")
    return make_relation_training_examples(theory, "R", 3, n_samples, seed=0)


def test_data_generator_produces_examples():
    examples = _examples()
    assert len(examples) >= 1


def test_example_tensor_shape():
    examples = _examples()
    for ex in examples[:20]:
        assert ex.input_tensor.shape == (4, 3, 3)


def test_target_decodes_to_legal_relation_edit():
    for ex in _examples()[:20]:
        edit = decode_action_index(ex.target_action, ex.n, ex.relation)
        assert edit.relation == "R"
        assert all(0 <= a < ex.n for a in edit.args)
        assert edit.value in (Truth.TRUE, Truth.FALSE)


def test_jsonl_roundtrip(tmp_path):
    examples = _examples()
    path = tmp_path / "data.jsonl"
    write_training_examples_jsonl(examples, path)
    loaded = read_training_examples_jsonl(path)
    assert len(loaded) == len(examples)
    assert loaded[0].target_action == examples[0].target_action
    assert loaded[0].input_tensor.shape == examples[0].input_tensor.shape
