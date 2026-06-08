import json
from pathlib import Path

from logical_gans import modelbuilder
from logical_gans.modelbuilder.cli import main

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
THEORIES = ROOT / "examples" / "theories"
CLAIMS = ROOT / "examples" / "claims"


def test_docs_files_exist():
    for name in (
        "modelbuilder_output_schema.md",
        "modelbuilder_architecture.md",
        "modelbuilder_v0_1_walkthrough.md",
    ):
        assert (DOCS / name).is_file(), name


def test_version_string_exists():
    assert modelbuilder.__version__ == "0.1.0-alpha"


def test_root_readme_mentions_modelbuilder():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "ModelBuilder" in text
    assert "src/logical_gans/modelbuilder/" in text


def test_simple_graph_theory_synthesizes(capsys):
    rc = main(["synthesize", "--theory", str(THEORIES / "simple_graph.json"),
               "--n", "3", "--policy", "sparse_horn"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "satisfied"


def test_walkthrough_synthesize_sparse(capsys):
    rc = main(["synthesize", "--theory", str(THEORIES / "preorder.json"),
               "--n", "3", "--policy", "sparse_horn", "--k", "1", "--budget", "20"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "satisfied"
    assert out["n"] == 3 and out["k"] == 1 and out["budget"] == 20
    assert out["policy"] == "sparse_horn"


def test_walkthrough_synthesize_maximal(capsys):
    rc = main(["synthesize", "--theory", str(THEORIES / "preorder.json"),
               "--n", "3", "--policy", "maximal_horn", "--k", "1", "--budget", "20"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "satisfied"
    assert out["policy"] == "maximal_horn"


def test_walkthrough_search_semigroup(capsys):
    rc = main(["search", "--theory", str(THEORIES / "semigroup.json"),
               "--n", "2", "--k", "2", "--budget", "100", "--max-nodes", "10000"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "satisfied"
    assert out["max_nodes"] == 10000


def test_walkthrough_refute_antisymmetry(capsys):
    rc = main(["refute", "--theory", str(THEORIES / "preorder.json"),
               "--claim", str(CLAIMS / "antisymmetry.json"),
               "--n", "2", "--policy", "maximal_horn"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "refuted"
    assert out["witness"]["assignment"] == {"x": 0, "y": 1}
