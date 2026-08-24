import json
from pathlib import Path


def test_analysis_notebook_has_expected_sections():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/analysis_pipeline.ipynb").read_text())
    assert notebook["nbformat"] == 4
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for section in (
        "High-level event features",
        "Constituent features",
        "DNN and interaction-network models",
        "Training",
        "Evaluation",
        "Save the experiment",
    ):
        assert section in text
