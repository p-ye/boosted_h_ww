# Archive

This directory preserves the original research materials. The active analysis
starts from `notebooks/analysis_pipeline.ipynb`; archived files are retained for
provenance and for reproducing historical results.

## Contents

- `original_scripts/`: early ROOT-to-CSV, filtering, and constituent-counting
  scripts. These document the original truth-level workflow.
- `notebooks/truth/`: truth-level GNN, combined-model, and constituent-validation
  notebooks.
- `notebooks/reco/`: reconstructed and truth-comparison training notebooks plus
  the large post-training analysis notebook.
- `research/slides/`: dated progress decks, kept as a chronological lab record.
- `research/plots/`: plot exports from the notebooks, including ambiguously named
  `Unknown*.png` files.

## Which notebooks were superseded?

- `truth/GNN-2.ipynb` was superseded by `truth/gnn_fjc_only-3.ipynb` and the
  maintained package.
- `reco/reco_training.ipynb` was superseded by the two larger `Copy` notebooks.
- The two `Copy` notebooks are truth/reco variants with extensive duplicated
  model code. Their maintained replacement is one configurable training command.
- `reco/reco_computing-3.ipynb` remains the fullest historical analysis record
  and contains the specialized post-training plots.

Archived files should be consulted when reproducing a historical plot or tracing
an old analysis decision. New model studies should begin with the active analysis
notebook.
