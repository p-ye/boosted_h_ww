# Boosted HWW classification studies

This repository contains the preprocessing and machine-learning workflow used to
study boosted Higgs bosons in the semileptonic
`H -> WW* -> qqlnu` channel. The immediate task is binary classification of
signal and background events using global event kinematics together with the
internal structure of the large-radius jet.

The analysis was developed first with truth-level samples and later with
reconstructed samples. Both sample types now use the same wide event format so
that selections, input features, architectures, and performance metrics can be
compared directly.

## Repository structure

```text
configs/pipeline.toml
    Sample locations, preprocessing settings, common cuts, and training defaults.

scripts/preprocess.py
    Chunked ROOT-to-Parquet preprocessing for truth and reconstructed samples.

notebooks/analysis_pipeline.ipynb
    Main research notebook: feature construction, constituent tensors, models,
    training, scoring, plots, and truth/reconstruction comparisons.

src/hww/
    Small helpers used by preprocessing: configuration and ROOT conversion.

docs/DATA_SCHEMA.md
    Expected columns in the common wide event table.

archive/
    Original scripts, exploratory notebooks, progress slides, and saved plots.

docs/references/
    Project proposal and the interaction-network/JEDI-net reference paper.
```

The active analysis is intentionally notebook-centered. Model definitions and
scientific choices remain visible and editable in the notebook rather than being
hidden behind a fixed training interface.

## Environment setup

Python 3.11 or newer is recommended.

From the repository root—the directory containing `pyproject.toml`—create an
environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Then start the notebook interface:

```bash
jupyter lab
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Input ROOT and Parquet files are not included in this repository.

## Configure the samples

Edit `configs/pipeline.toml` before preprocessing. Each sample has a name,
domain, binary label, input ROOT path, and output Parquet path.

```toml
[samples.reco_signal]
domain = "reco"
label = 1
input = "data/raw/reco/signal.root"
output = "data/processed/reco/signal.parquet"
```

Use `domain = "truth"` for truth-level samples. Several background productions
can be listed separately; the notebook gives each background sample equal total
weight before balancing signal and background.

Paths in the configuration are resolved relative to the repository root.

## Preprocess ROOT samples

Process every configured sample in one domain:

```bash
python scripts/preprocess.py --domain reco
python scripts/preprocess.py --domain truth
```

Or select individual samples:

```bash
python scripts/preprocess.py \
  --sample truth_signal \
  --sample truth_background
```

The preprocessor reads the `analysis` tree in bounded chunks, keeps the first 30
configured fat-jet tracks, applies loose I/O cuts, and writes one wide Parquet
table per sample. A JSON manifest is written beside each output.

Existing files are protected by default. Use `--overwrite` only when replacing a
processed sample intentionally.

The active preprocessor requires one wide row per selected ROOT event. Truth
samples therefore need the same wide branch organization as reconstructed
samples. The earlier component-based truth pipeline is preserved under
`archive/original_scripts/` for reproducing older studies.

## Run the analysis notebook

Open `notebooks/analysis_pipeline.ipynb` and run it section by section.

At the top of the notebook, choose:

```python
DOMAIN = "reco"          # or "truth"
MODEL_NAME = "gdnn"
RUN_NAME = "baseline_v1"
```

The notebook contains the complete analysis sequence:

1. Load all configured signal and background samples for the chosen domain.
2. Apply the common training selection.
3. Define the high-level event features.
4. Construct the `30 x 4` constituent representation.
5. Build event weights and stratified train/validation/test splits.
6. Fit the high-level and constituent scalers on the training split.
7. Define, modify, and instantiate a model.
8. Train with weighted binary cross-entropy and early stopping.
9. Inspect ROC curves, background rejection, and the confusion matrix.
10. Score the complete selected dataset and save the experiment.

The available model names are:

- `dnn`: high-level event features only
- `gnn`: fully connected constituent interaction network
- `gdnn`: interaction-network embedding plus high-level features
- `transformer`: constituent Transformer
- `tdnn`: Transformer embedding plus high-level features
- `deepsets`: constituent Deep Sets model
- `hybrid_deepsets`: Deep Sets embedding plus high-level features

The architecture classes are ordinary notebook cells. They can be copied or
modified for new layer sizes, pooling rules, attention setups, constituent
features, losses, and ablation studies. The archived reconstructed notebooks also
contain the earlier ParticleNet experiments.

## Plots and diagnostics

Running the main training and evaluation cells produces:

- training and validation loss versus epoch
- weighted ROC curve and test AUC
- signal efficiency versus background rejection
- weighted test-set confusion matrix
- signal and background classifier-score distributions

The **Extended plotting toolbox** section provides reusable functions for:

- signal efficiency versus a kinematic feature at fixed background acceptance,
  including statistical error bars
- background rejection versus a kinematic feature at fixed signal efficiency,
  including statistical error bars
- truth/reconstruction or signal/background feature distributions with a ratio panel
- one- and two-dimensional constituent and event-variable distributions
- constituent `dEta`, `dPhi`, `dR`, and relative-`pT` comparisons
- learning curves such as test AUC versus training-set size
- event-selection cut-flow charts
- overlays of weighted ROC and background-rejection curves from saved models
- truth-versus-reconstruction and standalone-versus-hybrid model comparisons

The functions accept arbitrary columns, so the same helpers can be used for
Higgs `pT`, reconstructed system `pT`, fat-jet properties, track ranks, angular
separations, model scores, or other stored variables.

Not every historical figure is generated automatically. The active notebook
reproduces the major plot families, but plots requiring a particular old scored
Parquet file, an old feature definition, or repeated training at several dataset
fractions still require the corresponding data and explicit function calls.
Exact legacy layouts and specialized difference panels remain in
`archive/notebooks/reco/reco_computing-3.ipynb` and the dated slides. This avoids
presenting old experimental plots as current results while preserving the code
needed to recover them.

## Default inputs and selection

The default high-level representation contains 25 variables covering track
counts, the lepton, fat jet, two small-radius jets, missing transverse momentum,
and relative angular quantities.

Each of the first 30 fat-jet tracks is represented by:

- track `pT / fatJetPt`
- track `dEta` relative to the fat jet
- wrapped track `dPhi` relative to the fat jet
- track `dR` relative to the fat jet

Missing track positions are zero padded. The GNN, Transformer, and Deep Sets
implementations mask padded entries.

The default tight selection is configured in `configs/pipeline.toml`:

- fat-jet transverse momentum above 200
- fat-jet mass above 40
- lepton/fat-jet `dR` below 1
- removal of events containing invalid padded jet values

Feature lists and tensor construction are kept in separate notebook cells so
that changes are explicit in each study.

## Experiment outputs

The notebook stores runs under:

```text
outputs/runs/<domain>/<model>/<run-name>/
```

For example:

```text
outputs/runs/reco/gdnn/baseline_v1/
    best_model.pt
    preprocessing.npz
    metrics.json
    training_history.csv
    test_predictions.parquet
    full_predictions.parquet
```

`test_predictions.parquet` contains only the held-out test events used for
reported performance. `full_predictions.parquet` contains every selected event
and marks it as `train`, `validation`, or `test` for later binned studies.

Runs are not overwritten automatically. Use a descriptive new `RUN_NAME` for a
new setup, especially after changing features, selections, weighting, or model
architecture.

## Comparing truth and reconstruction

Run the notebook once with `DOMAIN = "truth"` and once with `DOMAIN = "reco"`,
using the same model and experiment name. The final notebook section loads the
saved test predictions and overlays weighted ROC and background-rejection curves.

For a meaningful comparison, keep the following identical:

- tight event selection
- high-level feature list
- constituent feature definition and ordering
- model architecture and hyperparameters
- weighting procedure
- evaluation metrics

Truth-only particle identity information may be retained during preprocessing,
but it is not part of the default constituent tensor because no equivalent input
is available in the reconstructed sample.

## Historical material

The `archive/` directory records the development of the analysis:

- early ROOT-to-CSV and truth-matching scripts
- truth constituent-GNN and combined-model notebooks
- reconstructed DNN/GNN/Transformer/Deep Sets/ParticleNet experiments
- detailed post-training comparison work
- dated progress slides and exported figures

These files are useful for tracing an old result or recovering a specialized
plot. New studies should start from `notebooks/analysis_pipeline.ipynb` and use
the archive only as a reference.
