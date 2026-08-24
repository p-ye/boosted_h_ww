# Common event-table schema

The maintained pipeline expects truth and reconstructed samples to use the same
wide event format. This is the format used by the later comparison notebooks.

## Metadata added during preprocessing

- `sample`: configured sample name
- `domain`: `truth` or `reco`
- `label`: `1` for signal, `0` for background
- `event_index`: original entry number within the configured ROOT sample

The active preprocessor requires exactly one output row per selected ROOT event.
If a retained branch is still jagged and would create multiple DataFrame rows,
preprocessing stops rather than silently changing event alignment.

## Required selection columns

- `fatJetPt`
- `fatJetM`
- `muon_fatjet_dr`

## Required particle-parent columns

- `fatJetPt`
- `fatJetEta`
- `fatJetPhi`

## Numbered constituent/track columns

For `N = 1..30`, the default representation uses:

- `fj_trackNPt`
- `fj_trackNEta` and `fj_trackNPhi`, or precomputed `fj_trackNdEta` and
  `fj_trackNdPhi`
- optional `fj_trackNdR`
- optional truth-only `fj_trackNTruthOrigin` and `fj_trackNTruthType`

The maintained baseline constructs four shared particle inputs:
`rel_pt`, `deta`, `dphi`, and `dr`. Missing track slots are zero padded and
masked by the maintained particle models.

## High-level features

The standard 25-variable list is defined in the **High-level event features**
cell of `notebooks/analysis_pipeline.ipynb`. Angular features are derived from
base eta/phi columns when they are not already stored.

If a ROOT production uses different branch names, convert or alias those names
before training. The pipeline fails with an explicit list of missing columns
rather than silently changing the feature set.
