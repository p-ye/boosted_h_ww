"""Shared ROOT-to-Parquet preprocessing for truth and reconstructed samples.

The newer analysis stores one event table per sample, with up to 30 fat-jet
tracks/constituents represented by numbered columns. Both truth and reco ROOT
files pass through this implementation. Truth-only identity branches are kept
when present and ignored otherwise.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PipelineConfig, SampleSpec


def _branches_to_keep(tree, max_tracks: int, track_fields: Iterable[str]) -> list[str]:
    """Keep all event branches but restrict the large ``fj_track`` family."""

    allowed_tracks = {
        f"fj_track{index}{field}"
        for index in range(1, max_tracks + 1)
        for field in track_fields
    }
    return [
        str(key)
        for key in tree.keys()
        if not str(key).startswith("fj_track") or str(key) in allowed_tracks
    ]


def _event_any(values, predicate):
    """Apply a predicate and reduce jagged object axes to one value per event."""

    import awkward as ak

    selected = predicate(values)
    return ak.any(selected, axis=1) if selected.ndim > 1 else selected


def preprocess_sample(
    config: PipelineConfig,
    sample: SampleSpec,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert one configured ROOT sample into a chunked Parquet event table."""

    import awkward as ak
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot

    settings = config.section("preprocessing")
    tree_name = str(settings.get("tree_name", "analysis"))
    step_size = settings.get("step_size", "500 MB")
    max_tracks = int(settings.get("max_tracks", 30))
    track_fields = settings.get("track_fields", ["Pt", "Eta", "Phi"])
    dr_branch = str(settings.get("dr_branch", "muon_fatjet_dr"))
    mass_branch = str(settings.get("fatjet_mass_branch", "fatJetM"))
    dr_max = float(settings.get("dr_max", 1.5))
    mass_min = float(settings.get("fatjet_mass_min", 20.0))

    if not sample.input_path.exists():
        raise FileNotFoundError(f"Input ROOT file not found: {sample.input_path}")
    if sample.output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {sample.output_path}. Pass --overwrite to replace it."
        )

    sample.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = sample.output_path.with_suffix(".parquet.incomplete")
    if temporary_output.exists():
        temporary_output.unlink()

    with uproot.open(sample.input_path) as root_file:
        tree = root_file[tree_name]
        branches = _branches_to_keep(tree, max_tracks, track_fields)

    missing_selection = [name for name in (dr_branch, mass_branch) if name not in branches]
    if missing_selection:
        raise KeyError(
            "Selection branch(es) missing from ROOT tree: " + ", ".join(missing_selection)
        )

    writer = None
    rows_written = 0
    entry_offset = 0
    try:
        iterator = uproot.iterate(
            f"{sample.input_path}:{tree_name}",
            expressions=branches,
            library="ak",
            step_size=step_size,
        )
        for chunk in iterator:
            mask = _event_any(chunk[dr_branch], lambda values: values < dr_max)
            mask = mask & _event_any(chunk[mass_branch], lambda values: values > mass_min)
            source_event_ids = np.arange(entry_offset, entry_offset + len(chunk), dtype=np.int64)
            entry_offset += len(chunk)
            selected = chunk[mask]
            if len(selected) == 0:
                continue

            frame = ak.to_dataframe(selected)
            if len(frame) != len(selected):
                raise ValueError(
                    f"Sample {sample.name!r} does not have one wide row per ROOT event. "
                    "At least one retained branch is still jagged after event selection. "
                    "Convert that production to the common wide schema before training."
                )
            selected_event_ids = source_event_ids[ak.to_numpy(mask)]
            frame = frame.reset_index(drop=True)
            if frame.empty:
                continue
            frame.insert(0, "event_index", selected_event_ids)
            frame.insert(1, "sample", sample.name)
            frame.insert(2, "domain", sample.domain)
            frame.insert(3, "label", sample.label)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_output, table.schema)
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows_written += len(frame)
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        if temporary_output.exists():
            temporary_output.unlink()
        raise RuntimeError(f"No events passed preprocessing for sample {sample.name!r}")

    temporary_output.replace(sample.output_path)
    manifest = {
        "sample": asdict(sample),
        "tree_name": tree_name,
        "max_tracks": max_tracks,
        "track_fields": list(track_fields),
        "loose_preselection": {
            dr_branch: {"maximum": dr_max},
            mass_branch: {"minimum": mass_min},
        },
        "rows_written": rows_written,
    }
    manifest["sample"]["input_path"] = str(sample.input_path)
    manifest["sample"]["output_path"] = str(sample.output_path)
    sample.output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return sample.output_path
