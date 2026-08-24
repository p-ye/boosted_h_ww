"""Configuration loading and path resolution.

The pipeline intentionally uses TOML because Python 3.11 can read it without an
extra dependency. Relative paths are always resolved from the repository root,
which makes commands independent of the caller's working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class SampleSpec:
    """One signal or background input configured in ``pipeline.toml``."""

    name: str
    domain: str
    label: int
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class PipelineConfig:
    """Parsed configuration plus the repository root used for path resolution."""

    path: Path
    root: Path
    values: dict[str, Any]
    samples: dict[str, SampleSpec]

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values.get(name, {}))

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def load_config(path: str | Path = "configs/pipeline.toml") -> PipelineConfig:
    """Load a pipeline configuration and resolve every sample path.

    The standard location is ``<repository>/configs/pipeline.toml``. For a
    configuration elsewhere, the repository root is assumed to be its parent,
    unless it is itself inside a directory named ``configs``.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        values = tomllib.load(stream)

    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    samples: dict[str, SampleSpec] = {}
    for name, raw in values.get("samples", {}).items():
        input_path = Path(raw["input"])
        output_path = Path(raw["output"])
        samples[name] = SampleSpec(
            name=name,
            domain=str(raw["domain"]),
            label=int(raw["label"]),
            input_path=input_path if input_path.is_absolute() else root / input_path,
            output_path=output_path if output_path.is_absolute() else root / output_path,
        )

    return PipelineConfig(config_path, root, values, samples)

