import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from custom_eval.config import JudgeConfig
from custom_eval.download import DataSpec
from inspect_ai import Task
from inspect_ai.scorer import Metric, mean, stderr


@dataclass(frozen=True)
class BuildContext:
    """Everything a benchmark needs to build its task for one run.

    Attributes:
        data_dir: Root data directory (Hub layout).
        answer_open: Opening answer marker.
        answer_close: Closing answer marker.
        judge: LLM judge config, if the run has one.
    """

    data_dir: Path
    answer_open: str
    answer_close: str
    judge: JudgeConfig | None


@dataclass(frozen=True)
class Benchmark:
    """A benchmark of the suite.

    The task's scorer must return a dict `Score.value` holding every key in
    `metrics` (each 0.0 or 1.0 per sample) and `Score.metadata["marker_found"]`
    (whether the answer markers were found in the final output). With
    `instruction_level`, `Score.metadata` also holds, for each check `c` in
    (`strict`, `loose`), the list `f"follow_{c}"` of per-instruction verdicts.

    Attributes:
        name: Name used in run configs.
        data: Dataset files the benchmark reads.
        build: Builds the task (dataset, scorer, sandbox; no solver, no
            epochs).
        metrics: Score keys, all per-sample pass/fail.
        headline: Key shown in the terminal table's strict column.
        headline_lenient: Key shown in the lenient column (None when the
            benchmark has a single extraction, e.g. code).
        label: What the headline measures, for the terminal table.
        instruction_level: Whether instruction-level accuracies are reported.
    """

    name: str
    data: tuple[DataSpec, ...]
    build: Callable[[BuildContext], Task]
    metrics: tuple[str, ...]
    headline: str
    headline_lenient: str | None
    label: str
    instruction_level: bool


def score_metrics(keys: tuple[str, ...]) -> list[Metric | dict[str, list[Metric]]]:
    """Inspect metrics (mean and stderr) for each score key."""
    return [{key: [mean(), stderr()] for key in keys}]


def read_jsonl(files: list[Path]) -> list[dict[str, Any]]:
    """Read rows from jsonl files, in file order."""
    rows: list[dict[str, Any]] = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def read_parquet(files: list[Path]) -> list[dict[str, Any]]:
    """Read rows from parquet files, in file order."""
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    return rows


def read_rows(spec: DataSpec, ctx: BuildContext) -> list[dict[str, Any]]:
    """Read a spec's rows (jsonl or parquet, chosen by extension)."""
    files = spec.files(ctx.data_dir)
    if not files:
        raise FileNotFoundError(
            f"No files matching {spec.pattern!r} under {ctx.data_dir / spec.repo_id}"
        )
    if all(f.suffix == ".jsonl" for f in files):
        return read_jsonl(files)
    if all(f.suffix == ".parquet" for f in files):
        return read_parquet(files)
    raise ValueError(f"Mixed or unsupported file types for {spec.repo_id}: {files}")
