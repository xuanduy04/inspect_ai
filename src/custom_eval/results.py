import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console
from rich.table import Table

from custom_eval.benchmarks import Benchmark
from custom_eval.benchmarks.if_tasks import CHECKS
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ContentReasoning, ContentText

TRUNCATED_STOP_REASONS = ("max_tokens", "model_length")
"""Stop reasons counted as truncated (token budget or context window exhausted)."""

RUN_DIR_TIME_FORMAT = "%Y%m%d-%H_%M_%S"
"""Start time ending a run folder's name, in local time. Fixed width, so among
folders with the same prefix (exp_name and model) alphabetical order is
chronological order."""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def clustered_stderr(per_problem: list[list[float]]) -> float | None:
    """Standard error of the overall mean, clustered by problem.

    The mean over repeats of each problem is one observation; the stderr is
    their sample standard deviation over sqrt(number of problems). With one
    repeat this is the usual binomial standard error.
    """
    means = [_mean(v) for v in per_problem]
    n = len(means)
    if n < 2:
        return None
    mu = _mean(means)
    return math.sqrt(sum((m - mu) ** 2 for m in means) / (n - 1) / n)


def ratio_stderr(followed: list[float], totals: list[float]) -> float | None:
    """Standard error of sum(followed)/sum(totals), clustered by problem (delta method)."""
    n = len(totals)
    if n < 2 or sum(totals) == 0:
        return None
    r = sum(followed) / sum(totals)
    resid = sum((y - r * x) ** 2 for y, x in zip(followed, totals, strict=True))
    return math.sqrt(n / (n - 1) * resid) / sum(totals)


def _reasoning(sample: EvalSample) -> str | None:
    content = sample.output.message.content if sample.output.choices else ""
    if isinstance(content, str):
        return None
    parts = [c.reasoning for c in content if isinstance(c, ContentReasoning)]
    return "\n".join(parts) if parts else None


def _messages(sample: EvalSample) -> list[dict[str, str]]:
    records = []
    for message in sample.messages:
        if message.role == "assistant":
            continue
        if isinstance(message.content, str):
            text = message.content
        else:
            text = "".join(
                c.text for c in message.content if isinstance(c, ContentText)
            )
        records.append({"role": message.role, "content": text})
    return records


def _the_score(sample: EvalSample) -> Any:
    if sample.scores is None or len(sample.scores) != 1:
        raise ValueError(
            f"Sample {sample.id} (repeat {sample.epoch}) has no single score: {sample.scores}"
        )
    return next(iter(sample.scores.values()))


def raw_record(benchmark: str, model: str, sample: EvalSample) -> dict[str, Any]:
    """One `output_raw/<benchmark>.jsonl` line: a generation and its judgement.

    A sample that errored (only in a failed benchmark) has no score: its
    `answer`, `score` and `judgement` are null and `error` holds the message.
    """
    score = _the_score(sample) if sample.error is None else None
    usage = sample.output.usage
    has_output = len(sample.output.choices) > 0
    return {
        "benchmark": benchmark,
        "model": model,
        "id": sample.id,
        "repeat": sample.epoch,
        "messages": _messages(sample),
        "target": sample.target,
        "output": sample.output.completion if has_output else None,
        "reasoning": _reasoning(sample) if has_output else None,
        "stop_reason": sample.output.stop_reason if has_output else None,
        "truncated": has_output and sample.output.stop_reason in TRUNCATED_STOP_REASONS,
        "usage": {
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "reasoning_tokens": usage.reasoning_tokens if usage else None,
        },
        "answer": score.answer if score is not None else None,
        "score": score.value if score is not None else None,
        "judgement": score.metadata if score is not None else None,
        "error": sample.error.message if sample.error else None,
    }


def benchmark_result(
    benchmark: Benchmark, model: str, url: str, repeats: int, samples: list[EvalSample]
) -> dict[str, Any]:
    """One `results_metadata.jsonl` line: a benchmark's metrics and run details.

    Every metric is the mean over all (problem, repeat) pairs, with its stderr
    clustered by problem. With several repeats, `pass@k` (k = repeats, any
    repeat correct) and the per-repeat means are added.
    """
    by_problem: dict[Any, list[EvalSample]] = {}
    for sample in samples:
        by_problem.setdefault(sample.id, []).append(sample)
    problems = list(by_problem.values())

    metrics: dict[str, dict[str, Any]] = {}
    for key in benchmark.metrics:
        per_problem = [[float(_the_score(s).value[key]) for s in p] for p in problems]
        entry: dict[str, Any] = {
            "mean": _mean([v for p in per_problem for v in p]),
            "stderr": clustered_stderr(per_problem),
        }
        if repeats > 1:
            entry[f"pass@{repeats}"] = _mean([float(max(p)) for p in per_problem])
            entry["per_repeat"] = [
                _mean(
                    [
                        float(_the_score(s).value[key])
                        for p in problems
                        for s in p
                        if s.epoch == r
                    ]
                )
                for r in range(1, repeats + 1)
            ]
        metrics[key] = entry

    if benchmark.instruction_level:
        for check in CHECKS:
            field = f"follow_{check}"
            followed = [
                float(sum(sum(_the_score(s).metadata[field]) for s in p))
                for p in problems
            ]
            totals = [
                float(sum(len(_the_score(s).metadata[field]) for s in p))
                for p in problems
            ]
            metrics[f"inst_{check}"] = {
                "mean": sum(followed) / sum(totals),
                "stderr": ratio_stderr(followed, totals),
            }

    output_tokens = [
        s.output.usage.output_tokens for s in samples if s.output.usage is not None
    ]
    return {
        "benchmark": benchmark.name,
        "status": "success",
        "model": model,
        "url": url,
        "n_problems": len(problems),
        "repeats": repeats,
        "n_samples": len(samples),
        "headline": benchmark.headline,
        "headline_lenient": benchmark.headline_lenient,
        "metrics": metrics,
        "format_rate": _mean(
            [float(_the_score(s).metadata["marker_found"]) for s in samples]
        ),
        "truncation_rate": _mean(
            [float(s.output.stop_reason in TRUNCATED_STOP_REASONS) for s in samples]
        ),
        "mean_output_tokens": _mean([float(t) for t in output_tokens])
        if output_tokens
        else None,
    }


def output_order(
    samples: list[EvalSample], sample_ids: list[int | str | None]
) -> list[EvalSample]:
    """Samples in `output_raw` order: repeat 1 of every problem, then repeat 2, ...

    Within a repeat, problems are in dataset order. So line k (k up to the
    number of problems) is problem k, repeat 1, whatever the number of repeats.

    Args:
        samples: A benchmark's samples, in any order.
        sample_ids: Its dataset's sample ids, in dataset order.

    Returns:
        The samples, sorted.
    """
    position = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    return sorted(samples, key=lambda s: (s.epoch, position[s.id]))


def write_benchmark(
    benchmark: Benchmark,
    log: EvalLog,
    sample_ids: list[int | str | None],
    model: str,
    url: str,
    repeats: int,
    results: TextIO,
    results_metadata: TextIO,
    raw: TextIO,
) -> dict[str, Any]:
    """Write a benchmark's lines to `results.jsonl`, `results_metadata.jsonl` and `output_raw/`.

    The `results.jsonl` line holds only the benchmark, its `score_mean` (the
    mean of its headline metric, the number the table's Score column shows) and
    its `score_stderr` (that metric's clustered stderr); both are None if the
    benchmark failed. Everything else goes to `results_metadata.jsonl`. The raw
    lines are in `output_order`.

    Args:
        benchmark: The benchmark.
        log: Its eval log, with samples (or a failed log).
        sample_ids: Its dataset's sample ids, in dataset order.
        model: Evaluated model id.
        url: Server URL.
        repeats: Repeats per problem.
        results: Open `results.jsonl`.
        results_metadata: Open `results_metadata.jsonl`.
        raw: Open `output_raw/<benchmark>.jsonl`.

    Returns:
        The `results_metadata.jsonl` line.
    """
    samples = output_order(log.samples or [], sample_ids)
    for sample in samples:
        raw.write(json.dumps(raw_record(benchmark.name, model, sample)) + "\n")
    if log.status != "success" or log.samples is None:
        result: dict[str, Any] = {
            "benchmark": benchmark.name,
            "status": log.status,
            "model": model,
            "url": url,
            "error": log.error.message if log.error else None,
            "log": log.location,
        }
        score_mean = None
        score_stderr = None
    else:
        result = benchmark_result(benchmark, model, url, repeats, samples)
        result["log"] = log.location
        headline = result["metrics"][benchmark.headline]
        score_mean = headline["mean"]
        score_stderr = headline["stderr"]
    results.write(
        json.dumps(
            {
                "benchmark": benchmark.name,
                "score_mean": score_mean,
                "score_stderr": score_stderr,
            }
        )
        + "\n"
    )
    results_metadata.write(json.dumps(result) + "\n")
    results.flush()
    results_metadata.flush()
    raw.flush()
    return result


def _pct(entry: dict[str, Any] | None) -> str:
    if entry is None:
        return "-"
    text = f"{100 * entry['mean']:.2f}"
    if entry["stderr"] is not None:
        text += f" ± {100 * entry['stderr']:.2f}"
    return text


def print_table(
    results: list[dict[str, Any]], benchmarks: dict[str, Benchmark], console: Console
) -> None:
    """Print the results table (percentages, ± clustered stderr)."""
    repeats = max(
        (r["repeats"] for r in results if r["status"] == "success"), default=1
    )
    table = Table(title="Results (%, mean ± stderr)")
    for column in ("Benchmark", "Metric", "Score", "Strict", "Lenient"):
        table.add_column(column)
    if repeats > 1:
        table.add_column(f"pass@{repeats} (strict)")
    for column in ("Format", "Truncated", "Problems", "Repeats"):
        table.add_column(column, justify="right")
    for result in results:
        benchmark = benchmarks[result["benchmark"]]
        if result["status"] != "success":
            row = [
                benchmark.name,
                benchmark.label,
                f"[red]{result['status']}[/red]",
                "-",
                "-",
            ]
            if repeats > 1:
                row.append("-")
            table.add_row(*row, "-", "-", "-", "-")
            continue
        metrics = result["metrics"]
        strict = metrics[benchmark.headline]
        lenient = (
            metrics[benchmark.headline_lenient]
            if benchmark.headline_lenient is not None
            else None
        )
        row = [
            benchmark.name,
            benchmark.label,
            _pct(strict),
            _pct(strict),
            _pct(lenient),
        ]
        if repeats > 1:
            key = f"pass@{result['repeats']}"
            row.append(f"{100 * strict[key]:.2f}" if key in strict else "-")
        row += [
            f"{100 * result['format_rate']:.1f}",
            f"{100 * result['truncation_rate']:.1f}",
            str(result["n_problems"]),
            str(result["repeats"]),
        ]
        table.add_row(*row)
    console.print(table)


def model_dir_name(model_id: str) -> str:
    """A model id as written in run folder names.

    Lowercase, with every run of characters other than `a-z` and `0-9` (such as
    `.`, `/` and `_`) replaced by one `-`, and no `-` at either end:
    `Qwen/Qwen3-8B` becomes `qwen-qwen3-8b`.
    """
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def run_dir_name(exp_name: str, model_id: str, started: datetime) -> str:
    """Name of a run folder: `{exp_name}-{model}-{YYYYMMDD-HH_MM_SS}`.

    Args:
        exp_name: Name prefix; empty for none (then the name starts with the
            model).
        model_id: Served model id, written as `model_dir_name(model_id)`.
        started: Start time of the run.

    Returns:
        The folder name.
    """
    prefix = f"{exp_name}-" if exp_name else ""
    return f"{prefix}{model_dir_name(model_id)}-{started.strftime(RUN_DIR_TIME_FORMAT)}"


def create_run_dir(output_dir: str | Path, exp_name: str, model_id: str) -> Path:
    """Create a new run folder in `output_dir`, named for the current local time.

    The folder is created atomically. If a run of the same name already took
    this second, the next second is used, so every run gets its own folder and
    a later run sorts after an earlier one.

    Args:
        output_dir: Parent directory (created if missing).
        exp_name: Name prefix; empty for none.
        model_id: Served model id.

    Returns:
        The new folder.
    """
    while True:
        started = datetime.now().astimezone()
        path = Path(output_dir) / run_dir_name(exp_name, model_id, started)
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            # wait for the next second
            time.sleep(1 - time.time() % 1)


def output_paths(run_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    """Paths of a run folder's `results.jsonl`, `results_metadata.jsonl`, `output_raw/` and log directory."""
    root = Path(run_dir)
    return (
        root / "results.jsonl",
        root / "results_metadata.jsonl",
        root / "output_raw",
        root / "logs",
    )
