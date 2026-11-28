import argparse
import json
import shutil
import sys
from pathlib import Path

from rich.console import Console

from custom_eval._score_pool import configure_score_workers
from custom_eval.benchmarks import BENCHMARKS, BuildContext
from custom_eval.config import ConfigError, RunConfig, exp_name, load_config, repeats
from custom_eval.download import DataMissingError, ensure_data
from custom_eval.judge import judge_model
from custom_eval.model import (
    ModelLookupError,
    api_base,
    connect_model,
    resolve_model_id,
)
from custom_eval.results import (
    create_run_dir,
    output_paths,
    print_table,
    write_benchmark,
)
from custom_eval.solver import system_prompt_generate
from inspect_ai import eval_set
from inspect_ai.log import read_eval_log

TABLE_WIDTH = 160
"""Console width used when stdout is not a terminal."""

RETRY_ON_ERROR = 3
"""Times a sample is retried after an error before its benchmark attempt fails."""

RETRY_ATTEMPTS = 3
"""Times a failed benchmark is resumed within one invocation before giving up."""

RESUME_KEYS = (
    "model",
    "system_prompt",
    "answer",
    "benchmarks",
    "repeats",
    "sampling",
    "judge",
)
"""Config keys that must be unchanged to resume a run (they change what is generated
or how it is scored). `url`, `data_dir`, `max_generation_workers` and
`max_score_workers` may change; `output_dir` and `exp_name` are not used by a
resume (the run folder is given)."""

RUN_FILE = "run.jsonl"
"""Records the resolved model id of a run (one line), for resume checks."""


def check_resume(config: RunConfig, run_dir: Path, model_id: str) -> None:
    """Check that the run folder `run_dir` holds an earlier attempt of the same run.

    Args:
        config: The new config.
        run_dir: The run folder given to `--resume`.
        model_id: The resolved model id.

    Raises:
        ConfigError: If `run_dir` holds no run, or a different run.
    """
    previous_path = run_dir / "config.yaml"
    if not previous_path.exists():
        raise ConfigError(f"--resume: {run_dir} holds no run (no config.yaml)")
    previous = load_config(previous_path)
    changed = [
        key
        for key in RESUME_KEYS
        if (key in previous) != (key in config)
        or (key in previous and previous[key] != config[key])  # type: ignore[literal-required]
    ]
    run_path = run_dir / RUN_FILE
    if run_path.exists():
        with open(run_path, encoding="utf-8") as f:
            previous_model = json.loads(f.readline())["model"]
        if previous_model != model_id:
            changed.append(f"served model ({previous_model} -> {model_id})")
    if changed:
        raise ConfigError(
            f"--resume: {run_dir} holds a different run (changed: "
            f"{', '.join(changed)}); restore the config to resume, or start a new "
            "run without --resume"
        )


def run(
    config: RunConfig, config_path: Path, resume_dir: Path | None, console: Console
) -> int:
    """Run one evaluation, or resume one: every benchmark of the config against one model.

    A new run gets a new run folder in `output_dir` (see `create_run_dir`).
    A resumed run continues in `resume_dir`: finished benchmarks are reused and
    failed or interrupted ones resume from their completed samples.
    `results.jsonl`, `results_metadata.jsonl` and `output_raw/` are rewritten
    from the logs each time.

    Args:
        config: Validated run config.
        config_path: Path of the config file (copied into a new run folder).
        resume_dir: Run folder to resume, or None for a new run.
        console: Console for the results table.

    Returns:
        Process exit code (0 if every benchmark succeeded).
    """
    unknown = [b for b in config["benchmarks"] if b not in BENCHMARKS]
    if unknown:
        raise ConfigError(
            f"config.benchmarks: unknown {unknown}; available: {sorted(BENCHMARKS)}"
        )
    benchmarks = [BENCHMARKS[b] for b in config["benchmarks"]]

    data_dir = Path(config["data_dir"])
    ensure_data([spec for b in benchmarks for spec in b.data], data_dir)

    url = config["url"]
    model_id = resolve_model_id(url, config["model"] if "model" in config else None)
    sampling = config["sampling"] if "sampling" in config else {}
    model = connect_model(url, model_id, sampling, config["max_generation_workers"])
    if resume_dir is not None:
        check_resume(config, resume_dir, model_id)
    console.print(
        f"{'Resuming' if resume_dir is not None else 'Evaluating'} "
        f"[bold]{model_id}[/bold] at {api_base(url)}"
    )

    judge = config["judge"] if "judge" in config else None
    if judge is not None:
        # connect now so a bad judge URL fails before generation starts
        judge_model(judge, config["max_generation_workers"])

    if resume_dir is None:
        run_dir = create_run_dir(config["output_dir"], exp_name(config), model_id)
        shutil.copyfile(config_path, run_dir / "config.yaml")
        with open(run_dir / RUN_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"model": model_id}) + "\n")
    else:
        run_dir = resume_dir
    # unwrapped, so the path can be copied for --resume
    console.print(f"Run folder: {run_dir}", soft_wrap=True, markup=False)
    results_path, metadata_path, raw_dir, log_dir = output_paths(run_dir)

    answer_open, answer_close = config["answer"]
    ctx = BuildContext(
        data_dir=data_dir,
        answer_open=answer_open,
        answer_close=answer_close,
        judge=judge,
    )
    configure_score_workers(config["max_score_workers"])
    tasks = [b.build(ctx) for b in benchmarks]
    n_repeats = repeats(config)
    seed = sampling["seed"] if "seed" in sampling else None

    # eval_set reuses the logs in log_dir: finished benchmarks are kept as they
    # are, failed or interrupted ones resume (completed samples are reused,
    # only the rest are generated and scored).
    # Generation and scoring are separate pools: every sample may be active at
    # once (max_samples), so a sample being scored never holds back another's
    # generation. Requests are capped by max_generation_workers (the model's
    # max_connections); CPU scoring by max_score_workers (score slots, and
    # max_subprocesses so Inspect's own subprocess cap is not lower).
    _, headers = eval_set(
        tasks,
        log_dir=str(log_dir),
        model=model,
        solver=system_prompt_generate(config["system_prompt"], seed),
        epochs=n_repeats,
        max_tasks=len(tasks),
        max_samples=max(len(task.dataset) for task in tasks) * n_repeats,
        max_subprocesses=config["max_score_workers"],
        retry_on_error=RETRY_ON_ERROR,
        retry_attempts=RETRY_ATTEMPTS,
    )

    results = []
    raw_dir.mkdir(exist_ok=True)
    with (
        open(results_path, "w", encoding="utf-8") as fr,
        open(metadata_path, "w", encoding="utf-8") as fm,
    ):
        by_task = {log.eval.task: log for log in headers}
        for benchmark, task in zip(benchmarks, tasks, strict=True):
            log = read_eval_log(by_task[benchmark.name].location)
            sample_ids = [sample.id for sample in task.dataset]
            with open(raw_dir / f"{benchmark.name}.jsonl", "w", encoding="utf-8") as fw:
                results.append(
                    write_benchmark(
                        benchmark, log, sample_ids, model_id, url, n_repeats, fr, fm, fw
                    )
                )
    print_table(results, BENCHMARKS, console)
    console.print(
        f"Results: {results_path}\nMetadata: {metadata_path}\n"
        f"Outputs: {raw_dir}\nLogs: {log_dir}"
    )
    return 0 if all(r["status"] == "success" for r in results) else 1


def main() -> None:
    """Entry point: `custom-eval <config.yaml> [--resume <run folder>]`."""
    parser = argparse.ArgumentParser(
        prog="custom-eval",
        description="Evaluate a vLLM-served model on the suite's benchmarks.",
    )
    parser.add_argument("config", type=Path, help="run config (YAML)")
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="continue the run in this run folder instead of starting a new run",
    )
    args = parser.parse_args()
    console = Console()
    if not console.is_terminal:
        # logs redirected to a file: don't squeeze the table into 80 columns
        console = Console(width=TABLE_WIDTH)
    try:
        config = load_config(args.config)
        code = run(config, args.config, args.resume, console)
    except ConfigError as ex:
        console.print(f"[red]Config error:[/red] {ex}")
        sys.exit(2)
    except (DataMissingError, ModelLookupError) as ex:
        console.print(f"[red]Error:[/red] {ex}")
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()
