from pathlib import Path
from typing import Any

import yaml
from typing_extensions import NotRequired, TypedDict


class JudgeConfig(TypedDict):
    """LLM judge served by vLLM (used by benchmarks that need a judge).

    Attributes:
        url: Base URL of the vLLM server serving the judge.
        model: Served model id to use. If absent, the server must serve exactly
            one model.
        prompt: Judge prompt template. `{question}`, `{response}` and
            `{target}` are replaced verbatim; no other text is touched.
        answer: `[open, close]` markers around the judge's verdict. The last
            marker in the judge's output is used.
        scores: Verdict (text between the markers, stripped) to score value.
            A verdict not listed here scores 0.
        sampling: Request parameters for the judge. Absent keys are not sent.
    """

    url: str
    model: NotRequired[str]
    prompt: str
    answer: list[str]
    scores: dict[str, float]
    sampling: NotRequired[dict[str, Any]]


class RunConfig(TypedDict):
    r"""One evaluation run: one model, a list of benchmarks and their system prompts.

    Attributes:
        url: Base URL of the vLLM server (with or without the trailing `/v1`).
        model: Served model id to evaluate. If absent, the server must serve
            exactly one model.
        default_system_prompt: System prompt inserted verbatim as the first
            message of every sample of every benchmark not in
            `benchmark_system_prompts`. `null` sends no system message.
        benchmark_system_prompts: System prompts of specific benchmarks, keyed
            by their names in `benchmarks`. Each replaces
            `default_system_prompt` for that benchmark; `null` sends no system
            message.
        answer: `[open, close]` markers around the final answer, e.g.
            `["<answer>", "</answer>"]` or `["\boxed{", "}"]`.
        data_dir: Root directory holding the datasets in the Hub layout
            (`<data_dir>/<org>/<name>/...`).
        output_dir: Parent directory of the run folders. Every run writes its
            `results.jsonl`, `results_metadata.jsonl`, `output_raw/` and logs
            to a new folder in it (see `custom_eval.results.create_run_dir`);
            `--resume` continues a run in its folder.
        exp_name: Prefix of the run folder's name; empty or null for none
            (default "").
        max_generation_workers: Maximum concurrent generation requests to the
            vLLM server, across all benchmarks.
        max_score_workers: Maximum concurrent scoring jobs (math checks,
            instruction checks, code tests), across all benchmarks. Scoring
            never holds a generation worker.
        benchmarks: Benchmarks to run
            (see `custom_eval.benchmarks.BENCHMARKS`).
        repeats: Generations per problem (default 1).
        sampling: Request parameters. Absent keys are not sent, so the server's
            defaults apply.
        judge: LLM judge, for benchmarks that need one.
    """

    url: str
    model: NotRequired[str]
    default_system_prompt: str | None
    benchmark_system_prompts: NotRequired[dict[str, str | None]]
    answer: list[str]
    data_dir: str
    output_dir: str
    exp_name: NotRequired[str | None]
    max_generation_workers: int
    max_score_workers: int
    benchmarks: list[str]
    repeats: NotRequired[int]
    sampling: NotRequired[dict[str, Any]]
    judge: NotRequired[JudgeConfig]


class ConfigError(ValueError):
    """Raised when a run config is invalid."""


def _check_keys(
    section: str,
    data: dict[str, Any],
    required: dict[str, tuple[type, ...]],
    optional: dict[str, tuple[type, ...]],
) -> None:
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {unknown}")
    missing = sorted(set(required) - set(data))
    if missing:
        raise ConfigError(f"{section}: missing required key(s) {missing}")
    for key, value in data.items():
        types = required[key] if key in required else optional[key]
        # bool is an int subclass; never accept it where a number is meant
        if isinstance(value, bool) and bool not in types:
            raise ConfigError(f"{section}.{key}: expected {types}, got {value!r}")
        if not isinstance(value, types):
            raise ConfigError(f"{section}.{key}: expected {types}, got {value!r}")


def _check_markers(section: str, answer: list[Any]) -> None:
    if len(answer) != 2 or not all(isinstance(m, str) for m in answer):
        raise ConfigError(
            f"{section}.answer: expected [open, close] strings, got {answer!r}"
        )
    if answer[0] == "":
        raise ConfigError(f"{section}.answer: the open marker must not be empty")


def _check_sampling(section: str, sampling: dict[str, Any]) -> None:
    for key in sampling:
        if not isinstance(key, str):
            raise ConfigError(f"{section}.sampling: keys must be strings, got {key!r}")
    if "n" in sampling:
        raise ConfigError(f"{section}.sampling: use 'repeats' instead of 'n'")
    if "stream" in sampling:
        raise ConfigError(f"{section}.sampling: 'stream' is not supported")


def validate_config(data: Any) -> RunConfig:
    """Validate a parsed run config.

    Args:
        data: Parsed YAML.

    Returns:
        The validated config.

    Raises:
        ConfigError: If the config is invalid.
    """
    if not isinstance(data, dict):
        raise ConfigError("config: expected a mapping at the top level")
    _check_keys(
        "config",
        data,
        required={
            "url": (str,),
            "default_system_prompt": (str, type(None)),
            "answer": (list,),
            "data_dir": (str,),
            "output_dir": (str,),
            "max_generation_workers": (int,),
            "max_score_workers": (int,),
            "benchmarks": (list,),
        },
        optional={
            "model": (str,),
            "benchmark_system_prompts": (dict,),
            "exp_name": (str, type(None)),
            "repeats": (int,),
            "sampling": (dict,),
            "judge": (dict,),
        },
    )
    _check_markers("config", data["answer"])
    if data["max_generation_workers"] < 1:
        raise ConfigError("config.max_generation_workers: must be >= 1")
    if data["max_score_workers"] < 1:
        raise ConfigError("config.max_score_workers: must be >= 1")
    if "repeats" in data and data["repeats"] < 1:
        raise ConfigError("config.repeats: must be >= 1")
    if not data["benchmarks"] or not all(
        isinstance(b, str) for b in data["benchmarks"]
    ):
        raise ConfigError("config.benchmarks: expected a non-empty list of names")
    if len(set(data["benchmarks"])) != len(data["benchmarks"]):
        raise ConfigError("config.benchmarks: duplicate benchmark names")
    if "benchmark_system_prompts" in data:
        for name, prompt in data["benchmark_system_prompts"].items():
            if name not in data["benchmarks"]:
                raise ConfigError(
                    f"config.benchmark_system_prompts: {name!r} is not in "
                    "config.benchmarks"
                )
            if not isinstance(prompt, str | None):
                raise ConfigError(
                    f"config.benchmark_system_prompts.{name}: expected a string "
                    f"or null, got {prompt!r}"
                )
    if "sampling" in data:
        _check_sampling("config", data["sampling"])
    if "judge" in data:
        judge = data["judge"]
        _check_keys(
            "config.judge",
            judge,
            required={
                "url": (str,),
                "prompt": (str,),
                "answer": (list,),
                "scores": (dict,),
            },
            optional={"model": (str,), "sampling": (dict,)},
        )
        _check_markers("config.judge", judge["answer"])
        for verdict, value in judge["scores"].items():
            if not isinstance(verdict, str) or isinstance(value, bool):
                raise ConfigError(
                    f"config.judge.scores: expected verdict: number, got {verdict!r}: {value!r}"
                )
            if not isinstance(value, int | float):
                raise ConfigError(
                    f"config.judge.scores: expected verdict: number, got {verdict!r}: {value!r}"
                )
        if "sampling" in judge:
            _check_sampling("config.judge", judge["sampling"])
    return data  # type: ignore[return-value]


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run config file (YAML).

    Args:
        path: Path to the YAML file.

    Returns:
        The validated config.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return validate_config(data)


def system_prompt(config: RunConfig, benchmark: str) -> str | None:
    """System prompt of one benchmark of the run.

    Args:
        config: The run config.
        benchmark: Benchmark name, as listed in `benchmarks`.

    Returns:
        The benchmark's `benchmark_system_prompts` entry if it has one, else
        `default_system_prompt` (None sends no system message).
    """
    if "benchmark_system_prompts" in config:
        prompts = config["benchmark_system_prompts"]
        if benchmark in prompts:
            return prompts[benchmark]
    return config["default_system_prompt"]


def repeats(config: RunConfig) -> int:
    """Generations per problem: `repeats` if set, else 1."""
    return config["repeats"] if "repeats" in config else 1


def exp_name(config: RunConfig) -> str:
    """Run folder name prefix: `exp_name` if set, else "" (also for null)."""
    if "exp_name" not in config:
        return ""
    name = config["exp_name"]
    return name if name is not None else ""
