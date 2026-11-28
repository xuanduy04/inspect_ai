import random
from types import ModuleType
from typing import Any

from custom_eval._score_pool import run_in_worker
from custom_eval.answer import extract_answer
from custom_eval.benchmarks._common import (
    Benchmark,
    BuildContext,
    read_rows,
    score_metrics,
)
from custom_eval.benchmarks._nltk import prepare_checkers
from custom_eval.download import DataSpec
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

IFEVAL = DataSpec("google/IFEval", "ifeval_input_data.jsonl")
IFBENCH = DataSpec("allenai/IFBench_test", "data/*.parquet")

CHECKS = ("strict", "loose")
IF_METRICS = tuple(f"prompt_{c}" for c in CHECKS)

CHECKER_SEED = 0
"""Seed for the checkers' random draws, reset before each instruction is built."""

G_CHECKER_RNG = random.Random(CHECKER_SEED)
"""Private RNG installed in the vendored checker modules in place of `random`."""

G_REGISTRIES: dict[str, ModuleType] = {}
"""Prepared `instructions_registry` module per checker, in this process."""


def _loose_variants(response: str) -> list[str]:
    # evaluation_lib.test_instruction_following_loose (IFEval and IFBench)
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    return [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]


def _build(
    registry: ModuleType, instruction_id: str, kwargs: dict[str, Any], prompt: str
) -> Any:
    # evaluation_lib.test_instruction_following_{strict,loose}. Builders draw
    # random arguments when one is missing or invalid (e.g. IFEval key 1129's
    # letter "!"); reseeding makes that draw the same in every run.
    G_CHECKER_RNG.seed(CHECKER_SEED)
    instruction = registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    return instruction


def follow_instructions(
    registry: ModuleType,
    response: str | None,
    prompt: str,
    instruction_ids: list[str],
    kwargs_list: list[dict[str, Any]],
) -> tuple[list[bool], list[bool]]:
    """Strict and loose per-instruction verdicts, as in the official evaluation_lib.

    A missing response follows no instruction (`empty_output`). The builder is
    re-created for each check, as upstream does, because building can draw
    random values when an argument is absent.

    Args:
        registry: `instructions_registry` module of the checker (IFEval or
            IFBench).
        response: Text to check, or None if there is none.
        prompt: The sample's prompt.
        instruction_ids: Instruction ids.
        kwargs_list: Instruction arguments (None values already removed).

    Returns:
        (strict verdicts, loose verdicts).
    """
    if response is None:
        return [False] * len(instruction_ids), [False] * len(instruction_ids)
    strict: list[bool] = []
    loose: list[bool] = []
    variants = _loose_variants(response)
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        instruction = _build(registry, instruction_id, kwargs, prompt)
        strict.append(bool(response.strip() and instruction.check_following(response)))
        instruction = _build(registry, instruction_id, kwargs, prompt)
        loose.append(
            any(v.strip() and instruction.check_following(v) for v in variants)
        )
    return strict, loose


def checker_registry(checker: str) -> ModuleType:
    """Prepare a checker in this process (once) and return its `instructions_registry`.

    Points NLTK at the shipped data and installs `G_CHECKER_RNG` in the checker
    modules.

    Args:
        checker: `ifeval` (Google's checkers) or `ifbench` (AllenAI's).
    """
    if checker in G_REGISTRIES:
        return G_REGISTRIES[checker]
    prepare_checkers()
    if checker == "ifeval":
        from custom_eval._vendor.ifeval import (
            instructions,
            instructions_registry,
            instructions_util,
        )

        modules: list[ModuleType] = [instructions, instructions_util]
    else:
        from custom_eval._vendor.ifbench import (  # type: ignore[no-redef]
            classic_instructions,
            instructions,
            instructions_registry,
            instructions_util,
        )

        modules = [instructions, classic_instructions, instructions_util]
    for module in modules:
        setattr(module, "random", G_CHECKER_RNG)
    G_REGISTRIES[checker] = instructions_registry
    return instructions_registry


def check_instructions(
    checker: str,
    response: str | None,
    prompt: str,
    instruction_ids: list[str],
    kwargs_list: list[dict[str, Any]],
) -> tuple[list[bool], list[bool]]:
    """`follow_instructions` with the named checker (runs in a score worker).

    Args:
        checker: `ifeval` or `ifbench`.
        response: Text to check, or None if there is none.
        prompt: The sample's prompt.
        instruction_ids: Instruction ids.
        kwargs_list: Instruction arguments (None values already removed).

    Returns:
        (strict verdicts, loose verdicts).
    """
    return follow_instructions(
        checker_registry(checker), response, prompt, instruction_ids, kwargs_list
    )


@scorer(metrics=score_metrics(IF_METRICS))
def instruction_following(checker: str, answer_open: str, answer_close: str) -> Scorer:
    """IFEval / IFBench verification of the whole final output.

    The answer markers play no part in what is checked: the whole final output
    (reasoning excluded) is checked, strictly and loosely, as in the official
    evaluation. `marker_found` is only recorded for the format rate. The checks
    run in score workers.

    Args:
        checker: `ifeval` (Google's checkers) or `ifbench` (AllenAI's).
        answer_open: Opening answer marker (format rate only).
        answer_close: Closing answer marker (format rate only).
    """
    # prepared here too, so a broken checker fails before generation starts
    checker_registry(checker)

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        meta = state.metadata
        strict, loose = await run_in_worker(
            check_instructions,
            (
                checker,
                completion,
                meta["prompt"],
                meta["instruction_id_list"],
                meta["kwargs"],
            ),
            None,
            None,
        )
        marker_found = extract_answer(completion, answer_open, answer_close) is not None
        return Score(
            value={
                "prompt_strict": float(all(strict)),
                "prompt_loose": float(all(loose)),
            },
            answer=completion,
            metadata={
                "marker_found": marker_found,
                "follow_strict": strict,
                "follow_loose": loose,
            },
        )

    return score


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Parquet stores every argument name for every instruction, unused ones as
    # None; the official evaluation drops None values (IFBench evaluation_lib).
    return {k: v for k, v in kwargs.items() if v is not None}


def _task(name: str, spec: DataSpec, ctx: BuildContext) -> Task:
    samples = [
        Sample(
            id=str(row["key"]),
            input=row["prompt"],
            metadata={
                "prompt": row["prompt"],
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": [_clean_kwargs(k) for k in row["kwargs"]],
            },
        )
        for row in read_rows(spec, ctx)
    ]
    return Task(
        name=name,
        dataset=MemoryDataset(samples, name=name),
        scorer=instruction_following(name, ctx.answer_open, ctx.answer_close),
    )


BENCHMARKS = [
    Benchmark(
        name="ifeval",
        data=(IFEVAL,),
        build=lambda ctx: _task("ifeval", IFEVAL, ctx),
        metrics=IF_METRICS,
        headline="prompt_strict",
        headline_lenient=None,
        label="prompt-level strict acc",
        instruction_level=True,
    ),
    Benchmark(
        name="ifbench",
        data=(IFBENCH,),
        build=lambda ctx: _task("ifbench", IFBENCH, ctx),
        metrics=IF_METRICS,
        headline="prompt_loose",
        headline_lenient=None,
        label="prompt-level loose acc",
        instruction_level=True,
    ),
]
