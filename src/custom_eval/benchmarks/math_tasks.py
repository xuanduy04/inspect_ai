from collections.abc import Callable

from custom_eval.benchmarks._common import Benchmark, BuildContext, read_rows
from custom_eval.benchmarks._math import MATH_METRICS, math_answer
from custom_eval.download import DataSpec
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample

MATH500 = DataSpec("HuggingFaceH4/MATH-500", "test.jsonl")
AIME24 = DataSpec("HuggingFaceH4/aime_2024", "data/*.parquet")
AIME25 = DataSpec("yentinglin/aime_2025", "data/*.parquet")
AIME26 = DataSpec("MathArena/aime_2026", "data/*.parquet")
HMMT_FEB25 = DataSpec("MathArena/hmmt_feb_2025", "data/*.parquet")
HMMT_FEB26 = DataSpec("MathArena/hmmt_feb_2026", "data/*.parquet")
HMMT_NOV25 = DataSpec("MathArena/hmmt_nov_2025", "data/*.parquet")
BEYONDAIME = DataSpec("ByteDance-Seed/BeyondAIME", "data/*.parquet")

MATH500_EXTRA_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    # "Express in base b". The checker reads an unbraced subscript as Python's
    # digit separator (`52_8` is 528) and drops a braced one (`52_{8}` is 52),
    # so the bare digits and the braced form failed against these golds.
    "test/number_theory/864.json": ("52_8", ("52",)),
    "test/number_theory/46.json": ("40_9", ("40",)),
    "test/number_theory/516.json": ("2516_8", ("2516",)),
    "test/number_theory/1257.json": ("204_5", ("204",)),
    "test/number_theory/368.json": ("4343_6", ("4343",)),
    # The same, braced: this gold reads as 4210, so `4210_5` (42105) failed.
    "test/number_theory/847.json": ("4210_{5}", ("4210_5",)),
    # "Enter all such integers, separated by commas": the checker compares a
    # list of plain numbers in order.
    "test/intermediate_algebra/1388.json": ("1,-2", ("-2,1",)),
    "test/intermediate_algebra/1102.json": ("-2,1", ("1,-2",)),
    # "Enter the letter of the correct option": a bare `E` parses as Euler's
    # number.
    "test/precalculus/452.json": ("\\text{(E)}", ("E",)),
}
"""MATH-500 gold answers whose notation makes Inspect's checker reject correct
answers: id -> (gold, extra accepted answers). A match with any target counts.

The gold is stored so that a changed dataset fails loudly instead of getting
another problem's extras. Problems where any extra target would also accept a
plausible wrong answer are left out (see `NOTES.md`).
"""


def math500_targets(unique_id: str, gold: str) -> str | list[str]:
    """A MATH-500 problem's target: its gold answer, then `MATH500_EXTRA_TARGETS`.

    Args:
        unique_id: The problem's `unique_id`.
        gold: The problem's gold answer.

    Returns:
        The gold alone for a problem not in the table (the target every other
        math benchmark has), else the list of the gold and its extras.

    Raises:
        ValueError: If the dataset's gold no longer matches the table.
    """
    if unique_id not in MATH500_EXTRA_TARGETS:
        return gold
    expected, extra = MATH500_EXTRA_TARGETS[unique_id]
    if gold != expected:
        raise ValueError(
            f"{unique_id}: gold {gold!r} differs from {expected!r} in MATH500_EXTRA_TARGETS"
        )
    return [gold, *extra]


def _task(name: str, samples: list[Sample], ctx: BuildContext) -> Task:
    return Task(
        name=name,
        dataset=MemoryDataset(samples, name=name),
        scorer=math_answer(ctx.answer_open, ctx.answer_close),
    )


def _math500(ctx: BuildContext) -> Task:
    samples = [
        Sample(
            id=row["unique_id"],
            input=row["problem"],
            target=math500_targets(row["unique_id"], row["answer"]),
            metadata={"subject": row["subject"], "level": row["level"]},
        )
        for row in read_rows(MATH500, ctx)
    ]
    missing = set(MATH500_EXTRA_TARGETS) - {s.id for s in samples}
    if missing:
        raise ValueError(
            f"MATH500_EXTRA_TARGETS ids not in the dataset: {sorted(missing)}"
        )
    return _task("math500", samples, ctx)


def _aime(name: str, spec: DataSpec, ctx: BuildContext) -> Task:
    samples = [
        Sample(id=str(row["id"]), input=row["problem"], target=str(row["answer"]))
        for row in read_rows(spec, ctx)
    ]
    return _task(name, samples, ctx)


def _matharena(name: str, spec: DataSpec, ctx: BuildContext) -> Task:
    samples = [
        Sample(
            id=str(row["problem_idx"]),
            input=row["problem"],
            target=str(row["answer"]),
        )
        for row in read_rows(spec, ctx)
    ]
    return _task(name, samples, ctx)


def _beyondaime(ctx: BuildContext) -> Task:
    # the dataset has no id column: the id is the row's position in the file
    samples = [
        Sample(id=str(i), input=row["problem"], target=str(row["answer"]))
        for i, row in enumerate(read_rows(BEYONDAIME, ctx))
    ]
    return _task("beyondaime", samples, ctx)


def _math_benchmark(
    name: str, spec: DataSpec, build: Callable[[BuildContext], Task]
) -> Benchmark:
    return Benchmark(
        name=name,
        data=(spec,),
        build=build,
        metrics=MATH_METRICS,
        headline="strict",
        headline_lenient="lenient",
        label="accuracy",
        instruction_level=False,
    )


BENCHMARKS = [
    Benchmark(
        name="math500",
        data=(MATH500,),
        build=_math500,
        metrics=MATH_METRICS,
        headline="strict",
        headline_lenient="lenient",
        label="accuracy",
        instruction_level=False,
    ),
    Benchmark(
        name="aime24",
        data=(AIME24,),
        build=lambda ctx: _aime("aime24", AIME24, ctx),
        metrics=MATH_METRICS,
        headline="strict",
        headline_lenient="lenient",
        label="accuracy",
        instruction_level=False,
    ),
    Benchmark(
        name="aime25",
        data=(AIME25,),
        build=lambda ctx: _aime("aime25", AIME25, ctx),
        metrics=MATH_METRICS,
        headline="strict",
        headline_lenient="lenient",
        label="accuracy",
        instruction_level=False,
    ),
    _math_benchmark("aime26", AIME26, lambda ctx: _matharena("aime26", AIME26, ctx)),
    _math_benchmark(
        "hmmt_feb25", HMMT_FEB25, lambda ctx: _matharena("hmmt_feb25", HMMT_FEB25, ctx)
    ),
    _math_benchmark(
        "hmmt_feb26", HMMT_FEB26, lambda ctx: _matharena("hmmt_feb26", HMMT_FEB26, ctx)
    ),
    _math_benchmark(
        "hmmt_nov25", HMMT_NOV25, lambda ctx: _matharena("hmmt_nov25", HMMT_NOV25, ctx)
    ),
    _math_benchmark("beyondaime", BEYONDAIME, _beyondaime),
]
