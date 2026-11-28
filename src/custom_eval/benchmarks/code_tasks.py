import atexit
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from custom_eval._vendor.lcb.check import decode_tests
from custom_eval.benchmarks._code import (
    PLUS_PROGRAM_TIMEOUT_S,
    PROGRAM_TIMEOUT_S,
    code_source,
    extract_code,
    run_lcb,
    run_program,
)
from custom_eval.benchmarks._common import (
    Benchmark,
    BuildContext,
    read_rows,
    score_metrics,
)
from custom_eval.download import DataSpec
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

HUMANEVAL = DataSpec("openai/openai_humaneval", "openai_humaneval/*.parquet")
MBPP = DataSpec("google-research-datasets/mbpp", "full/test-*.parquet")
HUMANEVAL_PLUS = DataSpec("evalplus/humanevalplus", "data/*.parquet")
MBPP_PLUS = DataSpec("evalplus/mbppplus", "data/*.parquet")
LCB_REPO = "lighteval/code_generation_lite"
LCB_V1 = DataSpec(LCB_REPO, "v1/*.parquet", include=("v1/*",))
LCB_V2 = DataSpec(LCB_REPO, "v2/*.parquet", include=("v2/*",))
LCB_V3 = DataSpec(LCB_REPO, "v3/*.parquet", include=("v3/*",))
LCB_V4 = DataSpec(LCB_REPO, "v4/*.parquet", include=("v4/*",))
LCB_V5 = DataSpec(LCB_REPO, "v5/*.parquet", include=("v5/*",))
LCB_V6 = DataSpec(LCB_REPO, "v6/*.parquet", include=("v6/*",))
"""Problems new in each LiveCodeBench release; `release_vN` is the union of `v1` .. `vN`."""

CODE_METRICS = ("pass",)

LCB_FORMAT_STARTER = (
    "You will use the following starter code to write the solution to the problem."
)
"""LiveCodeBench's starter-code instruction, without its "enclose your code within delimiters"."""

LCB_FORMAT_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)
"""LiveCodeBench's stdin instruction, without its "enclose your code within delimiters as follows"."""

HUMANEVAL_PLUS_TEST_FIXES = {
    "HumanEval/32": (
        "assert _poly(*candidate(*inp), inp) <= 0.0001",
        "assert abs(_poly(*inp, candidate(*inp))) <= 0.0001",
    ),
}
"""Corrections to the `test` field of `evalplus/humanevalplus`: task id -> (old, new).

HumanEval/32 (`find_zero`): the shipped line unpacks the returned float and
fails for every solution, including the reference one. The fix applies
EvalPlus's own oracle for this task, `abs(_poly(*inp, out)) <= atol`.
"""


def _test_cache(name: str) -> Path:
    """Temporary directory for a task's test files, removed at exit.

    Tests stay on disk so they are neither held in memory nor copied into the
    logs (sample metadata only holds the file path).
    """
    cache = Path(tempfile.mkdtemp(prefix=f"custom-eval-{name}-"))
    atexit.register(shutil.rmtree, cache, ignore_errors=True)
    return cache


def humaneval_program(prompt: str, entry_point: str, test: str, code: str) -> str:
    """HumanEval test program.

    If the code defines `entry_point`, it is run after the prompt (so helper
    functions and imports from the prompt are available and the code's
    definition wins); otherwise the code is taken as the function body and
    appended to the prompt.
    """
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE):
        solution = f"{prompt}\n    pass\n\n\n{code}"
    else:
        solution = f"{prompt}{code}"
    return f"{solution}\n\n\n{test}\n\ncheck({entry_point})\n"


def mbpp_program(code: str, setup: str, tests: list[str]) -> str:
    """MBPP test program: the code, the setup code, then the asserts."""
    return f"{code}\n\n{setup}\n\n" + "\n".join(tests) + "\n"


def humaneval_plus_test(task_id: str, test: str) -> str:
    """A HumanEval+ test, with `HUMANEVAL_PLUS_TEST_FIXES` applied.

    Raises:
        ValueError: If a fix no longer matches the dataset's test.
    """
    if task_id not in HUMANEVAL_PLUS_TEST_FIXES:
        return test
    old, new = HUMANEVAL_PLUS_TEST_FIXES[task_id]
    if test.count(old) != 1:
        raise ValueError(f"{task_id}: expected test line not found: {old!r}")
    return test.replace(old, new)


def _read_test(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


@scorer(metrics=score_metrics(CODE_METRICS))
def code_pass(benchmark: str, answer_open: str, answer_close: str) -> Scorer:
    """Run the extracted code against the benchmark's tests.

    The code comes from between the last answer markers if present, else from
    the whole final output; within that, the last fenced code block if any.

    Args:
        benchmark: `humaneval`, `mbpp`, `humaneval_plus`, `mbpp_plus` or `lcb`.
        answer_open: Opening answer marker.
        answer_close: Closing answer marker.
    """

    async def score(state: TaskState, target: Target) -> Score:
        source, marker_found = code_source(
            state.output.completion, answer_open, answer_close
        )
        code = extract_code(source)
        meta = state.metadata
        if benchmark == "humaneval":
            passed, details = await run_program(
                humaneval_program(
                    meta["prompt"], meta["entry_point"], meta["test"], code
                ),
                PROGRAM_TIMEOUT_S,
            )
        elif benchmark == "mbpp":
            passed, details = await run_program(
                mbpp_program(code, meta["test_setup_code"], meta["test_list"]),
                PROGRAM_TIMEOUT_S,
            )
        elif benchmark == "humaneval_plus":
            passed, details = await run_program(
                humaneval_program(
                    meta["prompt"],
                    meta["entry_point"],
                    _read_test(meta["test_path"]),
                    code,
                ),
                PLUS_PROGRAM_TIMEOUT_S,
            )
        elif benchmark == "mbpp_plus":
            passed, details = await run_program(
                mbpp_program(
                    code, meta["test_imports"], [_read_test(meta["test_path"])]
                ),
                PLUS_PROGRAM_TIMEOUT_S,
            )
        else:
            passed, details = await run_lcb(meta["problem_path"], meta["n_tests"], code)
        return Score(
            value={"pass": float(passed)},
            answer=code,
            metadata={"marker_found": marker_found, **details},
        )

    return score


def _humaneval(ctx: BuildContext) -> Task:
    samples = [
        Sample(
            id=row["task_id"],
            input=f"Complete the following Python function.\n\n```python\n{row['prompt']}```",
            metadata={
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "test": row["test"],
            },
        )
        for row in read_rows(HUMANEVAL, ctx)
    ]
    return Task(
        name="humaneval",
        dataset=MemoryDataset(samples, name="humaneval"),
        scorer=code_pass("humaneval", ctx.answer_open, ctx.answer_close),
        sandbox="local",
    )


def _mbpp(ctx: BuildContext) -> Task:
    samples = [
        Sample(
            id=str(row["task_id"]),
            input=f"{row['text']}\nYour code should pass these tests:\n\n"
            + "\n".join(row["test_list"]),
            metadata={
                "test_setup_code": row["test_setup_code"],
                "test_list": list(row["test_list"]),
            },
        )
        for row in read_rows(MBPP, ctx)
    ]
    return Task(
        name="mbpp",
        dataset=MemoryDataset(samples, name="mbpp"),
        scorer=code_pass("mbpp", ctx.answer_open, ctx.answer_close),
        sandbox="local",
    )


def _humaneval_plus(ctx: BuildContext) -> Task:
    cache = _test_cache("humaneval_plus")
    samples: list[Sample] = []
    for row in read_rows(HUMANEVAL_PLUS, ctx):
        path = cache / f"{len(samples):04d}.py"
        path.write_text(humaneval_plus_test(row["task_id"], row["test"]), "utf-8")
        samples.append(
            Sample(
                id=row["task_id"],
                input=f"Complete the following Python function.\n\n```python\n{row['prompt']}```",
                metadata={
                    "prompt": row["prompt"],
                    "entry_point": row["entry_point"],
                    "test_path": str(path),
                },
            )
        )
    return Task(
        name="humaneval_plus",
        dataset=MemoryDataset(samples, name="humaneval_plus"),
        scorer=code_pass("humaneval_plus", ctx.answer_open, ctx.answer_close),
        sandbox="local",
    )


def _mbpp_plus(ctx: BuildContext) -> Task:
    cache = _test_cache("mbpp_plus")
    samples: list[Sample] = []
    for row in read_rows(MBPP_PLUS, ctx):
        path = cache / f"{len(samples):04d}.py"
        path.write_text(row["test"], "utf-8")
        samples.append(
            Sample(
                id=str(row["task_id"]),
                input=f"{row['prompt']}\nYour code should pass these tests:\n\n"
                + "\n".join(row["test_list"]),
                metadata={
                    "test_imports": "\n".join(row["test_imports"]),
                    "test_path": str(path),
                },
            )
        )
    return Task(
        name="mbpp_plus",
        dataset=MemoryDataset(samples, name="mbpp_plus"),
        scorer=code_pass("mbpp_plus", ctx.answer_open, ctx.answer_close),
        sandbox="local",
    )


def lcb_prompt(question: str, starter_code: str) -> str:
    """LiveCodeBench's generic prompt, minus its code-delimiter instructions."""
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMAT_STARTER}\n```python\n{starter_code}\n```\n"
    else:
        prompt += f"### Format: {LCB_FORMAT_STDIN}\n"
    return prompt


@dataclass(frozen=True)
class LcbVariant:
    """A LiveCodeBench problem set: every problem of the release folders it reads.

    Attributes:
        name: Benchmark name.
        data: Release folders to read (`LCB_V1` .. `LCB_V6`).
    """

    name: str
    data: tuple[DataSpec, ...]


def _lcb(variant: LcbVariant, ctx: BuildContext) -> Task:
    import pyarrow.parquet as pq

    # Tests stay encoded on disk; the runner decodes them per grading.
    cache = _test_cache(variant.name)
    samples: list[Sample] = []
    for spec in variant.data:
        for file in spec.files(ctx.data_dir):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                for row in batch.to_pylist():
                    samples.append(_lcb_sample(row, cache / f"{len(samples):04d}.json"))
    return Task(
        name=variant.name,
        dataset=MemoryDataset(samples, name=variant.name),
        scorer=code_pass("lcb", ctx.answer_open, ctx.answer_close),
        sandbox="local",
    )


def _lcb_sample(row: dict[str, Any], path: Path) -> Sample:
    """Build a LiveCodeBench sample, writing its encoded tests to `path`."""
    metadata = json.loads(row["metadata"])
    problem = {
        "public_test_cases": row["public_test_cases"],
        "private_test_cases": row["private_test_cases"],
        "fn_name": metadata["func_name"] if "func_name" in metadata else None,
    }
    n_tests = len(decode_tests(row["public_test_cases"])) + len(
        decode_tests(row["private_test_cases"])
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(problem, f)
    return Sample(
        id=row["question_id"],
        input=lcb_prompt(row["question_content"], row["starter_code"]),
        metadata={
            "problem_path": str(path),
            "n_tests": n_tests,
            "platform": row["platform"],
            "difficulty": row["difficulty"],
            "contest_date": row["contest_date"],
        },
    )


LCB_VARIANTS = [
    LcbVariant("lcb_v5", (LCB_V1, LCB_V2, LCB_V3, LCB_V4, LCB_V5)),
    LcbVariant("lcb_v6", (LCB_V1, LCB_V2, LCB_V3, LCB_V4, LCB_V5, LCB_V6)),
]
"""LiveCodeBench problem sets. `lcb_v5` / `lcb_v6`: the whole release v5 / v6
(folders `v1` .. `v5` / `v1` .. `v6`; 880 / 1055 problems)."""


def _code_benchmark(
    name: str, data: tuple[DataSpec, ...], build: Callable[[BuildContext], Task]
) -> Benchmark:
    return Benchmark(
        name=name,
        data=data,
        build=build,
        metrics=CODE_METRICS,
        headline="pass",
        headline_lenient=None,
        label="pass@1",
        instruction_level=False,
    )


def _lcb_benchmark(variant: LcbVariant) -> Benchmark:
    return _code_benchmark(variant.name, variant.data, lambda ctx: _lcb(variant, ctx))


BENCHMARKS = [
    Benchmark(
        name="humaneval",
        data=(HUMANEVAL,),
        build=_humaneval,
        metrics=CODE_METRICS,
        headline="pass",
        headline_lenient=None,
        label="pass@1",
        instruction_level=False,
    ),
    Benchmark(
        name="mbpp",
        data=(MBPP,),
        build=_mbpp,
        metrics=CODE_METRICS,
        headline="pass",
        headline_lenient=None,
        label="pass@1",
        instruction_level=False,
    ),
    _code_benchmark("humaneval_plus", (HUMANEVAL_PLUS,), _humaneval_plus),
    _code_benchmark("mbpp_plus", (MBPP_PLUS,), _mbpp_plus),
    *[_lcb_benchmark(variant) for variant in LCB_VARIANTS],
]
