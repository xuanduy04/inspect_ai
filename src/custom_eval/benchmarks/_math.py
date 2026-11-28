from custom_eval._score_pool import run_in_worker
from custom_eval.answer import extract_answer
from custom_eval.benchmarks._common import score_metrics
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.scorer._math import (
    _check_dependency,
    _MathLimitError,
    _MathParseError,
    _parse_targets_worker,
    _score_answer_worker,
    _WorkerScore,
)
from inspect_ai.solver import TaskState

MATH_METRICS = ("strict", "lenient")

TIMEOUT_S = 10.0
"""Budget for one math checker call in a score worker: parsing the targets, or
checking an answer (which parses the targets again, see `check_answer`)."""

WARM_UP_TARGET = "\\frac{1}{2}"
"""Target parsed and checked once per score worker, to pay the lazy sympy / latex2sympy2 imports."""


def warm_up() -> None:
    """Pay the math checker's lazy imports in a score worker (outside any timeout)."""
    parsed, error = _parse_targets_worker((WARM_UP_TARGET,))
    if error is not None:
        raise ValueError(f"Math checker warm-up failed: {error}")
    _score_answer_worker(WARM_UP_TARGET, parsed)


def target_error(targets: tuple[str, ...]) -> str | None:
    """Parse the targets in a score worker and return the parse error, if any.

    Only the error is returned: parsed targets never leave the worker
    (see `check_answer`).

    Args:
        targets: The sample's targets.

    Returns:
        The parse error, or None if the targets parse.
    """
    _, error = _parse_targets_worker(targets)
    return error


def check_answer(text: str, targets: tuple[str, ...]) -> _WorkerScore:
    """Check `text` against the targets, parsing both in this score worker.

    Parsed targets are never sent between processes: unpickling rebuilds a
    SymPy expression evaluated (`Mul(-1, 2)` becomes `-2`), and the checker
    compares tuples, sets and intervals structurally, so a target parsed in
    another process no longer equals the same answer parsed here (`(-2,1)`
    scored wrong against itself).

    An answer the checker cannot handle scores wrong. Inspect's checker raises
    its parse and limit errors from two places outside the `try` that turns
    them into a status: the size check on the whole output, and the fallback
    that retries an unmatched prose answer's other candidates. Those are turned
    into the same statuses here.

    Args:
        text: Text holding the answer.
        targets: The sample's targets.

    Returns:
        The checker's verdict.

    Raises:
        ValueError: If the targets cannot be parsed.
    """
    parsed, error = _parse_targets_worker(targets)
    if error is not None:
        raise ValueError(f"Could not parse math target {targets!r}: {error}")
    try:
        return _score_answer_worker(text, parsed)
    except _MathLimitError as ex:
        return _WorkerScore("answer_limit", None, str(ex))
    except _MathParseError as ex:
        return _WorkerScore("answer_parse_error", None, str(ex))


async def _validate_targets(targets: tuple[str, ...]) -> None:
    error = await run_in_worker(target_error, (targets,), TIMEOUT_S, warm_up)
    if error is not None:
        raise ValueError(f"Could not parse math target {targets!r}: {error}")


async def _check(text: str, targets: tuple[str, ...]) -> tuple[bool, str, str | None]:
    """Compare `text` with the targets using Inspect's math checker.

    Returns:
        (correct, status, extracted answer).
    """
    try:
        result = await run_in_worker(check_answer, (text, targets), TIMEOUT_S, warm_up)
    except TimeoutError:
        return False, "answer_timeout", None
    return result.status == "correct", result.status, result.answer


@scorer(metrics=score_metrics(MATH_METRICS))
def math_answer(answer_open: str, answer_close: str) -> Scorer:
    r"""Math equivalence on the answer between the run's markers.

    `strict`: the text between the last answer markers in the final output
    (reasoning excluded) must be mathematically equivalent to the target; no
    markers means incorrect. `lenient`: same when the markers are present;
    otherwise Inspect's default extraction (last `\boxed{}`, "answer is ...",
    last number, ...) over the whole final output.

    Args:
        answer_open: Opening answer marker.
        answer_close: Closing answer marker.
    """
    _check_dependency()

    async def score(state: TaskState, target: Target) -> Score:
        targets = tuple(target)
        await _validate_targets(targets)
        completion = state.output.completion
        answer = extract_answer(completion, answer_open, answer_close)
        if answer is not None:
            strict, status, extracted = await _check(answer, targets)
            lenient, lenient_status, lenient_extracted = strict, status, extracted
        else:
            strict, status, extracted = False, "no_marker", None
            lenient, lenient_status, lenient_extracted = await _check(
                completion, targets
            )
        return Score(
            value={"strict": float(strict), "lenient": float(lenient)},
            answer=answer,
            metadata={
                "marker_found": answer is not None,
                "strict_status": status,
                "strict_parsed": extracted,
                "lenient_status": lenient_status,
                "lenient_parsed": lenient_extracted,
            },
        )

    return score
