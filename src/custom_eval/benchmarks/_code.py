import json
import secrets
import sys

from custom_eval._score_pool import score_slots
from custom_eval.answer import extract_answer
from inspect_ai.util import sandbox

PROGRAM_TIMEOUT_S = 10
"""Wall-clock limit for one HumanEval / MBPP test program."""

PLUS_PROGRAM_TIMEOUT_S = 60
"""Wall-clock limit for one HumanEval+ / MBPP+ test program (all EvalPlus inputs
in one run; the slowest reference solution takes about 13 s)."""

LCB_TEST_TIMEOUT_S = 6
"""LiveCodeBench per-test timeout (the official default)."""

LCB_TIMEOUT_MARGIN_S = 60
"""Slack over the LiveCodeBench runner's own global timeout before the process is killed."""

MAX_DETAIL_CHARS = 2000
"""Truncation for stderr / error text kept in score metadata."""


def code_source(
    completion: str, answer_open: str, answer_close: str
) -> tuple[str, bool]:
    """Text that holds the code: between the last answer markers, or the whole output.

    Args:
        completion: Final model output (reasoning excluded).
        answer_open: Opening answer marker.
        answer_close: Closing answer marker.

    Returns:
        (text, whether the markers were found).
    """
    answer = extract_answer(completion, answer_open, answer_close)
    if answer is None:
        return completion, False
    return answer, True


def extract_code(text: str) -> str:
    """Code in `text`: the last fenced block if there is one, else the text itself.

    Fences are found as in LiveCodeBench's `extract_code`: lines containing
    three backticks; the code is what lies between the last two such lines.

    Args:
        text: Text holding the code.

    Returns:
        The code.
    """
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) >= 2:
        return "\n".join(lines[fences[-2] + 1 : fences[-1]])
    return text


async def run_program(program: str, timeout: int) -> tuple[bool, dict[str, object]]:
    """Run a Python test program in the sample's local sandbox, holding a scoring slot.

    Args:
        program: Solution followed by its tests.
        timeout: Wall-clock limit in seconds.

    Returns:
        (passed, details for score metadata).
    """
    token = secrets.token_hex(16)
    await sandbox().write_file("program.py", program)
    try:
        async with score_slots():
            result = await sandbox().exec(
                [sys.executable, "-m", "custom_eval._pyexec", "program.py", token],
                timeout=timeout,
                timeout_retry=False,
            )
    except TimeoutError:
        return False, {"status": "timeout"}
    passed = result.returncode == 0 and token in result.stdout.split()
    return passed, {
        "status": "passed" if passed else "failed",
        "returncode": result.returncode,
        "stderr": result.stderr[-MAX_DETAIL_CHARS:],
    }


async def run_lcb(
    problem_path: str, n_tests: int, code: str
) -> tuple[bool, dict[str, object]]:
    """Grade code with the vendored LiveCodeBench runner in the local sandbox, holding a scoring slot.

    Args:
        problem_path: JSON file with the problem's encoded tests and `fn_name`.
        n_tests: Number of tests (sets the kill timeout).
        code: Extracted code.

    Returns:
        (passed, details for score metadata).
    """
    await sandbox().write_file("solution.py", code)
    runner_timeout = (LCB_TEST_TIMEOUT_S + 1) * n_tests + 5
    try:
        async with score_slots():
            result = await sandbox().exec(
                [
                    sys.executable,
                    "-m",
                    "custom_eval._vendor.lcb.check",
                    problem_path,
                    "solution.py",
                    "result.json",
                    str(LCB_TEST_TIMEOUT_S),
                ],
                timeout=runner_timeout + LCB_TIMEOUT_MARGIN_S,
                timeout_retry=False,
            )
    except TimeoutError:
        return False, {"status": "runner_timeout"}
    try:
        output = json.loads(await sandbox().read_file("result.json"))
    except FileNotFoundError:
        raise RuntimeError(
            f"LiveCodeBench runner wrote no result (exit {result.returncode}): "
            f"{result.stderr[-MAX_DETAIL_CHARS:]}"
        ) from None
    results = output["results"]
    # lcb_runner/evaluation/pass_k_utils.py: a generation passes when every result is > 0
    passed = len(results) > 0 and all(
        isinstance(r, bool | int | float) and r > 0 for r in results
    )
    metadata = output["metadata"]
    return passed, {
        "status": "passed" if passed else "failed",
        "tests_passed": sum(1 for r in results if r is True),
        "n_tests": output["n_tests"],
        "error_code": metadata["error_code"] if "error_code" in metadata else None,
        "error_message": str(metadata["error_message"])[:MAX_DETAIL_CHARS]
        if "error_message" in metadata
        else None,
    }
