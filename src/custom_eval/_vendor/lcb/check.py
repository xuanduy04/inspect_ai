# Runner entry point for LiveCodeBench grading, run as a fresh process:
#
#     python -m custom_eval._vendor.lcb.check <problem.json> <code.py> <result.json> <timeout>
#
# `check_correctness` and `_temp_run` are copied from LiveCodeBench
# lcb_runner/evaluation/compute_code_generation_metrics.py (MIT, see LICENSE)
# with `debug` defaulting to False and the multiprocessing context pinned to
# "fork" (the runner relies on fork and Unix signals); on a global timeout it
# returns explicit metadata instead of indexing the empty metadata list (both
# grade the generation as failed). Pass/fail follows
# lcb_runner/evaluation/pass_k_utils.py: every test result must be > 0. Test decoding follows
# lcb_runner/benchmarks/code_generation.py (CodeGenerationProblem.__post_init__
# and get_evaluation_sample). Results go to a file, not stdout, because the
# graded code and the runner's timeout handler both print.

import base64
import json
import multiprocessing
import pickle
import sys
import zlib

sys.set_int_max_str_digits(50000)

from .testing_util import run_test  # noqa: E402


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(sample, generation, timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    ctx = multiprocessing.get_context("fork")
    manager = ctx.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = ctx.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(
        timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
    )
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
        return list(result[0]), {"error_code": -1, "error_message": "Global timeout"}
    return list(result[0]), dict(metadata_list[0])


def decode_tests(encoded):
    """Decode LiveCodeBench test cases (plain JSON, or base64 + zlib + pickle + JSON)."""
    try:
        return json.loads(encoded)
    except Exception:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(encoded.encode("utf-8"))))
        )


def evaluation_sample(problem):
    """Build the `sample` dict `run_test` expects (public tests, then private)."""
    tests = decode_tests(problem["public_test_cases"]) + decode_tests(
        problem["private_test_cases"]
    )
    return {
        "input_output": json.dumps(
            {
                "inputs": [t["input"] for t in tests],
                "outputs": [t["output"] for t in tests],
                "fn_name": problem["fn_name"],
            }
        )
    }


def main():
    problem_path, code_path, result_path, timeout = sys.argv[1:5]
    with open(problem_path, encoding="utf-8") as f:
        problem = json.load(f)
    with open(code_path, encoding="utf-8") as f:
        code = f.read()
    sample = evaluation_sample(problem)
    try:
        results, metadata = check_correctness(sample, code, timeout=int(timeout))
        fixed = []
        for e in results:
            if hasattr(e, "item"):
                e = e.item()
            fixed.append(e)
        results = fixed
    except Exception as e:
        results = [-2]
        metadata = {
            "error": repr(e),
            "error_code": -5,
            "error_message": "TestRunnerError",
        }
    n_tests = len(json.loads(sample["input_output"])["inputs"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "metadata": metadata, "n_tests": n_tests},
            f,
            default=str,
        )


if __name__ == "__main__":
    main()
