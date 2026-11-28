import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from custom_eval.answer import extract_answer
from custom_eval.benchmarks._code import extract_code
from custom_eval.benchmarks.code_tasks import humaneval_program, lcb_prompt
from custom_eval.benchmarks.if_tasks import _clean_kwargs, follow_instructions
from custom_eval.config import ConfigError, exp_name, repeats, validate_config
from custom_eval.download import DataMissingError, DataSpec, ensure_data
from custom_eval.judge import judge_prompt
from custom_eval.model import api_base, generate_config
from custom_eval.results import (
    clustered_stderr,
    create_run_dir,
    model_dir_name,
    output_order,
    ratio_stderr,
    run_dir_name,
)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "url": "https://h:8000/v1",
        "system_prompt": "Put the answer in <answer></answer>.",
        "answer": ["<answer>", "</answer>"],
        "data_dir": "/data",
        "output_dir": "/out",
        "max_generation_workers": 8,
        "max_score_workers": 4,
        "benchmarks": ["aime24"],
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "text,open_marker,close_marker,expected",
    [
        ("<answer>1</answer> then <answer>2</answer>", "<answer>", "</answer>", "2"),
        ("no tags here, \\boxed{3}", "<answer>", "</answer>", None),
        ("<answer>unterminated", "<answer>", "</answer>", None),
        ("<answer>1</answer> <answer>2", "<answer>", "</answer>", None),
        ("x \\boxed{\\frac{1}{2}} y", "\\boxed{", "}", "\\frac{1}{2}"),
        ("\\boxed{1} and \\boxed{\\{a, b\\}}", "\\boxed{", "}", "\\{a, b\\}"),
        ("\\boxed{\\frac{1}{2}", "\\boxed{", "}", None),
        ("Answer: 42", "Answer:", "", " 42"),
        ("### 1 ### 2 ###", "###", "###", " 2 "),
        ("### 1", "###", "###", None),
        ("<answer></answer>", "<answer>", "</answer>", ""),
    ],
)
def test_extract_answer(
    text: str, open_marker: str, close_marker: str, expected: str | None
) -> None:
    assert extract_answer(text, open_marker, close_marker) == expected


def test_extract_code_last_fenced_block() -> None:
    text = "```python\na = 1\n```\ntext\n```\nb = 2\n```"
    assert extract_code(text) == "b = 2"
    assert extract_code("def f():\n    return 1") == "def f():\n    return 1"


def test_humaneval_program_body_and_full_function() -> None:
    prompt = 'def f(x):\n    """Doc."""\n'
    body = "    return x + 1\n"
    assert humaneval_program(prompt, "f", "def check(c): pass", body).startswith(
        prompt + body
    )
    full = "def f(x):\n    return x + 1\n"
    program = humaneval_program(prompt, "f", "def check(c): pass", full)
    assert program.index(full) > program.index(prompt)
    assert program.rstrip().endswith("check(f)")


def test_pyexec_runs_programs_like_the_official_graders(tmp_path: Path) -> None:
    # human-eval and EvalPlus run code with exec(code, {}): an answer's own
    # `if __name__ == "__main__":` block (e.g. unittest.main()) does not run.
    # Code that exits before the end, or fails an assert, still fails.
    import subprocess
    import sys

    def passes(program: str) -> bool:
        path = tmp_path / "program.py"
        path.write_text(program)
        result = subprocess.run(
            [sys.executable, "-m", "custom_eval._pyexec", str(path), "TOKEN"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0 and "TOKEN" in result.stdout.split()

    main_block = "if __name__ == '__main__':\n    raise SystemExit(1)\n"
    assert passes(f"def f():\n    return 1\n\n{main_block}\nassert f() == 1\n")
    assert not passes("import sys\nsys.exit(0)\nassert True\n")
    assert not passes("assert 1 == 2\n")


def test_humaneval_plus_find_zero_fix() -> None:
    from custom_eval.benchmarks.code_tasks import humaneval_plus_test

    shipped = (
        "import math\n"
        "def _poly(xs, x):\n"
        "    return sum([c * math.pow(x, i) for i, c in enumerate(xs)])\n"
        "def check(candidate):\n"
        "    inputs = [[[-6, 11, -6, 1]], [[1, 2]]]\n"
        "    for i, inp in enumerate(inputs):\n"
        "        assert _poly(*candidate(*inp), inp) <= 0.0001\n"
    )
    fixed = humaneval_plus_test("HumanEval/32", shipped)
    assert humaneval_plus_test("HumanEval/0", shipped) == shipped
    namespace: dict[str, object] = {}
    exec(fixed, namespace)
    check = namespace["check"]
    assert callable(check)
    roots = {(-6, 11, -6, 1): 2.0, (1, 2): -0.5}
    check(lambda xs: roots[tuple(xs)])  # true roots pass
    with pytest.raises(AssertionError):
        check(lambda xs: 0.0)  # not a root
    with pytest.raises(ValueError):
        humaneval_plus_test("HumanEval/32", "def check(candidate): pass")


def test_lcb_variants_registered() -> None:
    from custom_eval.benchmarks import BENCHMARKS
    from custom_eval.benchmarks.code_tasks import LCB_VARIANTS

    assert [v.name for v in LCB_VARIANTS] == ["lcb_v5", "lcb_v6"]
    assert [name for name in BENCHMARKS if name.startswith("lcb")] == [
        "lcb_v5",
        "lcb_v6",
    ]


def test_lcb_v5_v6_are_whole_releases() -> None:
    from custom_eval.benchmarks import BENCHMARKS
    from custom_eval.benchmarks.code_tasks import (
        LCB_V1,
        LCB_V2,
        LCB_V3,
        LCB_V4,
        LCB_V5,
        LCB_V6,
        LCB_VARIANTS,
    )

    release_v5 = (LCB_V1, LCB_V2, LCB_V3, LCB_V4, LCB_V5)
    variants = {v.name: v for v in LCB_VARIANTS}
    for name, data in (("lcb_v5", release_v5), ("lcb_v6", (*release_v5, LCB_V6))):
        assert variants[name].data == data
        assert BENCHMARKS[name].data == data


def test_lcb_v5_v6_build_every_problem_of_the_release(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from custom_eval.benchmarks import BENCHMARKS, BuildContext

    # one problem per release folder; v4 also holds an older contest, as the
    # real v4 does, and nothing is cut by date
    dates = {
        "v1": ["2023-05-07T00:00:00"],
        "v2": ["2024-03-09T00:00:00"],
        "v3": ["2024-06-01T00:00:00"],
        "v4": ["2023-08-26T00:00:00", "2024-10-05T00:00:00"],
        "v5": ["2025-01-04T00:00:00"],
        "v6": ["2025-04-06T00:00:00"],
    }
    tests = json.dumps([{"input": "1\n", "output": "1\n", "testtype": "stdin"}])
    for folder, contest_dates in dates.items():
        rows = [
            {
                "question_title": "t",
                "question_content": f"{folder} problem {i}",
                "platform": "atcoder",
                "question_id": f"{folder}_{i}",
                "contest_id": "c",
                "contest_date": contest_date,
                "starter_code": "",
                "difficulty": "easy",
                "public_test_cases": tests,
                "private_test_cases": tests,
                "metadata": "{}",
            }
            for i, contest_date in enumerate(contest_dates)
        ]
        path = tmp_path / "lighteval/code_generation_lite" / folder / "test.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    ctx = BuildContext(tmp_path, "<answer>", "</answer>", None)

    release_v5 = ["v1_0", "v2_0", "v3_0", "v4_0", "v4_1", "v5_0"]
    for name, ids in (("lcb_v5", release_v5), ("lcb_v6", [*release_v5, "v6_0"])):
        task = BENCHMARKS[name].build(ctx)
        assert task.name == name
        assert [sample.id for sample in task.dataset] == ids
        assert all(sample.metadata["n_tests"] == 2 for sample in task.dataset)


def test_lcb_prompt_has_no_delimiter_instruction() -> None:
    assert "delimiters" not in lcb_prompt("Q", "")
    assert "delimiters" not in lcb_prompt("Q", "class Solution: pass")
    assert "class Solution: pass" in lcb_prompt("Q", "class Solution: pass")


def test_validate_config_ok_and_repeats_default() -> None:
    config = validate_config(_config())
    assert repeats(config) == 1
    assert repeats(validate_config(_config(repeats=4))) == 4
    assert validate_config(_config(system_prompt=None))["system_prompt"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": 1},
        {"answer": ["<answer>"]},
        {"answer": ["", "</answer>"]},
        {"max_generation_workers": 0},
        {"max_generation_workers": True},
        {"max_score_workers": 0},
        {"max_score_workers": True},
        {"max_connections": 8},
        {"repeats": 0},
        {"benchmarks": []},
        {"benchmarks": ["a", "a"]},
        {"sampling": {"n": 4}},
        {"judge": {"url": "u", "prompt": "p", "answer": ["<v>", "</v>"]}},
    ],
)
def test_validate_config_rejects(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        validate_config(_config(**overrides))


@pytest.mark.parametrize(
    "key", ["system_prompt", "max_generation_workers", "max_score_workers"]
)
def test_validate_config_missing_key(key: str) -> None:
    config = _config()
    del config[key]
    with pytest.raises(ConfigError):
        validate_config(config)


def test_exp_name_defaults_to_empty() -> None:
    assert exp_name(validate_config(_config())) == ""
    assert exp_name(validate_config(_config(exp_name=None))) == ""
    assert exp_name(validate_config(_config(exp_name=""))) == ""
    assert exp_name(validate_config(_config(exp_name="sft-a"))) == "sft-a"
    with pytest.raises(ConfigError):
        validate_config(_config(exp_name=1))


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("Qwen/Qwen3-8B", "qwen-qwen3-8b"),
        ("zai-org/GLM-4.7-Flash", "zai-org-glm-4-7-flash"),
        ("unsloth/Magistral-Small-2507", "unsloth-magistral-small-2507"),
        ("/ckpt/Qwen3_8B.sft/step-100/", "ckpt-qwen3-8b-sft-step-100"),
    ],
)
def test_model_dir_name(model_id: str, expected: str) -> None:
    assert model_dir_name(model_id) == expected


def test_run_dir_name() -> None:
    started = datetime(2026, 11, 25, 9, 5, 7, tzinfo=UTC)
    assert run_dir_name("", "Qwen/Qwen3-8B", started) == (
        "qwen-qwen3-8b-20261125-09_05_07"
    )
    assert run_dir_name("sft-a", "Qwen/Qwen3-8B", started) == (
        "sft-a-qwen-qwen3-8b-20261125-09_05_07"
    )


def test_run_dir_names_sort_by_start_time() -> None:
    # zero-padded, largest unit first, 24-hour clock
    starts = [
        datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 9, 59, 59, tzinfo=UTC),
        datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 13, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 10, 0, 0, 0, tzinfo=UTC),
        datetime(2026, 10, 1, 0, 0, 0, tzinfo=UTC),
    ]
    for prefix in ("", "sft-a"):
        names = [run_dir_name(prefix, "Qwen/Qwen3-8B", s) for s in starts]
        assert sorted(names) == names


def test_create_run_dir_takes_the_next_second_when_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custom_eval import results

    # local time, as create_run_dir reads it
    start = datetime(2026, 11, 25, 9, 5, 7).astimezone()
    clock = iter([start, start, start + timedelta(seconds=1)])

    class Clock:
        @staticmethod
        def now() -> datetime:
            return next(clock)

    monkeypatch.setattr(results, "datetime", Clock)
    first = create_run_dir(tmp_path / "out", "", "Qwen/Qwen3-8B")
    second = create_run_dir(tmp_path / "out", "", "Qwen/Qwen3-8B")
    assert first.name == "qwen-qwen3-8b-20261125-09_05_07"
    assert second.name == "qwen-qwen3-8b-20261125-09_05_08"


def test_create_run_dir_spammed_runs_get_their_own_folders(tmp_path: Path) -> None:
    folders = [create_run_dir(tmp_path / "out", "", "Qwen/Qwen3-8B") for _ in range(3)]
    assert all(f.is_dir() for f in folders)
    names = [f.name for f in folders]
    assert len(set(names)) == 3 and sorted(names) == names


def test_check_resume(tmp_path: Path) -> None:
    import yaml

    from custom_eval.cli import RUN_FILE, check_resume

    config = validate_config(_config())
    with pytest.raises(ConfigError):  # not a run folder
        check_resume(config, tmp_path, "m")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / RUN_FILE).write_text(json.dumps({"model": "m"}) + "\n")
    check_resume(config, tmp_path, "m")
    # where new runs go, the server URL and the worker counts may change
    moved = _config(
        url="https://other:8000",
        output_dir="/elsewhere",
        exp_name="x",
        max_generation_workers=1,
        max_score_workers=1,
    )
    check_resume(validate_config(moved), tmp_path, "m")
    with pytest.raises(ConfigError):
        check_resume(validate_config(_config(system_prompt="other")), tmp_path, "m")
    with pytest.raises(ConfigError):  # the server now serves another model
        check_resume(config, tmp_path, "other")


def test_output_order_is_repeat_major_in_dataset_order() -> None:
    from inspect_ai.log import EvalSample

    dataset_ids: list[int | str | None] = ["b", "a", "c"]
    samples = [
        EvalSample(id=i, epoch=e, input="q", target="t")
        for e in (2, 1)
        for i in ("c", "a", "b")
    ]
    ordered = [(s.epoch, s.id) for s in output_order(samples, dataset_ids)]
    assert ordered == [(1, "b"), (1, "a"), (1, "c"), (2, "b"), (2, "a"), (2, "c")]
    # line k is the same problem whatever the number of repeats
    one_repeat = [s for s in samples if s.epoch == 1]
    assert [s.id for s in output_order(one_repeat, dataset_ids)] == ["b", "a", "c"]


def test_generate_config_split() -> None:
    config = generate_config(
        {
            "temperature": 0.6,
            "top_k": 20,
            "stop": "</s>",
            "chat_template_kwargs": {"a": 1},
        },
        16,
    )
    assert config.temperature == 0.6
    assert config.stop_seqs == ["</s>"]
    assert config.extra_body == {"top_k": 20, "chat_template_kwargs": {"a": 1}}
    assert config.max_connections == 16
    empty = generate_config({}, None)
    assert empty.temperature is None and empty.max_tokens is None
    assert empty.extra_body is None and empty.seed is None


def test_pool_limits_raise_inspects_pool_to_max_generation_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from custom_eval.model import NoAuthVLLMAPI, pool_limits
    from inspect_ai.model import GenerateConfig

    monkeypatch.delenv("INSPECT_HTTP_POOL_CONNECTIONS", raising=False)
    monkeypatch.delenv("INSPECT_HTTP_POOL_KEEPALIVE_CONNECTIONS", raising=False)
    assert pool_limits(None).max_connections == 1000
    assert pool_limits(256).max_connections == 1000  # never lowered
    raised = pool_limits(100_000)
    assert raised.max_connections == raised.max_keepalive_connections == 100_000
    monkeypatch.setenv("INSPECT_HTTP_POOL_CONNECTIONS", "5000")
    assert pool_limits(256).max_connections == 5000
    assert pool_limits(100_000).max_connections == 100_000

    # the evaluated model's client is built with them
    api = object.__new__(NoAuthVLLMAPI)
    api._init_config = GenerateConfig(max_connections=100_000)
    client = api._create_http_client()
    assert client._transport._kwargs["limits"].max_connections == 100_000


def test_api_base() -> None:
    assert api_base("https://h:8000") == "https://h:8000/v1"
    assert api_base("https://h:8000/v1/") == "https://h:8000/v1"


def test_judge_prompt_is_plain_replacement() -> None:
    template = "Q: {question}\nR: {response}\nT: {target}\n{other} {}"
    assert (
        judge_prompt(template, "q {x}", "r", "t") == "Q: q {x}\nR: r\nT: t\n{other} {}"
    )


def test_stderr() -> None:
    assert clustered_stderr([[1.0], [0.0]]) == pytest.approx(0.5)
    assert clustered_stderr([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]) == pytest.approx(
        (((0.5 - 0.5) ** 2 + 0.25 + 0.25) / 2 / 3) ** 0.5
    )
    assert clustered_stderr([[1.0]]) is None
    assert ratio_stderr([1.0, 2.0], [2.0, 2.0]) is not None


def test_follow_instructions_missing_response_fails_all() -> None:
    from custom_eval._vendor.ifeval import instructions_registry
    from custom_eval.benchmarks._nltk import prepare_checkers

    prepare_checkers()
    ids = ["punctuation:no_comma", "change_case:english_lowercase"]
    strict, loose = follow_instructions(instructions_registry, None, "p", ids, [{}, {}])
    assert strict == [False, False] and loose == [False, False]
    strict, loose = follow_instructions(
        instructions_registry,
        "this is a plain english sentence written entirely in lowercase letters",
        "p",
        ids,
        [{}, {}],
    )
    assert strict == [True, True] and loose == [True, True]


def test_clean_kwargs_drops_none() -> None:
    assert _clean_kwargs({"a": None, "b": 0, "c": "x"}) == {"b": 0, "c": "x"}


def test_ensure_data_downloads_with_hfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = tmp_path / "calls.log"
    hfd = tmp_path / "hfd.sh"
    hfd.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> {calls}\n'
        'dest=""; while [ $# -gt 0 ]; do [ "$1" = --local-dir ] && dest="$2"; shift; done\n'
        'mkdir -p "$dest/v5" && echo x > "$dest/v5/test.parquet"\n'
    )
    spec = DataSpec("org/name", "v5/*.parquet", include=("v5/*",))
    data_dir = tmp_path / "data"

    monkeypatch.delenv("PATH_TO_CUSTOM_HFD", raising=False)
    with pytest.raises(DataMissingError):
        ensure_data([spec], data_dir)

    monkeypatch.setenv("PATH_TO_CUSTOM_HFD", str(hfd))
    ensure_data([spec], data_dir)
    assert spec.files(data_dir)
    assert calls.read_text().split() == [
        "org/name",
        "--dataset",
        "--local-dir",
        str(data_dir / "org/name"),
        "--include",
        "v5/*",
    ]
    # present files are not downloaded again
    ensure_data([spec], data_dir)
    assert len(calls.read_text().splitlines()) == 1


def test_vendored_checkers_never_download() -> None:
    import nltk

    from custom_eval.benchmarks._nltk import prepare_checkers

    prepare_checkers()
    with pytest.raises(RuntimeError):
        nltk.download("punkt")
    assert len(nltk.data.path) == 1
    assert os.path.isdir(nltk.data.path[0])
    assert json.dumps(nltk.sent_tokenize("A b. C d.")) == '["A b.", "C d."]'


def test_resolve_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from custom_eval import model as model_module
    from custom_eval.model import ModelLookupError, resolve_model_id

    monkeypatch.setattr(model_module, "served_models", lambda url: ["a"])
    assert resolve_model_id("u", None) == "a"
    assert resolve_model_id("u", "a") == "a"
    with pytest.raises(ModelLookupError):
        resolve_model_id("u", "b")
    monkeypatch.setattr(model_module, "served_models", lambda url: ["a", "lora"])
    with pytest.raises(ModelLookupError):
        resolve_model_id("u", None)
    assert resolve_model_id("u", "lora") == "lora"


async def test_requests_carry_no_authorization() -> None:
    import httpx2

    from custom_eval.model import _drop_authorization

    request = httpx2.Request(
        "POST", "https://h/v1/chat/completions", headers={"Authorization": "Bearer x"}
    )
    await _drop_authorization(request)
    assert "Authorization" not in request.headers


def test_raw_record_errored_sample() -> None:
    from custom_eval.results import raw_record
    from inspect_ai.log import EvalError, EvalSample
    from inspect_ai.model import ChatMessageUser

    sample = EvalSample(
        id="x",
        epoch=1,
        input="q",
        target="t",
        messages=[ChatMessageUser(content="q")],
        error=EvalError(message="boom", traceback="", traceback_ansi=""),
    )
    record = raw_record("math500", "m", sample)
    assert record["error"] == "boom"
    assert record["score"] is None and record["output"] is None
    assert record["truncated"] is False


async def test_score_pool_math_checks_in_worker_processes() -> None:
    from custom_eval._score_pool import configure_score_workers
    from custom_eval.benchmarks._math import _check, _validate_targets

    configure_score_workers(2)
    targets = ("\\frac{3}{4}",)
    await _validate_targets(targets)
    assert (await _check("0.75", targets))[:2] == (True, "correct")
    assert (await _check("0.5", targets))[:2] == (False, "incorrect")
    with pytest.raises(ValueError):
        await _validate_targets(("",))


@pytest.mark.parametrize(
    "gold",
    [
        "(-2,1)",
        "1,-2",
        "3 \\pm 2 \\sqrt{2}",
        "x \\in [-2,7]",
        "\\left[ \\frac{\\pi^2}{8}, \\frac{5 \\pi^2}{4} \\right]",
        "\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}",
    ],
)
async def test_score_pool_math_composite_targets_match_themselves(gold: str) -> None:
    # MATH-500 / HMMT gold answers. Targets parsed in one process and checked in
    # another lost their unevaluated SymPy structure in pickling, so these
    # scored wrong against themselves.
    from custom_eval._score_pool import configure_score_workers
    from custom_eval.benchmarks._math import _check, _validate_targets

    configure_score_workers(2)
    await _validate_targets((gold,))
    assert (await _check(gold, (gold,)))[:2] == (True, "correct")


async def test_score_pool_math_over_limit_answer_scores_wrong() -> None:
    # Inspect's checker raises _MathLimitError outside its own try when an
    # answer's prose candidate does not match and a later candidate exceeds a
    # size limit, and when the output is over 1,000,000 characters. Such
    # answers errored their sample (and then the benchmark); they score wrong.
    from custom_eval._score_pool import configure_score_workers
    from custom_eval.benchmarks._math import _check

    configure_score_workers(2)
    long_last_line = "The answer is not clear yet.\n" + "1 + " * 2000 + "1"
    assert await _check(long_last_line, ("83",)) == (False, "answer_limit", None)
    assert await _check("x" * 1_000_001, ("83",)) == (False, "answer_limit", None)


async def test_score_pool_math500_extra_targets_accept_and_reject() -> None:
    from custom_eval._score_pool import configure_score_workers
    from custom_eval.benchmarks._math import _check, _validate_targets
    from custom_eval.benchmarks.math_tasks import MATH500_EXTRA_TARGETS

    configure_score_workers(2)
    for gold, extra in MATH500_EXTRA_TARGETS.values():
        targets = (gold, *extra)
        await _validate_targets(targets)
        for answer in targets:
            assert (await _check(answer, targets))[:2] == (True, "correct"), answer
    # the braced form, rejected by the gold alone
    assert (await _check("52_{8}", ("52_8", "52")))[:2] == (True, "correct")
    # wrong answers stay wrong: 52_8 is 42 in base 10, other digits in base 5,
    # a different list, another option
    wrong = {
        "test/number_theory/864.json": "42",
        "test/number_theory/847.json": "4201_5",
        "test/intermediate_algebra/1388.json": "1,2",
        "test/precalculus/452.json": "D",
    }
    for unique_id, answer in wrong.items():
        gold, extra = MATH500_EXTRA_TARGETS[unique_id]
        assert (await _check(answer, (gold, *extra)))[0] is False, answer


def test_math500_build_applies_extra_targets(tmp_path: Path) -> None:
    from custom_eval.benchmarks import BuildContext
    from custom_eval.benchmarks.math_tasks import MATH500_EXTRA_TARGETS, _math500

    def write(rows: list[dict[str, str]]) -> BuildContext:
        path = tmp_path / "HuggingFaceH4" / "MATH-500" / "test.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return BuildContext(tmp_path, "<answer>", "</answer>", None)

    def row(unique_id: str, answer: str) -> dict[str, str]:
        return {
            "unique_id": unique_id,
            "problem": unique_id,
            "answer": answer,
            "subject": "s",
            "level": "1",
        }

    rows = [row(u, gold) for u, (gold, _) in MATH500_EXTRA_TARGETS.items()]
    rows.append(row("test/algebra/1.json", "5"))
    targets = {s.id: s.target for s in _math500(write(rows)).dataset}
    assert targets["test/number_theory/1257.json"] == ["204_5", "204"]
    assert targets["test/algebra/1.json"] == "5"
    with pytest.raises(ValueError):  # a gold changed under the table
        _math500(write([*rows[1:], row(rows[0]["unique_id"], "999")]))
    with pytest.raises(ValueError):  # an id of the table is gone
        _math500(write(rows[1:]))


def test_score_workers_start_when_stderr_has_no_fileno() -> None:
    # Inspect's full display (Textual) replaces sys.stderr with an object whose
    # fileno() is -1. multiprocessing's resource tracker, launched by the first
    # worker start, passed that to fork_exec: "bad value(s) in fds_to_keep" on
    # every math and IF sample. A fresh interpreter, so no tracker runs yet.
    import subprocess
    import sys

    script = (
        "import sys, anyio\n"
        "from custom_eval._score_pool import configure_score_workers, run_in_worker\n"
        "configure_score_workers(1)\n"
        "class Capture:\n"
        "    def write(self, text): return len(text)\n"
        "    def flush(self): pass\n"
        "    def fileno(self): return -1\n"
        "sys.stderr = Capture()\n"
        "async def main():\n"
        "    try:\n"
        "        print(await run_in_worker(abs, (-3,), 60.0, None))\n"
        "    except Exception as ex:\n"
        "        print(repr(ex))\n"
        "anyio.run(main)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.stdout.strip() == "3", result.stdout + result.stderr


async def test_score_pool_timeout_kills_worker_and_errors_propagate() -> None:
    import time

    from custom_eval._score_pool import configure_score_workers, run_in_worker

    configure_score_workers(1)
    with pytest.raises(TimeoutError):
        await run_in_worker(time.sleep, (30,), 1.0, None)
    # the killed worker is replaced: the single slot still works
    assert await run_in_worker(abs, (-3,), 10.0, None) == 3
    with pytest.raises(ValueError):
        await run_in_worker(int, ("x",), 10.0, None)


async def test_score_pool_instruction_checks() -> None:
    from custom_eval._score_pool import configure_score_workers, run_in_worker
    from custom_eval.benchmarks.if_tasks import check_instructions

    configure_score_workers(2)
    strict, loose = await run_in_worker(
        check_instructions,
        (
            "ifeval",
            "hello world",
            "Answer in lowercase.",
            ["change_case:english_lowercase"],
            [{}],
        ),
        None,
        None,
    )
    assert strict == [True] and loose == [True]


def test_unreachable_localhost_server_is_retried() -> None:
    import httpx
    from openai import APIConnectionError

    from custom_eval.model import NoAuthVLLMAPI

    api = object.__new__(NoAuthVLLMAPI)
    ex = APIConnectionError(
        request=httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
    )
    assert api.should_retry(ex).retry
    assert not api.should_retry(RuntimeError("CUDA out of memory")).retry
