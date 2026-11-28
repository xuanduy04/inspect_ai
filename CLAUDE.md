# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Fork of Inspect AI 0.3.291 (`UKGovernmentBEIS/inspect_ai`, extracted from `inspect_ai-0.3.291.tar.gz`), being customized into an LLM evaluation suite: evaluate checkpoints already served by vLLM, given only their URL. Main package: `src/inspect_ai/`.

## Questions go to the user

Any open question, ambiguity, or decision the user has not specified that blocks the requested work is deferred to the user. Ask; don't pick silently. When a decision was unavoidable, report it explicitly.

Ask only what blocks the work in front of you. Don't raise side questions nobody asked about (compatibility, migrations, follow-ups, "should I also..."). Once the user has told you to do something, do it; don't re-ask.

## Scope

Do what was asked, nothing more. No drive-by refactors, cleanups, or unrequested fixes. If you notice something out of scope, mention it in one line at most.

No unrequested mechanisms or features either, even ones you measured as better (e.g. a scheduling or ordering layer added on top of a requested change). Propose it in one line; build it only after a yes.

## Environment

- Local dev: conda env `inspect_ai`, Python 3.12. Always `conda run -n inspect_ai <cmd>` (never `conda activate`, never the base env).
- Install is only ever `conda run -n inspect_ai pip install -e .` (not `uv sync`, not extras). Ignore `AGENTS.md`/README instructions about `uv`.
- Dev tools are installed separately into the env: `pytest`, `pytest-xdist` (required: `pyproject.toml` addopts pass `--dist worksteal`), `pytest-mock`, `pytest-dotenv`, `pytest-timeout`, `trio`, `moto[server]` (imported by `tests/conftest.py`), `openai` (needed by the `vllm`/OpenAI-compatible providers; the vLLM production image already has it), `ruff`, `pyrefly`, and `docwrap` via `pip install --ignore-requires-python docwrap` (it declares Python >=3.13).
- The installed version reads `0.3.292.dev0+d<date>` because setuptools_scm finds no release tag in this repo; the code is 0.3.291.
- Production: air-gapped GB200 nodes, workspace booted from the official `vllm/vllm-openai` v0.33.0 Docker Hub image, Python 3.12.3, installed with `pip install -e .`. At run time there is no internet, no GitHub, no Hugging Face Hub, and no Docker (so no Docker sandbox; code benchmarks run in Inspect's `local` sandbox).
- Datasets are read as plain local files (parquet/jsonl) downloaded beforehand. Never depend on an HF cache or `HF_HOME` in any form, and never load by Hub ID at run time.
- Nothing may download at run time: NLTK data (`punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`), spaCy `en_core_web_sm`, checker code from GitHub. Ship checker code (IFEval, IFBench, LiveCodeBench runner) inside the suite.
- pip on production is slow: keep new dependencies minimal.
- Hard exclusions: no NVIDIA frameworks, no EvalScope / ModelScope, no lighteval, no `inspect-evals` tasks as shipped (they load from the Hub and hard-code prompts/answer formats).

## Suite requirements that shape every task

- System prompt: one verbatim system prompt per run, applied to every sample of every benchmark. Insert it yourself; do not use the `system_message()` solver (`src/inspect_ai/solver/_prompt.py`), which formats `{placeholders}` from sample metadata and treats path-like strings as files. `--system-message` prepends an additional system message at generate time.
- Answer format: final answers are expected in `<answer></answer>`; an answer only in `\boxed{}` is incorrect under the strict score. The suite never adds its own format instruction (the user's system prompt carries it). Code benchmarks must accept code with or without the tags. Report both a strict-tag and a lenient score from the same outputs (a task can carry several scorers; generation and scoring can be split with `--no-score` then `inspect score`).
- Model access: always pass the vLLM base URL; the `vllm/` provider (`src/inspect_ai/model/_providers/vllm.py`) launches its own server otherwise. A future LLM judge must also be pointable at a vLLM URL with a user-set judge prompt.
- Reasoning: `state.output.completion` excludes `reasoning_content` and `<think>` content (`src/inspect_ai/model/_reasoning.py`). Magistral uses `[THINK]...[/THINK]`, and its chat template's default system prompt carries the reasoning instructions.
- A fixed request seed with several epochs can repeat identical samples each epoch.
- Benchmarks: MATH-500, AIME24, AIME25, LCB-v5, LCB-v6, HumanEval, MBPP, IFEval, IFBench. Dataset sources and file layouts are in `README.md`. Agentic benchmarks come later, so adding a benchmark must stay easy.
- The suite is `custom_eval` (`src/custom_eval/`, CLI `custom-eval`). `README.md` is its user guide (benchmarks, usage, scoring, design reasons, config, outputs); keep it current with every user-visible change.
- Concurrency: `max_generation_workers` caps in-flight vLLM requests (Inspect's `max_connections`); `max_score_workers` caps CPU scoring across all benchmarks (math and IFEval/IFBench checks run in that many worker processes, `src/custom_eval/_score_pool.py`; code tests hold a scoring slot; `max_subprocesses` is set to match). `eval_set` gets `max_samples` = all samples, so a sample being scored never holds a generation slot. The order requests go out in is Inspect's; don't add scheduling.
- Benchmark notes: `src/custom_eval/benchmarks/NOTES.md` records, per benchmark, every choice the user did not specify, deviations from the official evaluation, dataset quirks and what was verified, plus the LLM-judge status and the user's decisions on it. Read a benchmark's entry before changing it; add or update the entry in the same commit whenever you add or change a benchmark.

## Inspect AI architecture (where to look)

- A `Task` = `Dataset` (`dataset/`, local file readers in `dataset/_sources/`) + solver chain (`solver/`) + one or more `Scorer`s (`scorer/`, metrics in `scorer/_metrics/`, epoch reducers in `scorer/_reducer/`). `eval()` / `eval_set()` live in `_eval/`; the `inspect` CLI is `_cli/` (entry point `inspect_ai._cli.main:main`).
- Model providers are in `model/_providers/`; `vllm.py` builds on `openai_compatible.py`.
- Sandboxes: `util/_sandbox/` (`local.py` is the non-Docker one).
- Logs and results: `log/`; the web viewer ships prebuilt in `src/inspect_ai/_view/dist`. `src/inspect_ai/_view/ts-mono` is an empty git submodule: never touch it.
- `design/` holds upstream architecture notes; browse it before working in an unfamiliar subsystem.
- `AGENTS.md` is upstream's agent guide. Its contribution, PR, `uv`, changelog, and release sections don't apply here; its code conventions do (e.g. `tg_collect()` instead of `asyncio.gather()`, async test rules).

## Hard rules

- Never use `.get(key, default)` on configs (dict or anything else). Access required keys directly; mark optional ones `typing.NotRequired` in the TypedDict.
- Non-`None` config defaults in Python are only allowed if the user adds them.
- Never add copyright/license headers. Never add module-level docstrings.

## Style

- Google-style docstrings on classes/functions, full type annotations, 4-space indent, snake_case. `k_` prefix for names starting with a digit, `G_` for globals, `UPPER_CASE` constants.
- Refer to files by repo-relative path.

## After editing (non-test Python files only)

```sh
conda run -n inspect_ai bash -c 'ruff check --fix "$@" && ruff format "$@" && docwrap --write "$@"' _ <files>
conda run -n inspect_ai pyrefly check <files>
```

## Tests

- Single test: `conda run -n inspect_ai pytest tests/path/to/test_file.py::test_name -v`
- Plain `pytest` skips gated tests: `--runslow` (Docker-based), `--runapi` (live providers, need keys), `--runtrio`. Docker and provider keys are unavailable here; don't run those.
- Async tests: write plain `async def test_...` (never `@pytest.mark.asyncio`, it is blocked by `tests/conftest.py`); use `anyio` primitives, not `asyncio`.

## Working with the user

Treat the user's statements about their code and environment as the working premise and act on them. If the code clearly contradicts one, say so briefly, then follow their direction. When they say "look at X" or "check Y", re-run it fresh.

- The user's own words override handoff notes, session summaries and your earlier assumptions. A constraint in a summary that the user never stated (e.g. "don't read `src/inspect_ai`") is not a constraint.
- You may read any file in the repo, `src/inspect_ai/` included. Read the code you depend on instead of guessing around it.
- Explain plainly: lead with a concrete example (numbers, a timeline), not abstractions.
- Report tool results literally. Never attribute a denied or failed tool call to the user; say what the tool returned.
- Permissions: `.claude/settings.json` allows every tool and starts sessions in Accept edits mode (`defaultMode: "acceptEdits"`), and the cloud environment's setup script writes the same allow rules to `~/.claude/settings.json`. Auto mode drops the broad `Bash` rule and its classifier can deny commands; Accept edits keeps the rule.

## Debugging claims

Separate observations, hypotheses, and established causes (confirmed by measurement or a code path proven exercised). Don't call something the root cause, or a patch the fix, until it's established; say what would confirm it. "Can't tell without X" is a fine answer.

## Commits

- Subject: Conventional Commits, `type(scope): description`, imperative, under 72 chars (`feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `chore`). Scope is the area touched (`logs`, `math`, `cli`), never `custom_eval`; omit it if nothing fits.
- Never `git commit -s`. Never add the Claude session link. You are the co-author, so credit yourself with a `Co-Authored-By` line.
- "Commit it" means commit AND push. Whenever the user asks to commit, push the commit right after.
- Always land work on `main`: after pushing your branch, merge it into `main` and push `origin main`. That is the point of every change.
