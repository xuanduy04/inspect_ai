# Custom Eval

Custom Eval evaluates a model that **vLLM already serves**, given only the
server's URL. It runs math, code and instruction-following benchmarks under
**your** system prompts and **your** answer format, and prints a results table
(mean ± standard error, pass@k, format and truncation rates).

- One run = one config file: one model, a list of benchmarks and their system
  prompts (a default one, plus any benchmark's own).
- Everything runs offline: datasets are plain local files, and every checker
  ships inside this repository.
- It is built on [Inspect AI](https://inspect.aisi.org.uk/) 0.3.291. This
  repository is a fork, and `src/inspect_ai/` is Inspect itself. Every run also
  writes full Inspect logs.

Contents: [Quick start](#quick-start) · [Benchmarks](#benchmarks) ·
[How answers are scored](#how-answers-are-scored) ·
[Why it works this way](#why-it-works-this-way) · [Config reference](#config-reference) ·
[Dataset files](#dataset-files) · [Outputs](#outputs) · [Resuming a run](#resuming-a-run) ·
[LLM judge](#llm-judge) · [Adding a benchmark](#adding-a-benchmark) ·
[Repository layout](#repository-layout)

## Quick start

**1. Install** on the GPU node, from the repository root (Python 3.11+; the
vLLM image's Python 3.12 works):

```bash
pip install -e .
```

This adds a few small dependencies: `pyarrow`, the math checker's
`latex2sympy2-extended` and `antlr4-python3-runtime`, and the IFEval/IFBench
checkers' `nltk`, `langdetect`, `emoji` and `syllapy`. Nothing is downloaded at
run time.

**2. Put the datasets** under one directory, in the Hugging Face layout
`<data_dir>/<org>/<name>/...`. [Dataset files](#dataset-files) lists exactly
which files each benchmark reads. If `PATH_TO_CUSTOM_HFD` points to your
`hfd.sh`, missing datasets are fetched automatically at the start of a run.

**3. Serve the model** with vLLM, as usual. For reasoning models, start vLLM
with its reasoning parser so the reasoning is kept out of the judged answer.

**4. Write a config.** Start from one of the two examples in
`examples/custom_eval/`. Both run `math500`, `aime24`, `aime25`, `humaneval`,
`mbpp`, `lcb_v5`, `lcb_v6`, `ifeval` and `ifbench`, and differ only in their
system prompts:
- `config.yaml`: the default system prompt asks for the final answer in
  `<answer></answer>` tags; IFEval and IFBench get the same prompt without
  that instruction (`benchmark_system_prompts`).
- `mistral_config.yaml`: the same for Mistral models. Its system prompts open
  with reasoning instructions in Mistral's `[THINK]...[/THINK]` format instead
  of the helpful-assistant text.

In short, a config looks like this:

```yaml
url: https://gpu-node-07:8000/v1
default_system_prompt: |
  Think step by step. Put your final answer between <answer> and </answer>.
benchmark_system_prompts:  # optional: these benchmarks get their own prompt
  ifeval: |
    Think step by step.
answer: ["<answer>", "</answer>"]
data_dir: /data/hf
output_dir: /results
exp_name: prompt-a
max_generation_workers: 256
max_score_workers: 64
benchmarks: [math500, aime25, aime26, humaneval, lcb_v6, ifeval]
repeats: 4
sampling: {temperature: 0.6, top_p: 0.95, max_tokens: 32768, seed: 0}
```

**5. Run it:**

```bash
custom-eval run.yaml
```

At the end a table is printed. For each benchmark it shows:
- Score: the benchmark's reported number (its headline metric) as % ± stderr,
  the same number `results.jsonl` holds;
- the headline metric as % ± stderr: strict and lenient for math; code and
  IFEval/IFBench have a single score, shown in the Strict column;
- pass@k (k = repeats);
- the format rate: the share of outputs that contain your answer markers;
- the truncation rate: the share of outputs stopped by `max_tokens` or the
  context window;
- the number of problems and repeats.

The same numbers, plus every individual generation, are written to a new run
folder in `output_dir`, e.g. `/results/prompt-a-qwen-qwen3-8b-20261125-14_30_05`
for `Qwen/Qwen3-8B` (see [Outputs](#outputs)). The exit code is 0 only if
every benchmark succeeded. If a run is interrupted, run the same command with
`--resume <run folder>` (see [Resuming a run](#resuming-a-run)).

To evaluate the same model under other system prompts, copy the config,
change `default_system_prompt` or `benchmark_system_prompts` (and `exp_name`,
to tell the runs apart), and run again. Every run gets its own folder, so the
same config can also be launched many times, even at once.

## Benchmarks

Put any of these names in `benchmarks:`.

**Math.** Each benchmark gives the problem as the user message, verbatim.
Scores: `strict` and `lenient` accuracy.

| Name | Problems | What it is |
|---|---|---|
| `math500` | 500 | MATH-500 |
| `aime24` | 30 | AIME 2024 |
| `aime25` | 30 | AIME 2025 |
| `aime26` | 30 | AIME 2026 (MathArena) |
| `hmmt_feb25` | 30 | HMMT February 2025 (MathArena) |
| `hmmt_feb26` | 33 | HMMT February 2026 (MathArena's final-answer selection) |
| `hmmt_nov25` | 30 | HMMT November 2025 (MathArena) |
| `beyondaime` | 100 | BeyondAIME (ByteDance Seed): problems at least as hard as AIME 11-15 |

**Code.** Score: `pass` (pass@1). Code is accepted with or without your answer
markers.

| Name | Problems | What it is |
|---|---|---|
| `humaneval` | 164 | HumanEval |
| `humaneval_plus` | 164 | HumanEval+ (EvalPlus): HumanEval with many more tests |
| `mbpp` | 500 | MBPP, full test split |
| `mbpp_plus` | 378 | MBPP+ (EvalPlus): MBPP with many more tests |
| `lcb_v5` | 880 | LiveCodeBench release v5: every problem of the release |
| `lcb_v6` | 1055 | LiveCodeBench release v6: every problem of the release |

**Instruction following.** The whole final output is checked; the answer
markers play no part. Scores: prompt-level and instruction-level accuracy,
each with the checkers' strict and loose modes.

| Name | Problems | What it is |
|---|---|---|
| `ifeval` | 541 | IFEval (headline: prompt-level strict) |
| `ifbench` | 300 | IFBench (headline: prompt-level loose) |

Each benchmark's implementation choices are listed in
[`src/custom_eval/benchmarks/NOTES.md`](src/custom_eval/benchmarks/NOTES.md):
the prompt wording, time limits, deviations from the official evaluation,
dataset quirks, what was verified, and what is still unreviewed.

## How answers are scored

**What is judged.** Only the final output is judged. Reasoning returned by
vLLM's reasoning parser (`reasoning_content`) and `<think>` blocks are left out,
but they are saved in `output_raw/<benchmark>.jsonl`.

**The answer.** The answer is the text between the **last** opening marker and
the next closing marker after it.
- If the opening marker ends in `{`, `(`, `[` or `<` and the closing marker is
  the matching bracket, brackets are balanced, so `\boxed{\frac{1}{2}}` gives
  `\frac{1}{2}`.
- An empty closing marker takes the rest of the output.
- An unterminated last marker means no answer.

**Strict and lenient (math).** Both come from the same outputs:
- `strict`: only the answer between your markers counts. No markers means
  wrong. This measures whether the model follows your format.
- `lenient`: the answer between the markers if they are present. Otherwise,
  the usual extraction over the whole output: the last `\boxed{}`,
  "the answer is ...", or the last number.

Code has a single score, because code is accepted either way. IFEval and
IFBench have a single score too: they always check the whole final output,
with or without markers.

| Benchmarks | User message | How it is checked |
|---|---|---|
| math (all) | the problem, verbatim | Inspect's math equivalence checker (sympy): `\frac{17}{2}` = `8.5`, roots in any order, `74` = `74^\circ`. For 9 MATH-500 problems whose gold answer is written in a way the checker misreads, a plain form also counts, e.g. `204` for `204_5` ("express in base 5"); the list is in `NOTES.md` |
| `humaneval`, `humaneval_plus` | `Complete the following Python function.` + the function stub in a python fence | the official tests; 10 s per program (60 s for the plus version) |
| `mbpp`, `mbpp_plus` | `{task}\nYour code should pass these tests:\n\n{3 asserts}` | the official tests; 10 s (60 s for the plus version) |
| `lcb_*` | LiveCodeBench's own prompt, minus its "enclose your code in delimiters" sentences | the official LiveCodeBench runner, public + private tests, 6 s per test |
| `ifeval`, `ifbench` | the prompt, verbatim | the official checkers (shipped in `src/custom_eval/_vendor/`), on the whole final output |

**Extracting code.** The code is taken from between the markers if they are
present, otherwise from the whole output. Within that text, the last fenced
code block is used if there is one, otherwise the text as-is. HumanEval and
MBPP code runs the way their official graders run it: an
`if __name__ == "__main__":` block in the answer (its own tests, a demo) is
not executed.

## Why it works this way

- **Only a URL.** The suite never launches or loads a model. It talks to your
  vLLM server's OpenAI-compatible API, checks the model name via `/v1/models`,
  and sends no API key and skips TLS verification.
- **Your system prompts, verbatim.** The system prompt is the thing you vary
  between runs, so every sample gets exactly the text you wrote: its
  benchmark's entry in `benchmark_system_prompts`, or else
  `default_system_prompt`. So one run can give benchmarks different prompts,
  e.g. IFEval and IFBench without the answer-tag instruction (as in the
  examples). A prompt is never templated and never read as a file. Inspect's
  own `system_message()` does both, which is why it isn't used.
- **No format instruction of our own.** The format instruction lives in your
  system prompt, so you decide its wording. That is why `strict` exists: it
  shows whether the model followed your format. `lenient` shows the
  underlying ability anyway.
- **Offline by construction.** The GPU nodes have no internet, GitHub or
  Hugging Face. So datasets are read as plain parquet/jsonl files, never through
  a Hugging Face cache. The IFEval, IFBench and LiveCodeBench checkers and the
  NLTK data they need are shipped in the repository, and NLTK's downloader is
  disabled, so a missing resource fails loudly instead of trying the network.
- **No Docker.** Code runs in Inspect's `local` sandbox (a subprocess with
  LiveCodeBench's guard against destructive calls), because Docker is not
  available on the nodes.
- **Repeats and honest error bars.** Small sets such as AIME's 30 problems
  swing by several points between runs. So `repeats` generates several answers
  per problem, and the standard error is clustered by problem: it reflects the
  number of problems, not the number of generations. With a `seed`, repeat k
  uses `seed + k - 1`, because a fixed seed would make every repeat identical.
- **Truncation and format rates.** Reasoning models often hit the token budget,
  and a truncated answer scores 0. These two rates tell you whether a low score
  means "can't solve it" or "ran out of tokens / ignored the format".
- **Generation and scoring are separate pools.** A sample that has finished
  generating never holds a generation worker while it is scored, so slow
  scoring (code tests, math checks) does not leave the server idle. Math and
  instruction checks run in separate processes, because sympy and the
  checkers are pure Python: in threads of one process they would take turns
  on one core (the GIL) and stall the requests.
- **A new folder per run.** Runs never share a folder, so many can be launched
  from one config (one per system prompt, all at once) without overwriting
  each other. Folder names end in the start time, so listing them shows one
  model's runs oldest first.
- **Resumable, and refusing to mix runs.** Long runs get interrupted, so
  `--resume` continues a run where it stopped. Changing anything that affects
  what is generated or scored would silently mix two experiments, so that is
  refused.

## Config reference

| Key | Required | Meaning |
|---|---|---|
| `url` | yes | vLLM server URL, with or without `/v1`. Requests carry no API key and skip TLS verification. |
| `model` | no | Served model id. Without it the server must serve exactly one model; with it, the model must be served. Checked via `/v1/models`. |
| `default_system_prompt` | yes | Text inserted verbatim as the system message of every sample of every benchmark not in `benchmark_system_prompts`. `null` sends no system message (the chat template's default then applies). |
| `benchmark_system_prompts` | no | System prompts of specific benchmarks, as `name: prompt`, keyed by the benchmark's name exactly as written in `benchmarks` (any other key is refused). Each replaces `default_system_prompt` for that benchmark; `null` sends no system message. To give several benchmarks one prompt without pasting it twice, use a YAML anchor: `&if_prompt` after `ifeval:` names its prompt, and `ifbench: *if_prompt` reuses it (as in the examples). |
| `answer` | yes | `[open, close]` final-answer markers, e.g. `["<answer>", "</answer>"]` or `["\\boxed{", "}"]`. |
| `data_dir` | yes | Dataset root (see [Dataset files](#dataset-files)). |
| `output_dir` | yes | Parent folder of the run folders: every run writes to a new folder in it (see [Outputs](#outputs)). `--resume` continues a run in its folder. |
| `exp_name` | no (`""`) | Prefix of the run folder's name. Empty or `null`: no prefix. |
| `max_generation_workers` | yes | Maximum concurrent generation requests to the server, across all benchmarks. The HTTP connection pool (Inspect's default: 1,000 connections) is raised to match. Each request in flight holds a socket, so the open-file limit (`ulimit -n`) must be well above the number in flight: with 1,000 in flight, runs failed under the common default of 1,024. |
| `max_score_workers` | yes | Maximum concurrent scoring jobs (math checks, IFEval/IFBench checks, code tests), across all benchmarks. Math and instruction checks run in this many worker processes; code tests in this many subprocesses. |
| `benchmarks` | yes | Benchmark names from [Benchmarks](#benchmarks). |
| `repeats` | no (1) | Generations per problem. |
| `sampling` | no | Request parameters. Keys left out are not sent, so vLLM's defaults (the checkpoint's `generation_config.json`) apply. `temperature`, `top_p`, `max_tokens`, `seed`, `presence_penalty`, `frequency_penalty` and `stop` are standard. Any other key (`top_k`, `min_p`, `repetition_penalty`, `chat_template_kwargs`, ...) is sent as-is in the request body. |
| `judge` | no | LLM judge (see [LLM judge](#llm-judge)). |

## Dataset files

Every dataset lives under `data_dir` in the Hugging Face layout:
`<data_dir>/<org>/<name>/...`.

| Benchmark | Hugging Face dataset | Files read | Size |
|---|---|---|---|
| `math500` | `HuggingFaceH4/MATH-500` | `test.jsonl` | 0.4 MB |
| `aime24` | `HuggingFaceH4/aime_2024` | `data/*.parquet` | 0.1 MB |
| `aime25` | `yentinglin/aime_2025` | `data/*.parquet` (not `part1/`, `part2/`) | 0.02 MB |
| `aime26` | `MathArena/aime_2026` | `data/*.parquet` | 0.01 MB |
| `hmmt_feb25` | `MathArena/hmmt_feb_2025` | `data/*.parquet` | 0.01 MB |
| `hmmt_feb26` | `MathArena/hmmt_feb_2026` | `data/*.parquet` | 0.01 MB |
| `hmmt_nov25` | `MathArena/hmmt_nov_2025` | `data/*.parquet` | 0.01 MB |
| `beyondaime` | `ByteDance-Seed/BeyondAIME` | `data/*.parquet` | 0.02 MB |
| `humaneval` | `openai/openai_humaneval` | `openai_humaneval/*.parquet` | 0.1 MB |
| `humaneval_plus` | `evalplus/humanevalplus` | `data/*.parquet` | 2.9 MB |
| `mbpp` | `google-research-datasets/mbpp` | `full/test-*.parquet` | 0.1 MB |
| `mbpp_plus` | `evalplus/mbppplus` | `data/*.parquet` | 1.1 MB |
| `lcb_*` | `lighteval/code_generation_lite` | some of `v1/` .. `v6/` (below) | below |
| `ifeval` | `google/IFEval` | `ifeval_input_data.jsonl` | 0.2 MB |
| `ifbench` | `allenai/IFBench_test` | `data/*.parquet` | 0.1 MB |

**LiveCodeBench.** The whole `lighteval/code_generation_lite` repository is over
100 GB, so fetch only the folders you need. Folder `vN/` holds the problems
added in release N, and release N is `v1/` .. `vN/` together.

| Benchmark | Folders |
|---|---|
| `lcb_v5` | `v1/` .. `v5/` (4.2 GB) |
| `lcb_v6` | `v1/` .. `v6/` (4.3 GB) |

**Automatic download.** If `PATH_TO_CUSTOM_HFD` holds the absolute path of your
`hfd.sh`, missing datasets are fetched at the start of a run:

```bash
export PATH_TO_CUSTOM_HFD=/abs/path/to/hfd.sh
```

Each missing dataset is fetched with
`bash $PATH_TO_CUSTOM_HFD <repo_id> --dataset --local-dir <data_dir>/<repo_id>`.
For LiveCodeBench, `--include` is added with only the folders the run needs,
e.g. `--include v5/* v6/*`. Without the variable, the run stops and lists what
is missing.

## Outputs

Every run writes to a new **run folder** in `output_dir`:

```
<output_dir>/<exp_name>-<model>-<YYYYMMDD-HH_MM_SS>/
```

- `<exp_name>-` is left out when `exp_name` is empty or `null`.
- `<model>` is the served model id in lowercase, with every run of characters
  other than `a-z` and `0-9` (`.`, `/`, `_`, ...) turned into one `-`:
  `Qwen/Qwen3-8B` gives `qwen-qwen3-8b`, `zai-org/GLM-4.7-Flash` gives
  `zai-org-glm-4-7-flash`.
- The time is when the run started, in the machine's local time (24-hour
  clock). For the same `exp_name` and model, alphabetical order is therefore
  chronological order: the latest run is listed last (unless the clock was
  turned back in between, e.g. at the end of daylight saving time). A run that
  would get the name of an existing folder (same second) takes the next
  second instead.
- The folder is printed when the run starts (`Run folder: ...`).

Inside it:

- `results.jsonl`: one line per benchmark, holding only `benchmark`,
  `score_mean` and `score_stderr`: the benchmark's reported number (the
  table's Score column) and its stderr, clustered by problem, as fractions
  (0.85 = 85%). Both are `null` if the benchmark failed; `score_stderr` is also
  `null` when it can't be computed (e.g. a single problem).
- `results_metadata.jsonl`: everything else, one line per benchmark.
  - Every metric has its `mean` over all (problem, repeat) pairs and its
    `stderr`, clustered by problem.
  - With repeats > 1, each metric also has `pass@k` (k = repeats: any repeat
    correct) and `per_repeat` (one mean per repeat).
  - Each line also has `format_rate`, `truncation_rate`, `mean_output_tokens`,
    `n_problems`, `repeats` and the log path.
  - IFEval and IFBench report the four official scores: `prompt_strict`,
    `prompt_loose`, `inst_strict` and `inst_loose` (strict and loose are the
    checkers' two modes, not the answer tags).
- `output_raw/<benchmark>.jsonl` (e.g. `output_raw/math500.jsonl`): one line
  per generation of that benchmark, with:
  - the messages sent, the output and the reasoning;
  - the stop reason and token usage;
  - the extracted answer and the scores;
  - the full judgement: math parse status, code test results and errors,
    per-instruction verdicts.

  The lines are in a fixed order: repeat 1 of every problem in dataset order,
  then repeat 2, and so on. So line k (k up to the number of problems) is the
  same problem in every run of that benchmark, whatever `repeats` is. A failed
  benchmark's file holds only the samples it has, in the same order.
- `logs/`: Inspect logs, one per benchmark. Browse them with
  `inspect view --log-dir <run folder>/logs`.
- `config.yaml`: a copy of the run config.
- `run.jsonl`: the served model id (one line), which `--resume` checks.

## Resuming a run

Run the same config again, with `--resume` and the run folder:

```bash
custom-eval run.yaml --resume /results/prompt-a-qwen-qwen3-8b-20261125-14_30_05
```

The resume picks up from `<run folder>/logs`:
- **Finished benchmarks** are reused as they are, with no requests sent.
- **Failed or interrupted benchmarks** keep every completed sample; only the
  rest are generated. This covers errors, Ctrl-C, `kill -9` and crashed nodes.
- **`results.jsonl`, `results_metadata.jsonl` and `output_raw/`** are
  rewritten, and the table is printed again.

Without `--resume`, a run always starts in a new folder.

**What you may change.** You may change `url` (the server moved),
`data_dir`, `max_generation_workers` and `max_score_workers` between attempts;
`output_dir` and `exp_name` are not used by a resume. Changing `model` (or the
model the server serves), `default_system_prompt`, `benchmark_system_prompts`,
`answer`, `benchmarks`, `repeats`, `sampling` or `judge` is refused, because it
would mix two different runs.
Start a new run (without `--resume`) for those.

**Errors within one run.**
- A failed request (server error, dropped connection, server unreachable,
  also on `localhost`) is re-sent until it succeeds, waiting 3 s, then
  longer each time (up to 30 min between tries). A vLLM restart is waited
  out.
- A sample that errors is retried 3 times.
- If it still fails, its benchmark is resumed automatically, up to 3 times.
- After that, the benchmark is reported as `error` and the command exits 1.
  The other benchmarks are unaffected. Resume the run (`--resume`) to
  continue it.

## LLM judge

No benchmark uses an LLM judge yet. A run can still configure one, served by
any vLLM server:

```yaml
judge:
  url: https://judge-node:8000/v1        # same no-key / no-TLS access
  model: Qwen/Qwen3-32B                  # optional, as for the evaluated model
  prompt: |                              # {question} {response} {target} are replaced verbatim
    Question: {question}
    Response: {response}
    Reference: {target}
    Reply <verdict>correct</verdict> or <verdict>incorrect</verdict>.
  answer: ["<verdict>", "</verdict>"]    # the last verdict marker pair is used
  scores: {correct: 1, incorrect: 0}     # verdict -> score; anything else scores 0
  sampling: {temperature: 0}             # optional
```

Judges are always open-weight models, so judged scores are not comparable to
leaderboards that use proprietary judges. Judge features get added only when a
benchmark needs them. The decisions so far are in
[`src/custom_eval/benchmarks/NOTES.md`](src/custom_eval/benchmarks/NOTES.md#llm-judge).

## Adding a benchmark

Add a module under `src/custom_eval/benchmarks/` (see `math_tasks.py`), with:
- a `DataSpec` for each set of files it reads;
- a `build(ctx) -> Task` that returns the dataset and a scorer. It has no
  solver: the suite adds the system prompt, generation and repeats.
- an entry in the module's `BENCHMARKS` list.

Then register the module in `src/custom_eval/benchmarks/__init__.py`. The
scorer returns a dict `Score.value` of pass/fail keys and
`metadata["marker_found"]`. Record the benchmark's choices and checks in
`src/custom_eval/benchmarks/NOTES.md`.

Tests: `pytest tests/custom_eval`.

## Repository layout

| Path | What |
|---|---|
| `src/custom_eval/` | the suite: CLI (`cli.py`), config, model access, results |
| `src/custom_eval/benchmarks/` | the benchmarks, and `NOTES.md` (per-benchmark decisions) |
| `src/custom_eval/_vendor/` | shipped checker code and NLTK data (see its `README.md`) |
| `examples/custom_eval/` | example run configs (see [Quick start](#quick-start)) |
| `tests/custom_eval/` | the suite's tests |
| `src/inspect_ai/`, `docs/`, `design/` | Inspect AI (upstream) and its docs: <https://inspect.aisi.org.uk/> |
