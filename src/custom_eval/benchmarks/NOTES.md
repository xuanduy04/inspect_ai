# Benchmark notes

What every benchmark's implementation decided that the user did not specify,
where it deviates from the official evaluation, dataset quirks, and what was
checked. `README.md` (the user guide) says *what* is scored; this file says *why*, and what is
still open.

Rules for agents:

- Read a benchmark's entry before changing it.
- When you add or change a benchmark, add or update its entry in the same
  commit: data source, every unrequested choice, every deviation from upstream,
  how you verified it (and the date).
- Items marked **unreviewed** have not been reviewed by the user. The user
  reviews them one benchmark at a time; record the decision here when they do
  (and drop the marker).

## Suite-wide (every benchmark)

Definitions are in `README.md` ("How answers are scored"). Choices behind them,
**unreviewed** unless a user decision is noted:

- Answer extraction: the text between the *last* opening marker and the next
  closing marker; bracket-balanced when the markers are `{`/`}`-style (so
  `\boxed{\frac{1}{2}}` works); an empty closing marker takes the rest of the
  output; an unterminated last marker means no answer.
- Reasoning (`reasoning_content`, `<think>`) is never judged; only the final
  output is.
- Code extraction (all code benchmarks): text between the markers if present,
  else the whole output; within it, the last fenced block (LiveCodeBench's
  `extract_code` rule), else the text as-is. Measured 2026-11-25 on real
  chat outputs (see Code): a model that follows its solution with a usage
  example in another fenced block gets the example graded. That was 150 of
  328 HumanEval outputs of two Nemotron-4B fine-tunes and 667 of 4,056 MBPP+
  outputs of four chat models. EvalPlus's `sanitize` keeps the block that
  defines the function instead. User decision (2026-11-25): the rule stays.
  A graded block that is not the solution is a problem of the model's
  answer.
- Code execution: Inspect's `local` sandbox, `sys.executable`, LiveCodeBench's
  `reliability_guard` applied first (`src/custom_eval/_pyexec.py`). A program passes
  only if it prints a random token at the end, so code that calls `exit()`
  early never passes. For humaneval, mbpp and their plus versions the
  program runs in a fresh globals dict without `__name__`, as human-eval
  and EvalPlus run it (`exec(code, {})`). A model's own
  `if __name__ == "__main__":` block (its tests, `unittest.main()`, a demo)
  therefore does not run. This is the user's decision of 2026-11-25. Before
  it, programs ran as `__main__`, and 12 of 1,632 real HumanEval outputs
  failed or timed out only because of that. Re-checked after the change:
  - Those 12 pass (8 on humaneval_plus), and no other verdict changed.
  - The suite and EvalPlus now give the same verdict on 1,630 of the 1,632
    on the same code. The other 2 are the module-level doctests (see
    humaneval).
  - None of the 6,408 real MBPP/MBPP+ outputs has such a block.
  - Reference solutions, wrong solutions and early exits score as before.
- Repeats with a `seed`: repeat k sends `seed + k - 1`.
- End-to-end, checked 2026-11-25 (CLI on a mock server: aime24, humaneval and
  ifbench; 3 repeats; seed 5; a system prompt holding `{name}`, `{answer}`, a
  path and a newline):
  - All 1,482 requests carried exactly [system, user], with the system
    prompt verbatim and no Authorization header.
  - Each prompt got seeds 5..7. The sampling keys were sent as given
    (`top_k`, `chat_template_kwargs` in the body; `stop` as a list).
  - Every metric in `results.jsonl` (mean, clustered stderr, pass@3,
    per-repeat means, format and truncation rates) equals a recomputation
    from `output_raw.jsonl`.
  - Rerunning the same config sends no request and gives the same results.
    A changed system prompt is refused (exit 2); a changed URL resumes. After
    a resume, the `log` field is a `file://` URI instead of a path; nothing
    else changes. (That was before run folders, below: a rerun now starts a
    new run, and resuming takes `--resume`.)
- Run folders (2026-11-25, from the user's request that output dirs support
  launching many runs): each run writes to a new
  `<output_dir>/<exp_name>-<model>-<YYYYMMDD-HH_MM_SS>` folder, with
  `output_raw/<benchmark>.jsonl` instead of one `output_raw.jsonl` and
  `run.jsonl` instead of `run.json` (Inspect's own `logs/*.json` stay, as the
  user said). User decisions: resuming is `--resume <run folder>`; the time
  is local time (the user has no daylight saving, so local and UTC sort the
  same). Not specified, chosen: the model id is lowercased with every run of
  characters other than `a-z0-9` turned into one `-`; a name already taken
  in the same second waits for the next second; raw lines are repeat-major
  in dataset order, so line k is the same problem whatever `repeats` is.
  Checked on a mock server with the example config's values (aime24,
  humaneval, ifeval, math500):
  - 3 runs of one config launched at once got 3 folders; two took the next
    second. Sorted names follow creation order.
  - Every `output_raw/` file is in (repeat, dataset order), line 5 of
    `math500.jsonl` is the same problem in runs with 1 and 2 repeats, and
    every metric in `results.jsonl` equals a recomputation from those files.
  - A run killed with `kill -9` after 307 of 1,235 requests resumed with 949
    new requests and complete outputs. Resuming a finished run sends no
    request; `output_raw/` is unchanged and `results.jsonl` differs only in
    `log`. A changed system prompt, a folder holding no run, and a server
    serving another model are refused (exit 2) before any request.
  - `max_generation_workers: 100000` kept at most 1,000 requests in flight
    (2,470 queued), Inspect's HTTP pool size; with
    `INSPECT_HTTP_POOL_CONNECTIONS=5000` all 2,470 were in flight.
- HTTP pool (2026-11-25, user request): `max_generation_workers` also raises
  the client's connection pool to its own size (`pool_limits` in
  `model.py`; never lowered below Inspect's 1,000 or
  `INSPECT_HTTP_POOL_CONNECTIONS`). Mock server, 30 s per request, 2,470
  requests: 100000 workers put all 2,470 in flight; 256 workers, 256. Each
  request in flight holds a socket: under an open-file soft limit of 1,024,
  runs with 1,000 or 2,470 in flight failed with `OSError(24, 'Too many open
  files')` (one benchmark ended in `error`), one with 256 succeeded. So a
  node needs `ulimit -n` well above the requests in flight; the suite does
  not raise it.
- Concurrency (2026-11-24, from the user's request to separate generation and
  scoring pools): generation is capped by `max_generation_workers`, scoring
  by `max_score_workers` (math checks and IFEval/IFBench checks run in that
  many worker processes; code tests hold a scoring slot while their
  subprocess runs, and Inspect's `max_subprocesses` is set to the same
  value). Every sample may be active at once, so a sample being scored never
  holds a generation worker. Verified with a mock server (64 generation
  workers, fast generations): idle generation capacity while requests waited
  went from 34% to 6%; per-sample scores identical to before.
- Bug fixed 2026-11-25, reported by the user from the GPU server. Run from
  a terminal, every math and IFEval/IFBench sample failed with
  `ValueError: bad value(s) in fds_to_keep`; code benchmarks were fine.
  - Cause: Inspect's default display (Textual, used when stdout is a
    terminal) replaces `sys.stderr` with an object whose `fileno()` is -1.
    The first score-worker start launched multiprocessing's resource
    tracker, which passes `sys.stderr.fileno()` to `fork_exec`, and -1 is
    refused.
  - Fix: `configure_score_workers` starts the tracker, before the display
    starts.
  - Earlier checks missed it because they ran with `INSPECT_DISPLAY=none` and
    no terminal. Reproduced in a pseudo-terminal with the default display
    (aime24 and ifeval: `error` before, `success` after, with the same
    verdicts as without a terminal).
- Per-benchmark system prompts (2026-11-28, user request: run the general
  and the IF configs as one run, the IF benchmarks under their own prompt;
  the user chose this over running several configs at once). User
  decisions: `system_prompt` is renamed `default_system_prompt`;
  `benchmark_system_prompts` maps a benchmark's name, exactly as written in
  `benchmarks`, to its prompt, and any other key is refused; the four
  example configs are merged into `config.yaml` and `mistral_config.yaml`.
  Not specified, chosen: a `benchmark_system_prompts` value may be `null`
  (no system message), as `default_system_prompt` may; both keys must be
  unchanged to resume; the merged configs list the general benchmarks, then
  ifeval and ifbench. Each task now carries its own solver (`task_with` in
  `cli.py`) instead of one solver passed to `eval_set`. Checked 2026-11-28
  on a mock server with the real aime24, humaneval, ifeval and ifbench
  files, 2 repeats, seed 5:
  - Every generation in `output_raw/` carried its benchmark's prompt
    verbatim: aime24 the default (holding `{name}`, a path and a newline),
    humaneval none (`null`), ifeval and ifbench the IF prompt given once
    with a YAML anchor. Seeds 5 and 6.
  - Resuming the finished run sent no request. A changed
    `default_system_prompt` or `benchmark_system_prompts` was refused
    (exit 2), and so was a key not in `benchmarks` (before any request, no
    run folder).
  - A run folder made by the previous code is refused by the new code,
    because its `config.yaml` has `system_prompt`. With that key renamed in
    that `config.yaml`, a run killed with `kill -9` after 807 of 1,035
    requests resumed with the remaining 236: the old logs were reused.
  - The merged example configs give exactly the prompts and other values of
    the four files they replace.
- Command-line `-u`/`--url` and `-m`/`--model`, and the saved config
  (2026-11-28, user request). User decisions: the flags; a run folder's
  `config.yaml` holds the config the run used, with the overrides and the
  served model id (comments may go); `run.jsonl` is dropped, so resume
  checks the model through `config.yaml`. Resume behavior the user set: a
  moved server serving the same model resumes, a server serving another
  model is refused, and adding `model: <the same id>` resumes. Not
  specified, chosen: an override is applied before validation, exactly as if
  written in the file, so the file may leave `url` out when `-u` is given;
  `config.yaml` is written in `RunConfig` key order without comments (YAML
  anchors written out in full, multi-line strings as block literals) and is
  not rewritten by a resume; the refusal names both models
  (`changed: model (a -> b)`). Run folders made before this change whose
  config had no `model` are refused (`changed: model`) until `model: <id>`
  (the id in their `run.jsonl`) is added to their `config.yaml`. Checked
  2026-11-28 on mock servers with the real aime24 and ifeval files:
  - `-u` over a dead `url` in the file ran on the given server. The saved
    `config.yaml` held that URL and `model: mock/model`, no comments, and
    read back to the same config. The round trip was also tested on prompts
    with CRLF, tabs, trailing spaces, blank lines and unicode.
  - A run killed with `kill -9` after 393 of 571 requests resumed on another
    server serving the same model with the remaining 179; `config.yaml` was
    unchanged.
  - Resuming the finished run with `-m mock/model`, or with
    `model: mock/model` in the file, sent no request. A server serving only
    `other/model`, and `-m other/model` on a server serving both, were
    refused (`changed: model (mock/model -> other/model)`).
  - A file without `url` was refused without `-u` and ran with it; `-m`
    picked one model on a server serving two.
  - A run folder of the previous code (its `config.yaml` without `model`) was
    refused, and resumed with no request once `model: mock/model` was added.
- The URL as a positional argument (2026-11-28, user request):
  `custom-eval <config> [URL] [options]` does the same as `-u`. User
  decisions: positional arguments come before the options; giving both a
  URL argument and `-u` is refused at once. Not specified, chosen: the
  refusal is an argparse usage error (exit 2), raised before the config is
  read. Checked 2026-11-28: the argument and `-u` each reached the URL given;
  both together exited 2 with `give the URL either as an argument or with
  -u/--url, not both` and ran nothing.
- `max_generation_workers` capped (2026-11-28, user request): a run uses
  min(`max_generation_workers`, problems x repeats) as the model's
  `max_connections`, and so for the HTTP pool size (`pool_limits`); the
  judge check gets the same value. Not specified, chosen: the problems are
  summed over the run's benchmarks (every sample may be in flight at once,
  and no more can be); a resume counts every sample, not only the remaining
  ones; the saved `config.yaml` keeps the configured value. The datasets are
  now loaded before the model is connected (the URL and model are still
  checked first). Checked 2026-11-28 on a mock server (aime24 and ifeval,
  571 problems) through the logs' `max_connections`: 100000 with 1 repeat
  gave 571, with 2 repeats 1142, and 300 gave 300.

## Math

All math benchmarks: user message = the problem verbatim; scored by Inspect's
own math checker (`inspect_ai.scorer._math`, sympy-based), not math-verify or
any model vendor's grader, so numbers can differ from published ones on
answers with unusual formats (**unreviewed**). Parses run in score worker
processes; each worker first parses and checks `\frac{1}{2}` to pay the lazy
sympy / latex2sympy2 imports, outside any timeout. One parse may then take
10 s; a timeout kills that worker and scores the answer wrong (a target
timeout errors the sample). An answer is checked in the worker call that also
parses the targets (every target in the suite parses in at most 0.12 s), so
parsed targets never cross processes: unpickling rebuilds a SymPy expression
evaluated (`Mul(-1, 2)` becomes `-2`), and the checker compares tuples, sets
and intervals structurally. Sending parsed targets to the workers (the
2026-11-24 pool change, fixed the same day) scored 12 MATH-500 gold answers
and HMMT Feb 2025 #10 wrong against themselves (`(-2,1)`, `3 \pm 2 \sqrt{2}`,
`x \in [-2,7]`, two roots).

Verified 2026-11-24, all 8 math benchmarks (783 problems): every gold answer
scores 1 strict and lenient in the markers (also as `$gold$` or
`\boxed{gold}` inside them), 0 strict and 1 lenient as `\boxed{gold}` without
markers, and a wrong answer scores 0; verdicts identical to the first
commit's in-thread checker. Common rewrites of the gold answers are accepted
(5193 of 5194: `\dfrac`, no `\left`/`\right`, no `\text{}`, no `^\circ`, no
spaces, `x = n`, `n.0`, prose or `\[...\]` around it). The miss is MATH-500
`test/precalculus/452.json`, gold `\text{(E)}`: a bare `E` or `(E)` parses as
Euler's number (`\text{E}` and `\textbf{(E)}` match). The CLI run on a mock
server also showed: reasoning holding another marked answer is ignored;
format and truncation rates, per-repeat means and pass@k come out as designed.

Re-checked 2026-11-25, from fresh downloads, every run through the CLI on a
mock server:
- Gold answers: same results. All 783 problems score as designed in 9 forms,
  including the gold only in `reasoning_content` or in a `<think>` block (0)
  and in an earlier marker pair (0). 3,067 of 3,076 rewrites are accepted;
  the misses are `\text{(E)}` and artificial `x = ` prefixes.
- Official verdicts on real outputs: MathArena's published outputs
  (`MathArena/<set>_outputs`, 22-67 models) for aime25, aime26, hmmt_feb25,
  hmmt_feb26 and hmmt_nov25, served as the completion, markers `\boxed{` `}`.
  On 25,518 outputs the strict verdict equals MathArena's on 25,378 (99.45%).
  MathArena accepts 138 that the suite rejects:
  - 56: no `\boxed{}` in the final output (strict needs the markers).
  - 17: hmmt_feb25 #10's two roots boxed separately (only the last box
    counts) or rewritten (see comma lists below).
  - 16: malformed `\begin{matrix}` boxes (Grok 3 Mini).
  - 13: an equation chain or formula in the box (`P-Q=7140-1200=5940`; a
    single `x = v` is accepted).
  - 11: 165 for aime26 #10, which MathArena accepts beside the official 156.
  - 10: Cardano forms for hmmt_feb26 #7 that exceed the 10 s budget.
  - 10: `79\%` for 79 (hmmt_nov25 #11 asks for x in "x% mercury").
  - 1 each: `√23−2√3`, read as `\sqrt23-2\sqrt3` (3√2); the expanded integer
    for `2^{25} \cdot 26!` (a target with a large factorial matches only
    structurally); a hedged text answer; an unparseable box; a wrong value
    MathArena accepted (0.568489 vs 0.568261).

  The suite accepts 2 that MathArena rejects: decimals to 10+ digits
  (`6.316624790\ldots` for `3+\sqrt{11}`; `6.3166` is rejected).
- OpenR1-Distill-7B's outputs from open-r1's lighteval details: 4 per
  MATH-500 problem, 64 per AIME problem, reasoning inline in `<think>`.
  - AIME24/AIME25, 3,264 outputs: every verdict equals "the last `\boxed{}`
    after `</think>` is the gold integer". lighteval's stored per-problem
    counts match no extraction rule (aime25 #18: 11/64 vs 57/64), so they
    were not used.
  - MATH-500, 1,999 outputs: on the same extracted answer, the suite and
    math-verify 0.9.0 agree on 1,963. The suite rejects 33 correct answers:
    base-n answers without the subscript (`204` for `204_5`; 19, from 5 of
    MATH-500's 6 such golds), `[-2, 7]` for `x \in [-2,7]` (4), `10\%` for a
    "by what percent" question (4), `E` for `\text{(E)}` (4), `(3/2, -13)`
    (1), and `-2, 1` for `1,-2` (1). math-verify rejects
    `12^{\mathrm{th}}` for 12 (3), which the suite accepts. The 2,000th
    output errored (the `check_answer` bug below).
    With the math500 extra targets (see math500), on all 2,000 outputs, the
    two agree on 1,988: 24 of the 33 now pass; the suite still rejects
    `[-2, 7]` (4), `10\%` (4) and `(3/2, -13)` (1).
- Comma lists: elements are compared structurally, not by value
  (`\frac{\sqrt{17}-1}{2}` matches `\frac{-1+\sqrt{17}}{2}` alone, not in a
  list; `(3/2, -13)` fails where `(\frac{3}{2}, -13)` passes). Lists of plain
  numbers are ordered (`5,3` fails for `3,5`; MATH-500's "find all" golds
  `1,-2` and `-2,1` are affected); lists with LaTeX are not
  (`-\sqrt{2}, \sqrt{2}` passes for `\sqrt{2}, -\sqrt{2}`).
- Bug fixed 2026-11-25 (`check_answer`), at the user's request. The checker
  raised `_MathLimitError` instead of returning a verdict in two places
  outside the `try` that turns its parse and limit errors into a status:
  - the size check on the whole output (over 1,000,000 characters);
  - the fallback that retries an unmatched prose answer's other candidates
    (one over 4,096 characters, or nested too deeply).

  `_check` caught only `TimeoutError`, so the sample errored. Its retries
  regenerated the answer (the same one, with a seed), and then the benchmark
  ended in `error`. That happened on 22 of the 31,374 real outputs above: long
  outputs without a final-answer marker, checked whole on the lenient path
  with the reasoning left in the content. Every run holding one reported its
  benchmark as `error`. `check_answer` now turns these errors into the
  checker's own statuses (`answer_limit`, `answer_parse_error`), so the answer
  scores wrong. Inspect 0.3.291 and 0.3.292 have the same gap in their own
  `math()` scorer.

  Re-running all 8 real-output runs: every benchmark succeeds, and the 22
  outputs score 0 (`answer_limit`). Every other score is unchanged. Two
  hmmt_feb26 #7 answers went from `answer_timeout` to `incorrect` (0 both
  times) because their check finished within the 10 s budget this time.

### math500
`HuggingFaceH4/MATH-500`, `test.jsonl`; id = `unique_id`. Gold answers
checked 2026-11-24 (see Math). The file is byte-identical to PRM800K's
`prm800k/math_splits/test.jsonl` (sha256 = its Git LFS oid; 2026-11-25).

Extra accepted answers, added 2026-11-25 at the user's request (a deviation
from the official evaluation): `MATH500_EXTRA_TARGETS` (`math_tasks.py`)
gives 9 problems, whose gold's notation makes the checker reject correct
answers, more targets; a match with any target counts.
- Base n. The checker reads an unbraced subscript as Python's digit
  separator (`52_8` is 528) and drops a braced one (`52_{8}` is 52). Against
  the 5 unbraced golds (`52_8`, `40_9`, `2516_8`, `204_5`, `4343_6`) the bare
  digits and the braced form failed: the extra is the bare digits. The
  braced `4210_{5}` reads as 4210, so `4210_5` failed: the extra is
  `4210_5`. Side effects: a wrong base in braces (`52_{10}`) passes, as it
  already did for `4210_{5}`; `42105` passes for `4210_{5}`, as `528`
  already did for `52_8`.
- `1,-2` and `-2,1` (intermediate_algebra/1388, /1102, "enter all such
  integers"): a list of plain numbers is compared in order; the extra is the
  other order.
- `\text{(E)}` (precalculus/452, "enter the letter"): a bare `E` failed (it
  is Euler's number); the extra is `E`, which also accepts `e`, `\mathrm{e}`
  and `\exp(1)`, not `2.718`.

Left out, because any extra target would also accept a plausible wrong
answer:
- algebra/101, `x \in [-2,7]` ("interval notation"; read as an interval): a
  plain `[-2,7]`, `(-2,7)` and `-2,7` all read as the pair (-2, 7), so a
  `[-2,7]` target also accepts the open interval.
- prealgebra/1233, gold `10` ("by what percent"): `10\%` reads as 0.1, so a
  `10\%` target also accepts `0.1`.

**Unreviewed**: the table the user approved also had algebra/101 (extra
`[-2,7]`) and, for `4210_{5}`, the extra `4210`, which changes nothing; both
were changed as above when the checks found this.

Each gold is stored with its extras: the build fails if a gold changes or an
id is missing. Problems outside the table keep a string target; the 9 have a
list in the outputs. The LLM judge (not used by math500) would put a list
target into its prompt joined into one string (`Target.text`).

Verified 2026-11-25: every gold and extra scores correct, and wrong answers
(`42` for `52_8`, `4201_5` for `4210_{5}`, `1,2`, `D`) stay wrong (tests).
Through the CLI on a mock server: the gold-answer run (7,500 outputs, 15
forms per problem) has 0 errors and the verdicts of the run with the
approved table: every form as designed, and of the rewrites only the 8
artificial `x = ` prefixes fail (the `(E)` rewrite now passes).
OpenR1-Distill-7B's 2,000 outputs (see Math), against the run without the
table: 0 errors, and exactly 24 verdicts change, all 0 to 1, all on table
problems and all correct per math-verify.

### aime24, aime25
`HuggingFaceH4/aime_2024` and `yentinglin/aime_2025` (`data/` only; `part1/` and
`part2/` are its halves). Gold answers checked 2026-11-24 (see Math).
Checked 2026-11-25:
- aime24's answers equal MathArena's `aime_2024_I`/`II` by value. 7 golds
  have a leading zero (e.g. `025`); `25` scores correct.
- aime25's answers equal MathArena's `aime_2025` (id k is their k+1). Its
  statements omit the figures (TikZ/Asymptote) that MathArena's include, and
  have small typos (`lie that order`, `Let $k$ be real numbers`).

### aime26, hmmt_feb25, hmmt_feb26, hmmt_nov25
MathArena datasets (`MathArena/aime_2026`, `hmmt_feb_2025`, `hmmt_feb_2026`,
`hmmt_nov_2025`); id = `problem_idx`. HMMT answers are LaTeX expressions (e.g.
`2^{25} \cdot 26!`, `74^\circ`, two roots separated by a comma);
`hmmt_feb26` has 33 problems, MathArena's selection of final-answer problems.
License CC BY-NC-SA 4.0.
Verified 2026-11-23: every gold answer, wrapped in the markers, scores 1 strict
and lenient; equivalent forms were also accepted (`74` for `74^\circ`, roots in
the other order, `\frac{17}{2}` for `8.5`, `33,124,147`). Re-checked
2026-11-24 after the worker fix (see Math).
Re-checked 2026-11-25 against MathArena's graded outputs (see Math).
- Some of MathArena's hmmt_nov25 #20 output rows were graded against a
  typo'd gold (`\frac{sqrt{7} + 1}{2}`); the dataset's gold is right.
- The official answer to aime26 #10 is 156 (AoPS); MathArena also accepted
  165.

### beyondaime
`ByteDance-Seed/BeyondAIME`, 100 problems, integer answers. The dataset has no
id column: id = row position in `data/test.parquet` (the rows are sorted by
answer) (**unreviewed**). Verified 2026-11-23: all gold answers score 1.
Re-checked 2026-11-25: all 100 answers are integers (3 to 33,124,147). No
graded model outputs are published for it; its integer path is the one
verified on the AIME outputs (see Math).

## Code

Real outputs used on 2026-11-25, all served through the CLI on a mock server
without markers:
- 328 HumanEval chat outputs of two Nemotron-4B fine-tunes, with their
  published EvalPlus results (`abhilash1910/nemosmith-evalplus-samples`).
- 1,304 HumanEval completions (`wentingzhao/humanevalplus_predictions_16`,
  the first 8 per task).
- 4,056 MBPP+ chat outputs of four models
  (`lmarena-ai/PPE-MBPP-Plus-Best-of-K`, the first 8 of 32 per row).

They were graded with EvalPlus 0.3.1 (PyPI) on the official releases. Chat
outputs were cleaned with `evalplus.sanitize`, as EvalPlus does for chat
models, and a plus pass needs both the base and the extra inputs to pass. A
fresh EvalPlus run reproduces all 328 published verdicts.

### humaneval
`openai/openai_humaneval`, 164. Prompt `Complete the following Python
function.` + the HumanEval prompt in a python fence: our wording, HumanEval
being a completion benchmark with no official chat prompt (**unreviewed**).
A full function definition is run after the prompt (helpers and imports
available); anything else is appended to the prompt as the body. The official
`check()` runs; 10 s for the whole program (**unreviewed**). Verified
2026-11-24 (CLI run, mock server): the 164 canonical solutions pass as a full
function (fenced, in the markers) and as a bare body (fenced, no markers); a
wrong function fails.
Re-checked 2026-11-25 (see Code for the sources):
- The HF file equals OpenAI's `HumanEval.jsonl.gz` (164/164 rows).
- The canonical solutions pass in both formats. Every task fails a wrong
  function, a function returning any of 9 constants (`None`, `[]`, `0`, ...)
  and `sys.exit(0)` before the tests.
- 328 chat outputs of two Nemotron-4B fine-tunes, against their published
  EvalPlus base verdicts: same verdict on 281. The suite fails 47 that pass
  officially, never the reverse. 45 of them are the last fenced block being a
  usage example (or another version of the function); 2 are the model's
  `unittest.main()` running. pass@1 is 22.0% and 19.5% here, 34.8% and 35.4%
  officially. On the same extracted code, EvalPlus and the suite agree on 326
  of 328.
- 1,304 completions (8 per task): same verdict as EvalPlus on the raw
  `prompt + completion` on 1,254. Of the other 50:
  - 40 pass here because the suite grades a fenced function the completion
    rambles into (a SyntaxError as raw text).
  - 2 pass here because a module-level `doctest.testmod()` breaks inside
    EvalPlus's sandbox.
  - 8 fail here because the model's `__main__` block runs.

  On the same code, the suite and EvalPlus agree on 1,292.

### mbpp
`google-research-datasets/mbpp`, `full/test` (500), not `sanitized` (257)
(**unreviewed**). Prompt `{text}\nYour code should pass these tests:\n\n{3
asserts}`: the common zero-shot MBPP wording, without the "You are an expert
Python programmer" sentence and without few-shot examples (**unreviewed**).
Program = code, `test_setup_code`, the 3 asserts; 10 s. Verified 2026-11-24
(CLI run, mock server): the 500 reference solutions pass, in the markers
without a fence and fenced without markers.
Re-checked 2026-11-25:
- `full/test` equals google-research's `mbpp.jsonl`, tasks 11-510 (only task
  367 has setup code).
- The 500 reference solutions pass in both formats. Every task fails a
  wrong program, a function returning any of 9 constants, and
  `sys.exit(0)` before the tests.
- 2,352 chat outputs of four models on 175 of the tasks (from
  `lmarena-ai/PPE-MBPP-Plus-Best-of-K`), against the standard MBPP execution
  (EvalPlus-sanitized code, setup code and the 3 asserts, exec'd as in
  human-eval, 3 s): same verdict on 1,973. The suite fails 379 that pass
  there, never the reverse, all because of the last-block extraction: on the
  suite's extracted code the same execution agrees on 2,352/2,352.

### humaneval_plus
`evalplus/humanevalplus` (HF copy, last updated 2024-05), 164. All
**unreviewed**:
- Prompt identical to humaneval. EvalPlus's own chat instruction is not used
  because it says "in a markdown code block", a format instruction, which the
  suite never adds.
- Program = humaneval's, with EvalPlus's `test` (base + plus inputs). One 60 s
  limit for the whole program. EvalPlus uses per-input limits, max(4 s, 4x
  the reference time) on GitHub `master` and max(1 s, 4x) in its PyPI
  release 0.3.1, and caps a task at 60 s (+1). The slowest reference program
  (HumanEval/139) takes about 8 s here.
- HumanEval/32 (`find_zero`): the shipped test line
  `assert _poly(*candidate(*inp), inp) <= 0.0001` raises for every solution.
  It is replaced by EvalPlus's own oracle, `abs(_poly(*inp, out)) <= 1e-4`
  (`HUMANEVAL_PLUS_TEST_FIXES` in `code_tasks.py`). Even then, the reference
  solution (Newton's method) fails 7 inputs, both in the HF copy and in the
  official v0.1.10 release, so a Newton-style solution cannot pass this task
  upstream either.
- The HF copy has the prompts, reference solutions and distinct inputs of the
  official release (EvalPlus `HumanEvalPlus` v0.1.10, GitHub). The release
  lists some inputs more than once (98 tasks, 1-11 extra copies); the HF copy
  lists each once. Corrected 2026-11-25: the earlier note read this as 1-2
  fewer inputs.
- The HF tests compare outputs with their own `assertion`, which lacks
  EvalPlus's type and length checks. An expected float or empty list is
  compared with `np.allclose`, so a function returning `[]` or `()` passes
  HumanEval/2, /4 and /45 (checked 2026-11-25).
Verified 2026-11-23: reference solutions pass 163/164 (all but HumanEval/32).
Same on 2026-11-24 through the CLI, in both humaneval formats.
Re-checked 2026-11-25 (see Code for the sources):
- Reference solutions: same, 163/164.
- 328 chat outputs against their published EvalPlus verdicts (base and plus
  inputs both passing): same verdict on 283. The other 45 fail here because
  of the last-block extraction. On the same extracted code, EvalPlus and the
  suite agree on 328/328.
- 1,304 completions: same verdict on 1,260; on the same code, 1,294 (8
  `__main__` blocks, 2 module-level doctests, as in humaneval).

### mbpp_plus
`evalplus/mbppplus`, 378. All **unreviewed**:
- Prompt as mbpp, with EvalPlus's `prompt` field as the description and its
  `test_list` (the original 3 asserts) as the examples.
- Program = code, `test_imports`, EvalPlus's `test` (base + plus inputs); 60 s,
  as humaneval_plus. The slowest reference program (Mbpp/599) takes about 13 s
  here.
- EvalPlus applies special oracles to some tasks. The HF tests already build
  in the set-equality ones; those for Mbpp/581 (`surface_Area`, tolerance
  against a reference function) and Mbpp/558 (`digit_distance_nums`, compare
  with a corrected reference) are not applied: outputs are compared exactly
  with the stored expected values. Mbpp/164 and Mbpp/295 are not in the 378.
- The HF tests are weaker than EvalPlus's checks (found 2026-11-25):
  - For EvalPlus's "output is not None" tasks (Mbpp/737 `check_str`, 787
    `text_match_three`, 794 `text_starta_endb`), their `assertion` computes
    the verdict but never asserts it. Any solution that does not raise
    passes those 3.
  - As in humaneval_plus, an expected float or empty list is compared with
    `np.allclose` without EvalPlus's type and length checks. A function
    returning `[]` or `()` passes 24 more tasks.
- The HF copy has the prompts, reference solutions and distinct inputs of the
  official release (EvalPlus `MbppPlus` v0.2.0). The release lists some
  inputs more than once (72 tasks). Mbpp/124 and Mbpp/252 hold complex inputs
  that the release stores as strings or ints for EvalPlus to convert; the HF
  copy has them converted.
Verified 2026-11-23: reference solutions pass 378/378 (without markers, so the
untagged path was exercised too). Same on 2026-11-24 through the CLI, in both
mbpp formats.
Re-checked 2026-11-25:
- Reference solutions: same, 378/378.
- 4,056 chat outputs of four models on 298 tasks
  (`lmarena-ai/PPE-MBPP-Plus-Best-of-K`, 8 per row), against EvalPlus 0.3.1
  on the sanitized code: same verdict on 3,355. 671 fail here because of the
  last-block extraction. 30 pass here and fail in EvalPlus, all on the same
  code:
  - 16 Mbpp/287: exact integers for n = 10^6. The reference divides with
    `/`, and EvalPlus then requires a float.
  - 8 Mbpp/787 and 794: no assertion (above).
  - 5 Mbpp/418: `None` accepted for an expected `[]`.
  - 1 Mbpp/123: EvalPlus 0.3.1's 1 s per-input limit; it passes with
    `master`'s 4 s.
- PPE's own scores (its harness on the same HF tests) agree with the suite on
  4,012.

### LiveCodeBench (lcb_*)
`lighteval/code_generation_lite` (parquet copy of the official data), read from
the per-release folders `v1/`..`v6/` (folder `vN` = problems new in release N;
release N = `v1`..`vN`). Prompt = LiveCodeBench's generic prompt minus its
format instructions: the "enclose your code within delimiters" sentences, the
`# YOUR CODE HERE` fence that shows that format (stdin problems), and the
closing "### Answer: (use the provided format with backticks)" line (checked
against LiveCodeBench `main` 2026-11-24); graded by the vendored official
runner, 6 s per test (official default), public + private tests.

- `lcb_v5`, `lcb_v6` (880 / 1055): the whole release v5 / v6, read from the
  folders `v1`..`v5` / `v1`..`v6`, with no date cut. User decision
  (2026-11-26, confirmed with their superiors): only the full release of each
  version matters, so `lcb_v5` / `lcb_v6` name the full releases. Before that
  they were the problems new in each release (folder `v5` / `v6` alone, 167 /
  175), the unreviewed pick for the handoff's "pick something sensible".
  `lcb_v5` and `lcb_v6` are the only LiveCodeBench benchmarks.
- Removed 2026-11-26 (user decision, after the change above): the windowed
  variants and `lcb_release_v6`, by then the same set as `lcb_v6`. They are
  described here because the verification records below refer to them. The
  windowed variants were added 2026-11-23 for labels that recent model
  reports use. A label "LCB vN (YYMM-YYMM)" was read as release N cut to
  contests from the first day of the first month to the last day of the last
  month, both inclusive:

  | Benchmark | Problems | Label it reproduced |
  |---|---|---|
  | `lcb_v5_2408_2501` | 279 | DeepSeek-R1 "2024-08 – 2025-01". Mapping it to release v5 is an inference (v6 did not exist yet). Over release v6 the same window has 323. |
  | `lcb_v5_2410_2502` | 166 | Qwen3 report "v5 (2024.10-2025.02)" |
  | `lcb_v6_2408_2505` | 454 | DeepSeek-V3.1, Kimi K2, Nemotron-Cascade-2, OpenReasoning-Nemotron "v6 (2408-2505)"; DeepSeek-V3.2 "2024.08-2025.04" is the same set |
  | `lcb_v6_2502_2505` | 131 | Qwen3-2507 "v6 (25.02-25.05)" |
  | `lcb_release_v6` | 1055 | the whole release (e.g. Kaggle's release_v6 leaderboard); the same set as `lcb_v6` since 2026-11-26 |

  Folder date ranges (`contest_date`): v1 2023-05-07..2024-03-02, v2
  2024-03-09..2024-05-25, v3 2024-06-01..2024-08-10, v4 2023-08-26..2024-10-05,
  v5 2024-09-22..2025-01-04, v6 2025-01-04..2025-04-06. Folders overlap in
  time, so a window was filtered by `contest_date`, not by folder; each
  windowed variant read only the folders that hold problems in its window.
  Labels left out then (window or release unclear): Magistral "LCB v5", Olmo
  3 "LCB v3 (2024)", Llama-3.3-Nemotron-Super-49B-v1.5 "24.10-25.02" (release
  unknown), GLM-4.7-Flash "LCB v6" (no window given).
- No gold solutions exist in the data. Verified 2026-11-23: counts and date
  ranges of every windowed variant except `lcb_release_v6` built from real
  data; on `abc391_a` a correct solution passes 42/42 and a wrong one fails
  with wrong-answer.
- Verified 2026-11-24, when `lcb_v5` / `lcb_v6` were folder `v5` / `v6`
  alone (167 / 175 problems; the other counts as above): every variant builds
  from the real files with those counts (`lcb_release_v6` too), and the
  vendored `testing_util.py` equals LiveCodeBench `main`. Officially graded
  generations from the leaderboard (`LiveCodeBench/submissions`, one per
  problem) were re-graded through the CLI on a mock server (4 score workers
  on 4 CPUs), about half of them inside the markers:
  - o4-mini (low) on `lcb_release_v6` (1055): same verdict on 1050 (817 pass,
    233 fail; 443/444 LeetCode function problems, 607/611 stdin problems).
    The other 5 pass here and were "Time Limit Exceeded" officially.
  - Gemini 2.5 Pro (06-05) on the six other variants (the old `lcb_v5`,
    `lcb_v6` and the four windows: 454 problems, each in two or more
    variants, 1372 samples): same verdict on 451, and every problem got the
    same verdict in each variant. 2 that were "Time Limit Exceeded"
    officially pass here; `abc390_g` timed out here under full load and
    passes when graded alone (43/43, 45 s).
  Solutions near the 6 s per-test limit flip with machine speed and load.
- Re-checked 2026-11-25, from fresh downloads and the GitHub submissions repo,
  with 4 score workers on 4 CPUs, `lcb_v5` / `lcb_v6` still being folder `v5` /
  `v6` alone:
  - Every variant builds with the counts of the time (167 / 175 for
    `lcb_v5` / `lcb_v6`, the others as above). The vendored
    `testing_util.py` still equals LiveCodeBench `main`, and the copied
    parts of `check.py` (test decoding, `check_correctness`, pass rule)
    match upstream apart from the listed changes.
  - The prompt and `extract_code` still match the description above.
    Upstream's `extract_code` returns nothing when there is no fenced block,
    where the suite takes the text as-is.
  - The same leaderboard generations were re-graded through the CLI: raw for
    about half the problems, the official extracted code in the markers for
    the rest.
  - o4-mini (low), `lcb_release_v6`: same verdict on 1,048 of 1,055. The
    other 7 pass here and were "Time Limit Exceeded" officially.
  - Gemini 2.5 Pro (06-05), the same six variants (the old `lcb_v5`, `lcb_v6`
    and the four windows; 1,372 samples, 454 problems): same verdict on
    1,360. Every difference is near the time limit. 3686, 3690 and
    `abc367_e` were "Time Limit Exceeded" officially and pass here.
    `abc373_f` (officially TLE) and `abc390_g` (47.9 s officially) passed in
    one variant and timed out in another.
  - On 1,253 raw outputs (534 o4-mini, 719 Gemini samples), the suite's code
    equals LiveCodeBench's extracted code except for 9 o4-mini refusals with
    no code block (both fail).
- Verified 2026-11-26, after `lcb_v5` / `lcb_v6` became the whole releases,
  from fresh downloads, with 4 score workers on 4 CPUs:
  - LiveCodeBench's loader (`code_generation_lite.py` of
    `livecodebench/code_generation_lite`) defines `release_v5` as
    `test.jsonl`..`test5.jsonl` and `release_v6` as `test.jsonl`..`test6.jsonl`,
    where `vN` is file N alone. `load_code_generation_dataset` cuts no dates
    unless asked. So release N is folders `v1`..`vN`, with no date cut.
  - Folders `v1`..`v5` and the dataset's own `release_v5/` hold the same 880
    rows, all 11 columns equal; `v1`..`v6` and `release_v6/` the same 1055.
    Every `lcb_v5` / `lcb_v6` sample's prompt, test count, contest date and
    test file match those rows. `lcb_v5` is the first 880 problems of
    `lcb_v6`, and `lcb_v6` equals `lcb_release_v6` (ids, order, prompts).
    Contest dates: `lcb_v5` 2023-05-07..2025-01-04, `lcb_v6`
    2023-05-07..2025-04-06. Each task takes about 2 minutes to build (peak
    3 GB of memory).
  - The o4-mini (low) leaderboard generations were re-graded through the CLI
    on a mock server, `lcb_v5` and `lcb_v6` in one run: raw for about half
    the problems, the official extracted code in the markers for the rest.
    Same verdict as official on 876 of 880 (`lcb_v5`) and 1,049 of 1,055
    (`lcb_v6`). Every difference was "Time Limit Exceeded" officially and
    passes here: `abc324_d`, `abc356_e`, `abc369_e` and `abc375_f` in both,
    3714 in `lcb_v6`, and `abc367_e`, which timed out in `lcb_v5` (as
    officially) and passed in `lcb_v6`. It is the only problem whose verdict
    differs between the two.
  - The table showed 880 and 1055 problems. All 1,935 requests carried
    [system, user], with the system prompt verbatim.
- Re-checked 2026-11-26 after removing the variants, on the same files and
  the same o4-mini (low) generations, with 4 score workers on 4 CPUs:
  - `lcb_v5` / `lcb_v6` build the same 880 / 1055 samples, each matching the
    official release rows (prompt, test count, contest date, test file). They
    are the only LiveCodeBench benchmarks. A config that names a removed one
    stops before anything runs (`config.benchmarks: unknown [...]`, exit 2,
    no run folder).
  - Re-graded through the CLI, `lcb_v5` and `lcb_v6` in one run: same verdict
    as official on 875 of 880 and 1,049 of 1,055. The differences are the
    "Time Limit Exceeded" problems listed above. `abc367_e` passed in both
    tasks this time, so no problem's verdict differs between them.

## Instruction following

### ifeval, ifbench
`google/IFEval` (`ifeval_input_data.jsonl`) and `allenai/IFBench_test`
(`data/`). User message = the prompt verbatim. Official checkers vendored,
with the changes listed in `src/custom_eval/_vendor/README.md` (seeded `langdetect`,
reseeded checker RNG, NLTK data shipped). Checks run in score worker
processes; each process prepares the NLTK data and the checker RNG itself, so
verdicts do not depend on which worker checks. `None` kwargs are dropped, as the
official evaluation does. The whole final output is checked (reasoning
excluded, as everywhere); the answer markers play no part.
- User decision (2026-11-28): IF benchmarks always check the whole output.
  The two extractions they had before (`strict`: the text between the
  markers, no markers failing every instruction; `lenient`: that text if
  present, else the whole output) were never the user's choice; they are
  removed, with their `strict_*`/`lenient_*` keys. `marker_found` is still
  recorded, for the format rate only (**unreviewed**).
- Headline: IFBench prompt-level loose (from the handoff); IFEval prompt-level
  strict (**unreviewed**).
- Dataset quirk: in `google/IFEval`, key 2785 asks for "at least one
  placeholder" while its checker requires 3 (`num_placeholders: 3`). Google
  fixed the prompt on GitHub on 2024-06-11 ("at least 3 placeholders"); the HF
  file still has the old text, and the suite uses it as is.
  `allenai/IFBench_test` equals upstream's `data/IFBench_test.jsonl` except
  trailing spaces in the prompts of keys 13 and 19.
- Verified 2026-11-24: the vendored checker code equals upstream
  (google-research `master`, allenai/IFBench `main` 4889186) apart from the
  changes in `src/custom_eval/_vendor/README.md`, and the shipped NLTK files are
  byte-identical to nltk_data's.
  - IFEval on Google's GPT-4 reference responses
    (`input_response_data_gpt4_20231107_145030.jsonl`), CLI run on a mock
    server: prompt/instruction strict 77.08/83.69, loose 79.67/85.61. Google's
    own `evaluation_lib` (`master`), run twice on the same responses and input
    file: strict 76.89/83.57 both times, loose 79.48/85.49 and 79.30/85.37 (the
    second equals the paper). Per instruction, strict and loose, the suite
    matches it on 539 of 541 prompts. The other two, keys 1122 and 1129
    (`keywords:letter_frequency` with the non-letters `#` and `!`), are where
    upstream draws a random letter, so its own verdict changes between runs;
    the suite's draw is fixed (`src/custom_eval/_vendor/README.md`).
  - IFBench on upstream's `data/sample_output.jsonl`: per-instruction verdicts
    equal upstream's committed `eval/eval_results_*.jsonl` (2025-10-17) on 326
    of 331 instructions (290 prompts present in both). The other 5 are on
    checkers upstream changed after those results: `format:list` (merged
    2026-11-13, markers counted at line starts only), `ratio:overlap` (merged
    2026-08-30, word instead of character trigrams), `ratio:sentence_type`
    (2025-12-16, `nltk.sent_tokenize` sentence splitter). The old logic reproduces the
    committed verdict on the 4 `format:list` / `ratio:overlap` cases.
  - In the CLI run, every logged per-instruction verdict equals a direct
    re-check of the logged output; `<think>` blocks and `reasoning` holding
    markers are not judged.
- Re-checked 2026-11-25 from fresh downloads. The vendored checker code
  still equals upstream apart from the listed changes, and the shipped NLTK
  files are byte-identical to nltk_data's packages. Both upstream
  `evaluation_lib`s were run twice on the suite's input rows. The suite got the
  same responses through the CLI twice: in the markers (strict extraction)
  and bare (lenient extraction).
  - IFEval, Google's GPT-4 reference responses: every per-instruction verdict,
    strict and loose, in both extractions, equals Google's on 540 of 541
    prompts in both runs. The exception is key 1122, where upstream's own
    verdicts changed between its two runs.
  - IFBench, all 300 responses of upstream's `data/sample_output.jsonl`:
    identical to the current upstream `evaluation_lib` on all 344
    instructions, strict and loose, in both extractions. Upstream's two runs
    were identical. This covers the 10 prompts the 2026-11-24 comparison
    missed and uses the checkers as vendored.
  - Bare responses fail every instruction under strict extraction, as
    designed then. That extraction was removed on 2026-11-28; the bare runs
    above are the whole-output check every output gets now.
- Checked 2026-11-28 after the change, with the real scorer and
  `results.py`, on four outputs for `change_case:english_lowercase`:
  `hello world` and `<answer>hello world</answer>` pass; `Sure! hello world`
  and `Sure! Here it is: <answer>hello world</answer>` fail (text outside the
  markers is checked). `results_metadata.jsonl` holds `prompt_strict`,
  `prompt_loose`, `inst_strict` and `inst_loose`; the table shows one IF
  score, the headline.

## LLM judge

Status: `judge:` in the run config and `custom_eval.judge` (single-answer grading:
`{question}`/`{response}`/`{target}` template, verdict markers, verdict ->
score map) exist; no benchmark uses them yet.

User decisions (2026-11-24):
- The judge is always an open-weight model served by vLLM, or the benchmark is
  not run. Judged scores are therefore not comparable to public leaderboards
  that use proprietary judges; the user states this in their reports.
- Keep it simple: no general framework (template overrides, verdict parsers,
  multiple judges). Implement judge features only when a benchmark needs them.
- A benchmark's reference outputs are the official ones (e.g. AlpacaEval's
  GPT-4-Turbo outputs). Custom references would no longer be that benchmark.
- Open: verdict from token logprobs (as AlpacaEval does officially) or from a
  sampled answer. Decide from what others do when an open-weight judge is used,
  at implementation time.

AlpacaEval 2.0 facts collected 2026-11-23 (`tatsu-lab/alpaca_eval`, GitHub):
- Official judge prompt: a fixed system + user message template,
  `src/alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt`.
  It shows the instruction and two outputs labelled `m` and `M` and asks for the
  label of the better one only. GitHub-only: it would have to be vendored.
- Annotator config `weighted_alpaca_eval_gpt4_turbo/configs.yaml`: judge
  `gpt-4-1106-preview`, `max_tokens: 1`, `logprobs`, `top_logprobs: 5`;
  preference = P(m) / (P(m) + P(M)); output order randomized. A thinking judge
  breaks the one-token logprob scheme unless its thinking is turned off.
- Reference outputs: HF `tatsu-lab/alpaca_eval`, `alpaca_eval_gpt4_baseline.json`
  (805, generator `gpt4_1106_preview`); instructions in the same file.
- Headline: length-controlled win rate, a logistic regression in
  `src/alpaca_eval/metrics/glm_winrate.py` (formula
  `tanh(std_delta_len) + instruction_difficulty + not_gamed_baseline - 1`);
  `instruction_difficulty.csv` is on the HF repo.
