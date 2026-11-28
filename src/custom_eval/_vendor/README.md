# Vendored checker code

Copied from upstream so nothing is fetched at run time. Each folder keeps its
upstream LICENSE. Fetched 2026-11-23.

| Folder | Upstream | Files | Changes |
|---|---|---|---|
| `lcb/` | LiveCodeBench/LiveCodeBench `main` (MIT) | `lcb_runner/evaluation/testing_util.py` -> `testing_util.py` | none |
| `lcb/` | same | `check.py` | new entry point; `check_correctness` from `compute_code_generation_metrics.py`, test decoding from `benchmarks/code_generation.py` (changes listed in its header) |
| `ifeval/` | google-research/google-research `master`, `instruction_following_eval/` (Apache-2.0) | `instructions.py`, `instructions_registry.py`, `instructions_util.py` | see below |
| `ifbench/` | allenai/IFBench `main`, `ifbench/` (Apache-2.0) | `instructions.py`, `classic_instructions.py`, `instructions_registry.py`, `instructions_util.py` | see below |
| `nltk_data/` | nltk/nltk_data `gh-pages` (Apache-2.0) | English parts of `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng` | only the English files are kept |

Changes to the IFEval / IFBench code (nothing that alters a verdict):

- Imports rewritten to this package (`custom_eval._vendor.ifeval` / `custom_eval._vendor.ifbench`).
- `absl.logging` replaced by the standard `logging`; `immutabledict` replaced by
  `types.MappingProxyType` (two fewer dependencies).
- IFBench: the `.nltk_data` folder creation inside the package and the import-time
  `nltk.download(...)` calls are removed; NLTK data comes from `nltk_data/` here
  (see `custom_eval.benchmarks._nltk`).
- `langdetect` is seeded (`DetectorFactory.seed = 0`) by `custom_eval.benchmarks._nltk`,
  so the language checks are deterministic (upstream leaves it unseeded).
- The checker modules' `random` is replaced at run time (by `custom_eval.benchmarks.if_tasks`)
  with a private RNG reseeded to 0 before each instruction is built. Upstream draws a
  random argument when one is missing or invalid (e.g. IFEval key 1129 asks for the
  "letter" `!`, which upstream replaces with a random letter), so its verdict could change
  between runs; here it cannot.
