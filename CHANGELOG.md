# Changelog

All notable changes to RCAEval are listed here, newest first. Versions follow the
[GitHub releases](https://github.com/phamquiluan/RCAEval/releases), where the longer
release notes live, and the [PyPI releases](https://pypi.org/project/RCAEval/#history).
A few early versions exist on only one of the two.

## [1.11.0] - 2026-10-31

RCAEval now ships the agents themselves: tools for investigating a case, a reference
LLM agent for any LiteLLM model, and an MCP server that lets agent CLIs such as
Claude Code investigate cases.

### Added
- Tools for investigating a case (`RCAEval.agent.toolkit`): `describe_case`, `list_metrics`,
  `get_metric`, `rank_anomalies` (BARO's ranking for metrics; also log template counts and
  trace latency and errors), `search_logs`, `trace_summary`, `call_graph` and `submit`, each
  returning a compact JSON answer, with every call logged.
- `rcaeval-agent run`: a reference LLM agent over those tools, for any LiteLLM model, with a
  shared, versioned prompt, step and cost limits, resume, per-case trajectories, and token and
  cost accounting.
- `rcaeval-agent mcp`: the same tools as an MCP server for one case. `rcaeval-agent run-cli`
  runs an agent CLI over a task view through it, with a Claude Code preset (no built-in tools,
  only RCAEval's) and a custom command template.
- `rcaeval-agent evaluate` reports `Inspected@root` from trajectories: how often the agent
  queried the root-cause service before submitting.
- Task prompt version 2, now the default for `run` and `run-cli`: the agent checks each top
  suspect's own evidence instead of trusting one anomaly ranking. On 10 RE2-OB cases it
  raised the reference agent's AC@1 (Claude Sonnet 5.5) from 0.60 (version 1) to 1.00, at
  about twice the cost, and Claude Code's through `run-cli` from 0.10 to 0.30.
  `--prompt-version 1` keeps the earlier prompt.
- New `agent` extra (`litellm`, `mcp`).

### Changed
- The task view also exports `log_templates.json`, the template behind each `logts.csv` column.
- `baro()` also returns each metric's score (`scores`); its ranking is unchanged.

## [1.10.0] - 2026-10-31

LLM agents can now be evaluated on RCAEval through a blind task view and a
predictions file.

### Added
- LLM agent onboarding ([docs/AGENTS.md](docs/AGENTS.md)). `rcaeval-agent export --dataset <name>`
  writes a blind task view of any RE1–RE3 subset: each case under an anonymised id
  with its telemetry and a `task.json` (inject time, candidate services and
  indicators), and the labels in a separate answers file. `rcaeval-agent evaluate`
  scores a predictions file (`{"case_id", "ranks"}` per line) with the same AC@k and
  Avg@k as `main.py`, at service and indicator level, and totals any reported token,
  cost and time usage.
- `docs/examples/baro_agent.py`, a reference agent that runs BARO over the task view.
  It ranks every case exactly as `main.py` does.
- `CHANGELOG.md`, covering every release since 0.0.1.

### Changed
- `main.py` takes its dataset paths, downloads and ranking-to-score conversion from the
  new `RCAEval.benchmark.datasets` and `RCAEval.benchmark.scoring` modules, which the
  agent evaluator shares. Its output is unchanged.

## [1.9.0] - 2026-10-29

Python 3.14 is now the default environment; 3.12 and 3.13 also work. RCD,
Multi-source RCD, HT and TORAI keep their Python 3.8 environments.

### Fixed
- RUN crashed under Python 3.14's `forkserver` start method and on NaN inputs, and its
  Pearson correlation overflowed on large integer memory metrics (numpy 1 wrapped this
  silently, giving wrong correlations).
- RUN no longer turns on torch anomaly mode for every torch-based method at import.
- MicroCause works with tigramite 5, pandas 3 and networkx 3.
- ε-Diagnosis runs without PyRCA through a bundled copy of PyRCA's implementation;
  PyRCA is still used when installed.
- `fci_randomwalk` fell back to column order on every case because FCI's circle edge
  marks were rejected.
- `*_randomwalk` wrote its transition matrix into a copy under pandas 3.
- `ges_pagerank` returned `(name, score)` tuples, which crashed `main.py`.
- `granger_pagerank` and `granger_randomwalk` work on statsmodels 0.14 and 0.15.
- `cmlp_pagerank` standardizes its input, so training no longer diverges to NaN.
- The sdist now ships the requirement files that `setup.py` reads.

### Changed
- `requirements.txt` lists only the 17 packages RCAEval imports (was 164), with one
  spec for Python 3.12–3.14; `statsmodels` stays below 0.15.
- New `core` extra (everything but torch) and `ntlr` extra (`gcastle`, for
  `ntlr_pagerank` / `ntlr_randomwalk`). `eventadl` now equals `core`, so EventADL runs
  in the default environment.
- Any Python 3.10 or newer passes the version check for the default environment.
- `python_requires=">=3.8"` and Python version classifiers.
- CI tests Python 3.14, 3.13 and 3.12 with CPU-only torch.
- Docs describe Python 3.14 as the default.

### Removed
- `easyrca` and `causalai` are no longer offered by `main.py`.
- `torai` is rejected outside its Python 3.8 environment.

## [1.8.1] - 2026-10-20

### Fixed
- CausalRCA ranked effects above their causes: PageRank now walks the reversed learned
  graph ([#82](https://github.com/phamquiluan/RCAEval/pull/82)).

## [1.8.0] - 2026-09-29

### Added
- Retrieval–reranking decomposition from *Where Root Cause Analysis Fails* (NeurIPS'26):
  `main.py --report-decomposition K` and `Evaluator.retrieval()` / `rerank()`.
- `Evaluator(max_k=...)` to track cutoffs beyond 5.

### Fixed
- RE1 disk faults are scored against latency, since RE1 has no disk I/O metrics
  ([#77](https://github.com/phamquiluan/RCAEval/issues/77)).
- PageRank-based methods and CIRCA silently ranked in column order after scikit-network
  and networkx API removals; the fallback now reports its error on stderr.

## [1.7.0] - 2026-09-03

### Added
- Chance-level baseline for Avg@k: `main.py --report-chance`, and `Evaluator`
  `n_candidates` / `answer_in_candidates` with `chance_accuracy`, `chance_average` and
  `lift` ([#79](https://github.com/phamquiluan/RCAEval/pull/79)).

## [1.6.0] - 2026-08-02

### Added
- All nine datasets on [Hugging Face](https://huggingface.co/datasets/phamquiluan/RCAEval)
  as Parquet, with a `cases.parquet` index.
- `read_metrics`, `read_logs` and `read_traces`, which read either the original or the
  Parquet layout.

### Fixed
- RE3-SS and RE3-TT fault types listed as `f1`–`f4`.

## [1.5.0] - 2026-07-30

### Added
- Python 3.14 support for the default environment
  ([#76](https://github.com/phamquiluan/RCAEval/pull/76)), with CI on 3.12 and 3.14.

## [1.4.0] - 2026-07-30

### Added
- Runnable multi-source benchmark in `main.py`: Multi-source BARO, CIRCA (new) and RCD,
  which now uses logs and traces as well as metrics
  ([#75](https://github.com/phamquiluan/RCAEval/pull/75)).

### Fixed
- PyPI wheels for 1.2.0 and 1.3.0 contained only the top-level package; wheels now
  include every subpackage.

## [1.3.0] - 2026-07-30

### Added
- TORAI, a multi-source RCA method (FSE'26), with dataset support and
  [docs/TORAI.md](docs/TORAI.md).
- EventADL, event-log root cause localization (FSE'26)
  ([#74](https://github.com/phamquiluan/RCAEval/pull/74)).

### Changed
- `main-ase.py` merged into `main.py`.
- RE2 sub-datasets download separately; CircleCI caches datasets.

### Fixed
- `IndexError` in evaluation when a rank entry has no underscore.
- pandas compatibility (`concat` instead of `_append`, `ffill` / `bfill`).

## [1.2.0] - 2025-06-07

### Changed
- Default environment moved from Python 3.10 to 3.12; requirements loosened and DVC
  requirements removed.
- Log parsing reorganized into the `RCAEval.logparser` module with `LogTemplate` and
  `EventTemplate`.

## [1.1.2] - 2025-04-24

### Changed
- Event templates renamed to log templates (`LogTemplate`).

## [1.1.1] - 2025-04-24

Version bump only.

## [1.1.0] - 2025-04-24

### Added
- Template-based log parser, with templates for Online Boutique and Sock Shop services
  and utilities to mask values in JSON logs.

## [1.0.1] - 2025-04-24

CI configuration change only.

## [1.0.0] - 2025-04-01

The version used in the WWW'25 paper, also kept in the
[www25 branch](https://github.com/phamquiluan/RCAEval/tree/www25).

## [0.4.5] - 2025-03-11

Packaging metadata and README updates.

## [0.4.4] - 2025-02-16

### Added
- Tests for HT and ε-Diagnosis, and setup instructions for the RCD, HT and ε-Diagnosis
  environment.

## [0.4.3] - 2025-01-10

### Added
- `main.py` checks the Python version (`is_py38`).

## [0.4.2] - 2025-01-05

### Added
- Code-level RCA tutorial notebook
  ([#45](https://github.com/phamquiluan/RCAEval/issues/45)).
- Per-system download functions for RE1 and RE2 (`download_re1ob_dataset`, ...).

## [0.4.1] - 2024-12-28

### Removed
- fGES dependencies ([#42](https://github.com/phamquiluan/RCAEval/pull/42)).

## [0.4.0] - 2024-12-27

Published to PyPI as 0.3.1; there is no 0.4.0 on PyPI.

### Added
- `main.py` (formerly `rq2.py`) for the RE1, RE2 and RE3 datasets.
- CausalRCA, MSCRED, TraceRCA, MicroRank, PDiagnose and Multi-source RCD.
- [docs/EXTENDING.md](docs/EXTENDING.md).

### Removed
- Synthetic dataset support.

## [0.2.1] - 2024-12-21

### Added
- Multi-source RCA demo notebook and `download_multi_source_sample`.

## [0.2.0] - 2024-12-21

The RCAEval benchmark ([arXiv:2412.17015](https://arxiv.org/abs/2412.17015)).

### Added
- RE1, RE2 and RE3 download functions (735 failure cases).
- Multi-source BARO.

## [0.1.6] - 2024-09-10

### Added
- Data-length analysis notebook under `legacy/`.

## [0.1.5] - 2024-08-25

README updates only.

## [0.1.4] - 2024-08-24

### Added
- Licenses for the bundled third-party methods (BARO, N-Sigma, Granger, MicroCause, RCD,
  ε-Diagnosis, causal-learn).

## [0.1.3] - 2024-08-24

README updates only.

## [0.1.2] - 2024-08-24

Version bump only.

## [0.1.1] - 2024-08-24

### Added
- BARO, `read_data`, and a basic functionality test.

## [0.1.0] - 2024-08-24

### Added
- `--test` flag for faster smoke runs, and RCD tests in CI with `script/link.sh`.

### Fixed
- [#5](https://github.com/phamquiluan/RCAEval/issues/5),
  [#9](https://github.com/phamquiluan/RCAEval/issues/9),
  [#11](https://github.com/phamquiluan/RCAEval/issues/11),
  [#13](https://github.com/phamquiluan/RCAEval/issues/13).

## [0.0.11] to 0.0.13 - 2024-08-15

Software artifact for the ASE'24 paper *Root Cause Analysis for Microservice System
based on Causal Inference: How Far Are We?*, also kept in the
[ase24 branch](https://github.com/phamquiluan/RCAEval/tree/ase24). Adds CircleCI and
[docs/SETUP.md](docs/SETUP.md).

## [0.0.10] - 2024-08-14

### Added
- Reproduction scripts and CI for all four research questions.
- `@rca` decorator so one failing case does not stop a run
  ([#3](https://github.com/phamquiluan/RCAEval/issues/3)).

## [0.0.9] - 2024-08-14

### Added
- Automatic download of the Online Boutique, Sock Shop and Train Ticket datasets, with
  tests and CI.

## [0.0.8] - 2024-08-11

### Added
- End-to-end methods: RUN, RCD, CIRCA, MicroCause, CloudRanger, EasyRCA, CausalAI, and
  PC / FCI / GES / LiNGAM / Granger with PageRank or random walk.
- Graph heads: PageRank, random walk, random selection, DFS and RHT.
- Third-party licenses.

## [0.0.7] - 2024-08-11

### Added
- Causal graph construction: PC, FCI, FGES, LiNGAM, Granger, PCMCI, NOTEARS, DAG-GNN and
  cMLP, with an evaluation script.

## [0.0.1] to 0.0.5 - 2024-08-10

Initial package: data I/O, utilities, graph heads, the PC algorithm and the benchmark
module.

[1.11.0]: https://github.com/phamquiluan/RCAEval/compare/1.10.0...1.11.0
[1.10.0]: https://github.com/phamquiluan/RCAEval/compare/1.9.0...1.10.0
[1.9.0]: https://github.com/phamquiluan/RCAEval/compare/1.8.1...1.9.0
[1.8.1]: https://github.com/phamquiluan/RCAEval/compare/1.8.0...1.8.1
[1.8.0]: https://github.com/phamquiluan/RCAEval/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/phamquiluan/RCAEval/compare/1.6.0...1.7.0
[1.6.0]: https://github.com/phamquiluan/RCAEval/compare/1.5.0...1.6.0
[1.5.0]: https://github.com/phamquiluan/RCAEval/compare/1.4.0...1.5.0
[1.4.0]: https://github.com/phamquiluan/RCAEval/compare/1.3.0...1.4.0
[1.3.0]: https://github.com/phamquiluan/RCAEval/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/phamquiluan/RCAEval/compare/1.1.1...1.2.0
[1.1.2]: https://pypi.org/project/RCAEval/1.1.2/
[1.1.1]: https://github.com/phamquiluan/RCAEval/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/phamquiluan/RCAEval/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/phamquiluan/RCAEval/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/phamquiluan/RCAEval/compare/0.4.5...1.0.0
[0.4.5]: https://github.com/phamquiluan/RCAEval/compare/0.4.4...0.4.5
[0.4.4]: https://github.com/phamquiluan/RCAEval/compare/0.4.3...0.4.4
[0.4.3]: https://github.com/phamquiluan/RCAEval/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/phamquiluan/RCAEval/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/phamquiluan/RCAEval/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/phamquiluan/RCAEval/compare/0.2.1...0.4.0
[0.2.1]: https://github.com/phamquiluan/RCAEval/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/phamquiluan/RCAEval/compare/0.1.6...0.2.0
[0.1.6]: https://github.com/phamquiluan/RCAEval/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/phamquiluan/RCAEval/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/phamquiluan/RCAEval/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/phamquiluan/RCAEval/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/phamquiluan/RCAEval/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/phamquiluan/RCAEval/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/phamquiluan/RCAEval/compare/0.0.13...0.1.0
[0.0.11]: https://github.com/phamquiluan/RCAEval/compare/0.0.10...0.0.13
[0.0.10]: https://github.com/phamquiluan/RCAEval/compare/0.0.9...0.0.10
[0.0.9]: https://github.com/phamquiluan/RCAEval/compare/0.0.8...0.0.9
[0.0.8]: https://github.com/phamquiluan/RCAEval/compare/0.0.7...0.0.8
[0.0.7]: https://github.com/phamquiluan/RCAEval/compare/0.0.5...0.0.7
[0.0.1]: https://github.com/phamquiluan/RCAEval/releases/tag/0.0.1
