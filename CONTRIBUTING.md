# Contributing to RCAEval

Thanks for helping improve RCAEval. Bug reports, fixes, new RCA methods, new datasets and documentation are all welcome. This guide covers how to set up a development environment, what we expect in a pull request, and the conventions the codebase follows.

By taking part you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and asking questions

Open an issue on [GitHub](https://github.com/phamquiluan/RCAEval/issues). For a bug, include:

- the exact command you ran, e.g. `python main.py --method baro --dataset re2-ob`
- the output you got and the output you expected (for a score, the numbers from the paper or README you compared against)
- your Python version, OS, which environment you installed (`default`, `rcd`, `agent`, ...) and the RCAEval version (`python setup.py --version` or `pip show RCAEval`)

If you think you have found a security problem, don't open a public issue; follow [SECURITY.md](SECURITY.md) instead.

## Development setup

RCAEval needs two environments because some baselines pin old dependencies. [docs/SETUP.md](docs/SETUP.md) has the full instructions; the short version is:

```bash
git clone https://github.com/<you>/RCAEval.git && cd RCAEval

# default environment: most methods, the evaluator and the agent tools
python3.14 -m venv env && . env/bin/activate
pip install --upgrade pip
pip install -e .[default] --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .[agent]        # only if you work on RCAEval/agent
pip install pytest
```

Python 3.12, 3.13 and 3.14 are all tested in CI. Ubuntu also needs a few system packages (`build-essential libxml2-dev zlib1g-dev python3-tk graphviz`); see SETUP.md.

The **RCD environment** (Python 3.8, `pip install -e .[rcd]`, then `bash script/link.sh`) is needed only for RCD, Multi-source RCD and HT. TORAI has its own Python 3.8 environment, described in [docs/TORAI.md](docs/TORAI.md). Optional extras: `ntlr` (NOTEARS low-rank methods), `eventadl`, `agent`.

Datasets are downloaded into `data/` on first use and are git-ignored. Don't commit data, outputs or notebooks' large cell outputs.

## Running the tests

Run the same suite CI runs on the default environment:

```bash
python -m pytest tests/test.py tests/test_parser* tests/test_evaluator.py \
  tests/test_agent.py tests/test_agent_toolkit.py tests/test_agent_react.py \
  tests/test_graph_methods.py tests/test_sli.py tests/test_version_gates.py

python -m pytest tests/test_agent_mcp.py      # needs the agent extra
```

In the RCD environment, run `python -m pytest tests/test_rcd.py`.

Some tests download sample data on the first run, so they need network access. `tests/test_datasets.py` downloads every dataset and is not part of the regular CI run; run it only when you change dataset download or loading code.

If your change can affect scores, also check a reproduction end to end. CI does this for BARO:

```bash
python main.py --method baro --dataset online-boutique
python main.py --method baro --dataset re2-ss
```

`python main.py --method <m> --dataset <d> --test` is a quick smoke test that doesn't run every case.

## Making a change

1. Fork the repository and create a branch from `main`.
2. Keep the pull request to one logical change. Unrelated refactors or reformatting make review harder; send them separately.
3. Add or update tests. A bug fix should come with a test that fails without the fix (see `tests/test_graph_methods.py` for the style: small synthetic data, a comment saying what used to go wrong).
4. Update the docs the change touches: `README.md`, `docs/*.md`, or the method's docstring.
5. Add an entry to `CHANGELOG.md` for anything a user would notice (new method, changed output, fixed wrong result). Use the existing `Added` / `Changed` / `Fixed` headings.
6. Open a pull request against `main` and describe what changed, why, and how you tested it. GitHub Actions runs the test suite on Python 3.12–3.14 and 3.8 (RCD), plus a BARO reproduction.

### Reproducibility comes first

RCAEval is a benchmark: people compare their numbers against ours. So:

- **Don't change published scores silently.** If a change alters the output of `main.py` for an existing method or dataset, say so in the pull request, explain why the new numbers are correct, and list before/after results. Refactors should leave the output identical, and the pull request should say you checked.
- Keep randomness seeded and deterministic where the method allows it.
- Don't change dataset contents or labels without discussing it in an issue first.

### Commit messages

Write a short subject line in the imperative mood ("Count gRPC failure codes as trace errors", not "Counted..." or "Fixes..."). Many commits use a [Conventional Commits](https://www.conventionalcommits.org/) prefix with the area in parentheses, e.g. `fix(e2e): ...`, `docs(readme): ...`, `ci: ...`; this is welcome but not required. Use the body to explain *why* when it isn't obvious.

## Code conventions

There is no enforced formatter or linter. Follow the style of the surrounding code (PEP 8, 4-space indentation, imports at the top) and keep diffs focused.

- Comments explain *why*, especially workarounds for upstream API changes or data quirks.
- Methods must not swallow errors silently. The `@rca` wrapper in `RCAEval/e2e/__init__.py` prints any exception to stderr before falling back; keep that behaviour if you write a fallback of your own.
- Code that needs an optional dependency (torch, gcastle, litellm, mcp, ...) must not break `import RCAEval.e2e` when that dependency is missing. Tests that need one should use `pytest.importorskip`.
- The default environment must work on any Python ≥ 3.10, not only the versions CI tests. Use the helpers in `RCAEval/utility` (`is_modern_py()`, `is_py38()`) instead of checking specific minor versions.

## Adding a new RCA method

[docs/EXTENDING.md](docs/EXTENDING.md) describes the method design in more detail. In this codebase, a method is wired in as follows:

1. Create `RCAEval/e2e/<method>.py` with a function of this form:

   ```python
   def my_method(data, inject_time=None, dataset=None, sli=None, anomalies=None, **kwargs):
       ...
       return {"ranks": ranked_candidates}   # most likely root cause first
   ```

   `data` is a `pandas.DataFrame` with a `time` column for metric-based methods, or a dict of frames (`metric`, `logts`, `tracets_err`, `tracets_lat`) for multi-source methods. Reuse the preprocessing helpers in `RCAEval.io.time_series` (`preprocess`, `drop_constant`, `drop_time`, ...) so your method sees the same input as the existing baselines.

2. Register it in `_METHOD_MODULES` in `RCAEval/e2e/__init__.py` (module name → exported function names). Methods that only run on Python 3.8 go in the `else` branch instead.
3. Add the function name to the method list in `main.py` so `--method my_method` works.
4. Add a test under `tests/` that runs the method on small synthetic data, and add it to the CI test command in `.github/workflows/build-and-test.yml` if it is a new file.
5. Add the method to "Available Baselines" in `README.md`, with a citation if it comes from a paper.
6. Put any new dependency in `requirements.txt` (or a new extra in `setup.py` if most users won't need it).

### Code from other projects

If you port or adapt code from another repository, check that its license allows it, add the license as `LICENSES/LICENSE-<Name>`, and list it under "Licensing" in `README.md`. Code we write ourselves is MIT-licensed; by contributing you agree that your contribution is distributed under the [MIT License](LICENSE).

## Adding a dataset

See the dataset section of [docs/EXTENDING.md](docs/EXTENDING.md). Open an issue first so we can agree on the format, hosting and labels before you do the work. Datasets are hosted outside the repository (Zenodo and [Hugging Face](https://huggingface.co/datasets/phamquiluan/RCAEval)); the download code lives in `RCAEval/utility` and `RCAEval.benchmark.datasets`.

## Releases (maintainers)

1. Bump `version` in `setup.py` in its own commit (`Bump version to X.Y.Z`) and move the changelog entries under a new `## [X.Y.Z] - YYYY-MM-DD` heading.
2. Tag it: `. script/tag.sh "<message>"`, then push the tag.
3. Publish a GitHub release for the tag. The `Upload Python Package` workflow builds the wheel and uploads it to PyPI.

## Contact

Questions that don't fit an issue: [phamquiluan@gmail.com](mailto:phamquiluan@gmail.com?subject=RCAEval).
