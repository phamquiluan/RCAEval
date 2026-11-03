<!--
Thanks for contributing to RCAEval! Please read CONTRIBUTING.md first:
https://github.com/phamquiluan/RCAEval/blob/main/CONTRIBUTING.md
Keep one logical change per pull request. Delete any section that doesn't apply.
-->

## What and why

<!-- What does this change, and why is it needed? Link the issue it addresses, e.g. "Fixes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New RCA method
- [ ] New or changed dataset support
- [ ] Evaluator, metrics or agent tooling
- [ ] Documentation
- [ ] CI, packaging or dependencies

## How it was tested

<!-- Commands you ran and their results, e.g. the pytest command from CONTRIBUTING.md, or a main.py run. -->

## Effect on scores

RCAEval is a benchmark, so any change to existing results has to be visible.

- [ ] This change does **not** alter the output of `main.py` for any existing method or dataset (and I checked).
- [ ] This change **does** alter existing results. Before/after numbers and the reason the new ones are correct are below.

<!--
| Method | Dataset | Before | After |
|--------|---------|--------|-------|
| baro   | re2-ob  | ...    | ...   |
-->

## Checklist

- [ ] Tests added or updated (a bug fix includes a test that fails without it)
- [ ] Optional dependencies stay optional: `import RCAEval.e2e` still works without them
- [ ] Docs updated (`README.md`, `docs/`, docstrings) where behaviour changed
- [ ] `CHANGELOG.md` entry added for user-visible changes
- [ ] Code ported from another project has its license in `LICENSES/` and is listed in the README

<!-- For a new method: also registered in _METHOD_MODULES (RCAEval/e2e/__init__.py) and in main.py's method list. -->
