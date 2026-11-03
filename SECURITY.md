# Security Policy

## Supported versions

Security fixes go into the latest release on [PyPI](https://pypi.org/project/RCAEval/) and the `main` branch.

| Version | Supported |
| ------- | --------- |
| 1.11.x (latest) | Yes |
| < 1.11  | No. Please upgrade |

The paper branches (`ase24`, `www25`, `fse26`) are frozen so that published results can be reproduced. They don't receive security fixes.

## Reporting a vulnerability

**Please don't report security vulnerabilities in public GitHub issues, discussions or pull requests.**

Email [phamquiluan@gmail.com](mailto:phamquiluan@gmail.com?subject=RCAEval%20security) with "RCAEval security" in the subject. Include:

- what the problem is and what an attacker could do with it
- the affected version or commit, and the component (e.g. dataset download, `rcaeval-agent mcp`)
- steps or a minimal proof of concept to reproduce it
- any fix you would suggest

RCAEval is maintained by a small research team, so we can't promise fixed response times. We aim to acknowledge reports within a week and keep you updated until the issue is resolved. Once a fix is released we will credit you in the changelog, unless you would rather stay anonymous. Please give us a reasonable chance to release a fix before you disclose the issue publicly.

## Scope

In scope: code in this repository, in particular

- **Dataset download and extraction** (`RCAEval/utility`, `RCAEval.benchmark.datasets`), e.g. an archive that writes files outside the target directory
- **The agent tooling** (`RCAEval/agent`): the MCP server, the reference agent and `rcaeval-agent run-cli`, e.g. a crafted task view or case file that makes a tool read or write files outside the case directory or run commands
- **The blind task view**: a way for an agent to recover the ground-truth labels from an exported task directory. This is a benchmark integrity problem rather than a classic vulnerability, but please report it the same way so it can be fixed before it skews published results

Out of scope:

- Vulnerabilities in third-party dependencies (numpy, torch, litellm, mcp, ...). Please report those upstream. If RCAEval uses a dependency in an unsafe way, or pins a version with a known vulnerability, that is in scope.
- Bugs in the original implementations of the baselines we include (see [LICENSES](LICENSES)), unless they come from our changes. Please report those to the original authors.
- Behaviour that is intended and documented. For example, `rcaeval-agent run-cli --agent custom --command ...` runs the command you give it, and any agent you run has the permissions of your user account. Only run agents and commands you trust.
- Wrong RCA results or crashes with no security impact. Please open a regular [issue](https://github.com/phamquiluan/RCAEval/issues) for those.

For everything else about contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).
