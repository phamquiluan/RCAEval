# Evaluating LLM agents on RCAEval

RCAEval can score any root cause analysis agent, whatever its framework or language. You export a dataset as a **blind task view**, run your agent over it, and score the **predictions file** it writes:

```bash
rcaeval-agent export --dataset re2-ob --out agent_tasks          # 1. blind task view + answers file
python my_agent.py agent_tasks/re2-ob preds.jsonl                  # 2. your agent, any stack
rcaeval-agent evaluate --tasks agent_tasks/re2-ob --predictions preds.jsonl   # 3. AC@k / Avg@k
```

`rcaeval-agent` is installed with RCAEval. `python -m RCAEval.agent` does the same.

You don't have to write the agent loop yourself. RCAEval ships [tools for investigating a case](#tools-for-investigating-a-case), a [reference LLM agent](#reference-llm-agent) that works with any LiteLLM provider, and an [MCP server](#mcp-server-and-agent-clis) that lets agent CLIs such as Claude Code investigate cases.

## Why a blind view

The raw datasets name each case directory after its answer, for example `data/RE2/RE2-OB/checkoutservice_cpu/1/`. An agent that can list files would see the root cause. The export:

- writes each case under an anonymised id such as `re2-ob-0042`
- includes only the case's telemetry and a `task.json`
- puts the labels in a separate answers file, next to the task directory and never inside it

## 1. Export

```bash
rcaeval-agent export --dataset re2-ob --out agent_tasks
```

`--dataset` accepts any of the nine subsets, `re1-ob` … `re3-tt`. Most published work uses `re2-ob`. The command downloads the dataset into `data/` if it is not already there. Use `--data-root` to point at a copy elsewhere.

```
agent_tasks/
  re2-ob/                       # give only this directory to the agent
    manifest.json               # dataset, number of cases, list of case ids, answer format
    cases/re2-ob-0001/
      task.json
      metrics.csv               # the metrics main.py feeds to methods (simple_metrics.csv / data.csv)
      metrics_raw.csv           # raw metrics, when the case has them (RE2/RE3)
      logs.csv  logts.csv       # logs and log time series, RE2/RE3
      traces.csv  tracets_err.csv  tracets_lat.csv   # traces, RE2/RE3 Online Boutique and Train Ticket
      log_templates.json        # the log template behind each logts.csv column, RE2/RE3
  re2-ob.answers.jsonl          # labels for the evaluator; keep it away from the agent
```

Case ids are deterministic: the same dataset exports to the same ids on every machine, so a predictions file made anywhere can be scored against anyone's answers file.

Telemetry files are hardlinked to the originals, never symlinked, because a symlink would reveal the source path. If hardlinking is not possible, they are copied, and `--copy` always copies. Writing to a hardlinked file changes the original, so mount the task directory read-only for your agent.

### task.json

```json
{
  "schema_version": 1,
  "case_id": "re2-ob-0001",
  "dataset": "re2-ob",
  "system": "Online Boutique",
  "inject_time": 1705312345,
  "standard_window": {"start": 1705311745, "end": 1705312945},
  "files": {"metrics": "metrics.csv", "logs": "logs.csv", "...": "..."},
  "candidate_services": ["adservice", "cartservice", "..."],
  "candidate_indicators": ["adservice_cpu", "adservice_mem", "adservice_latency", "..."],
  "answer_format": "..."
}
```

- `inject_time` is when the fault started (Unix seconds). It is the same anomaly-detection time that classic methods receive.
- `standard_window` covers the 10 minutes on each side of `inject_time`. Metrics are sampled once per second, and these are the rows `main.py` gives to methods. Your agent may look wider.
- `candidate_services` and `candidate_indicators` are the closed vocabulary to answer in. They are read from the metrics header the way `main.py` presents it: time columns are removed, p50 latency is dropped, and `*_latency-90` becomes `*_latency`.

## 2. Write predictions

Write one JSON line per case:

```json
{"case_id": "re2-ob-0007", "ranks": ["checkoutservice_cpu", "cartservice_mem", "frontend"], "reasoning": "...", "model": "my-agent/gpt-x", "usage": {"input_tokens": 51234, "output_tokens": 2310, "cost_usd": 0.21, "wall_time_s": 48.2, "tool_calls": 17}}
```

- `ranks` (required) lists the most likely root cause first. An entry is either a service from `candidate_services`, which is scored at service level only, or a `<service>_<indicator>` column from `candidate_indicators`, which is scored at both levels. A single answer is just a one-element list.
- `reasoning` and `model` (optional) are kept with the predictions but not scored.
- `usage` (optional) values are totalled and averaged in the report, so cost can be reported next to accuracy.

Names are trimmed and lower-cased before matching. There is no fuzzy matching, so answer with names from the candidate lists.

[`docs/examples/baro_agent.py`](examples/baro_agent.py) is a complete reference "agent" that runs BARO on every case and writes this file. Replace its BARO call with your agent:

```bash
python docs/examples/baro_agent.py agent_tasks/re2-ob preds.jsonl
```

## 3. Evaluate

```bash
rcaeval-agent evaluate --tasks agent_tasks/re2-ob --predictions preds.jsonl --report-chance
```

The reference BARO agent on RE1-OB prints:

```
--- Evaluation results (re1-ob, 125 cases) ---
Coverage:    125/125 cases predicted
             service    indicator
AC@1:        0.74       0.62
AC@3:        0.93       0.82
AC@5:        0.98       0.90
Avg@5:       0.90       0.79
Chance@5:    0.22       0.06
Lift@5:      0.67       0.73
--- Avg@5 by fault (service level) ---
Avg@5-CPU:   0.98
...
--- Usage ---
wall_time_s:    total 10.738, mean 0.085904 over 125 cases
---
```

- **Service level** asks whether the root-cause service is within the top k. **Indicator level** asks whether the exact `<service>_<indicator>` is: CPU faults are scored against `cpu`, memory against `mem`, delay and loss against `latency`, disk against `diskio` (`latency` on RE1, which has no disk metric), and socket against `socket`. The RE3 code-level faults (f1–f5) have no metric indicator, so their indicator level is n/a.
- Scoring uses the same functions as `main.py`, so BARO scores the same on the blind view as through `main.py`; the parity test in `tests/test_agent.py` checks this. The one difference is `Chance@5`: here its denominator is the full candidate list in `task.json`, while `main.py` uses the columns left after constant ones are dropped.
- A case with no line in the predictions file counts as a miss and is reported in the coverage line. Malformed lines, unknown or duplicate case ids, and a `ranks` that is not a list of strings stop the evaluation with every problem listed (exit code 1).
- `--per-case-out results.jsonl` writes, for each case, the answer, your top 5 and the rank at which the answer appeared. `--json-out summary.json` writes the summary.
- **Evidence.** When predictions carry a `trajectory` (the built-in agents write one), the report adds `Inspected@root`: the share of cases where a tool call targeted the root-cause service before `submit`, through a `service` argument or a metric of that service. It also shows AC@1 for cases with and without that inspection, which separates answers backed by evidence from lucky guesses.

## Tools for investigating a case

`RCAEval.agent.toolkit.CaseToolkit` opens one case of a task view and answers small questions about it. Each answer is a JSON object of at most about 8,000 characters, so it fits in an LLM context.

| Tool | Returns |
|---|---|
| `describe_case()` | system, inject time, standard window, available telemetry, candidate services |
| `list_metrics(service)` | metric names, per service or for one service |
| `get_metric(name, start, end, points)` | one metric averaged into at most `points` buckets, with mean and max before and after the inject time |
| `rank_anomalies(modality, top_k)` | series ranked by how far they move after the inject time (BARO's robust z-score) for `metrics`, `logs` (log template counts, with template text), `trace_latency` or `trace_errors`. For metrics this is exactly BARO's ranking. |
| `search_logs(service, pattern, start, end, limit)` | log lines matching a regex, grouped by pattern (numbers and ids masked), with counts before and after the inject time |
| `trace_summary(service, start, end, top_k)` | requests, errors and p50/p90 latency per service and operation, before and after the inject time |
| `call_graph()` | which service calls which, with call counts |
| `submit(ranks, reasoning)` | checks the ranking against the candidate lists and records it |

Times are Unix seconds and default to the standard window. A tool that needs telemetry the case lacks (RE1 has no logs or traces; Sock Shop has no traces) returns `{"error": ...}`.

```python
from RCAEval.agent.toolkit import CaseToolkit

tk = CaseToolkit("agent_tasks/re2-ob/cases/re2-ob-0001")
tk.call("rank_anomalies", {"modality": "metrics", "top_k": 5})
tk.call("search_logs", {"pattern": "error|exception"})
tk.call("submit", {"ranks": ["cartservice_cpu", "cartservice"]})
tk.calls  # every call: tool, arguments, ok/error, size, milliseconds
```

`RCAEval.agent.toolkit.TOOLS` describes every tool as a JSON schema, ready for any function-calling API.

## Reference LLM agent

```bash
pip install RCAEval[agent]
export ANTHROPIC_API_KEY=...    # or the key for your provider
rcaeval-agent run --tasks agent_tasks/re2-ob --model anthropic/claude-sonnet-5-5 --out preds.jsonl
rcaeval-agent evaluate --tasks agent_tasks/re2-ob --predictions preds.jsonl
```

The agent is a tool-calling loop over the tools above.
- **Prompt:** every model gets the same versioned prompt (`RCAEval/agent/prompt.py`). Each prediction records the prompt version.
- **Models:** `--model` takes any [LiteLLM](https://docs.litellm.ai/docs/providers) model name, e.g. `openai/...`, `gemini/...`, `openrouter/...`, or `ollama/...` for local models.
- **Limits:** a case ends when the model's `submit` is accepted. `--max-steps` (default 30 model turns) or `--max-cost-usd` (default 2.0 per case) force one last turn that must submit. If it still doesn't, the case counts as a miss.
- **Output:** predictions are appended as each case finishes. Rerunning the same command skips finished cases, so an interrupted run never pays twice. Each case's full conversation and tool calls go to `<out>.trajectories/<case_id>.jsonl`, and tokens, cost and steps go into `usage`.
- **Other options:** `--cases` limits the run to some case ids, and `--workers` runs cases in parallel.

## MCP server and agent CLIs

`rcaeval-agent mcp --case <case_dir> --record <dir>` serves one case's tools over MCP (stdio). Any MCP client can use it. Each call is logged to `<dir>/calls.jsonl`, and the accepted `submit` is written to `<dir>/submission.json`.

`rcaeval-agent run-cli` runs an agent CLI over a whole task view. For each case it:
1. starts the MCP server for that case;
2. runs the CLI with the same prompt as the reference agent, in an empty temporary directory;
3. records the submission and appends a predictions line.

Resume, `--cases`, `--workers` and trajectories work as in `run`.

```bash
rcaeval-agent run-cli --agent claude-code --model sonnet --tasks agent_tasks/re2-ob --out preds.jsonl
```

The `claude-code` preset runs `claude -p` with:
- `--bare`, which skips your CLAUDE.md, memory and hooks, and authenticates with `ANTHROPIC_API_KEY`;
- `--tools ""`, which removes every built-in tool, including shell and file access;
- `--strict-mcp-config`, so only the RCAEval tools are available.

Claude Code therefore reaches the case only through RCAEval's tools. Cost, tokens and turns come from its JSON output.

For other MCP-capable agents, use `--agent custom` with a command template. The runner fills in `{prompt}`, `{mcp_config}` (a JSON file in the standard `mcpServers` format), `{workdir}` and `{model}`:

```bash
rcaeval-agent run-cli --agent custom --tasks agent_tasks/re2-ob --out preds.jsonl \
  --command "my-agent --mcp-config {mcp_config} --prompt {prompt}"
```

Agents that keep MCP servers in their own configuration format, such as Codex or OpenCode, need a small wrapper script that turns `{mcp_config}` into that format. These have not been tested with RCAEval yet. The tool calls the server logs are the trajectory, so `Inspected@root` works for any agent.

## Keeping the evaluation honest

- Give the agent only the task directory (`agent_tasks/re2-ob/`), read-only, for example as a read-only container mount. Never give it the answers file or `data/`. With `run` and the `claude-code` preset of `run-cli`, the agent has no file access at all and sees only the tools' answers.
- If the agent can browse the web, note that the raw datasets are public. The blind view prevents accidental and agent-side leakage. It does not stop someone deliberately matching telemetry against the published data.
- Report the model, the agent version, the dataset and the cost (`usage`) with your results. Report a classic baseline such as BARO on the same view.
