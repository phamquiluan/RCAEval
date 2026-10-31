"""Run an MCP-capable agent CLI over a task view.

    rcaeval-agent run-cli --agent claude-code --tasks agent_tasks/re2-ob --out preds.jsonl

For each case the runner writes an MCP config that starts `rcaeval-agent mcp`
on that case, runs the agent CLI in an empty temporary directory with the
canonical prompt, and reads the submission the server recorded. The agent sees
the case only through the MCP tools.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from os.path import abspath, exists, join

from RCAEval.agent.batch import run_batch
from RCAEval.agent.prompt import PROMPT_VERSION, full_prompt

# --bare skips CLAUDE.md, auto-memory and hooks and authenticates with
# ANTHROPIC_API_KEY; --tools "" removes every built-in tool (shell, file
# reading), so only the rcaeval MCP tools remain
PRESETS = {
    "claude-code": ["claude", "-p", "{prompt}", "--bare", "--tools", "", "--mcp-config", "{mcp_config}",
                    "--strict-mcp-config", "--allowedTools", "mcp__rcaeval", "--output-format", "json",
                    "--no-session-persistence"],
}


def build_command(agent, command, prompt, mcp_config, workdir, model=None):
    if agent == "custom":
        if not command:
            raise ValueError("--agent custom needs --command, e.g. \"mycli --mcp {mcp_config} -p {prompt}\"")
        template = shlex.split(command)
    else:
        template = list(PRESETS[agent])
        if model:
            template += ["--model", model]
    fields = {"prompt": prompt, "mcp_config": mcp_config, "workdir": workdir, "model": model or ""}
    return [part.format(**fields) for part in template]


def claude_usage(stdout):
    """Cost and tokens from `claude -p --output-format json`, when parseable."""
    try:
        out = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return {}
    u = out.get("usage") or {}
    usage = {"cost_usd": out.get("total_cost_usd"), "turns": out.get("num_turns")}
    if u:
        usage["input_tokens"] = sum(int(u.get(k) or 0) for k in
                                    ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        usage["output_tokens"] = int(u.get("output_tokens") or 0)
    return {k: v for k, v in usage.items() if v is not None}


def run_case(case_dir, record_dir, agent="claude-code", command=None, model=None, timeout=900,
             prompt_version=PROMPT_VERSION):
    """Run the agent CLI on one case; returns (prediction, trajectory records)."""
    with open(join(case_dir, "task.json")) as f:
        task = json.load(f)
    if exists(record_dir):
        shutil.rmtree(record_dir)  # a rerun of a case that did not finish
    os.makedirs(record_dir)
    workdir = tempfile.mkdtemp(prefix="rcaeval-agent-")
    mcp_config = join(workdir, ".rcaeval-mcp.json")
    with open(mcp_config, "w") as f:
        json.dump({"mcpServers": {"rcaeval": {
            "command": sys.executable,
            "args": ["-m", "RCAEval.agent", "mcp", "--case", abspath(case_dir), "--record", abspath(record_dir)],
        }}}, f)

    cmd = build_command(agent, command, full_prompt(task, prompt_version), mcp_config, workdir, model)
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout)
        stop_reason = None if proc.returncode == 0 else f"exit_{proc.returncode}"
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        stop_reason = "timeout"
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    wall = round(time.time() - started, 2)
    shutil.rmtree(workdir, ignore_errors=True)
    with open(join(record_dir, "agent_stdout.txt"), "w") as f:
        f.write(stdout)
    with open(join(record_dir, "agent_stderr.txt"), "w") as f:
        f.write(stderr[-20000:])

    calls = []
    if exists(join(record_dir, "calls.jsonl")):
        with open(join(record_dir, "calls.jsonl")) as f:
            calls = [json.loads(line) for line in f if line.strip()]
    submission = None
    if exists(join(record_dir, "submission.json")):
        with open(join(record_dir, "submission.json")) as f:
            submission = json.load(f)
    if stop_reason is None:
        stop_reason = "submit" if submission else "no_submit"

    usage = {"wall_time_s": wall, "tool_calls": len(calls)}
    if agent == "claude-code":
        usage.update(claude_usage(stdout))
    usage.setdefault("cost_usd", None)
    submission = submission or {"ranks": [], "reasoning": None}
    prediction = {
        "case_id": task["case_id"],
        "ranks": submission["ranks"],
        "reasoning": submission.get("reasoning"),
        "model": agent + (f":{model}" if model else ""),
        "prompt_version": prompt_version,
        "stop_reason": stop_reason,
        "usage": usage,
    }
    records = ([{"type": "meta", "case_id": task["case_id"], "model": prediction["model"],
                 "prompt_version": prompt_version}] + calls + [{"type": "result", **prediction}])
    return prediction, records


def run_tasks(tasks_dir, out, agent="claude-code", command=None, model=None, timeout=900,
              cases=None, trajectories=None, workers=1, prompt_version=PROMPT_VERSION):
    if agent == "claude-code" and shutil.which("claude") is None:
        raise SystemExit("the claude-code preset needs the `claude` CLI on PATH")
    runs = out + ".runs"

    def solve(case_id, case_dir):
        return run_case(case_dir, join(runs, case_id), agent=agent, command=command, model=model, timeout=timeout,
                        prompt_version=prompt_version)

    return run_batch(tasks_dir, out, solve, cases=cases, trajectories=trajectories, workers=workers)
