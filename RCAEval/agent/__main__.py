"""rcaeval-agent: blind task views, LLM agents, and predictions scoring.

    rcaeval-agent export --dataset re2-ob --out agent_tasks
    rcaeval-agent run --tasks agent_tasks/re2-ob --model anthropic/claude-sonnet-5-5 --out preds.jsonl
    rcaeval-agent run-cli --agent claude-code --tasks agent_tasks/re2-ob --out preds.jsonl
    rcaeval-agent mcp --case agent_tasks/re2-ob/cases/re2-ob-0001 --record runs/re2-ob-0001
    rcaeval-agent evaluate --tasks agent_tasks/re2-ob --predictions preds.jsonl
"""
import argparse
import sys

from RCAEval.agent.export import EXPORTABLE, export
from RCAEval.agent.predictions import PredictionsError, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rcaeval-agent", description="Evaluate LLM agents on RCAEval")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("export", help="write a blind task view of a dataset and its answers file")
    p.add_argument("--dataset", required=True, choices=EXPORTABLE)
    p.add_argument("--out", default="agent_tasks", help="output directory (default: agent_tasks)")
    p.add_argument("--data-root", default=None,
                   help="the dataset's directory, if not at its default location under data/ (skips the download)")
    p.add_argument("--copy", action="store_true", help="copy telemetry files instead of hardlinking them")
    p.add_argument("--force", action="store_true", help="replace an existing export")

    p = sub.add_parser("evaluate", help="score a predictions file")
    p.add_argument("--predictions", required=True, help="predictions JSONL, one line per case")
    p.add_argument("--tasks", default=None, help="the exported task directory, e.g. agent_tasks/re2-ob")
    p.add_argument("--answers", default=None, help="answers file (default: <tasks>.answers.jsonl)")
    p.add_argument("--report-chance", action="store_true",
                   help="also print the Avg@5 a random ranking over the candidates would reach, and the lift over it")
    p.add_argument("--per-case-out", default=None, help="write per-case results as JSONL")
    p.add_argument("--json-out", default=None, help="write the summary as JSON")

    p = sub.add_parser("run", help="run the reference LLM agent (LiteLLM) over a task view")
    p.add_argument("--tasks", required=True, help="the exported task directory")
    p.add_argument("--model", required=True, help="LiteLLM model name, e.g. anthropic/claude-sonnet-5-5")
    p.add_argument("--out", required=True, help="predictions file (appended; finished cases are skipped)")
    p.add_argument("--cases", nargs="*", help="only these case ids (default: all)")
    p.add_argument("--max-steps", type=int, default=30, help="model turns per case (default 30)")
    p.add_argument("--max-cost-usd", type=float, default=2.0, help="cost per case before submit is required (default 2.0)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=1, help="cases run in parallel (default 1)")
    p.add_argument("--trajectories", default=None, help="trajectory directory (default: <out>.trajectories)")

    p = sub.add_parser("run-cli", help="run an MCP-capable agent CLI over a task view")
    p.add_argument("--tasks", required=True, help="the exported task directory")
    p.add_argument("--agent", default="claude-code", choices=["claude-code", "custom"])
    p.add_argument("--command", default=None,
                   help="for --agent custom: command template with {prompt}, {mcp_config}, {workdir}, {model}")
    p.add_argument("--model", default=None, help="model passed to the agent CLI")
    p.add_argument("--out", required=True, help="predictions file (appended; finished cases are skipped)")
    p.add_argument("--cases", nargs="*", help="only these case ids (default: all)")
    p.add_argument("--timeout", type=int, default=900, help="seconds per case (default 900)")
    p.add_argument("--workers", type=int, default=1, help="cases run in parallel (default 1)")
    p.add_argument("--trajectories", default=None, help="trajectory directory (default: <out>.trajectories)")

    p = sub.add_parser("mcp", help="serve one case's tools over MCP (stdio)")
    p.add_argument("--case", required=True, help="a case directory of a task view")
    p.add_argument("--record", required=True, help="directory for calls.jsonl and submission.json")

    args = parser.parse_args(argv)

    if args.subcommand == "run":
        from RCAEval.agent.react import run_tasks

        run_tasks(args.tasks, args.model, args.out, cases=args.cases, trajectories=args.trajectories,
                  workers=args.workers, max_steps=args.max_steps, max_cost_usd=args.max_cost_usd,
                  temperature=args.temperature)
        return 0
    if args.subcommand == "run-cli":
        from RCAEval.agent.runner import run_tasks

        run_tasks(args.tasks, args.out, agent=args.agent, command=args.command, model=args.model,
                  timeout=args.timeout, cases=args.cases, trajectories=args.trajectories, workers=args.workers)
        return 0
    if args.subcommand == "mcp":
        from RCAEval.agent.mcp_server import serve

        serve(args.case, args.record)
        return 0

    if args.subcommand == "export":
        task_dir, answers_path = export(args.dataset, out_dir=args.out, copy=args.copy, force=args.force,
                                        data_root=args.data_root)
        print(f"Task view:    {task_dir}  (give this directory, read-only, to the agent)")
        print(f"Answers file: {answers_path}  (keep it out of the agent's reach)")
        return 0

    if args.tasks is None and args.answers is None:
        parser.error("evaluate needs --tasks or --answers")
    try:
        run(args.predictions, tasks_dir=args.tasks, answers_path=args.answers,
            report_chance=args.report_chance, per_case_out=args.per_case_out, json_out=args.json_out)
    except PredictionsError as e:
        print(f"error: {e}", file=sys.stderr)
        for line in e.errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
