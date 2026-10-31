"""Score a predictions file against the answers written by the export.

predictions.jsonl holds one line per case:
    {"case_id": "re2-ob-0007", "ranks": ["checkoutservice_cpu", "frontend", ...]}
with optional "reasoning", "model" and "usage" fields. A case without a line is
scored as an empty ranking, i.e. a miss at every k.
"""
import json
from os.path import exists, join

from RCAEval.agent import trajectory
from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.benchmark.scoring import indicator_ranks, service_ranks
from RCAEval.classes.graph import Node

# the order main.py prints per-fault results in; other faults (RE3's f1-f5) follow
FAULT_ORDER = ["cpu", "mem", "disk", "socket", "delay", "loss"]
USAGE_KEYS = ["input_tokens", "output_tokens", "cost_usd", "wall_time_s", "tool_calls"]


class PredictionsError(Exception):
    """The predictions file is invalid; .errors lists every problem found."""

    def __init__(self, errors):
        super().__init__(f"{len(errors)} problem(s) in the predictions file")
        self.errors = errors


def read_jsonl(path):
    with open(path) as f:
        return [(n, line) for n, line in enumerate(f, start=1) if line.strip()]


def load_answers(path):
    return [json.loads(line) for _, line in read_jsonl(path)]


def load_predictions(path, case_ids):
    """case_id -> prediction dict, after checking every line; raises
    PredictionsError listing all malformed, unknown or duplicate lines."""
    known = set(case_ids)
    predictions, errors = {}, []
    for n, line in read_jsonl(path):
        try:
            p = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not valid JSON ({e.msg})")
            continue
        if not isinstance(p, dict):
            errors.append(f"line {n}: expected a JSON object")
            continue
        case_id, ranks = p.get("case_id"), p.get("ranks")
        if not isinstance(case_id, str):
            errors.append(f"line {n}: missing or non-string case_id")
            continue
        if case_id not in known:
            errors.append(f"line {n}: unknown case_id {case_id!r}")
            continue
        if case_id in predictions:
            errors.append(f"line {n}: duplicate case_id {case_id!r}")
            continue
        if not isinstance(ranks, list) or not all(isinstance(r, str) for r in ranks):
            errors.append(f"line {n}: ranks must be a list of strings")
            continue
        usage = p.get("usage")
        if usage is not None and not isinstance(usage, dict):
            errors.append(f"line {n}: usage must be an object")
            continue
        p["ranks"] = [r.strip().lower() for r in ranks if r.strip()]
        predictions[case_id] = p
    if errors:
        raise PredictionsError(errors)
    return predictions


def rank_of(answer, ranks):
    """1-based position of answer in ranks, or None."""
    return ranks.index(answer) + 1 if answer in ranks else None


def evaluate(answers, predictions, max_k=5, predictions_path=None):
    """Score predictions (case_id -> dict) against answers; returns a summary
    dict and the per-case rows. With predictions_path, trajectories referenced
    by the predictions feed the evidence metric."""
    s_all, f_all = Evaluator(max_k=max_k), Evaluator(max_k=max_k)
    s_by_fault = {}
    rows = []
    usage_totals = {}
    evidence = []  # (inspected the root-cause service, service rank 1) per case with a trajectory

    for a in answers:
        p = predictions.get(a["case_id"])
        ranks = p["ranks"] if p else []
        s_ranks = service_ranks(ranks)
        s_answer = Node(a["service"], "unknown")
        s_all.add_case(ranks=s_ranks, answer=s_answer, n_candidates=a["n_candidate_services"],
                       answer_in_candidates=a["answer_in_candidates"])
        s_by_fault.setdefault(a["fault"], Evaluator(max_k=max_k)).add_case(
            ranks=s_ranks, answer=s_answer, n_candidates=a["n_candidate_services"],
            answer_in_candidates=a["answer_in_candidates"])

        f_rank = None
        if a["indicator"] is not None:
            f_ranks = indicator_ranks(ranks)
            f_answer = Node(a["service"], a["indicator"])
            f_all.add_case(ranks=f_ranks, answer=f_answer, n_candidates=a["n_candidate_indicators"],
                           answer_in_candidates=a["indicator_in_candidates"])
            f_rank = rank_of(f_answer, f_ranks)

        for key, value in ((p or {}).get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total, count = usage_totals.get(key, (0, 0))
                usage_totals[key] = (total + value, count + 1)

        inspected = None
        if predictions_path is not None:
            calls = trajectory.load_calls(predictions_path, p)
            if calls is not None:
                inspected = trajectory.inspected(calls, a["service"])
                evidence.append((inspected, rank_of(s_answer, s_ranks) == 1))

        rows.append({
            "case_id": a["case_id"],
            "predicted": p is not None,
            "inspected_root": inspected,
            "service": a["service"],
            "fault": a["fault"],
            "indicator": a["indicator"],
            "top5": ranks[:5],
            "service_rank": rank_of(s_answer, s_ranks),
            "indicator_rank": f_rank,
        })

    def scores(e):
        if e.num == 0:
            return None
        return {
            "n": e.num,
            **{f"AC@{k}": e.accuracy(k) for k in (1, 3, 5)},
            "Avg@5": e.average(5),
            "Chance@5": e.chance_average(5),
            "Lift@5": e.lift(5),
        }

    faults = [f for f in FAULT_ORDER if f in s_by_fault] + sorted(f for f in s_by_fault if f not in FAULT_ORDER)
    summary = {
        "n_cases": len(answers),
        "n_predicted": sum(r["predicted"] for r in rows),
        "service": scores(s_all),
        "indicator": scores(f_all),
        "service_by_fault": {f: scores(s_by_fault[f]) for f in faults},
        "usage": {k: {"total": t, "mean": t / c, "n": c} for k, (t, c) in usage_totals.items()},
        "evidence": evidence_summary(evidence),
    }
    return summary, rows


def evidence_summary(evidence):
    if not evidence:
        return None
    hit = [top1 for seen, top1 in evidence if seen]
    miss = [top1 for seen, top1 in evidence if not seen]
    return {
        "n": len(evidence),
        "inspected_root": len(hit) / len(evidence),
        "AC@1_inspected": sum(hit) / len(hit) if hit else None,
        "AC@1_not_inspected": sum(miss) / len(miss) if miss else None,
    }


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def print_report(summary, dataset, report_chance=False):
    n, n_pred = summary["n_cases"], summary["n_predicted"]
    s, f = summary["service"], summary["indicator"]
    print(f"--- Evaluation results ({dataset}, {n} cases) ---")
    coverage = f"{n_pred}/{n} cases predicted"
    if n_pred < n:
        coverage += f" ({n - n_pred} missing, scored as misses)"
    print("Coverage:".ljust(12), coverage)
    print("".ljust(12), "service".ljust(10), "indicator")
    keys = ["AC@1", "AC@3", "AC@5", "Avg@5"] + (["Chance@5", "Lift@5"] if report_chance else [])
    for key in keys:
        print(f"{key}:".ljust(12), fmt(s[key]).ljust(10), fmt(f[key] if f else None))
    if f is None:
        print("(indicator level is n/a: these faults have no metric indicator)")
    elif f["n"] < n:
        print(f"(indicator level over the {f['n']} cases whose fault has a metric indicator)")

    print("--- Avg@5 by fault (service level) ---")
    for fault, sc in summary["service_by_fault"].items():
        print(f"Avg@5-{fault.upper()}:".ljust(12), fmt(sc["Avg@5"]))
        if report_chance:
            print(f"Chance@5-{fault.upper()}:".ljust(12), fmt(sc["Chance@5"]))
            print(f"Lift@5-{fault.upper()}:".ljust(12), fmt(sc["Lift@5"]))

    ev = summary.get("evidence")
    if ev:
        print(f"--- Evidence ({ev['n']} cases with trajectories) ---")
        print("Inspected@root:".ljust(22), fmt(ev["inspected_root"]),
              "(a tool call targeted the root-cause service before submit)")
        print("AC@1 if inspected:".ljust(22), fmt(ev["AC@1_inspected"]))
        print("AC@1 if not inspected:".ljust(22), fmt(ev["AC@1_not_inspected"]))

    if summary["usage"]:
        print("--- Usage ---")
        keys = [k for k in USAGE_KEYS if k in summary["usage"]] + sorted(k for k in summary["usage"] if k not in USAGE_KEYS)
        for key in keys:
            u = summary["usage"][key]
            print(f"{key}:".ljust(15), f"total {u['total']:.6g}, mean {u['mean']:.6g} over {u['n']} cases")
    print("---")


def answers_path_for(tasks_dir):
    return tasks_dir.rstrip("/\\") + ".answers.jsonl"


def run(predictions_path, tasks_dir=None, answers_path=None, report_chance=False,
        per_case_out=None, json_out=None):
    """CLI body of `rcaeval-agent evaluate`; returns the summary dict."""
    if answers_path is None:
        if tasks_dir is None:
            raise ValueError("give --tasks or --answers")
        answers_path = answers_path_for(tasks_dir)
    if not exists(answers_path):
        raise FileNotFoundError(f"answers file not found: {answers_path}")
    answers = load_answers(answers_path)
    case_ids = [a["case_id"] for a in answers]

    dataset = answers[0]["dataset"] if answers else "?"
    if tasks_dir is not None:
        with open(join(tasks_dir, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["case_ids"] != case_ids:
            raise ValueError(f"{answers_path} does not belong to {tasks_dir} (case ids differ)")

    predictions = load_predictions(predictions_path, case_ids)
    summary, rows = evaluate(answers, predictions, predictions_path=predictions_path)
    print_report(summary, dataset, report_chance=report_chance)

    if per_case_out:
        with open(per_case_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if json_out:
        with open(json_out, "w") as f:
            json.dump({"dataset": dataset, **summary}, f, indent=2)
    return summary
