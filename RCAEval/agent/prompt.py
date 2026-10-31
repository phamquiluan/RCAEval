"""The canonical task prompt, shared by the reference agent and the CLI runner,
so every agent is compared on the same instructions."""

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """You are an SRE investigating a failure in a microservice system. A fault was \
injected into one service at a known time; your job is to find the root cause.

You see the system only through the tools. Work like this:
1. Call describe_case to see the candidate services and which telemetry exists.
2. Find what changed after the inject time: call rank_anomalies on each available modality, then \
confirm with get_metric, search_logs, trace_summary and call_graph. Failures propagate along calls, \
so prefer the service where the change starts over services that only feel its effects.
3. Call submit with 5 to 10 candidates ranked most likely first. Each candidate is a service from \
candidate_services or a <service>_<indicator> metric name such as cartservice_cpu; the indicator \
(cpu, mem, diskio, socket, latency, ...) says what kind of fault you believe it is, so name the \
metric that best shows the fault first.

Tool calls are limited, so be efficient. Your investigation ends when submit is accepted."""


def case_prompt(task):
    telemetry = ", ".join(sorted(task["files"]))
    return (
        f"Case {task['case_id']}: a fault was injected into the {task['system']} system at "
        f"{task['inject_time']} (Unix seconds). The standard window is {task['standard_window']['start']} "
        f"to {task['standard_window']['end']}. Available telemetry: {telemetry}. "
        "Investigate with the tools and submit your ranking."
    )


def full_prompt(task):
    """System and case prompt in one message, for agent CLIs that take a single prompt."""
    return SYSTEM_PROMPT + "\n\n" + case_prompt(task)
