import random
from collections import Counter, defaultdict

import networkx as nx

from RCAEval.eventadl.actor import get_actor
from RCAEval.eventadl.extract_resources import extract_resources


def _temporal_aware_random_walk_rca(G, num_walks=100, walk_length=10, from_anomalies=True):
    """Rank root-cause nodes by walking backwards from anomaly nodes through
    the actor -> resource -> anomaly graph, preferring edges that occurred
    before the anomaly they lead to."""
    anomaly_nodes = [node for node in G.nodes() if node.startswith("anom-")] if from_anomalies else []

    if not anomaly_nodes:
        start_nodes = list(G.nodes())
        if not start_nodes:
            return []

        visit_counts = Counter()
        for _ in range(num_walks):
            current_node = random.choice(start_nodes)
            for _ in range(walk_length):
                visit_counts[current_node] += 1
                predecessors = list(G.predecessors(current_node))
                if not predecessors:
                    current_node = random.choice(start_nodes)
                    continue
                current_node = random.choice(predecessors)

        return sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)

    anomaly_times = {}
    for node in G.nodes():
        if node in anomaly_nodes:
            continue
        for neighbor in G.neighbors(node):
            if neighbor not in anomaly_nodes:
                continue
            for _, edge_data in G[node][neighbor].items():
                if "event_time" in edge_data:
                    anomaly_time = edge_data["event_time"][0]
                    if neighbor not in anomaly_times or anomaly_time < anomaly_times[neighbor]:
                        anomaly_times[neighbor] = anomaly_time

    visit_counts = Counter()
    for _ in range(num_walks):
        current_node = random.choice(anomaly_nodes)
        anomaly_time = anomaly_times.get(current_node)

        for _ in range(walk_length):
            visit_counts[current_node] += 1

            valid_predecessors = []
            for predecessor in G.predecessors(current_node):
                for _, edge_data in G[predecessor][current_node].items():
                    edge_times = edge_data.get("event_times") or edge_data.get("event_time")
                    if edge_times and anomaly_time and any(t <= anomaly_time for t in edge_times):
                        valid_predecessors.append(predecessor)
                        break

            if not valid_predecessors:
                current_node = random.choice(anomaly_nodes)
                anomaly_time = anomaly_times.get(current_node)
                continue

            current_node = random.choice(valid_predecessors)

    root_cause_scores = [
        (node, score) for node, score in visit_counts.items() if node not in anomaly_nodes
    ]
    return sorted(root_cause_scores, key=lambda x: x[1], reverse=True)


def _build_graph(events):
    G = nx.MultiDiGraph()
    edge_lookup = defaultdict(lambda: defaultdict(dict))  # actor -> resource -> action -> edge_data

    write_events = [e for e in events if not e.get("readOnly", False)]
    for event in write_events:
        action = event.get("eventName")
        event_time = event.get("eventTime")

        try:
            actor = get_actor(event)
            resources = extract_resources(event)
        except (ValueError, KeyError):
            continue

        for resource in resources or []:
            res = resource["arn"]
            if action in edge_lookup[actor][res]:
                edge_lookup[actor][res][action]["event_times"].append(event_time)
            else:
                edge_data = {"action": action, "event_times": [event_time]}
                edge_lookup[actor][res][action] = edge_data
                G.add_edge(actor, res, **edge_data)

    anomalies = [e for e in events if e.get("errorCode")]
    for event in anomalies:
        event_id = event.get("eventID")
        event_time = event.get("eventTime")

        try:
            resources = extract_resources(event)
        except (ValueError, KeyError):
            continue

        for resource in resources or []:
            res = resource["arn"]
            G.add_edge(res, f"anom-{event_id[-4:]}", event_time=[event_time])

    return G


def _has_outgoing_action(G, node):
    """Only nodes that act as an actor toward some resource (an outgoing
    edge carrying an `action`) are valid root-cause candidates — resource
    nodes only ever receive edges and must be excluded from the ranking."""
    for neighbor in G.neighbors(node):
        for _, edge_data in G[node][neighbor].items():
            if "action" in edge_data:
                return True
    return False


def eventadl(data, inject_time=None, dataset=None, sli=None, anomalies=None, num_walk=100, **kwargs):
    """EventADL root cause localization over CloudTrail-style events.

    `data` is expected to be a dict with an "events" key holding the list of
    raw CloudTrail event dicts for one test case (see main.py's eventadl data
    loading branch). Anomalous events are identified via each event's own
    `errorCode`, and the final ranked root-cause entities are returned,
    matching the standard `{"ranks": [...]}` contract used by every other
    RCAEval method.
    """
    events = data["events"] if isinstance(data, dict) else data

    G = _build_graph(events)
    ranked = _temporal_aware_random_walk_rca(G, num_walks=num_walk)
    ranks = [node for node, _ in ranked if _has_outgoing_action(G, node)]
    if not ranks:
        # The from-anomalies walk can surface only resource nodes (no
        # outgoing "action" edges), leaving nothing after filtering; retry
        # without anchoring to anomaly nodes, same as the reference impl.
        ranked = _temporal_aware_random_walk_rca(G, num_walks=num_walk, from_anomalies=False)
        ranks = [node for node, _ in ranked if _has_outgoing_action(G, node)]
    return {"node_names": list(G.nodes()), "ranks": ranks}
