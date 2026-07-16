# EventADL: Open-Box Anomaly Detection and Localization for Events in Cloud-Based Service Systems

EventADL is a root cause localization method over CloudTrail-style API-call event logs (rather than metrics/traces). It builds an actor -> resource -> anomaly graph from write/error events and ranks root-cause entities with a temporal-aware random walk.

**Table of Contents**

  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Basic Usage](#basic-usage)
  * [Data Format](#data-format)

## Installation

EventADL uses Python 3.12 with a dependency set (`numpy==2.2.5`, `networkx==3.4.2`, ...) that conflicts with RCAEval's default environment (`numpy<2`, `networkx==2.5`), so it lives in its own virtual environment.

```bash
# create and activate the virtual environment
python3.12 -m venv .venv-eventadl
source .venv-eventadl/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements_eventadl.lock

# install RCAEval
pip install -e .[eventadl]
```

## Dataset

EventADL ships three datasets: `falcon`, `flask`, `live`. Like TORAI, the data is expected to already be present locally -- place each dataset under `data/eventadl-<name>/` (e.g. `data/eventadl-falcon/`) before running `main.py`, matching the layout described in [Data Format](#data-format) below.

## Basic Usage

```bash
source .venv-eventadl/bin/activate

python main.py --method eventadl --dataset eventadl-falcon
python main.py --method eventadl --dataset eventadl-flask
python main.py --method eventadl --dataset eventadl-live
```

## Data Format

```
data/eventadl-falcon/
  events/
    {id}.json        # raw CloudTrail events for test case {id}
  rca.json            # {"test_cases": [{"id": ..., "ground_truth": "<actor/resource ARN>"}, ...]}
```

`RCAEval.e2e.eventadl.eventadl()` takes a dict `{"events": [...]}` for a single test case, builds the actor -> resource -> anomaly graph, and returns `{"ranks": [...]}`, a ranked list of root-cause entities, following the same contract as every other RCAEval method (see `docs/EXTENDING.md`).
