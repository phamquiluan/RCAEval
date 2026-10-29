# EventADL: Open-Box Anomaly Detection and Localization for Events in Cloud-Based Service Systems

EventADL is a root cause localization method over CloudTrail-style API-call event logs (rather than metrics/traces). It builds an actor -> resource -> anomaly graph from write/error events and ranks root-cause entities with a temporal-aware random walk.

**Table of Contents**

  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Basic Usage](#basic-usage)
  * [Data Format](#data-format)

## Installation

EventADL runs in RCAEval's default environment (Python 3.14; see the [README](../README.md#installation)):

```bash
python3.14 -m venv env
. env/bin/activate

pip install -e .[default]
# or, without torch, which EventADL does not need:
pip install -e .[eventadl]
```

`requirements_eventadl.lock` is the historical Python 3.12 environment EventADL used before the default environment moved to numpy 2 and networkx 3. It is no longer needed and is kept for reference only.

## Dataset

EventADL ships three datasets: `falcon`, `flask`, `live`, published on [Zenodo](https://zenodo.org/records/19433493). `main.py` downloads the requested dataset automatically on first use, extracting only the event logs and ground truth into `data/eventadl-<name>/` (e.g. `data/eventadl-falcon/`), matching the layout described in [Data Format](#data-format) below. To download manually instead:

```python
from RCAEval.utility import download_eventadl_dataset

download_eventadl_dataset()          # all three datasets
download_eventadl_dataset("falcon")  # a single dataset
```

## Basic Usage

```bash
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
