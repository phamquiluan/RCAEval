# 🕵️ RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems

[![DOI](https://zenodo.org/badge/840137303.svg)](https://doi.org/10.5281/zenodo.13294048)
[![pypi package](https://img.shields.io/pypi/v/RCAEval.svg)](https://pypi.org/project/RCAEval)
[![Downloads](https://static.pepy.tech/personalized-badge/rcaeval?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/rcaeval)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/phamquiluan/RCAEval/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/phamquiluan/RCAEval/tree/main)
[![Build and test](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml)

[FSE 2026](https://arxiv.org/abs/2604.13522)
[WWW 2025](https://dl.acm.org/doi/10.1145/3701716.3715290)
[ASE 2024](https://dl.acm.org/doi/abs/10.1145/3691620.3695065)
[FSE 2024](https://dl.acm.org/doi/full/10.1145/3660805)

RCAEval is an open-source benchmark that offers nine datasets with 735 real failure cases, and an evaluation framework for root cause analysis (RCA) in microservice systems. It includes 15 reproducible baselines covering metric-based, trace-based, and multi-source RCA methods.



<p align="center">
<img width=1000 src= "./docs/readme.jpg"/>
</p>

**Table of Contents** 
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [How-to-use](#how-to-use)
    + [Data format](#data-format)
    + [Basic usage example](#basic-usage-example)
  * [Available Datasets](#available-datasets)
  * [Available Baselines](#available-baselines)
  * [Benchmark](#benchmark)
  * [For TORAI Paper](#for-torai-paper)
  * [For EventADL Paper](#for-eventadl-paper)
  * [Licensing](#licensing)
  * [Acknowledgments](#acknowledgments)
  * [Change Logs](#change-logs)
  * [Citation](#citation)
  * [Contact](#contact)

## Prerequisites

We recommend using machines equipped with at least 8 cores, 16GB RAM, and ~50GB available disk space with Ubuntu 22.04 or Ubuntu 20.04, and **Python3.12**.

## Installation

The `default` environment, which is used for most methods, can be easily installed as follows. Detailed installation instructions for all methods are in [SETUP.md](docs/SETUP.md).


Open your terminal and run the following commands

```bash
sudo apt update -y
sudo apt install -y build-essential \
  libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz
```

Clone RCAEval from GitHub

```bash
git clone https://github.com/phamquiluan/RCAEval.git && cd RCAEval
```

Create virtual environment with Python 3.12 (refer [SETUP.md](docs/SETUP.md) to see how to install Python3.12 on Linux)

```bash
python3.12 -m venv env
. env/bin/activate
```

Install RCAEval using pip

```bash
pip install -e .[default]
```

Or, install RCAEval from PyPI

```bash
# Install RCAEval from PyPI
pip install RCAEval[default]
```

Test the installation

```bash
python -m pytest tests/test.py::test_basic
```

Expected output after running the above command (it takes less than 1 minute)

```bash 
$ pytest tests/test.py::test_basic
============================== test session starts ===============================
platform linux -- Python 3.12.12, pytest-7.3.1, pluggy-1.0.0
rootdir: /home/ubuntu/RCAEval
plugins: dvc-2.57.3, hydra-core-1.3.2
collected 1 item                                                                 

tests/test.py .                                                            [100%]

=============================== 1 passed in 3.16s ================================
```

## How-to-use

### Data format

The telemetry data must be presented as `pandas.DataFrame`. We require the data to have a column named `time` that stores the timestep. A sample of valid data could be downloaded using the `download_data()` or `download_multi_source_data()` method that we will demonstrate shortly below.

### Basic usage example

A basic example to use BARO, a metric-based RCA baseline, to perform RCA are presented as follows,

```python
# You can put the code here to a file named test.py
from RCAEval.e2e import baro
from RCAEval.utility import download_data, read_data

# download a sample data to data.csv
download_data()

# read data from data.csv
data = read_data("data.csv")
anomaly_detected_timestamp = 1692569339

# perform root cause analysis
root_causes = baro(data, anomaly_detected_timestamp)["ranks"]

# print the top 5 root causes
print("Top 5 root causes:", root_causes[:5])
```

Expected output after running the above code (it takes around 1 minute)

```
$ python test.py
Downloading data.csv..: 100%|████████████████████| 570k/570k [00:00<00:00, 19.8MiB/s]
Top 5 root causes: ['emailservice_mem', 'recommendationservice_mem', 'cartservice_mem', 'checkoutservice_latency', 'cartservice_latency']
```

A tutorial of using Multi-source BARO to diagnose failure using multi-source telemetry data (metrics, logs, and traces) is presented in [docs/multi-source-rca-demo.ipynb](docs/multi-source-rca-demo.ipynb). 

A tutorial of using BARO to diagnose code-level faults is presented in [docs/code-level-rca.ipynb](docs/code-level-rca.ipynb).


## Available Datasets

RCAEval benchmark includes nine datasets organized into three benchmark suites (RE1, RE2, RE3), each covering three microservice systems (Online Boutique, Sock Shop, Train Ticket). Together, these datasets feature 735 failure cases with 11 fault types. Each failure case includes annotated root cause service and root cause indicator (e.g., specific metric or log indicating the root cause).

| Dataset | System | Cases | Fault Types | Metrics | Logs | Traces |
|---------|--------|-------|-------------|---------|------|--------|
| RE1-OB | Online Boutique | 125 | cpu, mem, disk, delay, loss | 49-59 | N/A | N/A |
| RE1-SS | Sock Shop | 125 | cpu, mem, disk, delay, loss | 57-63 | N/A | N/A |
| RE1-TT | Train Ticket | 125 | cpu, mem, disk, delay, loss | 198-238 | N/A | N/A |
| RE2-OB | Online Boutique | 90 | cpu, mem, disk, delay, loss, socket | 69-77 | Yes | Yes |
| RE2-SS | Sock Shop | 90 | cpu, mem, disk, delay, loss, socket | 74-82 | Yes | N/A |
| RE2-TT | Train Ticket | 90 | cpu, mem, disk, delay, loss, socket | 340-376 | Yes | Yes |
| RE3-OB | Online Boutique | 30 | f1, f2, f3, f4, f5 | 68-101 | Yes | Yes |
| RE3-SS | Sock Shop | 30 | f1, f2, f3, f4, f5 | 80-107 | Yes | N/A |
| RE3-TT | Train Ticket | 30 | f1, f2, f3, f4, f5 | 294-322 | Yes | Yes |
| TORAI-OB | Online Boutique | 90 | cpu, mem, disk, delay, loss, socket | 69-77 | Yes | Yes |
| TORAI-SS | Sock Shop | 90 | cpu, mem, disk, delay, loss, socket | 74-82 | Yes | N/A |
| TORAI-TT | Train Ticket | 90 | cpu, mem, disk, delay, loss, socket | 340-376 | Yes | Yes |

**RE1 Datasets (375 cases):** Metric-only data supporting metric-based RCA methods. Five fault types (CPU, MEM, DISK, DELAY, LOSS) across five services per system, with five repetitions per fault-service pair.

**RE2 Datasets (270 cases):** Multi-source data (metrics, logs, traces) supporting multi-source RCA methods. Six fault types (adds SOCKET) across five services per system, with three repetitions per fault-service pair.

**RE3 Datasets (90 cases):** Multi-source data focusing on code-level faults (F1-F5). Supports diagnosing code-level faults through telemetry data, e.g., leveraging stack traces in logs or response codes in traces.


### File Structure

Each dataset directory follows the naming convention: `{benchmark}_{service}_{fault}_{instance}`
- `metrics.json`: Time-series metrics data
- `inject_time.txt`: Fault injection timestamp (Unix timestamp)
- `logs.csv`: Log data (RE2 and RE3 only)
- `traces.csv`: Trace data (RE2 and RE3 only)

Our datasets and their description are publicly available with the following information:
- Figshare (**recommended**): [https://figshare.com/articles/dataset/RCAEval_A_Benchmark_for_Root_Cause_Analysis_of_Microservice_Systems/31048672](https://figshare.com/articles/dataset/RCAEval_A_Benchmark_for_Root_Cause_Analysis_of_Microservice_Systems/31048672) (more structured format)
- Zenodo: [https://zenodo.org/records/14590730](https://zenodo.org/records/14590730) (DOI: https://doi.org/10.5281/zenodo.14590730)

We also provide utility functions to download our datasets using Python. The downloaded datasets will be available at directory `data`.

```python
from RCAEval.utility import (
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset,
)

download_re1_dataset()
download_re2_dataset()
download_re3_dataset()
```
<details>
<summary>Expected output after running the above code (it takes half an hour to download and extract the datasets. )</summary>

```
$ python test.py
Downloading RE1.zip..: 100%|█████████████████████| 390M/390M [01:02<00:00, 6.22MiB/s]
Downloading RE2.zip..: 100%|███████████████████| 4.21G/4.21G [11:23<00:00, 6.17MiB/s]
Downloading RE3.zip..: 100%|█████████████████████| 534M/534M [01:29<00:00, 5.97MiB/s]
```
</details>


## Available Baselines 

RCAEval stores all the RCA methods in the `e2e` module (implemented in `RCAEval.e2e`). There are many RCA baselines available: RUN, CausalRCA, CIRCA, RCD, MicroCause, EasyRCA, MSCRED, BARO, 𝜖-Diagnosis, TraceRCA, MicroRank, PDiagnose, Multi-source BARO, Multi-source RCD, Multi-source CIRCA, TORAI, EventADL.

## Benchmark 

We provide a script named `main.py` to assist in reproducing the results from [our RCAEval paper](https://arxiv.org/pdf/2412.17015). This script can be executed using Python with the following syntax: 

```
python main.py [-h] [--dataset DATASET] [--method METHOD]
```

The available options and their descriptions are as follows:

```
options:
  -h, --help            Show this help message and exit
  --dataset DATASET     Choose a dataset. Valid options:
                        [re2-ob, re2-ss, re2-tt, etc.]
  --method METHOD       Choose a method (`causalrca`, `microcause`, `e_diagnosis`, `baro`, `rcd`, `circa`, etc.)
```

For example, in Table 6, BARO achieves Avg@5 of 0.72, 0.99, 1, 0.83, 0.64, and 0.8 for CPU, MEM, DISK, SOCKET, DELAY, LOSS, and AVERAGE on the Train Ticket dataset. To reproduce these results, you can run the following commands:

```bash
python  main.py --method baro --dataset re2-tt
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
$ python  main.py --method baro --dataset re2-tt --length 20
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90/90 [00:45<00:00,  1.98it/s]
--- Evaluation results ---
Avg@5-CPU:   0.72
Avg@5-MEM:   0.99
Avg@5-DISK:  1.0
Avg@5-SOCKET: 0.83
Avg@5-DELAY: 0.63
Avg@5-LOSS:  0.64
---
Avg speed: 0.51
```

We can replace the baro method with other methods (e.g., circa) and substitute re2-tt with other datasets to replicate the corresponding results shown in Table 6. This reproduction process is also integrated into our Continuous Integration (CI) setup. For more details, refer to the [.circleci/config.yml](.circleci/config.yml) file.


## For TORAI Paper

**TORAI Datasets (270 cases):** A processed version of the RE2 datasets, containing pre-aggregated time-series for metrics, logs, and traces (where available). These datasets are used by the TORAI method and are available separately on Figshare: [https://doi.org/10.6084/m9.figshare.31925976](https://doi.org/10.6084/m9.figshare.31925976). 

TORAI requires a separate Python 3.8 environment and its own datasets. For full instructions, see [docs/TORAI.md](docs/TORAI.md) for installation and reproducibility instructions.

```bash
python main.py --method torai --dataset torai-ob --length 10
```

<details>
<summary>Expected output (it takes less than 1 minute)</summary>

```
--- Evaluation results ---
Avg@5-CPU:   0.96
Avg@5-MEM:   0.93
Avg@5-DISK:  1.0
Avg@5-SOCKET: 0.93
Avg@5-DELAY: 0.8
Avg@5-LOSS:  0.84
```
</details>


## For EventADL Paper

EventADL localizes root-cause actors/resources over CloudTrail-style API-call event logs, rather than the metrics/traces/logs used by other RCAEval methods. It requires a separate Python 3.12 environment. For full instructions, see [docs/EVENTADL.md](docs/EVENTADL.md) for installation and usage instructions.

```bash
python main.py --method eventadl --dataset eventadl-falcon
```

## Licensing

This repository includes code from various sources with different licenses. We have included their corresponding LICENSE into the [LICENSES](LICENSES) directory:

- **BARO**: Licensed under the [MIT License](LICENSES/LICENSE-BARO). Original source: [BARO GitHub Repository](https://github.com/phamquiluan/baro/blob/main/LICENSE).
- **CausalRCA**: No License. Original source: [CausalRCA GitHub Repository](https://github.com/AXinx/CausalRCA_code).
- **CIRCA**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE-CIRCA). Original source: [CIRCA GitHub Repository](https://github.com/NetManAIOps/CIRCA/blob/master/LICENSE).
- **E-Diagnosis**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE-E-Diagnosis). Original source: [PyRCA GitHub Repository](https://github.com/salesforce/PyRCA/blob/main/LICENSE).
- **MicroCause**: Licensed under the [Apache License 2.0](LICENSES/LICENSE-MicroCause). Original source: [MicroCause GitHub Repository](https://github.com/PanYicheng/dycause_rca/blob/main/LICENSE).
- **RCD**: Licensed under the [MIT License](LICENSES/LICENSE-RCD). Original source: [RCD GitHub Repository](https://github.com/azamikram/rcd).
- **RUN**: No License. Original source: [RUN GitHub Repository](https://github.com/zmlin1998/RUN).

**For the code implemented by us and for our datasets, we distribute them under the [MIT LICENSE](LICENSE)**.

## Acknowledgments

We would like to express our sincere gratitude to the researchers and developers who created the baselines used in our study. Their work has been instrumental in making this project possible. We deeply appreciate the time, effort, and expertise that have gone into developing and maintaining these resources. This project would not have been feasible without their contributions.

## Change Logs
- [Jul 2026] Added EventADL, an event-log root cause localization method accepted at FSE'26 (only the localization component is ported; the anomaly detection components are not included).
- [Jun 2026] The version of RCAEval with TORAI used in our FSE'26 paper is available in the [fse26 branch](https://github.com/phamquiluan/RCAEval/tree/fse26).
- [Apr 2026] Added TORAI, a multi-source RCA method accepted at FSE'26.
- [Mar 2025] The version of RCAEval used in our WWW'25 paper are available in the [www25 branch](https://github.com/phamquiluan/RCAEval/tree/www25).
- [Dec 2024] The prior version of RCAEval used in our ASE'24 paper are available in the [ase24 branch](https://github.com/phamquiluan/RCAEval/tree/ase24).

## Citation

```bibtex
@inproceedings{pham2026eventadl,
  title={EventADL: Open-Box Anomaly Detection and Localization Framework for Events in Cloud-Based Service Systems},
  author={Luan Pham},
  year={2026},
  journal={Proceedings of the ACM on Software Engineering},
  number={FSE}
}
```

```bibtex
@inproceedings{pham2026torai,
  title={TORAI: Multi-source Root Cause Analysis for Blind Spots in Microservice Service Call Graph}, 
  author={Luan Pham and Huong Ha and Xiuzhen Zhang and Hongyu Zhang},
  year={2026},
  journal={Proceedings of the ACM on Software Engineering},
  number={FSE}
}
```

```bibtex
@inproceedings{pham2025rcaeval,
  title={RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data},
  author={Pham, Luan and Zhang, Hongyu and Ha, Huong and Salim, Flora and Zhang, Xiuzhen},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={777--780},
  year={2025}
}
```

```bibtex
@inproceedings{pham2024root,
  title={Root Cause Analysis for Microservice System based on Causal Inference: How Far Are We?},
  author={Pham, Luan and Ha, Huong and Zhang, Hongyu},
  booktitle={Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering},
  pages={706--715},
  year={2024}
}
```

```bibtex
@inproceedings{pham2024baro,
  title={BARO: Robust root cause analysis for microservices via multivariate bayesian online change point detection},
  author={Pham, Luan and Ha, Huong and Zhang, Hongyu},
  journal={Proceedings of the ACM on Software Engineering},
  volume={1},
  number={FSE},
  pages={2214--2237},
  year={2024},
}
```

## Contact

[phamquiluan\@gmail.com](mailto:phamquiluan@gmail.com?subject=RCAEval)
