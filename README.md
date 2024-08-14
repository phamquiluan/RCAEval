# 🕵️ Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?

[![DOI](https://zenodo.org/badge/840137303.svg)](https://zenodo.org/doi/10.5281/zenodo.13294048)
[![pypi package](https://img.shields.io/pypi/v/RCAEval.svg)](https://pypi.org/project/RCAEval)
[![Build and test](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml)

This repository includes artifacts for reuse and reproduction of experimental results presented in our ASE'24 paper titled "Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?".

**Table of Contents** 
  * [Installation](#installation)
  * [How-to-use](#how-to-use)
    + [Data format](#data-format)
    + [Basic usage example](#basic-usage-example)
  * [Reproducibility](#reproducibility)
    + [Reproduce RQ1 - Causal Discovery Performance](#reproduce-rq1---causal-discovery-performance)
    + [Reproduce RQ2 - Root Cause Analysis Performance](#reproduce-rq2---root-cause-analysis-performance)
    + [Reproduce RQ3 - Efficiency](#reproduce-rq3---efficiency)
    + [Reproduce RQ4 - Data lengths](#reproduce-rq4---data-lengths)
  * [Download Experimental Data - Supplementary Material](#download-experimental-data---supplementary-material)
  * [Licensing](#licensing)
  * [Acknowledgments](#acknowledgments)


## Installation

We maintain 3 separate environments (`default`, `rcd` and `fges`) due to the dependency constraints of certain methods. Detailed installation instructions for all environments can be found in [INSTALL.md](INSTALL.md). The `default` environment, are used for most methods, can be easily installed as follows.


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

Install RCAEval from PyPI

```bash
# Install RCAEval from PyPI
pip install RCAEval[default]
```

OR, build RCAEval from source

```bash
pip install pip==20.0.2
pip install -e .[default]
```

## How-to-use

### Data format

The data must be a `pandas.DataFrame` that consists of multivariate time series metrics data. We require the data to have a column named `time` that stores the timestep. Each other column stores a time series for metrics data with the name format of `<service>_<metric>`. For example, the column `cart_cpu` stores the CPU utilization of service `cart`. A sample of valid data could be downloaded using the `download_data()` method that we will demonstrate shortly below.



### Basic usage example

RCAEval stores all the RCA methods in the `e2e` module (implemented in `RCAEval.e2e`). The basic sample commands to perform root cause analysis using RCAEval are presented as follows,

```python
TBD
```




## Reproducibility

### Reproduce RQ1 - Causal Discovery Performance

We provide a script named `rq1.py` to assist in reproducing the RQ1 results from our paper. This script can be executed using Python with the following syntax: 

```
python rq1.py [-h] [--dataset] [--method] [--length LENGTH]
```

The available options and their descriptions are as follows:

```
options:
  -h, --help            Show this help message and exit
  --dataset             Choose a dataset. Valid options:
                        [circa10, circa50, rcd10, rcd50, causil10, causil50]
  --method METHOD       Choose a method (e.g. `pc`, `fci`, etc.)
  --length LENGTH       Specify the length of the time series (used for RQ4)
```

For example, in Table 3, PC achieves F1, F1-S, and SHD scores of 0.49, 0.65, and 16 on the CIRCA 10 dataset. To reproduce these results, you can run the following commands:

```bash
python rq1.py --dataset circa10 --method pc
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
F1:   0.49
F1-S: 0.65
SHD:  16
Avg speed: 0.08
```

We can replace the pc method with other methods (e.g., fci, granger) and substitute circa10 with other datasets to replicate the corresponding results shown in Table 3. This reproduction process is also integrated into our Continuous Integration (CI) setup. For more details, refer to the [.github/workflows/reproduce.yml](.github/workflows/reproduce.yml) file.


### Reproduce RQ2 - Root Cause Analysis Performance

We provide a script named `rq2.py` to assist in reproducing the RQ2 results from our paper. This script can be executed using Python with the following syntax: 

```
python rq2.py [-h] [--dataset] [--method] [--tbias TBIAS] [--length LENGTH] 
```

The available options and their descriptions are as follows:

```
options:
  -h, --help            Show this help message and exit
  --dataset             Choose a dataset. Valid options:
                        [online-boutique, sock-shop-1, sock-shop-2, train-ticket,
                         circa10, circa50, rcd10, rcd50, causil10, causil50]
  --method METHOD       Choose a method (e.g. `nsigma`, `baro`, etc.)
  --tdelta              Specify $t_delta$ to simulate delay in anomaly detection
  --length LENGTH       Specify the length of the time series (used for RQ4)
```

For example, in Table 5, BARO [ $t_\Delta = 0$ ] achieves Avg@5 of 0.97, 1, 0.91, 0.98, and 0.67 for CPU, MEM, DISK, DELAY, and LOSS fault types on the Online Boutique dataset. To reproduce these results, you can run the following commands:

```bash
python rq2.py --dataset online-boutique --method baro 
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
--- Evaluation results ---
Avg@5-CPU:   0.97
Avg@5-MEM:   1.0
Avg@5-DISK:  0.91
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.67
---
Avg speed: 0.07
```

As presented in Table 5, BARO [ $t_\Delta = 60$ ] achieves Avg@5 of 0.94, 0.99, 0.87, 0.99, and 0.6 for CPU, MEM, DISK, DELAY, and LOSS fault types on the Online Boutique dataset. To reproduce these results, you can run the following commands:

```bash
python rq2.py --dataset online-boutique --method baro --tdelta 60
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
--- Evaluation results ---
Avg@5-CPU:   0.94
Avg@5-MEM:   0.99
Avg@5-DISK:  0.87
Avg@5-DELAY: 0.99
Avg@5-LOSS:  0.6
---
Avg speed: 0.07
```

We can replace the baro method with other methods (e.g., nsigma, fci_randomwalk) and substitute online-boutique with other datasets to replicate the corresponding results shown in Table 5. This reproduction process is also integrated into our Continuous Integration (CI) setup. For more details, refer to the [.github/workflows/reproduce.yml](.github/workflows/reproduce.yml) file.


### Reproduce RQ3 - Efficiency

The running time of each method is recorded in the scripts of RQ1 and RQ2 as we described and presented above.

### Reproduce RQ4 - Data lengths

Our RQ4 relies on the scripts of RQ1 and RQ2 as we described and presented above, with the option `--length`. 

As presented in Figure 3, BARO maintains stable accuracy on the Online Boutique dataset when we vary the data length from 60 to 600. To reproduce these results, for example, you can run the following Bash script:

```bash
for length in 60 120 180 240 300 360 420 480 540 600; do
    python rq2.py --dataset online-boutique --method baro --length $length
done
```


<details>
<summary>Expected output after running the above code (it takes few minutes)</summary>

<br />

The output list presents the `Avg@5` scores when we vary the data length. You can see that BARO can maintain a stable performance.


```
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.64it/s]
--- Evaluation results ---
Avg@5-CPU:   0.84
Avg@5-MEM:   0.98
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 120 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.37it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 180 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.46it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 240 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.47it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 300 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.40it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 360 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.42it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 420 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.53it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 480 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.37it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 540 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.50it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 600 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.49it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
```
</details>


## Download Experimental Data - Supplementary Material

Our datasets and their description are publicly available in Zenodo repository with the following information:
- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13305663.svg)](https://doi.org/10.5281/zenodo.13305663)
- Dataset URL: https://zenodo.org/records/13305663

In addition, you can download our supplementary material and all the data we used for this research from this [Google Drive](https://drive.google.com/drive/folders/1BG2P1ETEyKW62dU0I1ZpE64Ng9fy5_ju?usp=sharing).

## Licensing

This repository includes code from various sources with different licenses:

- **CIRCA**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_CIRCA). Original source: [CIRCA GitHub Repository](https://github.com/NetManAIOps/CIRCA).
- **RCD**: Licensed under the [MIT License](LICENSES/LICENSE_RCD). Original source: [RCD GitHub Repository](https://github.com/azamikram/rcd).
- **E-Diagnosis**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_E-Diagnosis). Original source: [PyRCA GitHub Repository](https://github.com/salesforce/PyRCA).
- **CausalAI**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_CausalAI). Original source: [CausalAI GitHub Repository](https://github.com/salesforce/causalai).
- **MicroCause**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_MicroCause). Original source: [MicroCause GitHub Repository](https://github.com/PanYicheng/dycause_rca).

We have included their corresponding LICENSE into the [LICENSES](LICENSES) directory. For the code implemented by us, we distribute them under the [MIT LICENSE](LICENSE).

## Acknowledgments

We would like to express our sincere gratitude to the researchers and developers who created the baselines used in our study. Their work has been instrumental in making this project possible. We deeply appreciate the time, effort, and expertise that have gone into developing and maintaining these resources. This project would not have been feasible without their contributions.
