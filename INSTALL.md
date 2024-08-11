# Installation

We assume the users already satisfy the [REQUIREMENTS.md](REQUIREMENTS.md) and ready to have Python 3.10 installed. Then, users can install RCAEval from PyPI or build RCAEval from source. In addition, users who familiar with Continuous Integration (CI) can take a look at our [build-and-test.yml](.github/workflows/build-and-test.yml) configuration to see how we install and test our RCAEval on Linux and Windows machine from Python 3.7 to 3.12.

**Table of contents**

  * [Installation Instruction](#installation-instruction)
    + [Install Python 3.10](#install-python-310)
    + [Clone RCAEval from GitHub](#clone-rcaeval-from-github)
    + [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)
    + [Install RCAEval from PyPI or Build RCAEval from source](#install-rcaeval-from-pypi-or-build-rcaeval-from-source)
  * [Test the installation](#test-the-installation)
  * [Basic usage example](#basic-usage-example)
## Installation Instruction

### Install Python 3.10

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv
```

### Clone RCAEval from GitHub


```bash
git clone https://github.com/phamquiluan/RCAEval.git && cd RCAEval
```


### Create and activate a virtual environment

```
# create a virtual environment
python3.10 -m venv env-dev

# activate the environment
. env-dev/bin/activate
```

### Install RCAEval from PyPI or Build RCAEval from source

```bash
# install RCAEval from PyPI
pip install RCAEval

# build RCAEval from source
pip install -e .
```

## Test the installation

Users can perform testing using the following commands:

```bash
pytest tests/test.py
```

<details>
<summary>The expected output would look like this</summary>

```bash

(ins)(env) luan@machine:~/ws/RCAEval$ pytest tests/test.py 
============================================ test session starts =============================================
platform linux -- Python 3.10.13, pytest-7.4.0, pluggy-1.3.0
rootdir: /home/luan/ws/RCAEval
collected 4 items                                                                                            

tests/test.py ....                                                                                     [100%]

======================================= 4 passed in 501.44s (0:08:21) ========================================
(ins)(env) luan@machine:~/ws/RCAEval$ 

```
</details>

## Basic usage example

Users can check a basic usage example of RCAEval in the [README.md#basic-usage-example](README.md#how-to-use) section.
