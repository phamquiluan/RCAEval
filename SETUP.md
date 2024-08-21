# Setup

We assume the users are on Ubuntu machine and ready to have Python 3.10 installed. Then, users can follow this instruction to install RCAEval using pip. In addition, users who familiar with Continuous Integration (CI) can take a look at our [.github/workflows](.github/workflows) and [.circleci](.circleci) to see how we install, test, and reproduce our RCAEval on different CI platforms including GitHub Action and CircleCI.

**Table of contents**

  * [Setup Instruction](#setup-instruction)
    + [Install Python 3.10](#install-python-310)
    + [Clone RCAEval from GitHub](#clone-rcaeval-from-github)
    + [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)
    + [Install RCAEval from PyPI or Build RCAEval from source](#install-rcaeval-from-pypi-or-build-rcaeval-from-source)
  * [Test the installation](#test-the-installation)
  * [Basic usage example](#basic-usage-example)


## Install Python 3.10 and required packages

```bash
# install Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv

# install required packages
sudo apt install -y build-essential \
  libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz
```

## Clone RCAEval from GitHub


```bash
git clone https://github.com/phamquiluan/RCAEval.git && cd RCAEval
```


## Create and activate a virtual environment

```
# create a virtual environment
python3.10 -m venv env

# activate the environment
. env/bin/activate
```

### Install RCAEval from PyPI or Build RCAEval from source

```bash
# install RCAEval from PyPI
pip install RCAEval

# build RCAEval from source
pip install -e .[default]
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
