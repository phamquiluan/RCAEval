# STATUS 

We apply for the Available and Reusable badges.
We believe our artifact meets the requirements for the two badges in the [artifact submission guideline](https://conf.researchr.org/track/ase-2024/ase-2024-artifact-evaluation-track).

* **Available**: Our Software and Dataset artifacts are "Available" as they are publicly accessible through the following repository:
  * Software Artifacts on GitHub: [https://github.com/phamquiluan/RCAEval/tree/0.0.8](https://github.com/phamquiluan/RCAEval/tree/0.0.8)
  * Software Artifacts on Zenodo (**immutable**): [https://zenodo.org/doi/10.5281/zenodo.13294048](https://zenodo.org/doi/10.5281/zenodo.13294048)
  * Dataset Artifacts on Zenodo: TBD


* **Reusable**: Our artifact is "Reusable" (and also "Functional") as we meet the following five criteria (the first four are the criteria for "Functional" badge) mentioned in the [artifact submission guideline](https://conf.researchr.org/track/ase-2024/ase-2024-artifact-evaluation-track).
  * _Documented_: We provide the following documents necessary for using our artifact: (1) [README.md](README.md), (2) [REQUIREMENTS.md](REQUIREMENTS.md), (3) [STATUS.md](STATUS.md), (4) [LICENSE](LICENSE), (5) [INSTALL.md](INSTALL.md), and (6) a copy of the accepted paper.
  * _Consistent & Complete_: We provide concrete steps for reproducing the main experimental results in the paper using our public artifacts, as described in Section [README.md#reproducibility](README.md#reproducibility).
  * _Exercisable_: We provide three tutorials at directory [./tutorials](tutorials) in the Jupyter Notebook format, which can also be opened using Google Colab. In addition, we also add unit tests for main functions at [./tests/test.py](./tests/test.py) file.
  * _Reusable for Future Researches_: We have published our RCAEval as a PyPI package [RCAEval](https://pypi.org/project/RCAEval) and provided concrete instructions to install and use RCAEval (as described above). In addition, we also provide unit-tests and adopt Continuous Integration (CI) tools to perform testing automatically in daily manner to detect and prevent code rot, see [.circleci/config.yml](.circleci/config.yml). Thus, we believe that other researchers can use RCAEval in their own research.
