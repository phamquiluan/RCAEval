from os.path import join, dirname, abspath
from setuptools import setup, find_packages


def read_requirements(path):
    with open(path) as f:
        lines = (line.split("#")[0].strip() for line in f)
        return [line for line in lines if line]


requirements = read_requirements("requirements.txt")
# everything but torch, which only causalrca, run, mscred and cmlp_pagerank need
core_requirements = [r for r in requirements if not r.startswith("torch")]
rcd_requirements = read_requirements("requirements_rcd.lock")

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="RCAEval",
    version="1.10.0",
    packages=find_packages(include=["RCAEval", "RCAEval.*"]),
    include_package_data=True,
    description="RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",  # 3.8 only for the rcd extra; the default env needs 3.11+
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    install_requires=[],
    entry_points={"console_scripts": ["rcaeval-agent=RCAEval.agent.__main__:main"]},
    extras_require={
        "default": requirements,
        "core": core_requirements,
        "rcd": rcd_requirements,
        # EventADL needs nothing beyond the core stack
        "eventadl": core_requirements,
        # ntlr_pagerank / ntlr_randomwalk (NOTEARS low-rank); slow on CPU
        "ntlr": ["gcastle>=1.0.4"],
        # the reference LLM agent (rcaeval-agent run) and the MCP server (mcp, run-cli)
        "agent": core_requirements + ["litellm>=1.106", "mcp>=2.3"],
    },
)
