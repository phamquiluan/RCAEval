from setuptools import setup

# parse requirements.txt to requirement list
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_rcd.lock") as f:
    rcd_requirements = f.read().splitlines()

setup(
    name="RCAEval",
    version="0.0.4",
    packages=["RCAEval"],
    include_package_data=True,
    install_requires=[],
    extras_require={"rcd": rcd_requirements, "dev": requirements},
)