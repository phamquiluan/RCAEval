from os.path import join, dirname, abspath
from setuptools import setup

VERSION = "0.0.11"

cwd = dirname(abspath(__file__))
version_path = join(cwd, "RCAEval", "version.py")
with open(version_path, "w") as ref:
    ref.write(f"__version__ = '{VERSION}'\n")

# parse requirements.txt to requirement list
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_rcd.lock") as f:
    rcd_requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name="RCAEval",
    version=VERSION,
    packages=["RCAEval"],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    extras_require={"rcd": rcd_requirements, "default": requirements},
)
