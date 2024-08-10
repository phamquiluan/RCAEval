from setuptools import setup

# parse requirements.txt to requirement list
with open("requirements.lock") as f:
    requirements = f.read().splitlines()

with open("requirements_rcd.lock") as f:
    rcd_requirements = f.read().splitlines()

setup(
    name="cfm",
    version="0.0.1",
    packages=["cfm"],
    include_package_data=True,
    install_requires=[],
    extras_require={"rcd": rcd_requirements, "dev": requirements},
)