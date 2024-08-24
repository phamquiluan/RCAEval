import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causalDiscovery",
    version="0.0.1",
    author="Adobe",
    description="A one-stop Causal Discovery package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayushchauhan/Mixed-Data-Causal-Discovery",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
