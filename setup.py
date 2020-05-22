# File: setup.py
# File Created: Wednesday, 2nd January 2019 3:55:50 pm
# Author: Steven Atkinson (212726320@ge.com)

import setuptools

from setuptools import setup, find_packages

description = (
    "Physics-Informed Artificial Intelligence Research "
    + "Assistant for Theory Extraction"
)

dependencies = [
    "numpy",
    "scikit-learn",
    "pandas",
    "deap",
    "matplotlib",  # For demo notebooks
    "jupyter",
    "ipykernel",
    "pytest",
    "seaborn",
    "tqdm",
    "gptorch",
]

try:
    import torch
except ImportError as e:
    print("Error importing PyTorch:\n{}\n".format(e))
    print("If Anaconda is available, we recommend:")
    print("$ conda install torch torchvision -c pytorch")
    print("See https://pytorch.org/ for additional installation options.")

packages = find_packages(".")

setup(
    name="pirate",
    version="0.1.0",
    description=description,
    url="https://github.com/sdatkinson/PIRATE",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    license="MIT",
    install_requires=dependencies,
    packages=packages,
    include_package_data=True,
    zip_safe=False,
)
