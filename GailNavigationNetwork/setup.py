#!/usr/bin/env python3
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="GailNavigationNetwork",
    version="0.1.0",
    author="Malika Navaratna",
    python_requires=">=3.8.10",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_namespace_packages(),
)
