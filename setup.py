# setup.py
from setuptools import setup, find_packages

setup(
    name="appcorr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "triton",
        "realesrgan",
    ],
)
