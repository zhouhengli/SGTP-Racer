import os

import setuptools
from setuptools import setup, find_packages

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="SGTP_Racer",
    version="1.0.0",
    author="anonymous",
    packages=find_packages(),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9.25",
        "License :: Free for non-commercial use",
    ],
    license="MIT",
)
