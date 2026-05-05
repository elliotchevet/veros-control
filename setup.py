#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages
from codecs import open
import sys
import os

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)


CLASSIFIERS = """
Development Status :: 1 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: IMT-Atlantique License
Programming Language :: Python :: 3.12.12
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

INSTALL_REQUIRES = [
    "veros>=1.6"
]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="veros_control",
    license="MIT",
    author="Elliot Chevet",
    author_email="elliot.chevet@imt-atlantique.fr",
    keywords="oceanography python machine-learning ocean-model veros plugin",
    description="Online data-driven bias correction for Veros",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elliotchevet/veros-control",
    python_requires=">=3.12.12",
    packages=find_packages(),
    version="0.1.0",
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "veros.setup_dirs": [
            "control = veros_control.setup"
        ]
    },
    package_data={"veros_control": ["setup/*/assets.json"]},
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
)
