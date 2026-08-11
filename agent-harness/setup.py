#!/usr/bin/env python3
"""Setup script for cli-anything-kronos

Install (dev mode):
    pip install -e .

Build:
    python -m build

Publish:
    twine upload dist/*
"""

from pathlib import Path
from setuptools import setup, find_namespace_packages

ROOT = Path(__file__).parent
README = ROOT / "cli_anything" / "kronos" / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="cli-anything-kronos",
    version="1.0.0",
    description="Agent-native CLI for Kronos — financial K-line prediction foundation model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="cli-anything contributors",
    url="https://github.com/HKUDS/CLI-Anything",

    project_urls={
        "Source": "https://github.com/HKUDS/CLI-Anything",
        "Tracker": "https://github.com/HKUDS/CLI-Anything/issues",
    },

    license="Apache-2.0",

    packages=find_namespace_packages(include=("cli_anything.*",)),

    python_requires=">=3.10",

    install_requires=[
        "click>=8.1",
        "prompt-toolkit>=3.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "torch>=2.0",
        "einops>=0.7",
        "huggingface-hub>=0.33",
        "safetensors>=0.6",
        "tqdm>=4.66",
        "pyyaml>=6.0",
    ],

    extras_require={
        "dev": [
            "pytest>=7",
            "pytest-cov>=4",
        ],
    },

    entry_points={
        "console_scripts": [
            "cli-anything-kronos=cli_anything.kronos.kronos_cli:main",
        ],
    },

    package_data={
        "cli_anything.kronos": ["README.md", "skills/*.md"],
    },

    zip_safe=False,
)
