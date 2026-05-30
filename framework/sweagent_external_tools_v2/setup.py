"""
Setup script for Agent-mem module.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agent-mem",
    version="0.1.0",
    author="Agent-mem Team",
    description="Work-Experience Memory module for SWE-agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/agent-mem",
    packages=find_packages(include=["agent_mem", "agent_mem.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "embedding": [
            "sentence-transformers>=2.2.0",
            "numpy>=1.24.0",
        ],
        "graph": [
            "networkx>=3.0",
            "neo4j>=5.0.0",  # Optional for advanced storage
        ],
    },
    entry_points={
        "console_scripts": [
            "agent-mem=agent_mem_main:main",
        ],
    },
    package_data={
        "agent_mem": ["config/*.yaml", "config/*.json"],
    },
)