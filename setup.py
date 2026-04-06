#!/usr/bin/env python3
"""
Setup script for Advanced Fraud Detection MCP
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

setup(
    name="fraud-detection-mcp",
    version="2.4.0",
    description="Fraud detection MCP server with behavioral biometrics, ML anomaly detection, and agent transaction protection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Marc Shade",
    author_email="contact@2acrestudios.com",
    url="https://github.com/marc-shade/fraud-detection-mcp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastmcp>=0.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "torch>=2.0.0",
        "networkx>=3.0",
        "scipy>=1.10.0",
        "category-encoders>=2.6.0",
        "shap>=0.43.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "prometheus-client>=0.16.0",
        "structlog>=23.0.0",
        "python-jose[cryptography]>=3.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "bandit>=1.7.5",
        ],
        "training": [
            "xgboost>=1.7.0",
            "imbalanced-learn>=0.11.0",
            "optuna>=3.4.0",
            "mlflow>=2.8.0",
        ],
        "gnn": [
            "torch-geometric>=2.4.0",
            "torchvision>=0.15.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detection-mcp=server:main",
            "fraud-detect=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="fraud detection, machine learning, behavioral biometrics, anomaly detection, mcp",
    project_urls={
        "Bug Reports": "https://github.com/marc-shade/fraud-detection-mcp/issues",
        "Source": "https://github.com/marc-shade/fraud-detection-mcp",
    },
)
