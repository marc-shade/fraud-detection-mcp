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
    version="1.0.0",
    description="Advanced fraud detection using cutting-edge machine learning and behavioral biometrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="2 Acre Studios",
    author_email="contact@2acrestudios.com",
    url="https://github.com/2-acre-studios/fraud-detection-mcp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastmcp>=0.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "torch>=2.0.0",
        "networkx>=3.0",
        "scipy>=1.10.0",
        "redis>=4.5.0",
        "asyncpg>=0.28.0",
        "cryptography>=40.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "performance": [
            "numba>=0.57.0",
            "cython>=0.29.0",
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
        "Bug Reports": "https://github.com/2-acre-studios/fraud-detection-mcp/issues",
        "Source": "https://github.com/2-acre-studios/fraud-detection-mcp",
        "Documentation": "https://github.com/2-acre-studios/fraud-detection-mcp/wiki",
    },
)
