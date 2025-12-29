#!/usr/bin/env python3
"""
Test runner for fraud-detection-mcp
Runs comprehensive test suite with coverage reporting
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run all tests with coverage"""
    print("=" * 70)
    print("Fraud Detection MCP - Test Suite")
    print("=" * 70)
    print()

    # Change to project directory
    project_dir = Path(__file__).parent

    # Test discovery and execution
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",                              # Verbose output
        "--tb=short",                      # Shorter traceback format
        "--cov=server",                    # Coverage for server.py
        "--cov-report=term-missing",       # Show missing lines
        "--cov-report=html",               # Generate HTML report
        "--cov-fail-under=80",             # Require 80% coverage
        "-ra",                             # Show summary of all test outcomes
    ]

    print("Running tests...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, cwd=project_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
