# Contributing to Fraud Detection MCP

Thank you for your interest in contributing to the Advanced Fraud Detection MCP.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-mcp
   cd fraud-detection-mcp
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Run Tests**
   ```bash
   python -m pytest tests/ -v --tb=short
   ```

4. **Run Linting**
   ```bash
   ruff check .
   ruff format --check .
   ```

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Write code and tests
3. Run the test suite: `python -m pytest tests/ -v`
4. Run linting: `ruff check . && ruff format --check .`
5. Submit a pull request with clear description

## Ways to Contribute

### Algorithm Development
- New detection methods for the Isolation Forest + Autoencoder pipeline
- Performance optimization of existing analyzers
- Bias reduction in fraud detection models

### Agent Protection Pipeline
- Support for additional agent protocols beyond the 9 currently recognized
- Improved collusion detection algorithms
- Behavioral fingerprinting enhancements

### Defense Compliance
- Additional NITTF behavioral indicators
- SIEM output format improvements
- Compliance module test coverage

### Testing & Quality
- Expand test coverage (currently 88%+)
- Integration tests for MCP tool workflows
- Performance benchmarks on real hardware

### Documentation
- Usage examples and tutorials
- Algorithm documentation
- API reference improvements

## Code Style

- Follow PEP 8 (enforced by `ruff`)
- Use type hints for function signatures
- Use Google-style docstrings
- All MCP tools follow the `_impl` pattern: `tool_name_impl()` for testable logic, `tool_name` for the `@mcp.tool()` wrapper

## Testing Guidelines

- Tests go in `tests/` and follow `test_*.py` naming
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
- Aim for test coverage above 80%
- Import `_impl` functions directly for unit tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=server --cov-report=term-missing

# Run by marker
python -m pytest -m unit
python -m pytest -m integration
```

## Security

- Never log sensitive transaction data
- Use parameterized queries for any database access
- Input sanitization via `security_utils.py`
- Report security vulnerabilities via GitHub Issues (private disclosure)

## Pull Request Checklist

- [ ] Tests pass (`python -m pytest tests/`)
- [ ] Linting passes (`ruff check .`)
- [ ] New features include tests
- [ ] Documentation updated if needed
- [ ] CLAUDE.md updated if architecture changed

## Communication

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions

---

**Ready to contribute?** Check [open issues](https://github.com/marc-shade/fraud-detection-mcp/issues) or open a new one to discuss your idea.
