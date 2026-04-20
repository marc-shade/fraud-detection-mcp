# Test Suite

831 tests across 28 test files. Coverage: 90%+ of `server.py`.

## Running

```bash
# Full suite (runs coverage automatically via pytest.ini)
python -m pytest tests/

# Single file
python -m pytest tests/test_transaction_analysis.py -v

# By marker
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m behavioral
python -m pytest -m transaction
python -m pytest -m explainability
python -m pytest -m synthetic
python -m pytest -m benchmark
python -m pytest -m security
python -m pytest -m velocity

# Without coverage (faster iteration)
python -m pytest tests/ --no-cov
```

## Markers

See `pytest.ini` for the full list. Most commonly used:

| Marker | Scope |
|--------|-------|
| `unit` | Individual function tests |
| `integration` | End-to-end workflows |
| `behavioral` | Keystroke, mouse, touch biometrics |
| `transaction` | 46-feature pipeline, ensemble scoring |
| `network` | Graph centrality, fraud rings |
| `explainability` | SHAP explanations |
| `synthetic` | Synthetic dataset generation |
| `benchmark` | Performance benchmarks |
| `security` | Input sanitization, rate limiting |
| `velocity` | User history, velocity analysis |

## Fixtures

`tests/conftest.py` provides shared fixtures and sample data. Tests import
`*_impl` functions directly from `server.py` — the MCP tool wrappers delegate
to these, which keeps tests decoupled from the FastMCP decorator.

## Coverage floor

CI and local `pytest.ini` both enforce `--cov-fail-under=60`. Actual coverage
is substantially higher; the floor is a regression guard, not a target.
