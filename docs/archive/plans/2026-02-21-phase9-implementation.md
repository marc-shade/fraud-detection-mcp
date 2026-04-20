# Phase 9: Synthetic Data Generation and Dataset Analysis Integration

## Overview
Integrate `SyntheticDataIntegration` from `integration.py` into the active MCP server (`server.py`), adding synthetic dataset generation and stored-dataset analysis as MCP tools with graceful degradation, input validation, and performance metrics.

## Tasks

### Task 1: Remove circular imports and clean up integration.py
- Removed `from server import transaction_analyzer, behavioral_analyzer, network_analyzer` (lines 24-28) that would cause circular imports when importing from server.py
- Removed standalone `FastMCP` server section, `@mcp.tool()` wrappers, `calculate_performance_metrics` function, and `if __name__` block
- Removed unused imports (`asyncio`, `json`, `FastMCP`, `Optional`)
- Commit: `refactor: Remove circular imports and MCP section from integration.py`

### Task 2: Add SYNTHETIC_DATA_AVAILABLE flag with graceful degradation
- Added try/except import of `SyntheticDataIntegration` from `integration` with `SYNTHETIC_DATA_AVAILABLE` flag
- Follows existing pattern (AUTOENCODER_AVAILABLE, TRAINING_AVAILABLE, EXPLAINABILITY_AVAILABLE)
- Initialized `synthetic_data_integration` singleton with error handling
- Commit: `feat: Add SYNTHETIC_DATA_AVAILABLE flag with graceful degradation`

### Task 3: Add generate_synthetic_dataset_impl and analyze_dataset_impl
- `generate_synthetic_dataset_impl`: Input validation (num_transactions, fraud_percentage, output_format), delegates to `SyntheticDataIntegration.generate_comprehensive_test_dataset()`
- `analyze_dataset_impl`: Reads CSV/JSON datasets, runs each transaction through `transaction_analyzer`, aggregates risk distribution, flags high-risk transactions, calculates performance metrics when ground truth labels exist
- `_calculate_performance_metrics`: Precision, recall, F1, accuracy from flagged vs actual fraud labels
- Added thin `@mcp.tool()` wrappers: `generate_synthetic_dataset` and `analyze_dataset`
- Updated MCP tool count test from 10 to 12
- Commit: `feat: Add generate_synthetic_dataset and analyze_dataset MCP tools`

### Task 4: Add synthetic_data status to health_check and get_model_status
- Added `synthetic_data` section to `health_check_impl` with `available`, `integration_loaded`, `output_dir`
- Added `synthetic_data` section to `get_model_status_impl` with same fields
- Commit: `feat: Add synthetic_data status to health_check and get_model_status`

### Task 5: Write comprehensive synthetic data tests
- 65 new tests in `tests/test_synthetic_data.py` across 12 test classes:
  - `TestSyntheticDataAvailability` (5 tests): Flag, import, singleton
  - `TestSyntheticDataIntegrationClass` (16 tests): Fraud patterns, schemas, data generation for all fraud types, behavioral data, network connections, schema compliance
  - `TestComprehensiveDatasetGeneration` (6 tests): End-to-end CSV/JSON, behavioral, network, fraud distribution
  - `TestGenerateSyntheticDatasetImpl` (8 tests): Input validation, graceful degradation
  - `TestAnalyzeDatasetImpl` (12 tests): CSV/JSON analysis, ground truth, edge cases, risk distribution
  - `TestCalculatePerformanceMetrics` (4 tests): Perfect, no-flagged, all-flagged, mixed predictions
  - `TestHealthCheckSyntheticData` (5 tests): synthetic_data section fields, unavailable state
  - `TestModelStatusSyntheticData` (3 tests): synthetic_data section fields, unavailable state
  - `TestSyntheticDataMCPRegistration` (3 tests): Tool registration, count
  - `TestEndToEndSyntheticFlow` (2 tests): Generate then analyze CSV/JSON
  - `TestGracefulDegradation` (2 tests): Unavailable and None states
- Added `synthetic` pytest marker to `pytest.ini`
- Commit: `test: Add Phase 9 synthetic data integration tests (65 tests)`

### Task 6: Final verification
- Fixed all ruff lint errors across server.py, integration.py, and tests
- Removed unused imports: `PaymentMethod` (server.py), `json`/`tempfile`/`numpy` (test), `asyncio`/`json`/`FastMCP`/`Optional` (integration.py)
- Fixed unused `base_flight` variables in integration.py
- Fixed `pd.DataFrame` type annotation in `_calculate_performance_metrics`
- Full test suite: 455 passed, 2 skipped
- Commit: `docs: Add Phase 9 plan document, fix lint errors`

## Results
- **Tests before**: 390 passed, 2 skipped
- **Tests after**: 455 passed, 2 skipped (+65 new tests)
- **MCP tools**: 10 -> 12 (generate_synthetic_dataset, analyze_dataset)
- **New features**: Synthetic fraud dataset generation with configurable fraud patterns, stored-dataset analysis with risk distribution and performance metrics, synthetic_data status in health check and model status
- **Graceful degradation**: Full backward compatibility when integration module unavailable
