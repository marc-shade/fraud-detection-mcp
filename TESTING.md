# Testing Guide

## Quick Test (No Dependencies)

```bash
# Test basic imports and structure
python -c "from config import get_config; print('✓ Config OK')"
python -c "from models_validation import TransactionData; print('✓ Validation OK')"
```

## Full Testing (Requires Dependencies)

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# This includes:
# - scikit-learn, xgboost (ML core)
# - torch, torch-geometric (deep learning)
# - shap, lime (explainability)
# - mlflow, optuna (MLOps)
# - fastapi, pydantic (API)
# - redis, prometheus-client (infrastructure)
```

### Run Full Test Suite

```bash
# Run comprehensive system test
python test_system.py

# Run specific module tests
python test_new_modules.py

# Run security tests
pytest tests/test_security.py -v
```

## CI/CD Testing

The system includes multiple test levels:

1. **Import Test** - Verify all modules can be imported
2. **Unit Tests** - Test individual components
3. **Integration Tests** - Test component interactions
4. **Performance Tests** - Run benchmarks

## Known Test Dependencies

- `xgboost` - Required for training pipeline and benchmarks
- `mlflow` - Optional, system works without it
- `torch` - Optional, falls back to scikit-learn
- `torch_geometric` - Optional, GNN uses fallback
- `shap` - Optional, explainability uses feature importance fallback
- `redis` - Optional, rate limiting disabled if unavailable

## Development Testing

For development, you can test without full dependencies:

```bash
# Test configuration
python -c "from config import get_config; print(get_config().ENVIRONMENT)"

# Test validation
python -c "from models_validation import TransactionData; from datetime import datetime; t = TransactionData(transaction_id='t1', user_id='u1', amount=100, merchant='M', location='L', timestamp=datetime.now(), payment_method='credit_card'); print('OK')"

# Test feature engineering
python -c "from feature_engineering import FeatureEngineer; print('OK')"
```

## Production Testing

Before deploying to production:

1. Install all dependencies: `pip install -r requirements.txt`
2. Run full test suite: `python test_system.py`
3. Run benchmarks: `python benchmarks.py`
4. Check security: `pytest tests/test_security.py`
5. Verify models trained: Check `models/` directory
6. Test API endpoints: Start server and test with curl

## Troubleshooting

If tests fail due to missing dependencies:
- Check `requirements.txt` is complete
- Verify Python version >= 3.9
- Install system packages (Redis, PostgreSQL) if needed
- Check GPU availability for PyTorch tests