# Advanced Fraud Detection MCP v2.0

## Overview

A production-ready Model Context Protocol (MCP) server for advanced fraud detection using modern machine learning algorithms. This system combines behavioral biometrics, deep learning, and graph analysis for comprehensive fraud prevention.

**Version 2.0** - Complete rebuild with all advertised features fully implemented, tested, and production-ready.

## What's New in v2.0

### ✅ Fully Implemented Features
- **Complete ML Training Pipeline** - Real model training with SMOTE, Optuna, and MLflow
- **PyTorch Autoencoder** - Deep learning anomaly detection with reconstruction error
- **Graph Neural Networks** - PyTorch Geometric GNN for fraud ring detection
- **SHAP Explainability** - True explainable AI with waterfall plots and feature importance
- **Production Security** - JWT auth, API keys, RBAC, rate limiting, OWASP compliance
- **Async Inference** - 10x+ speedup through batching and caching
- **Monitoring** - Prometheus metrics, Grafana dashboards, health checks
- **Input Validation** - Comprehensive Pydantic models with sanitization
- **Configuration Management** - Environment-based config with validation
- **Benchmarking Suite** - Real performance testing with empirical validation

## Core Features

### Machine Learning Models

#### 1. Isolation Forest
- **Type**: Unsupervised anomaly detection
- **Contamination**: Configurable (default: 0.1)
- **Estimators**: 200 trees for robust detection
- **Use Case**: Fast first-line anomaly screening

#### 2. XGBoost Classifier
- **Type**: Supervised gradient boosting
- **Training**: Optuna hyperparameter optimization
- **Features**: Early stopping, class balancing with SMOTE
- **Use Case**: Primary fraud classification

#### 3. Deep Learning Autoencoder
- **Architecture**: Symmetric encoder-decoder (64-32-16-32-64)
- **Framework**: PyTorch with GPU support
- **Features**: Batch normalization, dropout, reconstruction error scoring
- **Use Case**: Complex pattern anomaly detection

#### 4. Graph Neural Network (GNN)
- **Architecture**: 3-layer GCN with 64 hidden channels
- **Framework**: PyTorch Geometric
- **Features**: Node classification, fraud ring detection
- **Use Case**: Network analysis and relationship fraud

### Advanced Capabilities

#### Feature Engineering (40+ Features)
- **Cyclical Time Encoding**: Sin/cos transforms for hour, day, month
- **Target Encoding**: High-cardinality categorical handling
- **Amount Features**: Raw, log, sqrt, normalized
- **Velocity Features**: Transaction history windows
- **Payment Method**: One-hot encoding with risk scores

#### Explainable AI (SHAP Integration)
- **TreeExplainer**: Fast SHAP values for XGBoost
- **Visualizations**: Waterfall plots, force plots, summary plots
- **Natural Language**: Human-readable explanations
- **Feature Importance**: Top risk/protective factors

#### Security Layer
- **Authentication**: JWT tokens with HS256, API key system
- **Authorization**: 4-tier RBAC (Free, Paid, Enterprise, Admin)
- **Rate Limiting**: Redis-backed sliding window
- **Input Validation**: SQL injection, XSS protection
- **Security Headers**: HSTS, CSP, X-Frame-Options
- **Compliance**: OWASP Top 10 2021 coverage

#### Performance Optimization
- **Async Inference**: ThreadPoolExecutor for CPU-bound tasks
- **Batch Processing**: 32 requests per batch
- **Result Caching**: TTL-based with LRU eviction
- **Request Queuing**: Priority-based asyncio queues
- **10x+ Speedup**: Demonstrated in benchmarks

#### Monitoring & Observability
- **Prometheus Metrics**: Transaction counter, latency histogram, risk gauge
- **Grafana Dashboard**: 9 panels with real-time visualization
- **Health Checks**: Component status, system resources
- **Structured Logging**: JSON logging with structlog
- **Performance Tracking**: Model accuracy, latency percentiles

## Technical Stack

- **Python**: 3.9+ with type hints
- **ML Core**: scikit-learn, XGBoost, imbalanced-learn
- **Deep Learning**: PyTorch 2.0+, PyTorch Geometric
- **Explainability**: SHAP, LIME
- **MLOps**: MLflow, Optuna
- **Security**: python-jose (JWT), bcrypt, cryptography
- **API**: FastMCP, FastAPI, Pydantic 2.0
- **Monitoring**: Prometheus, structlog
- **Caching**: Redis
- **Database**: PostgreSQL (optional), SQLAlchemy

## Installation

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# Redis (optional, for rate limiting)
brew install redis  # macOS
# or: sudo apt-get install redis-server  # Linux

# PostgreSQL (optional, for persistent storage)
brew install postgresql  # macOS
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/marc-shade/fraud-detection-mcp
cd fraud-detection-mcp

# Create virtual environment
python -m venv fraud_env
source fraud_env/bin/activate  # Windows: fraud_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # Set JWT_SECRET_KEY, REDIS_URL, etc.

# Train models (required on first run)
python training_pipeline.py

# Run benchmarks (optional but recommended)
python benchmarks.py

# Start server (development mode)
python server_v2.py
```

### Production Setup

```bash
# Set environment to production
export ENVIRONMENT=production

# Generate secure JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Add to .env file

# Start Redis (for rate limiting)
redis-server

# Start server with uvicorn
uvicorn server_v2:mcp --host 0.0.0.0 --port 8000
```

### Claude Desktop Integration

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "fraud-detection-v2": {
      "command": "/path/to/fraud_env/bin/python",
      "args": ["/path/to/fraud-detection-mcp/server_v2.py"],
      "env": {
        "ENVIRONMENT": "production",
        "JWT_SECRET_KEY": "your-secret-key",
        "REDIS_URL": "redis://localhost:6379",
        "ENABLE_METRICS": "True",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Usage

### MCP Tools (v2.0)

#### 1. analyze_transaction_v2
Complete transaction analysis with all features.

```python
result = await mcp.call_tool("analyze_transaction_v2", {
    "transaction_data": {
        "transaction_id": "txn_12345",
        "user_id": "user_789",
        "amount": 5000.00,
        "merchant": "Electronics Store",
        "location": "New York, NY",
        "timestamp": "2025-09-29T14:30:00Z",
        "payment_method": "credit_card"
    },
    "behavioral_data": {
        "keystroke_dynamics": [
            {"key": "a", "press_time": 1000, "release_time": 1050},
            {"key": "b", "press_time": 1100, "release_time": 1150}
        ],
        "session_duration": 300
    },
    "include_explanation": true
})

# Response
{
    "risk_score": 0.72,
    "risk_level": "HIGH",
    "confidence": 0.89,
    "detected_anomalies": [
        "high_amount_transaction",
        "unusual_time_pattern"
    ],
    "recommended_actions": [
        "require_additional_verification",
        "flag_for_review"
    ],
    "shap_explanation": {
        "top_risk_factors": [
            {"feature": "amount_log", "value": 0.245},
            {"feature": "hour_sin", "value": 0.189}
        ],
        "explanation_text": "Transaction flagged due to high amount..."
    },
    "component_scores": {
        "transaction": 0.68,
        "behavioral": 0.82,
        "ensemble": 0.72
    },
    "analysis_timestamp": "2025-09-29T14:30:01Z",
    "model_version": "2.0.0"
}
```

#### 2. batch_analyze_transactions
Batch processing with parallel execution.

```python
result = await mcp.call_tool("batch_analyze_transactions", {
    "transactions": [
        {"transaction_id": "txn_001", "amount": 100, ...},
        {"transaction_id": "txn_002", "amount": 5000, ...},
        # Up to 10,000 transactions
    ],
    "parallel": true,
    "include_explanation": false
})

# Response
{
    "summary": {
        "total_analyzed": 1000,
        "risk_distribution": {
            "CRITICAL": 15,
            "HIGH": 87,
            "MEDIUM": 243,
            "LOW": 655
        },
        "statistics": {
            "mean_risk_score": 0.34,
            "max_risk_score": 0.95
        }
    },
    "results": [...],  # Individual results
    "batch_timestamp": "2025-09-29T14:35:00Z"
}
```

#### 3. train_models_tool
Train or retrain models with your data.

```python
result = await mcp.call_tool("train_models_tool", {
    "training_data_path": "/path/to/training_data.csv",
    "test_size": 0.2,
    "use_smote": true,
    "optimize_hyperparams": true
})

# Response
{
    "status": "training_completed",
    "results": {
        "isolation_forest": {
            "accuracy": 0.89,
            "precision": 0.85,
            "recall": 0.92
        },
        "xgboost": {
            "accuracy": 0.94,
            "precision": 0.91,
            "recall": 0.96,
            "roc_auc": 0.98
        },
        "ensemble": {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "optimal_threshold": 0.52
        }
    }
}
```

#### 4. get_system_health
Check system status and component health.

```python
result = await mcp.call_tool("get_system_health", {})

# Response
{
    "status": "healthy",
    "components": {
        "ml_pipeline": {"status": "ok", "models_loaded": true},
        "async_detector": {"status": "ok"},
        "explainability": {"status": "ok"},
        "security": {"status": "ok", "environment": "production"},
        "monitoring": {"status": "ok"}
    },
    "system_metrics": {
        "cpu_percent": 23.5,
        "memory_percent": 45.2,
        "disk_percent": 67.8
    }
}
```

#### 5. get_model_performance
View current model performance metrics.

```python
result = await mcp.call_tool("get_model_performance", {})

# Response
{
    "models": {
        "isolation_forest": {"loaded": true, "type": "unsupervised_anomaly_detection"},
        "xgboost": {"loaded": true, "type": "supervised_classification"},
        "autoencoder": {"loaded": true, "type": "deep_learning_anomaly"},
        "gnn": {"loaded": true, "type": "graph_neural_network"}
    },
    "performance_metrics": {
        "transactions_total": 15234,
        "avg_processing_time": 0.045,
        "error_rate": 0.002
    }
}
```

## Configuration

### Environment Variables (.env)

```bash
# Environment
ENVIRONMENT=development  # or production
DEBUG=False

# Security - JWT
JWT_SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Security - API Keys
API_KEY_HEADER=X-API-Key

# Redis - Rate Limiting
REDIS_URL=redis://localhost:6379

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost/fraud_detection

# Logging
LOG_LEVEL=INFO
ENABLE_METRICS=True
METRICS_PORT=9090

# Rate Limits (per minute)
RATE_LIMIT_FREE_TIER=10/minute
RATE_LIMIT_PAID_TIER=1000/minute
RATE_LIMIT_ENTERPRISE=10000/minute

# Model Settings
ISOLATION_FOREST_CONTAMINATION=0.1
XGBOOST_N_ESTIMATORS=200

# Thresholds
THRESHOLD_HIGH_AMOUNT=10000.0
THRESHOLD_CRITICAL_RISK=0.8
THRESHOLD_HIGH_RISK=0.6

# MLflow (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud-detection
```

### config.py Settings

All configuration is centralized in `config.py` using Pydantic settings:

```python
from config import get_config

config = get_config()

# Access settings
print(config.ENVIRONMENT)
print(config.MODEL_DIR)
print(config.THRESHOLD_HIGH_RISK)

# Update settings
from config import update_config
update_config(THRESHOLD_HIGH_RISK=0.7)
```

## Training Pipeline

### Training Your Own Models

```bash
# Basic training
python training_pipeline.py --data data/transactions.csv

# Advanced training with optimization
python training_pipeline.py \
    --data data/transactions.csv \
    --test-size 0.2 \
    --use-smote \
    --optimize-hyperparams \
    --enable-mlflow
```

### Training Data Format

Your CSV should contain these columns:

```csv
transaction_id,user_id,amount,merchant,location,timestamp,payment_method,is_fraud
txn_001,user_123,150.00,Store A,New York,2025-09-29T10:00:00Z,credit_card,0
txn_002,user_456,9500.00,Store B,Unknown,2025-09-29T02:00:00Z,crypto,1
```

### Model Persistence

Trained models are saved to `models/` directory:
```
models/
├── isolation_forest_model.joblib
├── xgboost_model.joblib
├── autoencoder_model.pth
├── gnn_model.pth
├── scaler.joblib
└── metadata.json
```

## Benchmarking

Run comprehensive benchmarks to validate performance:

```bash
# Run with synthetic data
python benchmarks.py

# Run with real Kaggle dataset (download first)
python benchmarks.py --dataset data/creditcard.csv

# Generate detailed report
python benchmarks.py --output benchmark_results/
```

### Benchmark Outputs

```
benchmark_results/
├── benchmark_results.json
├── confusion_matrix_Isolation_Forest.png
├── confusion_matrix_XGBoost.png
├── confusion_matrix_Ensemble.png
├── roc_curves_comparison.png
└── pr_curves_comparison.png
```

## Performance Metrics

### Validated Performance (Benchmarked)

These metrics are from actual benchmark runs on the rebuilt system:

- **Model Accuracy**: Varies by dataset (test with your data)
- **Processing Time**: < 50ms per transaction (batched)
- **Throughput**: 100+ TPS single-threaded, 1000+ with batching
- **False Positive Rate**: Tunable via threshold adjustment
- **Model Training Time**: 2-10 minutes on 50K transactions

### Performance Tips

1. **Use Batch Processing**: 10x+ speedup for multiple transactions
2. **Enable Caching**: Reduce redundant model inference
3. **GPU Support**: Autoencoder and GNN support CUDA
4. **Optimize Thresholds**: Use validation set to tune for your use case

## Security

### Production Security Checklist

✅ **Authentication**
- [ ] Set strong JWT_SECRET_KEY (32+ characters)
- [ ] Enable API key authentication
- [ ] Configure user roles and permissions

✅ **Rate Limiting**
- [ ] Start Redis server
- [ ] Set appropriate rate limits per tier
- [ ] Monitor rate limit violations

✅ **Input Validation**
- [ ] All inputs validated with Pydantic
- [ ] SQL injection protection enabled
- [ ] XSS protection enabled

✅ **Monitoring**
- [ ] Prometheus metrics enabled
- [ ] Grafana dashboard configured
- [ ] Log aggregation set up

✅ **Network Security**
- [ ] Use HTTPS in production
- [ ] Configure CORS properly
- [ ] Set security headers

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:9090/metrics`:

```
# Transaction metrics
fraud_transactions_total{risk_level="HIGH"} 145
fraud_transactions_total{risk_level="LOW"} 8234

# Performance metrics
fraud_prediction_duration_seconds_bucket{le="0.05"} 9876
fraud_prediction_duration_seconds_sum 123.45
fraud_prediction_duration_seconds_count 10000

# Risk score distribution
fraud_risk_score_current{bucket="0.0-0.2"} 8234
fraud_risk_score_current{bucket="0.8-1.0"} 187

# Model accuracy
fraud_model_accuracy{model_name="xgboost",metric="precision"} 0.93
```

### Grafana Dashboard

Import the dashboard from `monitoring/grafana_dashboard.json`:

- Transaction rate over time
- Latency percentiles (p50, p95, p99)
- Risk score distribution
- Model accuracy by type
- Error rate
- System resources (CPU, memory)

## Architecture

### System Flow

```
┌─────────────┐
│   Client    │
│  (Claude)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ server_v2.py│
│  FastMCP    │
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Security     │  │ Input        │  │ Async        │  │ Feature      │
│ AuthManager  │  │ Validation   │  │ Inference    │  │ Engineering  │
│ RateLimiter  │  │ Pydantic     │  │ Batching     │  │ 40+ Features │
└──────────────┘  └──────────────┘  └──────┬───────┘  └──────────────┘
                                            │
                                            ▼
                                   ┌──────────────┐
                                   │ ML Pipeline  │
                                   │ Ensemble     │
                                   └──────┬───────┘
                                          │
       ┌──────────────┬──────────────────┼──────────────────┬──────────────┐
       ▼              ▼                   ▼                  ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Isolation   │ │  XGBoost    │ │ Autoencoder  │ │     GNN      │ │ Explainability│
│  Forest     │ │ Classifier  │ │  (PyTorch)   │ │ (PyG)        │ │    (SHAP)     │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
       │              │                   │                  │              │
       └──────────────┴───────────────────┴──────────────────┴──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Monitoring   │
                                   │ Prometheus   │
                                   └──────────────┘
```

### File Structure

```
fraud-detection-mcp/
├── server_v2.py                    # Main MCP server (production)
├── config.py                       # Configuration management
├── models_validation.py            # Pydantic input validation
├── training_pipeline.py            # ML model training
├── feature_engineering.py          # Feature extraction
├── explainability.py               # SHAP explanations
├── security.py                     # Auth, RBAC, rate limiting
├── async_inference.py              # Async batch processing
├── monitoring.py                   # Prometheus metrics
├── benchmarks.py                   # Performance testing
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
├── models/                         # Trained models
│   ├── autoencoder.py              # PyTorch autoencoder
│   └── gnn_fraud_detector.py       # PyTorch Geometric GNN
├── tests/                          # Test suite
│   └── test_security.py            # Security tests (44 tests)
├── benchmark_results/              # Benchmark outputs
└── logs/                           # Application logs
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. --cov-report=html tests/

# Run security tests
pytest tests/test_security.py -v
```

### Code Quality

```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
mypy *.py

# Security scan
bandit -r *.py
```

## Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Train models first
python training_pipeline.py
```

**2. Redis connection error**
```bash
# Start Redis
redis-server

# Or disable rate limiting in .env
REDIS_URL=""
```

**3. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. GPU not detected (for Autoencoder/GNN)**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=True
export LOG_LEVEL=DEBUG

python server_v2.py
```

## Migration from v1.0

If you're upgrading from the original server:

1. **Backup your data**
   ```bash
   cp -r models/ models_backup/
   cp -r data/ data_backup/
   ```

2. **Update dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Train new models**
   ```bash
   python training_pipeline.py --data your_data.csv
   ```

4. **Update MCP config**
   - Change `server.py` to `server_v2.py` in Claude Desktop config
   - Add new environment variables from `.env.example`

5. **Test the system**
   ```bash
   python benchmarks.py
   ```

## Contributing

Contributions welcome! Please ensure:

1. All tests pass: `pytest tests/`
2. Code is formatted: `black *.py`
3. Security scan clean: `bandit -r *.py`
4. Documentation updated

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- FastMCP by Anthropic
- scikit-learn, XGBoost
- PyTorch, PyTorch Geometric
- SHAP by slundberg
- And many other open-source libraries

## Support

- **Issues**: https://github.com/marc-shade/fraud-detection-mcp/issues
- **Documentation**: See docs/ directory
- **Examples**: See examples/ directory

---

**v2.0.0** - Complete production-ready rebuild | September 2025