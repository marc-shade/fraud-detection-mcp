[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/marc-shade-fraud-detection-mcp-badge.png)](https://mseep.ai/app/marc-shade-fraud-detection-mcp)

# Advanced Fraud Detection MCP

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python-3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Part of Agentic System](https://img.shields.io/badge/Part_of-Agentic_System-brightgreen)](https://github.com/marc-shade/agentic-system-oss)

> **Fraud detection and anomaly analysis for financial security — with AI agent-to-agent transaction protection.**

Part of the [Agentic System](https://github.com/marc-shade/agentic-system-oss) - a 24/7 autonomous AI framework with persistent memory.

## Overview

<img src="assets/bloodhound_avatar.png" alt="Fraud Detection Bloodhound" width="300" style="border-radius: 15px;" align="right">An open-source Model Context Protocol (MCP) server for fraud detection using machine learning and behavioral analysis. This system combines behavioral biometrics, anomaly detection, network graph analysis, and **AI agent behavioral fingerprinting** for fraud prevention.

## Key Features

### Core Detection (Active by Default)
- **Isolation Forest**: Anomaly detection for real-time transaction screening
- **Autoencoder Ensemble**: PyTorch-based deep learning anomaly detection using reconstruction error scoring. Ensemble with Isolation Forest (configurable 60/40 weighting).
- **Behavioral Biometrics**: Keystroke dynamics (Isolation Forest), mouse movement patterns (One-Class SVM), touch screen patterns (LOF)
- **Network Graph Analysis**: NetworkX-based graph centrality metrics (degree, clustering coefficient, betweenness, closeness) for fraud ring detection
- **46-Feature Pipeline**: Comprehensive feature engineering with cyclical encoding, z-scores, and velocity features

### Agent-to-Agent Transaction Protection
- **Traffic Classification**: Automatic detection of AI agent vs human traffic (Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, PayPal, Coinbase, OpenAI, Anthropic, x402)
- **Agent Identity Verification**: API key format validation, JWT token expiry checks, JSON-backed agent registry
- **Agent Behavioral Fingerprinting**: Per-agent Isolation Forest baselines (8 features, max 1000 observations/agent)
- **Mandate Compliance**: Spending limits, merchant whitelists/blocklists, time windows, geographic restrictions
- **Collusion Detection**: Directed graph analysis for circular flows, temporal clustering, and volume anomalies
- **Agent Reputation Scoring**: Longitudinal trust from history, consistency, and collusion safety (weighted 40/25/25/10)

### Defense Insider Threat Compliance
- **28 Behavioral Indicators** from the NITTF Insider Threat Guide (EO 13587)
- **SIEM Integration**: CEF (ArcSight), LEEF (QRadar), Syslog RFC 5424 with MITRE ATT&CK enrichment
- **Cleared Personnel Monitoring**: SEAD 4/6 continuous evaluation, 13 adjudicative guidelines
- **Compliance Dashboard**: NITTF maturity scoring, KRIs, model drift detection

### Optional (Requires Manual Invocation)
- **XGBoost**: Available via the `train_models` tool when `training_pipeline.py` dependencies are installed. Not part of the default detection path.
- **SMOTE Resampling**: Class balancing for imbalanced fraud datasets (via `train_models`)
- **Optuna Hyperparameter Tuning**: Automated hyperparameter optimization (via `train_models`)
- **MLflow Tracking**: Experiment tracking for training runs (via `train_models`)

### Experimental (Disabled by Default)
- **Graph Neural Network**: `models/gnn_fraud_detector.py` exists but requires `torch-geometric` and is disabled (`train_gnn=False`). Not used in the default detection pipeline.

### Explainability
- **SHAP-based Explanations**: Feature importance and decision reasoning with agent-specific context. Graceful fallback when SHAP is unavailable.

## Architecture

### Core Components (server.py)

1. **`TransactionAnalyzer`** — 46-feature extraction, Isolation Forest + Autoencoder ensemble scoring with configurable weights. Supports model persistence and hot-reload.

2. **`BehavioralBiometrics`** — Keystroke dynamics (Isolation Forest), mouse patterns (One-Class SVM), touch patterns (LOF). Extracts 10 statistical features per modality.

3. **`NetworkAnalyzer`** — Builds NetworkX graphs of entity connections. Calculates degree, clustering coefficient, betweenness/closeness centrality.

4. **`UserTransactionHistory`** — Thread-safe, bounded per-user transaction history for velocity analysis with LRU eviction.

### Agent Protection Pipeline (server.py)

5. **`TrafficClassifier`** — Classifies transactions as human/agent/unknown. Recognizes 9 agent protocols.

6. **`AgentIdentityRegistry`** — Thread-safe JSON-backed registry tracking agent trust scores and transaction history.

7. **`AgentIdentityVerifier`** — Validates credentials via registry lookup, API key format, and JWT token expiry.

8. **`AgentBehavioralFingerprint`** — Per-agent Isolation Forest baselines (8 features, min 10 observations before activation).

9. **`MandateVerifier`** — Stateless mandate compliance: max_amount, daily_limit, allowed/blocked merchants, time windows.

10. **`CollusionDetector`** — Directed graph for circular flows, temporal clustering, volume anomalies with LRU eviction.

11. **`AgentReputationScorer`** — Longitudinal reputation: trust (40%), history (25%), behavioral consistency (25%), collusion safety (10%).

### Risk Scoring

**Human traffic**: Transaction 50%, Behavioral 30%, Network 20%.
**Agent traffic**: Equal weighting across all available components.
Thresholds: CRITICAL >= 0.8, HIGH >= 0.6, MEDIUM >= 0.4, LOW < 0.4.

## Technical Specifications

- **Language**: Python 3.10+
- **ML Libraries**: scikit-learn, PyTorch (autoencoder)
- **Graph Processing**: NetworkX
- **API**: FastMCP (Model Context Protocol)
- **Testing**: pytest (830+ tests, 88%+ coverage)

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/marc-shade/fraud-detection-mcp
cd fraud-detection-mcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install optional training/dev dependencies
pip install -r requirements-dev.txt      # Testing and linting
pip install -r requirements-optional.txt  # XGBoost, GNN, benchmarking
```

### Claude Code Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "fraud-detection-mcp": {
      "command": "/path/to/fraud-detection-mcp/venv/bin/python",
      "args": ["/path/to/fraud-detection-mcp/server.py"],
      "env": {
        "FRAUD_DETECT_MODEL_PATH": "/path/to/fraud-detection-mcp/models",
        "FRAUD_DETECT_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## MCP Tools (24 total)

### Core Fraud Detection (5 tools)
| Tool | Description |
|------|-------------|
| `analyze_transaction` | Transaction fraud analysis using 46-feature Isolation Forest + Autoencoder ensemble |
| `detect_behavioral_anomaly` | Behavioral biometrics anomaly detection (keystroke, mouse, touch) |
| `assess_network_risk` | Graph centrality-based fraud ring detection |
| `generate_risk_score` | Weighted composite risk score (agent-aware weighting) |
| `explain_decision` | SHAP-based explainable AI with agent-specific reasoning |

### Agent-to-Agent Transaction Protection (6 tools)
| Tool | Description |
|------|-------------|
| `classify_traffic_source` | Detect human vs AI agent traffic (9 agent protocols) |
| `verify_agent_identity` | Validate agent credentials (API keys, JWT, registry) |
| `analyze_agent_transaction` | Full agent-aware pipeline (identity + fingerprint + mandate + transaction) |
| `verify_transaction_mandate` | Check transactions against agent spending mandates |
| `detect_agent_collusion` | Directed graph detection of coordinated agent behavior |
| `score_agent_reputation` | Longitudinal reputation from trust, history, consistency |

### Model Management & Operations (8 tools)
| Tool | Description |
|------|-------------|
| `train_models` | ML training pipeline with SMOTE and optional Optuna tuning (requires optional deps) |
| `get_model_status` | Model source, configuration, and saved model paths |
| `analyze_batch` | Batch transaction analysis with prediction caching |
| `get_inference_stats` | Inference engine statistics and cache performance |
| `generate_synthetic_dataset` | Generate labeled fraud datasets (CSV/JSON) for evaluation |
| `analyze_dataset` | Analyze stored CSV/JSON datasets for fraud patterns |
| `run_benchmark` | Performance benchmark with throughput and latency percentiles |
| `health_check` | System health with model status, cache stats, resource usage |

### Defense Compliance (5 tools)
| Tool | Description |
|------|-------------|
| `assess_insider_threat` | Insider threat assessment (28 NITTF behavioral indicators) |
| `generate_siem_events` | Export events in CEF/LEEF/Syslog with MITRE ATT&CK enrichment |
| `evaluate_cleared_personnel` | SEAD 4/6 cleared personnel analytics and CE checks |
| `get_compliance_dashboard` | NITTF maturity, KRIs, compliance posture, executive summary |
| `generate_threat_referral` | Formal case referral or personnel security action report |

### Example Usage

```python
# Analyze a transaction
result = mcp_client.call("analyze_transaction", {
    "transaction_id": "txn_123",
    "amount": 5000.00,
    "merchant": "Electronics Store",
    "location": "New York, NY",
    "timestamp": "2025-09-26T14:30:00Z"
})

# Result includes risk score, confidence, and explanation
{
    "risk_score": 0.72,
    "risk_level": "HIGH",
    "confidence": 0.85,
    "is_anomaly": true,
    "details": {
        "isolation_forest_score": -0.3,
        "autoencoder_score": 0.8,
        "ensemble_score": 0.72
    }
}
```

## Algorithm Details

### Isolation Forest (Active)
- **Purpose**: Fast anomaly detection for real-time transaction screening
- **Complexity**: O(n log n), handles high-dimensional data
- **Use Case**: First-line defense, scores all transactions by default

### Autoencoder (Active)
- **Purpose**: Deep learning anomaly detection via reconstruction error
- **Architecture**: PyTorch fully-connected autoencoder
- **Use Case**: Ensemble member with Isolation Forest for improved detection

### Behavioral Biometrics (Active)
- **Keystroke Dynamics**: Timing patterns via Isolation Forest (5 dwell + 5 flight features)
- **Mouse Biometrics**: Movement patterns via One-Class SVM
- **Touch Analytics**: Screen interaction via Local Outlier Factor (LOF)

### Network Graph Analysis (Active)
- **Graph Centrality**: Degree, clustering coefficient, betweenness, closeness centrality
- **Fraud Ring Detection**: Identifies suspicious clusters via graph metrics
- **Implementation**: NetworkX directed/undirected graphs

### XGBoost (Optional — requires `train_models`)
- **Purpose**: Gradient boosting for supervised fraud classification
- **Availability**: Only active after training with labeled data via `train_models` tool
- **Dependencies**: Requires `xgboost`, `imbalanced-learn`, `optuna` (in `requirements-optional.txt`)

## Graceful Degradation

The server starts even when optional dependencies are missing:

| Module | Flag | Required For |
|--------|------|-------------|
| `monitoring.py` | `MONITORING_AVAILABLE` | Prometheus metrics, structured logging |
| `training_pipeline.py` | `TRAINING_AVAILABLE` | `train_models` tool |
| `models/autoencoder.py` | `AUTOENCODER_AVAILABLE` | Autoencoder ensemble member |
| `explainability.py` | `EXPLAINABILITY_AVAILABLE` | SHAP-based explanations |
| `integration.py` | `SYNTHETIC_DATA_AVAILABLE` | Synthetic dataset generation, benchmarks |
| `security_utils.py` | `SECURITY_UTILS_AVAILABLE` | Input sanitization, rate limiting |

## Security Notice

- **Transport Security**: The MCP server does not provide TLS. If exposed over a network, configure a reverse proxy with TLS termination.
- **Authentication**: The server does not implement authentication. Access control must be handled by the MCP client or network layer.
- **Data at Rest**: Transaction data and model files are stored unencrypted. Apply filesystem-level encryption if required by your security policy.
- **Input Sanitization**: When `security_utils.py` is available, inputs are sanitized against XSS/SQLi patterns and rate-limited.
- **Dependencies**: Review `requirements.txt` and pin versions for production deployments. Run `bandit -r . -x ./tests` for security scanning.

## Known Limitations

- **Models initialize with synthetic data**: Default models have not been trained on real fraud data. Use `train_models` with your own labeled dataset for production accuracy.
- **Performance is environment-dependent**: No benchmark numbers are published because results vary significantly by hardware, data distribution, and model configuration. Run `run_benchmark` on your own infrastructure to measure.
- **No continuous/adaptive learning**: Model retraining is manual via the `train_models` tool. There is no online learning or automatic model refresh.
- **No encryption at rest**: The server does not encrypt stored data. This must be handled at the infrastructure level.
- **No GDPR/PCI-DSS/SOX compliance**: The server does not implement regulatory compliance controls. The defense compliance modules (insider threat, SIEM, cleared personnel) address federal insider threat standards, not financial regulatory compliance.

## Defense Compliance Architecture

```
compliance/
  __init__.py                  # Package exports
  insider_threat.py            # EO 13587 / NITTF insider threat detection (28 indicators)
  siem_integration.py          # CEF/LEEF/Syslog event generation & correlation
  cleared_personnel.py         # SEAD 4/6 cleared personnel analytics
  dashboard_metrics.py         # NITTF maturity, KRIs, compliance posture
```

All compliance modules run locally with no external service dependencies, are thread-safe, and use graceful degradation.

## Contributing

This is an open-source project. Contributions welcome for:
- New detection algorithms
- Performance optimizations
- Test coverage improvements
- Documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See LICENSE file for details

---

## Part of the MCP Ecosystem

This server integrates with other MCP servers for comprehensive AGI capabilities:

| Server | Purpose |
|--------|---------|
| [enhanced-memory-mcp](https://github.com/marc-shade/enhanced-memory-mcp) | 4-tier persistent memory with semantic search |
| [agent-runtime-mcp](https://github.com/marc-shade/agent-runtime-mcp) | Persistent task queues and goal decomposition |
| [agi-mcp](https://github.com/marc-shade/agi-mcp) | Full AGI orchestration with 21 tools |
| [cluster-execution-mcp](https://github.com/marc-shade/cluster-execution-mcp) | Distributed task routing across nodes |
| [node-chat-mcp](https://github.com/marc-shade/node-chat-mcp) | Inter-node AI communication |
| [ember-mcp](https://github.com/marc-shade/ember-mcp) | Production-only policy enforcement |

See [agentic-system-oss](https://github.com/marc-shade/agentic-system-oss) for the complete framework.
