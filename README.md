# Advanced Fraud Detection MCP

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python-3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Part of Agentic System](https://img.shields.io/badge/Part_of-Agentic_System-brightgreen)](https://github.com/marc-shade/agentic-system-oss)

> **Fraud detection and anomaly analysis for financial security — now with AI agent-to-agent transaction protection.**

Part of the [Agentic System](https://github.com/marc-shade/agentic-system-oss) - a 24/7 autonomous AI framework with persistent memory.

## Overview

<img src="assets/bloodhound_avatar.png" alt="Fraud Detection Bloodhound" width="300" style="border-radius: 15px;" align="right">A sophisticated, open-source Model Context Protocol (MCP) server for advanced fraud detection using cutting-edge algorithms and techniques. This system combines behavioral biometrics, machine learning, real-time anomaly detection, and **AI agent behavioral fingerprinting** for comprehensive fraud prevention.

**Built for the Modern Threat Landscape** - Designed to detect sophisticated fraud patterns including synthetic identities, account takeovers, AI-generated attacks, and the emerging threat of **agent-to-agent transaction fraud** (McKinsey projects $3-5T in agent-mediated commerce by 2030).

## Key Features

### Core Detection Algorithms
- **Isolation Forest**: Fast anomaly detection for real-time processing
- **XGBoost Ensemble**: High-performance gradient boosting for pattern recognition
- **Autoencoders**: Deep learning-based anomaly detection for complex patterns
- **Graph Neural Networks**: Network analysis for fraud ring detection
- **Behavioral Biometrics**: Keystroke dynamics, mouse patterns, and interaction analysis

### Agent-to-Agent Transaction Protection
- **Traffic Classification**: Automatic detection of AI agent vs human traffic (Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, and more)
- **Agent Identity Verification**: API key format validation, JWT token expiry checks, agent registry
- **Agent Behavioral Fingerprinting**: Per-agent Isolation Forest baselines replacing human biometrics
- **Mandate Compliance**: Spending limits, merchant whitelists, time windows, geographic restrictions
- **Collusion Detection**: Graph-based circular flow, temporal clustering, and volume anomaly detection
- **Agent Reputation Scoring**: Longitudinal trust from history, consistency, and collusion safety

### Advanced Capabilities
- **Real-time Processing**: Sub-second transaction analysis
- **Adaptive Learning**: Continuous model improvement from new data
- **Multi-modal Analysis**: Combines transaction data, behavioral patterns, and network analysis
- **Explainable AI**: Clear reasoning for fraud decisions with agent-specific explanations
- **Privacy-First**: On-device processing with minimal data exposure

## Architecture

### Core Components

1. **Anomaly Detection Engine**
   - Isolation Forest for fast outlier detection
   - One-Class SVM for boundary-based detection
   - Local Outlier Factor (LOF) for density-based detection

2. **Behavioral Analysis Module**
   - Keystroke dynamics profiling
   - Mouse movement pattern analysis
   - Touch interaction biometrics
   - Session behavior tracking

3. **Transaction Pattern Engine**
   - Velocity analysis (transaction frequency/amounts)
   - Geographic anomaly detection
   - Merchant pattern analysis
   - Time-based pattern recognition

4. **Network Analysis System**
   - Graph-based fraud ring detection
   - Community detection algorithms
   - Relationship scoring
   - Entity resolution

5. **Agent Transaction Pipeline** (New)
   - Traffic source classification (human/agent/unknown)
   - Agent identity registry and credential verification
   - Behavioral fingerprinting with per-agent Isolation Forest baselines
   - Mandate verification (spending limits, merchant/location/time constraints)
   - Collusion detection via directed graph analysis
   - Longitudinal reputation scoring

6. **Risk Scoring Framework**
   - Multi-factor risk calculation with agent-aware weighting
   - Confidence intervals
   - Threshold management
   - Alert prioritization

## Technical Specifications

- **Language**: Python 3.10+
- **ML Libraries**: scikit-learn, XGBoost, PyTorch
- **Graph Processing**: NetworkX
- **API**: FastMCP (Model Context Protocol)
- **Testing**: pytest (727 tests, 80%+ coverage)

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/marc-shade/fraud-detection-mcp
cd fraud-detection-mcp

# Create virtual environment (recommended)
python -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
python setup.py install
```

### Claude Code Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "fraud-detection-mcp": {
      "command": "/path/to/fraud-detection-mcp/fraud_env/bin/python",
      "args": ["/path/to/fraud-detection-mcp/server.py"],
      "env": {
        "FRAUD_DETECT_MODEL_PATH": "/path/to/fraud-detection-mcp/models",
        "FRAUD_DETECT_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Usage

### MCP Tools Available (19 tools)

#### Core Fraud Detection
| Tool | Description |
|------|-------------|
| `analyze_transaction` | Real-time transaction fraud analysis |
| `detect_behavioral_anomaly` | Behavioral biometrics anomaly detection |
| `assess_network_risk` | Graph-based fraud ring detection |
| `generate_risk_score` | Comprehensive weighted risk assessment |
| `explain_decision` | Explainable AI reasoning (human + agent-aware) |

#### Agent-to-Agent Transaction Protection
| Tool | Description |
|------|-------------|
| `classify_traffic_source` | Detect human vs AI agent traffic (supports Stripe ACP, Visa TAP, Mastercard, Google AP2, PayPal, Coinbase, OpenAI, Anthropic) |
| `verify_agent_identity` | Validate agent credentials (API keys, JWT tokens, registry lookup) |
| `analyze_agent_transaction` | Full agent-aware analysis pipeline (identity + fingerprint + mandate + transaction) |
| `verify_transaction_mandate` | Check transactions against agent spending mandates |
| `detect_agent_collusion` | Graph-based detection of coordinated agent behavior |
| `score_agent_reputation` | Longitudinal reputation from trust, history, consistency, collusion safety |

#### Model Management & Operations
| Tool | Description |
|------|-------------|
| `train_models` | ML training pipeline with SMOTE and optional Optuna tuning |
| `get_model_status` | Model source, configuration, and saved model paths |
| `analyze_batch` | Batch transaction analysis with prediction caching |
| `get_inference_stats` | Inference engine statistics and cache performance |
| `generate_synthetic_dataset` | Generate realistic test datasets |
| `analyze_dataset` | Analyze stored CSV/JSON datasets for fraud patterns |
| `run_benchmark` | Performance benchmarking of the detection pipeline |
| `health_check` | System health with model status, cache stats, resource usage |

### Example Usage

```python
# Analyze a transaction
result = mcp_client.call("analyze_transaction", {
    "transaction_id": "txn_123",
    "amount": 5000.00,
    "merchant": "Electronics Store",
    "location": "New York, NY",
    "timestamp": "2025-09-26T14:30:00Z",
    "behavioral_data": {
        "keystroke_dynamics": [...],
        "mouse_patterns": [...],
        "session_data": {...}
    }
})

# Result includes risk score, confidence, and explanation
{
    "risk_score": 0.85,
    "risk_level": "HIGH",
    "confidence": 0.92,
    "detected_anomalies": [
        "unusual_amount_for_merchant",
        "abnormal_keystroke_dynamics",
        "geographic_anomaly"
    ],
    "explanation": "Transaction shows multiple risk factors...",
    "recommended_action": "require_additional_verification"
}
```

## Algorithm Details

### 1. Isolation Forest
- **Purpose**: Fast anomaly detection for real-time processing
- **Advantage**: O(n log n) complexity, handles high-dimensional data
- **Use Case**: First-line defense for transaction screening

### 2. XGBoost Ensemble
- **Purpose**: Pattern recognition with high accuracy
- **Features**: Handles imbalanced datasets, feature importance
- **Use Case**: Primary classification for known fraud patterns

### 3. Behavioral Biometrics
- **Keystroke Dynamics**: Timing patterns between keystrokes
- **Mouse Biometrics**: Movement velocity, acceleration, click patterns
- **Touch Analytics**: Pressure, swipe patterns, gesture recognition

### 4. Graph Neural Networks
- **Network Analysis**: Entity relationships and fraud rings
- **Community Detection**: Identifying suspicious clusters
- **Entity Resolution**: Linking related accounts/devices

## Performance Metrics

- **Detection Rate**: >95% for known fraud patterns
- **False Positive Rate**: <2% with proper tuning
- **Response Time**: <100ms for real-time analysis
- **Throughput**: 10,000+ transactions per second
- **Model Accuracy**: 97%+ on benchmark datasets

## Privacy and Security

- **On-Device Processing**: Sensitive data never leaves local environment
- **Differential Privacy**: Noise injection for model training
- **Encryption**: All data encrypted at rest and in transit
- **Audit Trails**: Complete decision logging
- **Compliance**: GDPR, PCI-DSS, SOX ready

## Contributing

This is an open-source project. Contributions welcome for:
- New detection algorithms
- Performance optimizations
- Additional behavioral biometrics
- Extended documentation
- Test coverage
- etc.

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
