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

### MCP Tools Available (24 tools)

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

## Defense Insider Threat Compliance

Comprehensive defense-grade insider threat detection and compliance modules aligned with federal standards and executive orders.

### Insider Threat Program

Per **Executive Order 13587** and **NITTF** (National Insider Threat Task Force) guidance:

- **28 Behavioral Indicators** from the NITTF Insider Threat Guide covering access anomalies, data movement, evasion, foreign nexus, counter-intelligence, physical security, and personal conduct
- **User Activity Monitoring (UAM)** aligned with CNSSD 504 requirements
- **Risk Scoring**: Weighted indicator aggregation producing 0-100 scores
- **Threat Levels**: 4-tier model aligned with DHS NTAS (BASELINE / ADVISORY / ELEVATED / IMMINENT)
- **Case Referral Reports**: Formal referral generation with executive summary, risk timeline, and legal notices

### SIEM Integration

Defense-grade security event correlation and multi-format output:

- **Common Event Format (CEF)** for ArcSight
- **Log Event Extended Format (LEEF)** for IBM QRadar
- **Syslog RFC 5424** with structured data
- **8 Correlation Rules** detecting multi-indicator attack patterns (data exfiltration sequences, credential compromise chains, foreign intelligence patterns, pre-departure exfiltration, and more)
- **MITRE ATT&CK Enrichment**: Maps all indicators to ATT&CK technique IDs with tactic and reference URLs
- **DoD 8570/8140 Classification**: Incidents categorized as CAT-1 through CAT-7
- **Batch Export**: JSON and CSV export for offline analysis with time/severity/user filters

### Cleared Personnel Monitoring

For users with security clearances per **SEAD 4** and **SEAD 6**:

- **Continuous Evaluation (CE)**: Real-time monitoring per SEAD 6 covering financial, criminal, foreign travel, foreign contacts, and public records
- **Whole Person Assessment**: All 13 adjudicative guidelines from SEAD 4:
  - (A) Allegiance, (B) Foreign Influence, (C) Foreign Preference, (D) Sexual Behavior, (E) Personal Conduct, (F) Financial Considerations, (G) Alcohol, (H) Drug Involvement, (I) Psychological Conditions, (J) Criminal Conduct, (K) Handling Protected Information, (L) Outside Activities, (M) Use of Information Technology
- **Need-to-Know Verification** per NIST 800-53 AC-3 and AC-25
- **SF-86 Consistency Checks**: Cross-reference questionnaire data against known records
- **Clearance Lifecycle**: Track PENDING / INTERIM / FINAL / SUSPENDED / REVOKED / EXPIRED states
- **Polygraph Compliance**: CI, Full Scope, and Lifestyle polygraph tracking
- **Personnel Security Action Reports** per EO 12968

### Compliance Dashboard

Metrics and reporting for executive briefings:

- **NITTF Maturity Scoring**: Five levels from Initial to Optimizing with per-criterion tracking
- **Key Risk Indicators (KRIs)**: MTTD, MTTR, false positive rate, true positive rate, alert volume with 7-day and 30-day trend analysis
- **Compliance Posture**: Weighted scoring against NIST 800-53 PS/PE/AC control families
- **Model Drift Detection**: Statistical z-score testing of model performance metrics
- **Executive Summary Reports**: Aggregated briefing-ready reports with recommendations
- **Export**: JSON and CSV export with optional historical data

### Defense Compliance MCP Tools

| Tool | Description |
|------|-------------|
| `assess_insider_threat` | Run insider threat assessment on user activity (28 behavioral indicators) |
| `generate_siem_events` | Export events in CEF/LEEF/Syslog with MITRE ATT&CK enrichment |
| `evaluate_cleared_personnel` | Run SEAD 4/6 cleared personnel analytics and CE checks |
| `get_compliance_dashboard` | Get NITTF maturity, KRIs, compliance posture, and executive summary |
| `generate_threat_referral` | Generate formal case referral or personnel security action report |

### NIST 800-53 Control Coverage

| Control ID | Control Name | Coverage |
|------------|-------------|----------|
| PS-1 | Personnel Security Policy and Procedures | Maturity assessment |
| PS-2 | Position Risk Designation | Maturity assessment |
| PS-3 | Personnel Screening | Insider threat indicators, CE checks, clearance validation |
| PS-4 | Personnel Termination | Post-termination access detection |
| PS-5 | Personnel Transfer | Access scope violation detection |
| PS-6 | Access Agreements | Reporting compliance, agreement violation detection |
| PS-7 | External Personnel Security | Maturity assessment |
| PS-8 | Personnel Sanctions | Maturity assessment |
| PE-2 | Physical Access Authorizations | Badge tailgating, after-hours physical access |
| PE-3 | Physical Access Control | Physical access anomaly detection |
| PE-6 | Monitoring Physical Access | Physical security integration |
| AC-2 | Account Management | Privilege escalation, dormant accounts, credential sharing |
| AC-3 | Access Enforcement | Need-to-know verification |
| AC-6 | Least Privilege | Access scope violation detection |
| AC-25 | Reference Monitor | Compartment access verification |

### Defense Compliance Architecture

```
compliance/
  __init__.py                  # Package exports
  insider_threat.py            # EO 13587 / NITTF insider threat detection
  siem_integration.py          # CEF/LEEF/Syslog event generation & correlation
  cleared_personnel.py         # SEAD 4/6 cleared personnel analytics
  dashboard_metrics.py         # NITTF maturity, KRIs, compliance posture
```

All compliance modules:
- Run entirely locally with no external service dependencies
- Are thread-safe for concurrent assessments
- Use graceful degradation (server runs without them if not installed)
- Follow existing server patterns (`@mcp.tool()`, `@_monitored()`, `Dict[str, Any]` returns)

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
