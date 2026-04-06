# Fraud Detection MCP — Quick Start

## 1. Install

```bash
git clone https://github.com/marc-shade/fraud-detection-mcp
cd fraud-detection-mcp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the MCP Server

```bash
python server.py
```

The server starts with models trained on synthetic data. For production use, train with your own data using the `train_models` tool.

## 3. Connect via Claude Desktop

Add to your Claude Desktop config (`~/.claude.json` or Claude Desktop settings):

```json
{
  "mcpServers": {
    "fraud-detection-mcp": {
      "command": "/path/to/fraud-detection-mcp/venv/bin/python",
      "args": ["/path/to/fraud-detection-mcp/server.py"]
    }
  }
}
```

## 4. Use the Tools

### Analyze a Transaction
```
analyze_transaction({"transaction_id": "txn_123", "amount": 5000.0, "merchant": "Electronics Store", "location": "New York, NY"})
```

### Check Agent Traffic
```
classify_traffic_source({"transaction_id": "txn_456", "amount": 100.0}, {"user_agent": "StripeACP/1.0"})
```

### Run a Benchmark
```
run_benchmark(num_transactions=100, fraud_percentage=10.0)
```

### Get System Health
```
health_check()
```

## Available Tools (24)

**Core Detection (5):** `analyze_transaction`, `detect_behavioral_anomaly`, `assess_network_risk`, `generate_risk_score`, `explain_decision`

**Agent Protection (6):** `classify_traffic_source`, `verify_agent_identity`, `analyze_agent_transaction`, `verify_transaction_mandate`, `detect_agent_collusion`, `score_agent_reputation`

**Operations (8):** `train_models`, `get_model_status`, `analyze_batch`, `get_inference_stats`, `generate_synthetic_dataset`, `analyze_dataset`, `run_benchmark`, `health_check`

**Defense Compliance (5):** `assess_insider_threat`, `generate_siem_events`, `evaluate_cleared_personnel`, `get_compliance_dashboard`, `generate_threat_referral`

## Default Detection Pipeline

The server uses **Isolation Forest + Autoencoder (PyTorch)** ensemble by default. Models are initialized on synthetic data at startup and can be replaced by training on real data:

```
train_models(data_path="your_data.csv", test_size=0.2)
```

## Optional Dependencies

```bash
# XGBoost training support
pip install -r requirements-optional.txt

# Development/testing
pip install -r requirements-dev.txt
```

## Run Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

## Troubleshooting

**Server won't start:** Check that `fastmcp` is installed (`pip install fastmcp`).

**SHAP not working:** Install SHAP (`pip install shap`) or the server falls back to feature importance.

**PyTorch not available:** The autoencoder ensemble member is disabled. Isolation Forest still runs.

See [README.md](README.md) for full documentation.
