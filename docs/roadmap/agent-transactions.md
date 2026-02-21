# AI Agent-to-Agent Transaction Fraud Detection

## Overview

AI agents are rapidly becoming autonomous economic actors. Stripe Agent-to-Agent Commerce Protocol (ACP), Visa Intelligent Commerce (TAP), Mastercard Agent Pay, Google A2P, PayPal Agent Ready, Coinbase x402, and Ethereum ERC-8004 are all shipping or announced for 2025-2026. Transmit Security projects a 500% increase in fraud losses from agent-mediated transactions.

This roadmap defines how `fraud-detection-mcp` adapts to detect fraud in agent-initiated and agent-to-agent transactions, where traditional human behavioral biometrics are meaningless.

## Why Agent Transactions Break Traditional Fraud Detection

| Signal | Human Transactions | Agent Transactions |
|--------|-------------------|-------------------|
| Keystroke dynamics | Strong signal | Does not exist |
| Mouse/touch patterns | Strong signal | Does not exist |
| Device fingerprint | Moderate signal | Shared/cloud infra |
| Geographic location | Strong signal | Datacenter IPs, no meaningful geo |
| Session timing | Normal hours, variable | 24/7, millisecond precision |
| Transaction velocity | ~10/day typical | 1,000+/day possible |
| Decision patterns | Hesitation, browsing | Instant, deterministic |
| Identity verification | KYC, 2FA, biometrics | API keys, OAuth tokens, certificates |

**Key insight**: `BehavioralBiometrics` (keystroke, mouse, touch) has near-zero relevance for agent traffic. `NetworkAnalyzer` (graph-based fraud ring detection) has HIGH relevance and becomes the primary detection layer.

## New Fraud Vectors Unique to Agent Transactions

### 1. Prompt Injection Attacks
Malicious instructions embedded in transaction metadata, product descriptions, or API responses that manipulate agent decision-making. An agent told to "approve all transactions from merchant X" via injected context.

### 2. Agent Impersonation
Forged agent identity tokens or stolen API credentials used to initiate transactions. Unlike human identity theft, agent credentials can be cloned perfectly with no behavioral deviation.

### 3. Agent Collusion Rings
Multiple compromised or malicious agents coordinating transactions to launder funds, inflate volumes, or exploit arbitrage. Graph analysis is the primary detection method.

### 4. Mandate Drift / Scope Creep
An agent authorized for "buy office supplies under $100" gradually executing transactions outside its mandate. Requires tracking authorization scope vs. actual behavior over time.

### 5. High-Frequency Micro-Transaction Abuse
Agents executing thousands of small transactions below monitoring thresholds. Traditional velocity checks tuned for human rates fail completely.

### 6. Session Smuggling
Hijacking an authenticated agent session to inject unauthorized transactions within a legitimate transaction stream.

### 7. Synthetic Merchant Fraud
Creating fake merchant endpoints that only agents interact with, invisible to human review processes.

### 8. Model Poisoning via Transaction History
Deliberately feeding agents misleading transaction patterns to bias future decisions.

## Existing Solutions in the Market

| Solution | Approach | Status |
|----------|----------|--------|
| BioCatch Connect 2.0 | Agent-aware behavioral detection, 3,000+ telemetry elements | Production 2025 |
| Oscilar | Real-time ML fraud detection with agent traffic classification | Production |
| Stytch Agent Ready | Agent identity verification and session management | Production 2025 |
| DataDome | Bot/agent traffic classification and intent analysis | Production |
| Stripe ACP | Built-in agent identity + merchant verification | Beta 2025 |
| Visa TAP | Token-based agent authentication standard | Announced 2025 |

**Gap**: No existing solution combines graph-based fraud ring detection with agent behavioral fingerprinting in an MCP-native interface. This is our niche.

## Architecture: What Changes

### Current Pipeline (Human Transactions)
```
Transaction → BehavioralBiometrics (30%) → TransactionAnalyzer (50%) → NetworkAnalyzer (20%) → Risk Score
```

### New Pipeline (Agent Transactions)
```
Transaction → TrafficClassifier → [human | agent | unknown]
                                       │           │
                                       │           ▼
                                       │    AgentIdentityVerifier (25%)
                                       │           │
                                       ▼           ▼
                              BehavioralBiometrics  AgentBehavioralFingerprint (25%)
                                    (30%)                  │
                                       │           ▼
                                       │    MandateVerifier (15%)
                                       │           │
                                       ▼           ▼
                              TransactionAnalyzer   TransactionAnalyzer
                                    (50%)              (20%)
                                       │           │
                                       ▼           ▼
                              NetworkAnalyzer       NetworkAnalyzer + CollusionDetector
                                    (20%)              (15%)
                                       │           │
                                       ▼           ▼
                                  Risk Score    Risk Score
```

### Weight Rebalancing for Agent Traffic

| Component | Human Weight | Agent Weight |
|-----------|-------------|-------------|
| Behavioral (human biometrics OR agent fingerprint) | 30% | 25% |
| Transaction analysis | 50% | 20% |
| Network/graph analysis | 20% | 15% |
| Agent identity verification | 0% | 25% |
| Mandate compliance | 0% | 15% |

## New MCP Tools

### 1. `classify_traffic_source`
Determine whether a transaction originates from a human, AI agent, or unknown source. Uses request metadata, timing patterns, session characteristics, and User-Agent analysis.

**Input**: Transaction metadata, session data, request headers
**Output**: `{ source: "human" | "agent" | "unknown", confidence: float, agent_type: string | null }`

### 2. `verify_agent_identity`
Validate agent credentials, certificate chains, and authorization tokens. Check against known agent registries (Stripe ACP registry, Visa TAP directory).

**Input**: Agent credentials (API key, OAuth token, certificate), claimed identity
**Output**: `{ verified: bool, identity: AgentIdentity, trust_score: float, warnings: list }`

### 3. `analyze_agent_transaction`
Specialized transaction analysis for agent-initiated transactions. Replaces behavioral biometrics with agent behavioral fingerprinting: API call timing consistency, decision pattern analysis, model-specific behavioral signatures.

**Input**: Transaction data, agent identity, historical agent behavior
**Output**: `{ risk_score: float, anomalies: list, fingerprint_match: float, mandate_compliance: float }`

### 4. `detect_agent_collusion`
Graph-based detection of coordinated agent behavior. Extends `NetworkAnalyzer` with temporal correlation, shared infrastructure detection, and transaction flow analysis.

**Input**: Agent identifiers, transaction history, time window
**Output**: `{ collusion_score: float, suspected_ring: list[AgentIdentity], evidence: list, graph_metrics: dict }`

### 5. `verify_transaction_mandate`
Check whether a transaction falls within an agent's authorized scope. Requires mandate definition (spending limits, merchant categories, time windows, geographic restrictions).

**Input**: Transaction data, agent mandate definition
**Output**: `{ compliant: bool, violations: list, drift_score: float, mandate_utilization: float }`

### 6. `score_agent_reputation`
Longitudinal reputation scoring based on transaction history, mandate compliance, identity verification history, and network trust signals.

**Input**: Agent identity, time window
**Output**: `{ reputation_score: float, history_length: int, violation_count: int, trust_network_score: float }`

## Implementation Phases

### Phase A: Foundation (Agent Traffic Classification)
- Add `classify_traffic_source` tool
- Add `is_agent_transaction` field to transaction data model
- Route agent vs. human transactions through appropriate pipelines
- Update `generate_risk_score` to apply different weights based on traffic source

### Phase B: Agent Identity Layer
- Add `verify_agent_identity` tool
- Implement agent credential validation (API keys, OAuth tokens, X.509 certificates)
- Build agent identity registry (local cache of known agents)
- Add identity score to risk calculation

### Phase C: Agent Behavioral Fingerprinting
- Add `analyze_agent_transaction` tool
- Replace `BehavioralBiometrics` with `AgentBehavioralFingerprint` for agent traffic
- Track: API timing patterns, decision consistency, request structure fingerprints
- Build per-agent behavioral baseline using Isolation Forest

### Phase D: Mandate and Collusion Detection
- Add `verify_transaction_mandate` tool
- Add `detect_agent_collusion` tool (extends NetworkAnalyzer)
- Implement temporal correlation analysis for multi-agent coordination
- Add mandate drift detection using sliding window analysis

### Phase E: Reputation and Integration
- Add `score_agent_reputation` tool
- Integrate all agent tools into unified `generate_risk_score` pipeline
- Add agent-specific thresholds and alerting
- Update `explain_decision` to cover agent-specific reasoning

## Research Sources

Key references informing this roadmap:

- Stripe Agent-to-Agent Commerce Protocol (ACP) documentation
- Visa Intelligent Commerce / Token Authentication Protocol (TAP) announcements
- Mastercard Agent Pay and Multi-Token Network specifications
- BioCatch Connect 2.0 agent behavioral detection whitepaper
- Transmit Security research on agent transaction fraud projections
- Oscilar real-time ML fraud detection for agentic commerce
- Stytch Agent Ready identity verification platform
- Coinbase x402 HTTP-native payment protocol specification
- Ethereum ERC-8004 agent authorization standard
- Google A2P (Agent-to-Pay) protocol documentation
- PayPal Agent Ready commerce framework
- Academic research on prompt injection attacks in financial contexts
- OWASP guidelines for AI agent security
