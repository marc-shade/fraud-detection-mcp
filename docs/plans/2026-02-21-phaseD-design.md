# Phase D: Mandate and Collusion Detection

## Goal
Add mandate verification and agent collusion detection. Two new MCP tools (#17, #18) that check whether agents operate within authorized scope and detect coordinated fraudulent behavior.

## Approach
1. Add `MandateVerifier` class — stateless, mandate passed per-call, checks constraints and calculates drift score
2. Add `CollusionDetector` class — directed graph of agent interactions, detects temporal clustering, circular flows, shared infrastructure
3. Add `verify_transaction_mandate` and `detect_agent_collusion` MCP tools following `_impl` pattern
4. Integrate mandate compliance into `generate_risk_score_impl` for agent traffic
