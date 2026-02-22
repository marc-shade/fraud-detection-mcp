# Phase C: Agent Behavioral Fingerprinting

## Goal
Add agent-specific behavioral analysis that replaces human biometrics for agent traffic. Track API timing, decision consistency, and request structure to build per-agent baselines.

## Approach
1. Add `AgentBehavioralFingerprint` class with Isolation Forest baselines
2. Add `analyze_agent_transaction` MCP tool following `_impl` pattern
3. Integrate behavioral fingerprint score into `generate_risk_score_impl` for agent traffic
