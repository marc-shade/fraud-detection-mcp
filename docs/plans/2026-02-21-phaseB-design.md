# Phase B: Agent Identity Layer

## Goal
Add agent credential validation, a local identity registry, and `verify_agent_identity` MCP tool. Integrate identity trust scores into risk calculation.

## Approach
1. Add `AgentIdentityRegistry` class backed by JSON file
2. Add `AgentIdentityVerifier` class for credential validation
3. Add `verify_agent_identity` MCP tool following `_impl` pattern
4. Integrate identity score into `generate_risk_score_impl` for agent traffic
