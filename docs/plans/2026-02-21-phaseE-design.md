# Phase E: Reputation and Integration

## Goal
Add longitudinal agent reputation scoring and update explain_decision for agent-specific reasoning. Completes the agent-to-agent transaction fraud detection roadmap.

## Approach
1. Add `AgentReputationScorer` class — computes reputation from existing registry, fingerprint, and collusion data (trust 40%, history 25%, consistency 25%, collusion 10%)
2. Add `score_agent_reputation` MCP tool (#19) following `_impl` pattern
3. Update `explain_decision_impl` to recognize agent-specific components (identity, fingerprint, mandate, collusion, reputation)
