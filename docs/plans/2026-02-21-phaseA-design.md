# Phase A: Agent Traffic Classification

## Goal
Add the ability to classify whether a transaction originates from a human, AI agent, or unknown source, and route it through the appropriate analysis pipeline with adjusted risk weights.

## Approach
1. Add `TrafficClassifier` class with heuristic-based classification
2. Add `classify_traffic_source` MCP tool following existing `_impl` pattern
3. Add optional agent fields to data models
4. Update `generate_risk_score_impl` to use different weights for agent traffic
