# Agent Commerce Threshold Calibration Report

**Date**: 2026-05-04T09:42:45.165058
**Sample size**: 200
**Random seed**: 42
**Optimal F1 operating threshold**: `0.45`
**Best F1**: `0.8585` (precision=0.8125, recall=0.9100)

## Operating curve

| threshold | precision | recall | f1 | tp | fp | tn | fn |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.05 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.10 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.15 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.20 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.25 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.30 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.35 | 0.500 | 1.000 | 0.667 | 100 | 100 | 0 | 0 |
| 0.40 | 0.503 | 1.000 | 0.669 | 100 | 99 | 1 | 0 |
| 0.45 | 0.812 | 0.910 | 0.858 | 91 | 21 | 79 | 9 |
| 0.50 | 1.000 | 0.740 | 0.851 | 74 | 0 | 100 | 26 |
| 0.55 | 1.000 | 0.380 | 0.551 | 38 | 0 | 100 | 62 |
| 0.60 | 1.000 | 0.150 | 0.261 | 15 | 0 | 100 | 85 |
| 0.65 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 0.70 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 0.75 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 0.80 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 0.85 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 0.90 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 0.95 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |
| 1.00 | 0.000 | 0.000 | 0.000 | 0 | 0 | 100 | 100 |

## Per-attack-type detection rate at operating threshold

| attack | n | detected | detection_rate | mean_risk | min | max |
|---|---|---|---|---|---|---|
| tampered_signature | 28 | 28 | 1.000 | 0.531 | 0.523 | 0.549 |
| high_amount_unusual_hour | 15 | 15 | 1.000 | 0.624 | 0.602 | 0.627 |
| no_signature_claimed | 25 | 16 | 0.640 | 0.453 | 0.437 | 0.468 |
| novel_merchant_location | 23 | 23 | 1.000 | 0.577 | 0.577 | 0.577 |
| forged_key | 9 | 9 | 1.000 | 0.517 | 0.497 | 0.530 |

## Calibrated config defaults

These are the values currently in `config.AppConfig`. They came from
this script's run on the date above.

```
ACP_API_KEY_INVALID_SIGNAL = 0.1
ACP_API_KEY_VALID_SIGNAL = 0.6
ACP_BACKEND = in_memory
ACP_FAILED_CONFIDENCE_DROP = 0.25
ACP_FINGERPRINT_ANOMALY_THRESHOLD = 0.6
ACP_IDEMPOTENCY_TTL_SECONDS = 86400
ACP_IDENTITY_VERIFIED_THRESHOLD = 0.5
ACP_JWT_EXP_ONLY_SIGNAL = 0.7
ACP_JWT_INVALID_SIGNAL = 0.1
ACP_JWT_NO_EXP_SIGNAL = 0.5
ACP_JWT_VERIFIED_SIGNAL = 0.85
ACP_NONCE_TTL_SECONDS = 480
ACP_PIPELINE_FAILED_TRUST_DROP = 0.3
ACP_PIPELINE_VERIFIED_TRUST_BOOST = 0.15
ACP_REGISTRY_NEW_AGENT_SIGNAL = 0.3
ACP_REPLAY_MAX_ENTRIES = 100000
ACP_VERIFIED_CONFIDENCE_BOOST = 0.15
```