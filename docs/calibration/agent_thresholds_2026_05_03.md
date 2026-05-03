# Agent Commerce Threshold Calibration Report

**Date**: 2026-05-03T18:48:15.668715
**Sample size**: 300
**Random seed**: 42
**Optimal F1 operating threshold**: `0.45`
**Best F1**: `0.8304` (precision=0.8633, recall=0.8000)

## Operating curve

| threshold | precision | recall | f1 | tp | fp | tn | fn |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.05 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.10 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.15 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.20 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.25 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.30 | 0.500 | 1.000 | 0.667 | 150 | 150 | 0 | 0 |
| 0.35 | 0.505 | 1.000 | 0.671 | 150 | 147 | 3 | 0 |
| 0.40 | 0.677 | 0.893 | 0.770 | 134 | 64 | 86 | 16 |
| 0.45 | 0.863 | 0.800 | 0.830 | 120 | 19 | 131 | 30 |
| 0.50 | 0.963 | 0.520 | 0.675 | 78 | 3 | 147 | 72 |
| 0.55 | 1.000 | 0.227 | 0.370 | 34 | 0 | 150 | 116 |
| 0.60 | 1.000 | 0.060 | 0.113 | 9 | 0 | 150 | 141 |
| 0.65 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 0.70 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 0.75 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 0.80 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 0.85 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 0.90 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 0.95 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |
| 1.00 | 0.000 | 0.000 | 0.000 | 0 | 0 | 150 | 150 |

## Per-attack-type detection rate at operating threshold

| attack | n | detected | detection_rate | mean_risk | min | max |
|---|---|---|---|---|---|---|
| no_signature_claimed | 36 | 6 | 0.167 | 0.415 | 0.387 | 0.483 |
| novel_merchant_location | 31 | 31 | 1.000 | 0.548 | 0.535 | 0.585 |
| high_amount_unusual_hour | 26 | 26 | 1.000 | 0.596 | 0.560 | 0.638 |
| tampered_signature | 42 | 42 | 1.000 | 0.495 | 0.460 | 0.535 |
| forged_key | 15 | 15 | 1.000 | 0.483 | 0.452 | 0.533 |

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