# Agent Commerce Threshold Calibration Report

**Date**: 2026-05-04T08:20:53.348512
**Sample size**: 300
**Random seed**: 42
**Optimal F1 operating threshold**: `0.45`
**Best F1**: `0.7867` (precision=0.7867, recall=0.7867)

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
| 0.35 | 0.515 | 1.000 | 0.680 | 150 | 141 | 9 | 0 |
| 0.40 | 0.582 | 0.920 | 0.713 | 138 | 99 | 51 | 12 |
| 0.45 | 0.787 | 0.787 | 0.787 | 118 | 32 | 118 | 32 |
| 0.50 | 0.975 | 0.527 | 0.684 | 79 | 2 | 148 | 71 |
| 0.55 | 1.000 | 0.227 | 0.370 | 34 | 0 | 150 | 116 |
| 0.60 | 1.000 | 0.047 | 0.089 | 7 | 0 | 150 | 143 |
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
| no_signature_claimed | 36 | 5 | 0.139 | 0.415 | 0.382 | 0.490 |
| novel_merchant_location | 31 | 31 | 1.000 | 0.540 | 0.527 | 0.577 |
| high_amount_unusual_hour | 26 | 26 | 1.000 | 0.586 | 0.551 | 0.627 |
| tampered_signature | 42 | 42 | 1.000 | 0.498 | 0.459 | 0.542 |
| forged_key | 15 | 14 | 0.933 | 0.491 | 0.446 | 0.529 |

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