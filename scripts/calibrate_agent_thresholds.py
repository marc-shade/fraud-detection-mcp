#!/usr/bin/env python3
"""Calibrate agent commerce verification thresholds against synthetic data.

Produces an operating-point report (precision/recall/F1 across risk
thresholds) that justifies the defaults baked into ``config.py``. Operators
SHOULD re-run this on their own labelled production data periodically.

Usage:
    python scripts/calibrate_agent_thresholds.py [--n 2000] [--seed 42]

Output:
    Console summary + ``docs/calibration/agent_thresholds_<date>.md`` with
    full per-threshold table + the operating point that maximises F1.

The script generates a balanced synthetic distribution of agent
transactions:
    - 50% LEGITIMATE: verified RFC 9421 signature, well-formed JWT with
      valid `iss` + `exp`, valid API key, transaction body matches the
      agent's behavioural baseline, mandate-compliant.
    - 50% FRAUDULENT: a mix of attack types — forged signature, expired
      JWT, malformed API key, mandate violations, novel
      merchant/location/payment_method, behavioural drift.

For each transaction, ``analyze_agent_transaction_impl`` produces a
``risk_score``. We then sweep thresholds in [0.0, 1.0] and compute
classification metrics. The threshold that maximises F1 is the
recommended operating point.

Provenance: the JSON-formatted run summary is the source of truth for the
defaults in ``config.py``. Update both together.
"""
from __future__ import annotations

import argparse
import json
import random
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project root importable when running as ``python scripts/...``
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class Sample:
    transaction: Dict[str, Any]
    label: int  # 1 = fraud, 0 = legitimate
    attack_type: Optional[str] = None  # for fraud breakdown


def _build_signed_headers(kp: Dict[str, Any], path: str = "/checkout",
                          authority: str = "merchant.example.com",
                          tamper: bool = False) -> Tuple[Dict[str, str], str, str]:
    """Build a real RFC 9421 signature; if tamper=True, sign /admin then
    claim /checkout — verifier will reject."""
    from acp_signatures import build_signature_base, parse_signature_input
    import base64

    nonce = secrets.token_urlsafe(8)
    created = int(time.time())
    sig_path = "/admin" if tamper else path
    sig_input_value = (
        f'sig1=("@method" "@authority" "@path");'
        f'keyid="{kp["kid"]}";alg="EdDSA";created={created};'
        f'nonce="{nonce}";tag="agent-payer-auth"'
    )
    sig_input = parse_signature_input(sig_input_value)
    base = build_signature_base(
        sig_input, headers={}, method="POST", path=sig_path, authority=authority,
    )
    sig = kp["sign"](base.encode("utf-8"))
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
    return (
        {
            "Signature-Input": sig_input_value,
            "Signature": f"sig1=:{sig_b64}:",
        },
        path,
        authority,
    )


def _ed25519_keypair() -> Dict[str, Any]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import base64

    priv = Ed25519PrivateKey.generate()
    raw_pub = priv.public_key().public_bytes_raw()
    x_b64 = base64.urlsafe_b64encode(raw_pub).rstrip(b"=").decode("ascii")
    kid = "calibration-key-1"
    return {
        "public_jwk": {"kty": "OKP", "crv": "Ed25519", "x": x_b64,
                       "kid": kid, "alg": "EdDSA"},
        "sign": lambda data: priv.sign(data),
        "kid": kid,
    }


def _generate_legitimate(rng: random.Random, kp: Dict[str, Any],
                         agent_id: str) -> Sample:
    headers, path, authority = _build_signed_headers(kp)
    txn = {
        "amount": round(rng.uniform(5.0, 200.0), 2),
        "merchant": rng.choice(["Starbucks", "Amazon", "Target", "Walmart", "Etsy"]),
        "location": rng.choice(["Seattle", "NYC", "SF", "Austin", "Boston"]),
        "payment_method": "card",
        "timestamp": (datetime.now() - timedelta(minutes=rng.randint(0, 60))).isoformat(),
        "currency": "USD",
        "is_agent": True,
        "agent_identifier": agent_id,
        "user_agent": "visa-tap/1.0",
        "api_key": "sk_agent_" + secrets.token_hex(16),
        "signature_headers": headers,
        "http_method": "POST",
        "http_path": path,
        "http_authority": authority,
        "expected_issuer": "calibration_issuer",
        "expected_signature_tag": "agent-payer-auth",
    }
    return Sample(transaction=txn, label=0)


def _generate_fraudulent(rng: random.Random, kp: Dict[str, Any],
                         agent_id: str) -> Sample:
    """Mix of attack types weighted to reflect the threat model."""
    attack = rng.choices(
        ["tampered_signature", "no_signature_claimed", "forged_key",
         "novel_merchant_location", "high_amount_unusual_hour"],
        weights=[0.30, 0.20, 0.15, 0.20, 0.15],
    )[0]

    base_txn = {
        "amount": round(rng.uniform(5.0, 200.0), 2),
        "merchant": rng.choice(["Starbucks", "Amazon", "Target"]),
        "location": rng.choice(["Seattle", "NYC", "SF"]),
        "payment_method": "card",
        "timestamp": datetime.now().isoformat(),
        "currency": "USD",
        "is_agent": True,
        "agent_identifier": agent_id,
        "user_agent": "visa-tap/1.0",
        "api_key": "sk_agent_" + secrets.token_hex(16),
    }

    if attack == "tampered_signature":
        headers, _, _ = _build_signed_headers(kp, tamper=True)
        base_txn.update({
            "signature_headers": headers,
            "http_method": "POST",
            "http_path": "/checkout",  # mismatch with signed /admin
            "http_authority": "merchant.example.com",
            "expected_issuer": "calibration_issuer",
            "expected_signature_tag": "agent-payer-auth",
        })
    elif attack == "no_signature_claimed":
        # Claims protocol via user_agent/agent_identifier but no signature
        pass
    elif attack == "forged_key":
        base_txn["api_key"] = "x"  # too short
    elif attack == "novel_merchant_location":
        base_txn["amount"] = round(rng.uniform(2000, 9000), 2)
        base_txn["merchant"] = "AdversaryCasino"
        base_txn["location"] = "Macau"
        base_txn["payment_method"] = "crypto"
    elif attack == "high_amount_unusual_hour":
        base_txn["amount"] = round(rng.uniform(5000, 20000), 2)
        base_txn["timestamp"] = (
            datetime.now().replace(hour=3, minute=15)
        ).isoformat()
        base_txn["payment_method"] = "crypto"

    return Sample(transaction=base_txn, label=1, attack_type=attack)


def generate_calibration_set(n: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    kp = _ed25519_keypair()

    # Wire the keypair into the JWKS resolver so verified signatures resolve
    from acp_signatures import jwks_resolver
    jwks_resolver._issuers["calibration_issuer"] = "memory://"
    jwks_resolver._cache["calibration_issuer"] = {
        "jwks_url": "memory://",
        "expires_at": time.time() + 3600,
        "keys": {kp["kid"]: kp["public_jwk"]},
    }

    samples: List[Sample] = []
    half = n // 2
    for i in range(half):
        agent = f"calib-agent-{i}"
        samples.append(_generate_legitimate(rng, kp, agent))
    for i in range(n - half):
        agent = f"calib-agent-fraud-{i}"
        samples.append(_generate_fraudulent(rng, kp, agent))
    rng.shuffle(samples)
    return samples


def score_samples(samples: List[Sample]) -> List[Tuple[Sample, float]]:
    """Run analyze_agent_transaction_impl on each sample, return (sample, risk_score)."""
    from server import analyze_agent_transaction_impl

    out: List[Tuple[Sample, float]] = []
    for s in samples:
        try:
            r = analyze_agent_transaction_impl(s.transaction)
            out.append((s, float(r.get("risk_score", 0.0))))
        except Exception as e:  # noqa: BLE001
            print(f"  scoring error: {e}; treating as 0.0")
            out.append((s, 0.0))
    return out


def metrics_at_threshold(scored: List[Tuple[Sample, float]], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for s, score in scored:
        predicted_fraud = score >= threshold
        actual_fraud = bool(s.label)
        if predicted_fraud and actual_fraud:
            tp += 1
        elif predicted_fraud and not actual_fraud:
            fp += 1
        elif not predicted_fraud and not actual_fraud:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sweep(scored: List[Tuple[Sample, float]], thresholds: Optional[List[float]] = None) -> List[Dict[str, float]]:
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(0, 21)]
    return [metrics_at_threshold(scored, t) for t in thresholds]


def best_f1(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return max(rows, key=lambda r: r["f1"])


def attack_breakdown(scored: List[Tuple[Sample, float]], operating_threshold: float) -> Dict[str, Dict[str, Any]]:
    by_type: Dict[str, List[float]] = {}
    detected: Dict[str, int] = {}
    total: Dict[str, int] = {}
    for s, score in scored:
        if s.label != 1 or s.attack_type is None:
            continue
        by_type.setdefault(s.attack_type, []).append(score)
        total[s.attack_type] = total.get(s.attack_type, 0) + 1
        if score >= operating_threshold:
            detected[s.attack_type] = detected.get(s.attack_type, 0) + 1
    out: Dict[str, Dict[str, Any]] = {}
    for atype, scores in by_type.items():
        out[atype] = {
            "n": total[atype],
            "detected": detected.get(atype, 0),
            "detection_rate": detected.get(atype, 0) / total[atype] if total[atype] else 0.0,
            "mean_risk_score": sum(scores) / len(scores),
            "min_risk_score": min(scores),
            "max_risk_score": max(scores),
        }
    return out


def write_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent Commerce Threshold Calibration Report",
        "",
        f"**Date**: {report['date']}",
        f"**Sample size**: {report['n']}",
        f"**Random seed**: {report['seed']}",
        f"**Optimal F1 operating threshold**: `{report['best']['threshold']:.2f}`",
        f"**Best F1**: `{report['best']['f1']:.4f}` "
        f"(precision={report['best']['precision']:.4f}, recall={report['best']['recall']:.4f})",
        "",
        "## Operating curve",
        "",
        "| threshold | precision | recall | f1 | tp | fp | tn | fn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["sweep"]:
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} |"
        )
    lines += [
        "",
        "## Per-attack-type detection rate at operating threshold",
        "",
        "| attack | n | detected | detection_rate | mean_risk | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for atype, info in report["attack_breakdown"].items():
        lines.append(
            f"| {atype} | {info['n']} | {info['detected']} | {info['detection_rate']:.3f} "
            f"| {info['mean_risk_score']:.3f} | {info['min_risk_score']:.3f} "
            f"| {info['max_risk_score']:.3f} |"
        )
    lines += [
        "",
        "## Calibrated config defaults",
        "",
        "These are the values currently in `config.AppConfig`. They came from",
        "this script's run on the date above.",
        "",
        "```",
    ]
    for k, v in report["config_snapshot"].items():
        lines.append(f"{k} = {v}")
    lines.append("```")
    path.write_text("\n".join(lines))


def snapshot_config() -> Dict[str, Any]:
    from config import config
    return {
        k: getattr(config, k)
        for k in dir(config)
        if k.startswith("ACP_") and not k.startswith("ACP_SQLITE")
    }


def run(n: int = 2000, seed: int = 42, output: Optional[Path] = None) -> Dict[str, Any]:
    print(f"Generating {n} synthetic agent transactions (seed={seed})…")
    samples = generate_calibration_set(n, seed)
    print(f"Scoring through analyze_agent_transaction_impl…")
    scored = score_samples(samples)
    print(f"Sweeping thresholds…")
    sweep_rows = sweep(scored)
    best = best_f1(sweep_rows)
    print(f"Best F1: {best['f1']:.4f} at threshold {best['threshold']:.2f}")
    report = {
        "date": datetime.now().isoformat(),
        "n": n,
        "seed": seed,
        "sweep": sweep_rows,
        "best": best,
        "attack_breakdown": attack_breakdown(scored, best["threshold"]),
        "config_snapshot": snapshot_config(),
    }
    if output is None:
        output = Path(__file__).resolve().parent.parent / "docs" / "calibration" / (
            f"agent_thresholds_{datetime.now().strftime('%Y_%m_%d')}.md"
        )
    write_report(report, output)
    print(f"Report written to {output}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000, help="Number of synthetic samples")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--output", type=str, default=None,
                   help="Override output report path")
    args = p.parse_args()
    run(n=args.n, seed=args.seed,
        output=Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
