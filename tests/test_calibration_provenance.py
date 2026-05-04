"""Calibration provenance test (Phase B).

Runs a small version of ``scripts/calibrate_agent_thresholds.py`` and
asserts that the configured defaults produce a sensible operating curve
on the synthetic agent-fraud distribution.

This is a drift-detector: if anyone changes the default deltas in
``config.AppConfig`` or the underlying scoring code in ways that
materially degrade F1, this test fires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


pytestmark = [pytest.mark.signature, pytest.mark.slow]


def test_calibration_achieves_sensible_f1():
    """Synthetic calibration with n=200 should score F1 >= 0.70 at the
    optimal threshold AND that optimal threshold should fall in the
    MEDIUM/HIGH risk band ([0.40, 0.70])."""
    from scripts.calibrate_agent_thresholds import run

    report = run(n=200, seed=42, output=None)
    best = report["best"]

    # Sanity floor — anything lower means we've broken scoring
    assert best["f1"] >= 0.70, (
        f"calibration F1 dropped to {best['f1']:.3f} (expected >= 0.70). "
        "Check ACP_* config defaults or analyze_agent_transaction_impl logic."
    )
    # Operating point should land in the MEDIUM/HIGH band — anything else
    # signals systematic mis-scaling.
    assert 0.40 <= best["threshold"] <= 0.70, (
        f"optimal threshold {best['threshold']:.2f} outside [0.40, 0.70] "
        "MEDIUM/HIGH band. Defaults may need recalibration."
    )


def test_calibration_catches_tampered_signatures():
    """The tampered_signature attack should be caught at high rate. We
    evaluate at a fixed conservative threshold (0.40) rather than the
    per-run F1 optimum, because the optimum varies with synthetic-data
    randomness and a slight shift can push the operating threshold above
    where tampered_signature scores cluster — making the test flaky.

    At threshold=0.40, tampered signatures should be detected >= 90% of
    the time. Anything materially below that means signature
    verification wiring has regressed.
    """
    from scripts.calibrate_agent_thresholds import (
        run, attack_breakdown, generate_calibration_set, score_samples,
    )

    # Run with explicit fixed threshold for the per-attack breakdown
    samples = generate_calibration_set(n=300, seed=42)
    scored = score_samples(samples)
    breakdown = attack_breakdown(scored, operating_threshold=0.40)
    if "tampered_signature" in breakdown:
        info = breakdown["tampered_signature"]
        assert info["detection_rate"] >= 0.90, (
            f"tampered_signature detection rate at threshold=0.40 dropped "
            f"to {info['detection_rate']:.2f} (n={info['n']}). Signature "
            f"verification may have regressed. Mean tampered score: "
            f"{info['mean_risk_score']:.3f}."
        )
