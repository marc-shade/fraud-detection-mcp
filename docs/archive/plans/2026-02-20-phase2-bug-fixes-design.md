# Phase 2: Bug Fixes and Edge Case Hardening

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement the corresponding implementation plan task-by-task.

**Goal:** Fix all 4 pre-existing test failures and harden edge cases to reach 213/213 tests passing.

**Architecture:** Targeted fixes to `BehavioralBiometrics` and `TransactionAnalyzer` classes in `server.py`, plus test assertion corrections where the tests had wrong expectations.

## 1. BehavioralBiometrics `invalid_data` Response Missing `error` Key

The `invalid_data` response `{'risk_score': 0.0, 'confidence': 0.0, 'status': 'invalid_data'}` is missing an `'error'` key. Two tests expect `'error' in result`.

**Fix:** Add `'error'` key with a description to the `invalid_data` response. Also add type validation at the top of `analyze_keystroke_dynamics` — reject non-list input before trying to process it.

**Affects:**
- `test_behavioral_analyzer_with_invalid_type` (status `invalid_data` but no `error` key)
- `test_error_handling_malformed_data` (same issue)

## 2. `_extract_keystroke_features` Returns `None` for Missing Timing Keys

When keystroke data has `press_time` but no `release_time`, no dwell/flight times are extracted, and the function returns `None`. The test expects zeroed features instead.

**Fix:** Return the 10-element zero vector `[0.0] * 10` instead of `None` when no timing data can be extracted. This is the "no signal" case, not a failure.

**Affects:**
- `test_zero_features_when_no_dwell_times`

## 3. Non-numeric Timing Values Not Validated

When `press_time` is `'invalid'` (string), subtraction would throw `TypeError`. Currently masked because the test only has 1 item (caught by `len < 2`).

**Fix:** Add type checking for individual timing values during extraction. Skip non-numeric values gracefully.

## 4. `TransactionAnalyzer.xgb_model` is Dead Placeholder

In Phase 1 we removed dead XGBoost training code. The `xgb_model = None` placeholder serves no purpose.

**Fix:** Remove the attribute entirely and update the test to not assert it.

## Out of Scope

- v2 module integration (Phase 3)
- New features or tools
- Model training pipeline
