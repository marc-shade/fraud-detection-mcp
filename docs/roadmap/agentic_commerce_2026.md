# Agentic Commerce 2026 — fraud-detection-mcp Feature Roadmap

**Date:** 2026-05-03
**Source:** Research synthesis from Stripe Sessions 2026 announcements, Visa TAP / Mastercard Verifiable Intent specs, Google AP2 open-source release, Coinbase x402 Foundation, OWASP LLM Top 10 2025, and 6 academic threat-model papers (arXiv 2601.22569, 2604.15367, 2603.17419, 2505.11717, 2410.05451, 2009.09497).
**Triggered by:** Nate B Jones video *"Walmart's ChatGPT Checkout Failed. Stripe Knows Why."* (2026-05-03)

---

## 1. The Five-Protocol Landscape

| Protocol | Wire substrate | Mandate location | Crypto | Status |
|---|---|---|---|---|
| **Stripe ACP** (OpenAI co-author) | OpenAPI 3.1, date-versioned `2026-04-17`. Headers: `Authorization` / `Idempotency-Key` / `Request-Id` / `Signature` / `Timestamp` / `API-Version` | NOT in ACP envelope — lives in Shared Payment Token (SPT) `allowance` object | Network tokenization (cryptogram + ECI), 3DS via `next_action` | GA. Walmart, Target, Sephora, Etsy, Shopify, Lowe's, Best Buy, Home Depot, Wayfair live |
| **Visa TAP** | RFC 9421 HTTP Message Signatures. `Signature-Input` + `Signature` headers. Body objects `agenticConsumer` + `agenticPaymentContainer` | No formal mandate yet; HTTP 402 "Browsing IOU" `{invoiceId, amount, cardAcceptorId}` | Ed25519 (recognition), PS256 / ES256 (consumer/payment). JWKS at `mcp.visa.com/.well-known/jwks` | GA spec, 2026 mainstream |
| **Mastercard Verifiable Intent + Agent Pay** | Same RFC 9421 substrate via Web Bot Auth (Cloudflare partnership) | 3-layer credential chain: L1 (user-key binding via `cnf.jwk`, FIDO-style), L2 (Immediate or constraint-bearing Mandates with 8 registered constraints), L3a (network) / L3b (merchant) | W3C Verifiable Credentials | 2026 announcement |
| **Google AP2** | Open source `github.com/google-agentic-commerce/AP2`. Built on A2A (messaging) + MCP (tool use) | 3 mandate types as W3C VCs: Intent Mandate (HNP), Cart Mandate (user-signed at purchase, hardware-backed key), Payment Mandate (network-visible) | ECDSA P-256 minimum | Open spec, payment-method-agnostic |
| **Coinbase x402** | Revives HTTP 402. `PAYMENT-REQUIRED` b64-JSON header (scheme/network/recipient/max amount/asset/resource/maxTimeoutSeconds). Client retries with `PAYMENT-SIGNATURE` | Per-call payload (`exact` scheme live; `upto` resource-metered theoretical) | EIP-3009 / Permit2 USDC, Solana, Stellar | x402 Foundation Oct 2025; Cloudflare + Stripe support |

**Microsoft + Meta:** consumers of the above. Microsoft Copilot Checkout + Brand Agents (Jan 2026) adopt ACP and integrate Mastercard Agent Pay + Visa Intelligent Commerce. Meta agentic commerce in WhatsApp/Instagram, no Llama Stack payment module documented.

**Anthropic:** no first-party payment primitive — strategy is MCP-as-substrate. PayPal-Anthropic partnership ships an MCP server in `claude.ai/directory`.

**Standardization:** W3C Web Payments WG actively discussing agent auth (no REC; EU dynamic-linking blocks fully autonomous payments today). 5 competing IETF drafts (`klrc-aiagent-auth`, `sharif-openid-agent-identity`, `ni-wimse-ai-agent-identity`, `drake-agent-identity-registry`, `larsson-aitlp`). All early-stage.

## 2. The Walmart Failure — Conversion, Not Fraud

**Verified data** (Daniel Danker, Walmart EVP, ~2026-03-23): ChatGPT Instant Checkout converted at **1.18% vs Walmart.com's ~3.5% (3× worse)**, 77.45% abandonment.

**Root causes** (none are fraud):
- Single-item cart only (no bundles → "five separate boxes" problem)
- No loyalty / account linking
- No US sales-tax remit infrastructure on OpenAI's side as of Feb 2026
- Refunds / cancellations / CS stayed on merchant — split-brain accountability

**Pivot 2026-03-24:** ACP rev now lets merchants run their own checkout (Stripe or Salesforce); Walmart embedded Sparky inside ChatGPT.

**Implication for this project:** Not a fraud product gap, but a "commerce-completeness scoring" feature could be valuable to merchants evaluating agent-checkout integrations. See Tier 2.3.

## 3. Threat Model — Academic Substrate

| Vector | Status | Primary source |
|---|---|---|
| Indirect prompt injection → wallet exfil | Demonstrated in shipping product (Brave Comet, Aug 2025) | brave.com/blog/comet-prompt-injection/ |
| Mandate forging on AP2 | Red-teamed: *Whispers of Wealth* (Branded Whisper, Vault Whisper) | arXiv 2601.22569 |
| SoK on agentic commerce security — 12+ vectors | Apr 2026 | arXiv 2604.15367 |
| Streaming-payment runaway (Nov 2025 LangChain $47K loop) | Documented incident | dev.to/waxell/the-47000-agent-loop |
| Behavioral drift adversary (Relative Loss of Robustness) | Formalized | arXiv 2009.09497 |
| WebInject — adversarial pixels in rendered web pages | arXiv 2505.11717 | (out of fraud-MCP scope but worth flagging) |
| TOCTOU: BIP70 refund-address bypass | Direct analog for mandate-check / settle race | arXiv 2103.08436 |
| Caging the Agents — credential-proxy sidecars + per-agent egress allowlists | 90-day production | arXiv 2603.17419 |
| Insider operator audit (Anthropic GTG-1002) | Threat reports Aug 2025, Nov 2025 | anthropic.com/news/disrupting-AI-espionage |

## 4. Critical Code Discrepancy (FIXED in this initiative)

`AgentBehavioralFingerprint` was documented as "8 features (log_amount, payment_method_hash, merchant_hash, location_hash, hour_of_day, field_completeness, timing_interval, amount_magnitude)" but implemented only 6 timing/decision features. Stolen-token replay by another agent at the same API timing distribution would sail through. **Tier 0.1 extends to 12 features** (the documented 8 plus the original 4 ML-ready timing features) — payment_method_hash / merchant_hash / location_hash / log_amount / hour_of_day / field_completeness now actually computed and fed to the Isolation Forest.

## 5. Tiered Roadmap

### Tier 0 — Production-grade (no theater, no stubs)

| # | Feature | Status |
|---|---|---|
| 0.1 | Behavioral fingerprint 6→**13 features** (cyclical hour sin/cos) | ✅ Implemented + wired in both call sites |
| 0.2 | RFC 9421 HTTP Message Signature verifier (`acp_signatures.py`) + JWKS resolver | ✅ Implemented |
| 0.3 | NonceCache (8-min TTL, Visa TAP-compatible) | ✅ Implemented |
| 0.4 | IdempotencyStore (ACP-required) | ✅ Implemented |
| 0.5 | TrafficClassifier `claimed_protocol` vs `verified_protocol` | ✅ Implemented + **wired through `analyze_agent_transaction_impl`** (F1) |
| 0.6 | AgentIdentityVerifier — real JWT signature verification (not just `exp`) | ✅ Implemented |
| F1  | `signature_headers` flows from `transaction_data` through pipeline → anomalies + identity_trust adjustment + verified_protocol in result | ✅ Implemented |
| F2  | Cyclical hour-of-day encoding (sin/cos) — 23:00↔00:00 close in feature space | ✅ Implemented |
| F3  | `validate_nonce` peek-by-default + `consume_nonce` atomic-record (TOCTOU-safe) | ✅ Implemented |
| F4  | `verified_protocol` reports protocol enum (`visa_tap`) not raw issuer (`visa`) | ✅ Implemented via `ISSUER_TO_PROTOCOL` |
| F5  | RFC 9421 Content-Digest (RFC 9530 sha-256/sha-512), `@query`, `@query-param;name="..."` | ✅ Implemented |
| F6  | JWKS fetch retry-with-backoff (3 attempts, 0.5/1/2s), User-Agent, structured logging | ✅ Implemented |
| F7  | Pre-register verified JWKS URLs (research-gated) | ✅ Visa only (others not publicly published — DID-based, on-chain, bilateral, or undocumented) |
| **P-A** | **Config-tunable thresholds** — every magic number now in `config.AppConfig` (env-overridable: `ACP_VERIFIED_CONFIDENCE_BOOST` etc.) | ✅ Implemented |
| **P-B** | **Synthetic calibration + provenance test** — `scripts/calibrate_agent_thresholds.py` writes `docs/calibration/agent_thresholds_<date>.md`; `test_calibration_provenance.py` is a drift-detector | ✅ Implemented |
| **P-C** | **Pluggable backends** — `NonceBackend` / `IdempotencyBackend` ABCs + InMemory + SQLite (WAL mode + busy_timeout) | ✅ Implemented |
| **P-D** | **Strong stolen-token replay test** — 60-obs baseline + 5-feature divergent attack must score risk_score >= 0.7 (HIGH) | ✅ Implemented (was >= 0.4 moderate, now >= 0.7 strong) |
| **P-E** | **Multi-process subprocess test** — two real Python children share a SQLite file; nonce consumed by A is rejected by B | ✅ Implemented |

### Tier 1 — Production-ready (3–6 weeks)

| # | Feature | Where |
|---|---|---|
| 1.1 | AP2 Verifiable Credential verifier (Intent / Cart / Payment mandates as JWS) | new `ap2_mandates.py` + 3 MCP tools |
| 1.2 | Stripe SPT `allowance` schema validator extending `MandateVerifier.verify` | `server.py:1665-1788` |
| 1.3 | Mastercard Verifiable Intent L1 / L2 / L3 parser | new `mastercard_vi.py` |
| 1.4 | Streaming-payment / rate-of-burn detector + budget enforcement hook | extend `UserTransactionHistory` |
| 1.5 | x402 challenge / response inspector | new `x402_validator.py` |
| 1.6 | Indirect-prompt-injection precursor signal | new `prompt_injection_signals.py` |
| 1.7 | 2-phase mandate reserve / settle / release (TOCTOU defense) | extend `MandateVerifier` |
| 1.8 | Risk-signal export in ACP `risk_signals[]` format | new `acp_risk_export.py` |

### Tier 2 — Differentiators (6–12 weeks)

| # | Feature | Where |
|---|---|---|
| 2.1 | Behavioral-drift adversarial robustness (RLR metric, sliding-window baseline) | `AgentBehavioralFingerprint` |
| 2.2 | Cross-merchant agent-ID privacy (per-merchant pseudonyms, k-anonymity) | new module |
| 2.3 | Agent commerce-completeness scorer (Walmart-failure axes) | new module |
| 2.4 | Webhook event family — `agent.mandate.exceeded`, `agent.token.replay_detected`, etc. | new `webhook_emitter.py` |
| 2.5 | Stripe Machine Payments Protocol (MPP) parser | when spec lands |
| 2.6 | Stripe Issuing-for-Agents single-use card detector | `feature_engineering.py` |
| 2.7 | W3C Verifiable Credential generic verifier (DRY across AP2 + Mastercard VI) | new `vc_verifier.py` |

## 6. Pydantic Models for `models_validation.py`

```
StripeACPHeaders        — Authorization, Idempotency-Key, Request-Id, Signature, Timestamp, API-Version
StripeAllowance         — reason, max_amount, currency, checkout_session_id, merchant_id, expires_at
VisaTAPSignature        — Signature-Input fields + agenticConsumer + agenticPaymentContainer
MastercardVI_Mandate    — L1 cnf.jwk, L2 constraints[], L3 view enum
AP2IntentMandate / AP2CartMandate / AP2PaymentMandate
X402PaymentRequired     — scheme, network, recipient, max_amount, asset, resource, maxTimeoutSeconds
RiskSignalACP           — type, score, action enum {blocked, manual_review, authorized}
AgentBudget             — agent_id, window_seconds, budget, current_burn, projected_overshoot_at_seconds
PromptInjectionContext  — recent_fetches[], recent_tool_calls[], untrusted_domains[]
```

## 7. New MCP Tools (target: 24 → 36)

Tier 0 adds 3; Tier 1 adds ~10. Naming follows existing `_impl` pattern.

```
verify_agent_signature           (RFC 9421)            ← Tier 0, this initiative
check_idempotency_key                                   ← Tier 0, this initiative
validate_nonce                                          ← Tier 0, this initiative
verify_ap2_intent_mandate                               ← Tier 1
verify_ap2_cart_mandate                                 ← Tier 1
verify_ap2_payment_mandate                              ← Tier 1
verify_mastercard_vi                                    ← Tier 1
analyze_x402_payment                                    ← Tier 1
enforce_agent_budget                                    ← Tier 1
reserve_mandate / settle_mandate / release_mandate      ← Tier 1
analyze_prompt_injection_precursors                     ← Tier 1
export_risk_signals_acp                                 ← Tier 1
emit_agent_event                                        ← Tier 2
score_acp_integration_completeness                      ← Tier 2
```

## 8. Test Markers

```
signature        — RFC 9421 / JWS / JWT verification
ap2              — AP2 mandate verification
mastercard_vi    — Mastercard Verifiable Intent
x402             — x402 challenge/response
streaming        — burn-rate / budget enforcement
prompt_injection — precursor detection
toctou           — 2-phase mandate reserve/settle race tests
```

Use *Whispers of Wealth* (arXiv 2601.22569) attacks as a published red-team baseline.

## 9. What NOT to Build

- Your own mandate format — AP2 + ACP allowance + Mastercard VI cover the space. Adopt, don't compete.
- Card-network-specific anti-fraud — Visa TAP + Mastercard Smart Authentication already do this. Stay above the network layer.
- Agent KYC / operator onboarding — Visa and Mastercard run their own JWKS registries.
- Crypto wallet directly — x402 is the protocol; we score and verify but don't custody.
- Replacing Stripe Radar — Radar runs on Stripe. We *export to* Radar's format (Tier 1.8) so non-Stripe stacks can consume the same shape.

## 10. Primary Sources (verified URLs)

**Stripe ACP**
- Spec repo: https://github.com/agentic-commerce-protocol/agentic-commerce-protocol
- Checkout OpenAPI (2026-04-17): https://raw.githubusercontent.com/agentic-commerce-protocol/agentic-commerce-protocol/main/spec/2026-04-17/openapi/openapi.agentic_checkout.yaml
- Webhook spec: https://raw.githubusercontent.com/agentic-commerce-protocol/agentic-commerce-protocol/main/spec/2026-04-17/openapi/openapi.agentic_checkout_webhook.yaml
- Delegated Payment (OpenAI mirror): https://developers.openai.com/commerce/specs/payment
- Stripe SPT concept docs: https://docs.stripe.com/agentic-commerce/concepts/shared-payment-tokens
- Sessions 2026 announcements: https://stripe.com/blog/everything-we-announced-at-sessions-2026

**Visa TAP**
- Spec: https://developer.visa.com/capabilities/trusted-agent-protocol/trusted-agent-protocol-specifications
- Reference impl: https://github.com/visa/trusted-agent-protocol
- Press release (Oct 2025): https://investor.visa.com/news/news-details/2025/Visa-Introduces-Trusted-Agent-Protocol-An-Ecosystem-Led-Framework-for-AI-Commerce/default.aspx

**Mastercard Verifiable Intent**
- Announcement: https://www.mastercard.com/us/en/news-and-trends/stories/2026/verifiable-intent.html
- Agent Pay docs: https://developer.mastercard.com/mastercard-checkout-solutions/documentation/use-cases/agent-pay/
- Cloudflare partnership: https://blog.cloudflare.com/secure-agentic-commerce/

**Google AP2**
- Spec: https://ap2-protocol.org/specification/
- Repo: https://github.com/google-agentic-commerce/AP2
- Announce: https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol
- Security analysis: https://cloudsecurityalliance.org/blog/2025/10/06/secure-use-of-the-agent-payments-protocol-ap2-a-framework-for-trustworthy-ai-driven-transactions

**Coinbase x402**
- Repo: https://github.com/coinbase/x402
- HTTP transport spec: https://github.com/coinbase/x402/blob/main/specs/transports-v2/http.md
- Whitepaper: https://www.x402.org/x402-whitepaper.pdf
- Cloudflare/Foundation: https://blog.cloudflare.com/x402/
- Stripe support: https://docs.stripe.com/payments/machine/x402

**Walmart × ChatGPT failure**
- Walmart corporate (announce, 2025-10-14): https://corporate.walmart.com/news/2025/10/14/walmart-partners-with-openai-to-create-ai-first-shopping-experiences
- CNBC pivot (2026-03-24): https://www.cnbc.com/2026/03/24/openai-revamps-shopping-experience-in-chatgpt-after-instant-checkout.html
- Search Engine Land (3× conversion data): https://searchengineland.com/walmart-chatgpt-checkout-converted-worse-472071
- Retail Dive (Sparky pivot): https://www.retaildive.com/news/walmart-sparky-chatgpt-instant-checkout/815647/

**Threat model**
- Whispers of Wealth (AP2 red-team): https://arxiv.org/abs/2601.22569
- SoK Agentic Commerce Security: https://arxiv.org/abs/2604.15367
- Caging the Agents: https://arxiv.org/abs/2603.17419
- WebInject: https://arxiv.org/abs/2505.11717
- SecAlign defense: https://arxiv.org/abs/2410.05451
- Adversarial concept drift (RLR): https://arxiv.org/abs/2009.09497
- BIP70 refund attack (TOCTOU analog): https://arxiv.org/abs/2103.08436
- OWASP LLM Top 10 2025: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Brave Comet disclosure: https://brave.com/blog/comet-prompt-injection/
- Anthropic GTG-1002 threat report: https://www.anthropic.com/news/disrupting-AI-espionage
- $47K LangChain runaway: https://dev.to/waxell/the-47000-agent-loop
