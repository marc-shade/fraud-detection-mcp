"""RFC 9421 HTTP Message Signature verification for agent commerce protocols.

Implements signature verification compatible with:
  - Visa Trusted Agent Protocol (TAP)
  - Mastercard Verifiable Intent / Web Bot Auth (Cloudflare partnership)
  - Stripe ACP (header-level Signature; algorithm bilateral)
  - Generic JWS-bearing protocols

The module is intentionally dependency-light: python-jose handles JWS / JWK,
and a small in-process JWKS cache avoids hammering issuer well-known endpoints.

Primary sources:
  - RFC 9421: https://www.rfc-editor.org/rfc/rfc9421
  - Visa TAP spec: https://developer.visa.com/capabilities/trusted-agent-protocol/trusted-agent-protocol-specifications
  - Cloudflare Web Bot Auth: https://blog.cloudflare.com/secure-agentic-commerce/
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _SigVerifyError(Exception):
    """Local exception type used when python-jose is unavailable."""


# Graceful degradation for crypto deps (already in requirements.txt)
try:
    from jose import jwk as _jose_jwk
    from jose.exceptions import JOSEError as _JOSEError, JWSError as _JWSError, JWKError as _JWKError

    JOSE_AVAILABLE = True
    JOSEError: type = _JOSEError
    JWSError: type = _JWSError
    JWKError: type = _JWKError
except ImportError:  # pragma: no cover - covered by environment policy
    JOSE_AVAILABLE = False
    _jose_jwk = None  # type: ignore[assignment]
    JOSEError = _SigVerifyError  # type: ignore[assignment,misc]
    JWSError = _SigVerifyError  # type: ignore[assignment,misc]
    JWKError = _SigVerifyError  # type: ignore[assignment,misc]


# Default JWKS URLs for known agent operators. Operators can override via
# JWKSResolver.register_issuer().
DEFAULT_JWKS_URLS: Dict[str, str] = {
    "visa": "https://mcp.visa.com/.well-known/jwks",
}

# RFC 9421 Signature-Input parameters we accept. Each maps key -> required.
SIGNATURE_INPUT_PARAMS = {
    "keyid": True,
    "alg": True,
    "created": True,
    "expires": False,
    "nonce": False,
    "tag": False,
}

# Algorithms allowed by the agent commerce protocols we support.
# See Visa TAP spec for crypto choices. Ed25519 is the recognition signature;
# PS256/ES256 are used for consumer/payment objects.
ALLOWED_ALGORITHMS = frozenset({"EdDSA", "PS256", "ES256", "RS256"})

# Maximum signature age (seconds) before rejection. Visa TAP mandates 8 minutes.
MAX_SIGNATURE_AGE_SECONDS = 8 * 60


@dataclass
class SignatureInput:
    """Parsed RFC 9421 Signature-Input header value."""

    label: str
    covered_components: List[str]
    keyid: str
    alg: str
    created: int
    expires: Optional[int] = None
    nonce: Optional[str] = None
    tag: Optional[str] = None
    raw: str = ""

    def is_fresh(self, now: Optional[float] = None) -> Tuple[bool, str]:
        """Return (fresh, reason) — RFC 9421 + Visa TAP timing rules."""
        now = now or time.time()
        age = now - self.created
        if age < -30:
            return False, f"signature created {-age:.1f}s in the future"
        if age > MAX_SIGNATURE_AGE_SECONDS:
            return False, f"signature is {age:.0f}s old (max {MAX_SIGNATURE_AGE_SECONDS})"
        if self.expires is not None and now > self.expires:
            return False, f"signature expired {now - self.expires:.0f}s ago"
        if self.expires is not None and self.expires - self.created > MAX_SIGNATURE_AGE_SECONDS:
            return False, "signature lifetime exceeds 8 minutes"
        return True, ""


_SIG_INPUT_RE = re.compile(
    r"""
    ^(?P<label>[A-Za-z0-9_-]+)      # label
    =\(\s*(?P<components>[^)]*)\)   # covered components
    (?P<params>(?:;[^;]+)*)         # ;key=value pairs
    \s*$
    """,
    re.VERBOSE,
)
_PARAM_RE = re.compile(r";\s*([a-zA-Z][a-zA-Z0-9_-]*)\s*=\s*(?:\"([^\"]*)\"|([^;\s]+))")


def parse_signature_input(header_value: str) -> SignatureInput:
    """Parse a single RFC 9421 Signature-Input header value.

    Handles the dictionary form `label=("@authority" "@path");keyid="..."`.
    For multi-label inputs the first label wins (matches Visa TAP single-label use).

    Raises ValueError on malformed input.
    """
    if not header_value or not isinstance(header_value, str):
        raise ValueError("empty Signature-Input header")

    # Take only the first dict entry (Visa TAP / Web Bot Auth use single label).
    primary = header_value.split(",", 1)[0].strip()
    m = _SIG_INPUT_RE.match(primary)
    if not m:
        raise ValueError(f"unparseable Signature-Input: {header_value[:80]!r}")

    label = m.group("label")
    components_raw = m.group("components").strip()
    components = [c.strip().strip('"') for c in components_raw.split() if c.strip()]

    params: Dict[str, str] = {}
    for pm in _PARAM_RE.finditer(m.group("params")):
        key = pm.group(1).lower()
        params[key] = pm.group(2) if pm.group(2) is not None else pm.group(3)

    for required, is_required in SIGNATURE_INPUT_PARAMS.items():
        if is_required and required not in params:
            raise ValueError(f"missing required parameter: {required}")

    try:
        created = int(params["created"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"invalid created timestamp: {e}") from None

    expires = int(params["expires"]) if "expires" in params else None

    alg = params["alg"]
    if alg not in ALLOWED_ALGORITHMS:
        raise ValueError(f"algorithm {alg!r} not in allowed set {sorted(ALLOWED_ALGORITHMS)}")

    return SignatureInput(
        label=label,
        covered_components=components,
        keyid=params["keyid"],
        alg=alg,
        created=created,
        expires=expires,
        nonce=params.get("nonce"),
        tag=params.get("tag"),
        raw=header_value,
    )


def build_signature_base(
    sig_input: SignatureInput,
    headers: Dict[str, str],
    method: Optional[str] = None,
    path: Optional[str] = None,
    authority: Optional[str] = None,
) -> str:
    """Construct the canonical signature base per RFC 9421 §2.3.

    Derived components (`@method`, `@path`, `@authority`, etc.) are sourced
    from the explicit args; HTTP header components are looked up in `headers`
    case-insensitively. Caller is responsible for canonical encoding of any
    structured-field values they pass in.
    """
    # Normalize headers to lowercase keys for case-insensitive lookup.
    norm_headers = {k.lower(): v for k, v in headers.items()}
    lines: List[str] = []
    for component in sig_input.covered_components:
        c = component.lower()
        if c == "@method":
            value = (method or "").upper()
        elif c == "@path":
            value = path or ""
        elif c == "@authority":
            value = (authority or "").lower()
        elif c.startswith("@"):
            # Other derived components — caller must supply via headers dict
            # under the same name (e.g. "@target-uri").
            value = norm_headers.get(c, "")
        else:
            value = norm_headers.get(c, "")
        lines.append(f'"{c}": {value}')
    # Trailing @signature-params line per spec
    sp_value = (
        "(" + " ".join(f'"{c}"' for c in sig_input.covered_components) + ")"
    )
    extras = []
    extras.append(f'created={sig_input.created}')
    if sig_input.expires is not None:
        extras.append(f'expires={sig_input.expires}')
    extras.append(f'keyid="{sig_input.keyid}"')
    extras.append(f'alg="{sig_input.alg}"')
    if sig_input.nonce:
        extras.append(f'nonce="{sig_input.nonce}"')
    if sig_input.tag:
        extras.append(f'tag="{sig_input.tag}"')
    sp_value += ";" + ";".join(extras)
    lines.append(f'"@signature-params": {sp_value}')
    return "\n".join(lines)


class JWKSResolver:
    """In-process JWKS cache with TTL.

    Fetches JWKS from issuer well-known endpoints and caches keys by
    (issuer, kid). Network errors are surfaced; the caller decides whether to
    fail-closed or accept unverified.

    The cache is intentionally per-process (no Redis dep). For a multi-process
    deployment, run an external JWKS proxy or warm the cache on startup.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        request_timeout_seconds: float = 5.0,
    ):
        self._lock = threading.Lock()
        # issuer -> {jwks_url, expires_at, keys: {kid -> jwk_dict}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._issuers: Dict[str, str] = dict(DEFAULT_JWKS_URLS)
        self._cache_ttl = cache_ttl_seconds
        self._timeout = request_timeout_seconds

    def register_issuer(self, issuer: str, jwks_url: str) -> None:
        """Register or override a JWKS URL for an issuer."""
        with self._lock:
            self._issuers[issuer] = jwks_url
            # Invalidate any cached entries for this issuer
            self._cache.pop(issuer, None)

    def known_issuers(self) -> Dict[str, str]:
        """Return a copy of the issuer -> JWKS URL mapping."""
        with self._lock:
            return dict(self._issuers)

    def get_key(self, issuer: str, kid: str) -> Optional[Dict[str, Any]]:
        """Return the JWK dict for (issuer, kid), refreshing JWKS if needed.

        Returns None if the issuer is not registered, JWKS fetch fails, or
        the kid is not present.
        """
        with self._lock:
            entry = self._cache.get(issuer)
            now = time.time()
            stale = entry is None or now >= entry.get("expires_at", 0)
            jwks_url = self._issuers.get(issuer)
        if not jwks_url:
            return None
        if stale:
            keys = self._fetch_jwks(jwks_url)
            if keys is None:
                return None
            with self._lock:
                self._cache[issuer] = {
                    "jwks_url": jwks_url,
                    "expires_at": time.time() + self._cache_ttl,
                    "keys": keys,
                }
        with self._lock:
            entry = self._cache.get(issuer, {})
            return entry.get("keys", {}).get(kid)

    def _fetch_jwks(self, jwks_url: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetch JWKS from a URL. Returns dict {kid -> jwk_dict} or None on error."""
        try:
            req = urllib.request.Request(
                jwks_url,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            logger.warning("JWKS fetch failed for %s: %s", jwks_url, e)
            return None
        keys = payload.get("keys") or []
        out: Dict[str, Dict[str, Any]] = {}
        for k in keys:
            kid = k.get("kid")
            if kid:
                out[kid] = k
        return out


# Module-level singleton (mirrors fraud-detection-mcp's other singletons)
jwks_resolver = JWKSResolver()


def verify_rfc9421_signature(
    headers: Dict[str, str],
    method: Optional[str] = None,
    path: Optional[str] = None,
    authority: Optional[str] = None,
    issuer: Optional[str] = None,
    expected_tag: Optional[str] = None,
    resolver: Optional[JWKSResolver] = None,
    nonce_cache: Optional[Any] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Verify an RFC 9421 HTTP Message Signature.

    Args:
        headers: HTTP headers including Signature-Input and Signature.
        method, path, authority: derived-component values per RFC 9421 §2.2.
        issuer: issuer name to resolve JWKS against (e.g. "visa", "mastercard").
                If None, signature_input.keyid must encode the issuer.
        expected_tag: if set, signature.tag must match (Visa TAP uses
                      "agent-browser-auth" / "agent-payer-auth").
        resolver: JWKSResolver instance (defaults to module singleton).
        nonce_cache: optional NonceCache; if provided, replay protection is
                     enforced.

    Returns dict with: verified (bool), reason (str), signature_input (parsed),
    keyid, issuer, algorithm, signature_age_seconds, warnings (List[str]).
    """
    resolver = resolver or jwks_resolver
    warnings: List[str] = []

    # Lowercase header lookup
    norm = {k.lower(): v for k, v in headers.items()}
    sig_input_raw = norm.get("signature-input")
    sig_raw = norm.get("signature")
    if not sig_input_raw or not sig_raw:
        return {
            "verified": False,
            "reason": "missing Signature-Input or Signature header",
            "warnings": warnings,
        }

    try:
        sig_input = parse_signature_input(sig_input_raw)
    except ValueError as e:
        return {
            "verified": False,
            "reason": f"signature_input_parse_error: {e}",
            "warnings": warnings,
        }

    # Freshness
    fresh, freshness_reason = sig_input.is_fresh(now=now)
    if not fresh:
        return {
            "verified": False,
            "reason": f"freshness_failed: {freshness_reason}",
            "signature_input": _siginput_to_dict(sig_input),
            "warnings": warnings,
        }

    # Tag binding (Visa TAP merchant-binding defense)
    if expected_tag and sig_input.tag != expected_tag:
        return {
            "verified": False,
            "reason": f"tag_mismatch: expected {expected_tag!r}, got {sig_input.tag!r}",
            "signature_input": _siginput_to_dict(sig_input),
            "warnings": warnings,
        }

    # Nonce replay protection
    if nonce_cache is not None and sig_input.nonce:
        seen = nonce_cache.seen(sig_input.keyid, sig_input.nonce)
        if seen:
            return {
                "verified": False,
                "reason": "nonce_replay_detected",
                "signature_input": _siginput_to_dict(sig_input),
                "warnings": warnings,
            }
    elif nonce_cache is not None and not sig_input.nonce:
        warnings.append("nonce_absent")

    # Crypto verification
    if not JOSE_AVAILABLE:
        return {
            "verified": False,
            "reason": "jose_library_unavailable",
            "signature_input": _siginput_to_dict(sig_input),
            "warnings": warnings,
        }

    # Resolve issuer: explicit arg wins; otherwise try keyid prefix (e.g. "visa:agent-7").
    resolved_issuer = issuer
    keyid_for_lookup = sig_input.keyid
    if not resolved_issuer and ":" in sig_input.keyid:
        resolved_issuer, keyid_for_lookup = sig_input.keyid.split(":", 1)

    if not resolved_issuer:
        return {
            "verified": False,
            "reason": "no_issuer_resolved",
            "signature_input": _siginput_to_dict(sig_input),
            "warnings": warnings,
        }

    jwk_dict = resolver.get_key(resolved_issuer, keyid_for_lookup)
    if not jwk_dict:
        return {
            "verified": False,
            "reason": f"jwks_key_not_found: issuer={resolved_issuer} kid={keyid_for_lookup}",
            "signature_input": _siginput_to_dict(sig_input),
            "warnings": warnings,
        }

    base = build_signature_base(
        sig_input, headers, method=method, path=path, authority=authority
    )

    try:
        # python-jose's compact-JWS verify wants a JWT-style detached payload.
        # For RFC 9421 we have a raw signature over the canonical base; build a
        # detached JWS by concatenating header.payload.signature where the
        # base is the payload and the signature is the b64-decoded value from
        # the Signature header (`label=:b64...:` per RFC).
        sig_b64 = _extract_signature_b64(sig_raw, sig_input.label)
        if not sig_b64:
            return {
                "verified": False,
                "reason": "signature_value_not_found_for_label",
                "signature_input": _siginput_to_dict(sig_input),
                "warnings": warnings,
            }
        verified_ok = _verify_with_jwk(
            jwk_dict, sig_input.alg, base.encode("utf-8"), _b64url_decode(sig_b64)
        )
        if not verified_ok:
            return {
                "verified": False,
                "reason": "cryptographic_verify_failed",
                "signature_input": _siginput_to_dict(sig_input),
                "warnings": warnings,
            }
    except Exception as e:  # noqa: BLE001  — jose may raise broad types
        return {
            "verified": False,
            "reason": f"verification_exception: {type(e).__name__}: {e}",
            "signature_input": _siginput_to_dict(sig_input),
            "warnings": warnings,
        }

    # Mark nonce as seen *after* successful verification
    if nonce_cache is not None and sig_input.nonce:
        nonce_cache.add(sig_input.keyid, sig_input.nonce)

    age = (now or time.time()) - sig_input.created
    return {
        "verified": True,
        "reason": "ok",
        "signature_input": _siginput_to_dict(sig_input),
        "keyid": sig_input.keyid,
        "issuer": resolved_issuer,
        "algorithm": sig_input.alg,
        "signature_age_seconds": float(age),
        "warnings": warnings,
    }


def _siginput_to_dict(s: SignatureInput) -> Dict[str, Any]:
    return {
        "label": s.label,
        "covered_components": list(s.covered_components),
        "keyid": s.keyid,
        "alg": s.alg,
        "created": s.created,
        "expires": s.expires,
        "nonce": s.nonce,
        "tag": s.tag,
    }


_SIG_VALUE_RE = re.compile(r"([A-Za-z0-9_-]+)\s*=\s*:([A-Za-z0-9+/=_-]+):")


def _extract_signature_b64(sig_header: str, label: str) -> Optional[str]:
    """Extract the b64-encoded signature value matching `label` from the
    Signature header (RFC 9421 dict form `label=:b64:`).
    """
    for m in _SIG_VALUE_RE.finditer(sig_header):
        if m.group(1) == label:
            return m.group(2)
    return None


def _b64url_decode(s: str) -> bytes:
    """Decode standard or URL-safe base64 with padding tolerance."""
    import base64

    pad = (-len(s)) % 4
    s_padded = s + ("=" * pad)
    # Try URL-safe first (RFC 9421 / JWS convention)
    try:
        return base64.urlsafe_b64decode(s_padded)
    except (ValueError, TypeError, binascii_error):
        return base64.b64decode(s_padded)


# binascii.Error fallback name to avoid binascii dep at import time
try:
    from binascii import Error as binascii_error
except ImportError:  # pragma: no cover
    binascii_error = ValueError  # type: ignore[assignment,misc]


def _verify_with_jwk(
    jwk_dict: Dict[str, Any], alg: str, message: bytes, signature: bytes
) -> bool:
    """Verify ``message`` against ``signature`` using the given JWK.

    EdDSA is handled directly via ``cryptography`` because python-jose has
    limited Ed25519 support. Other algorithms (PS256, ES256, RS256) route
    through python-jose for compactness.
    """
    if alg == "EdDSA":
        return _verify_ed25519_jwk(jwk_dict, message, signature)
    if not JOSE_AVAILABLE or _jose_jwk is None:
        return False
    try:
        key = _jose_jwk.construct(jwk_dict, alg)
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(key.verify(message, signature))
    except Exception:  # noqa: BLE001
        return False


def _verify_ed25519_jwk(
    jwk_dict: Dict[str, Any], message: bytes, signature: bytes
) -> bool:
    """Verify Ed25519 (EdDSA) JWK directly via ``cryptography``.

    JWK shape per RFC 8037: ``{"kty": "OKP", "crv": "Ed25519", "x": <b64url>}``.
    """
    if jwk_dict.get("kty") != "OKP" or jwk_dict.get("crv") != "Ed25519":
        return False
    x_b64 = jwk_dict.get("x")
    if not x_b64:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError:  # pragma: no cover
        return False
    try:
        raw_pub = _b64url_decode(x_b64)
        pubkey = Ed25519PublicKey.from_public_bytes(raw_pub)
        pubkey.verify(signature, message)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
