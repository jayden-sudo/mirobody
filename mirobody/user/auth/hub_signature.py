"""A hub's word that a batch of device readings is genuine.

One Ed25519 public key is configured; the hub signs seven lines (tag, audience,
method, path, timestamp, nonce, sha256 of the exact body). This module only
decides whether the signature is good. It touches no database; the one-time
nonce is spent by the caller, after this passes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re

from collections.abc import Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PATH = "/api/v1/ingest/batch"
SKEW_S = 300
TAG = "mirobody-hub-ingest-v1"

_REFUSED = "The ingest request was not accepted."
_B64U = re.compile(r"^[A-Za-z0-9_-]+$")


class Rejected(Exception):
    """The request does not carry the hub's signature. The message is fixed."""

    def __init__(self) -> None:
        super().__init__(_REFUSED)


def load_public_key(b64: str) -> Ed25519PublicKey:
    """The configured key: base64 of the 32 raw bytes. ValueError when it is not."""
    try:
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(b64.strip(), validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("INGEST_PUBLIC_KEY must be base64 of a 32-byte Ed25519 public key") from exc


def _b64u(value: str | None, lo: int, hi: int) -> bytes:
    if not value or not _B64U.fullmatch(value):
        raise Rejected
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        raise Rejected from None
    if not lo <= len(raw) <= hi:
        raise Rejected
    return raw


def verify(key: Ed25519PublicKey, audience: str, headers: Mapping[str, str], body: bytes, now: float) -> str:
    """Return the request's nonce when it is signed, fresh and well formed; else raise Rejected."""
    try:
        ts = headers.get("X-Ingest-Timestamp") or ""
        nonce = headers.get("X-Ingest-Nonce") or ""
        sig = _b64u(headers.get("X-Ingest-Signature"), 64, 64)
        _b64u(nonce, 16, 64)
        if not (ts.isascii() and ts.isdigit() and len(ts) <= 12) or abs(int(ts) - now) > SKEW_S:
            raise Rejected
        message = "\n".join([TAG, audience, "POST", PATH, ts, nonce, hashlib.sha256(body).hexdigest()])
        key.verify(sig, message.encode("utf-8"))
    except Rejected:
        raise
    except Exception:  # InvalidSignature, or anything a hostile header can provoke
        raise Rejected from None
    return nonce
