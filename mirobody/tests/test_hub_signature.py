"""The hub's request signature: what is accepted and what is refused."""

import base64
import hashlib
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mirobody.user.auth.hub_signature import PATH, Rejected, load_public_key, verify

AUD = "mirobody-dev"
b64u = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()  # noqa: E731

KEY = Ed25519PrivateKey.generate()
PUB = load_public_key(base64.b64encode(KEY.public_key().public_bytes_raw()).decode())


def signed(body=b"{}", *, ts=None, nonce=None, audience=AUD, key=KEY):
    ts = str(int(time.time()) if ts is None else ts)
    nonce = b64u(b"n" * 16) if nonce is None else nonce
    msg = "\n".join(["mirobody-hub-ingest-v1", audience, "POST", PATH, ts, nonce,
                     hashlib.sha256(body).hexdigest()])
    return {"X-Ingest-Timestamp": ts, "X-Ingest-Nonce": nonce,
            "X-Ingest-Signature": b64u(key.sign(msg.encode()))}


def test_valid_request_returns_its_nonce():
    assert verify(PUB, AUD, signed(), b"{}", time.time()) == b64u(b"n" * 16)


@pytest.mark.parametrize("mutate", [
    lambda h: h | {"X-Ingest-Timestamp": str(int(h["X-Ingest-Timestamp"]) + 1)},
    lambda h: h | {"X-Ingest-Nonce": b64u(b"m" * 16)},
    lambda h: {k: v for k, v in h.items() if k != "X-Ingest-Signature"},
    lambda h: h | {"X-Ingest-Signature": b64u(b"x" * 63)},
    lambda h: h | {"X-Ingest-Signature": "!!!"},
    lambda h: h | {"X-Ingest-Timestamp": "12.5"},
    lambda h: h | {"X-Ingest-Signature": "eyJhbGciOiJIUzI1NiJ9.e30.abc"},
])
def test_tampered_or_malformed_headers_are_refused(mutate):
    with pytest.raises(Rejected):
        verify(PUB, AUD, mutate(signed()), b"{}", time.time())


def test_changed_body_or_audience_or_key_is_refused():
    h = signed()
    for args in [(PUB, AUD, h, b"{ }"), (PUB, "other", h, b"{}")]:
        with pytest.raises(Rejected):
            verify(*args, time.time())
    with pytest.raises(Rejected):
        verify(PUB, AUD, signed(key=Ed25519PrivateKey.generate()), b"{}", time.time())


@pytest.mark.parametrize("delta,ok", [(-299, True), (299, True), (-301, False), (301, False)])
def test_clock_skew_window(delta, ok):
    now = time.time()
    h = signed(ts=int(now) + delta)
    if ok:
        verify(PUB, AUD, h, b"{}", now)
    else:
        with pytest.raises(Rejected):
            verify(PUB, AUD, h, b"{}", now)


@pytest.mark.parametrize("nonce", [b64u(b"n" * 15), b64u(b"n" * 65), b64u(b"n" * 16) + "=", "a+b/" * 6])
def test_bad_nonces_are_refused(nonce):
    with pytest.raises(Rejected):
        verify(PUB, AUD, signed(nonce=nonce), b"{}", time.time())


def test_absurdly_long_timestamp_is_a_refusal_not_a_crash():
    h = signed() | {"X-Ingest-Timestamp": "9" * 5000}
    with pytest.raises(Rejected):
        verify(PUB, AUD, h, b"{}", time.time())


def test_all_refusals_share_one_message():
    msgs = set()
    for h in (signed(), signed(ts=1), {}):
        try:
            verify(PUB, AUD, h, b"x", time.time())
        except Rejected as e:
            msgs.add(str(e))
    assert len(msgs) == 1


@pytest.mark.parametrize("bad", ["", "not base64!", base64.b64encode(b"x" * 31).decode()])
def test_load_public_key_rejects_bad_keys(bad):
    with pytest.raises(ValueError):
        load_public_key(bad)
