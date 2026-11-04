"""A sibian-os node's word that one of its people is asking to sign in.

sibian-os has no accounts to log in to: a person is an Ed25519 key and a
did:webvh, and a node proves who is asking by checking a one-time request the
person signed in the browser (`os.detach`, `server/ai.py`'s `ai_proof` there).
Checking that signature here would mean porting did:webvh, key rotation and
witness receipts, which the node already does and tests against the official
vectors. So the node vouches instead: it answers with a short EdDSA JWT whose
`sub` is the identity number and whose `aud` is this server, and this module
decides whether to believe it. Nothing here reaches the database; turning a
believed subject into an account is `user.ensure_sibian_user`.
"""

from __future__ import annotations

import asyncio
import logging
import time

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import jwt

logger = logging.getLogger(__name__)


class SibianAssertionRejected(Exception):
    """The assertion does not speak for anyone. The message is a fixed
    sentence safe to hand back to the caller."""


@dataclass(frozen=True)
class SibianClaims:
    subject: str
    name: str
    issuer: str
    jti: str
    expires_at: int


#: The node issues for 300 s. A token that claims to live longer was not cut
#: by a node following the contract, and a long-lived one is a credential
#: worth stealing: the exchange is meant to be repeated, not cached.
MAX_LIFETIME = 600

#: Clock skew between a node and this server.
LEEWAY = 30

_REFUSED = "The sign-in assertion was not accepted."

#: How long a node's JWKS is trusted before it is read again.
JWKS_TTL = 3600

#: A key id the cached JWKS does not hold costs at most one refetch per this
#: many seconds per node. That is how a rotated key is picked up at once,
#: while a stream of made-up `kid`s cannot turn this server into a client
#: that hammers the node on the attacker's behalf.
MISS_REFETCH_INTERVAL = 60


class SibianAssertionVerifier:
    """Believes an assertion only when a node on `issuers` signed it with a key
    from its published JWKS, for `audience`, and it is current.

    `issuers` maps each trusted node's `iss` to its JWKS URL: the list is the
    trust decision, so it comes from configuration and never from the token.
    `fetch_jwks` is how a JWKS is read, injected so the network stays outside;
    `clock` is the monotonic clock the cache ages by.
    """

    def __init__(
        self,
        issuers: dict[str, str],
        audience: str,
        fetch_jwks: Callable[[str], Awaitable[dict]],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._issuers = dict(issuers)
        self._clock = clock
        self._audience = audience
        self._fetch_jwks = fetch_jwks
        # url -> (jwks, fetched_at); url -> when a missing kid last forced a read.
        self._jwks: dict[str, tuple[dict, float]] = {}
        self._miss_refetch: dict[str, float] = {}
        # One read of a node's JWKS at a time: sign-ins arriving together on
        # a cold or expired cache wait for that read instead of each making
        # their own (and each paying a slow node's timeout).
        self._locks: dict[str, asyncio.Lock] = {}

    async def verify(self, token: str) -> SibianClaims:
        try:
            return await self._verify(token)
        except SibianAssertionRejected:
            raise
        except Exception as e:
            # A malformed token, a bad signature, an unreachable node: all one
            # answer to the caller. The type name is for us; it carries no value.
            logger.info("sibian assertion refused: reason=%s", type(e).__name__)
            raise SibianAssertionRejected(_REFUSED) from None

    async def _verify(self, token: str) -> SibianClaims:
        header = jwt.get_unverified_header(token)
        # `iss` is read unverified only to pick which node's keys to try; the
        # decode below checks it again against the same value.
        iss = jwt.decode(token, options={"verify_signature": False}).get("iss")
        # The node, as configured: `issuers_from_config` drops a trailing "/",
        # so the token's is dropped too for the lookup. The decode below still
        # holds the token to its own exact `iss`.
        node = iss.rstrip("/") if isinstance(iss, str) else None
        jwks_url = self._issuers.get(node) if node else None
        if not jwks_url:
            logger.info("sibian assertion refused: reason=untrusted_issuer")
            raise SibianAssertionRejected(_REFUSED)

        key = await self._key(jwks_url, header.get("kid"))
        # The algorithm is pinned here, never taken from the header: a token
        # naming HS256 would otherwise be checked with the public key as an
        # HMAC secret, which anyone can compute.
        claims = jwt.decode(
            token, key, algorithms=["EdDSA"], audience=self._audience, issuer=iss,
            leeway=LEEWAY, options={"require": ["sub", "exp", "iat", "jti"]},
        )
        if claims["exp"] - claims["iat"] > MAX_LIFETIME:
            logger.info("sibian assertion refused: reason=lifetime")
            raise SibianAssertionRejected(_REFUSED)
        if not isinstance(claims["sub"], str) or not claims["sub"] or not isinstance(claims["jti"], str):
            raise SibianAssertionRejected(_REFUSED)

        name = claims.get("name")
        return SibianClaims(subject=claims["sub"], name=name if isinstance(name, str) else "",
                            issuer=node, jti=claims["jti"], expires_at=int(claims["exp"]))


    async def _key(self, url: str, kid: object):
        async with self._locks.setdefault(url, asyncio.Lock()):
            return await self._key_locked(url, kid)

    async def _key_locked(self, url: str, kid: object):
        now = self._clock()
        cached = self._jwks.get(url)
        fresh = cached is None or now - cached[1] >= JWKS_TTL
        if cached is None:
            cached = await self._read(url, now)
        elif fresh:
            # Refresh, but keep serving the keys already held when the node
            # cannot be reached: an hour-old key the node has not withdrawn is
            # still its key, and refusing everyone because the node blinked at
            # the moment the cache aged out locks every person out at once.
            try:
                cached = await self._read(url, now)
            except Exception as e:
                logger.warning("sibian jwks refresh failed, using cached keys: reason=%s", type(e).__name__)
                # Try again after the miss interval, not on every sign-in: each
                # attempt at a dead node costs a request its whole timeout.
                cached = (cached[0], now - JWKS_TTL + MISS_REFETCH_INTERVAL)
                self._jwks[url] = cached
        key = _key_for(cached[0], kid)
        # A kid missing from keys read just now will not be there on a second
        # read either; only a cache read earlier can be out of date.
        if key is None and not fresh and now - self._miss_refetch.get(url, float("-inf")) >= MISS_REFETCH_INTERVAL:
            self._miss_refetch[url] = now
            key = _key_for((await self._read(url, now))[0], kid)
        if key is None:
            raise LookupError("no such key")
        return key

    async def _read(self, url: str, now: float) -> tuple[dict, float]:
        jwks = await self._fetch_jwks(url)
        self._jwks[url] = (jwks if isinstance(jwks, dict) else {}, now)
        return self._jwks[url]


def _key_for(jwks: dict, kid: object):
    for k in jwks.get("keys", []):
        if isinstance(k, dict) and k.get("kid") == kid and k.get("kty") == "OKP":
            return jwt.PyJWK(k, algorithm="EdDSA").key
    return None


#-----------------------------------------------------------------------------

#: Where a sibian node publishes the keys it signs assertions with.
JWKS_PATH = "/.well-known/sibian-health/jwks.json"



def issuers_from_config(value: object) -> dict[str, str]:
    """`SIBIAN_ISSUERS` as `{iss: jwks_url}`.

    Either a list of node addresses (each publishes at `JWKS_PATH`) or a
    mapping from address to a JWKS URL of its own. Every URL must be https:
    whoever can swap the JWKS in transit can sign in as anyone. Plain http is
    accepted only for a node on this machine, which is development.
    """
    if value in (None, "", [], {}):
        return {}
    if isinstance(value, (list, tuple)):
        pairs = [(str(v), None) for v in value]
    elif isinstance(value, dict):
        pairs = [(str(k), str(v)) for k, v in value.items()]
    else:
        raise ValueError("SIBIAN_ISSUERS must be a list of node addresses or a mapping to JWKS URLs")

    issuers: dict[str, str] = {}
    for iss, jwks in pairs:
        iss = iss.strip().rstrip("/")
        jwks = (jwks or iss + JWKS_PATH).strip()
        for url in (iss, jwks):
            _require_safe_url(url)
        issuers[iss] = jwks
    return issuers


def _require_safe_url(url: str) -> None:
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme == "https" and parts.netloc:
        return
    if parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"):
        return
    raise ValueError(f"SIBIAN_ISSUERS: {url!r} is not an https URL")


#: A JWKS is a few hundred bytes; anything this large is not one.
JWKS_MAX_BYTES = 64 * 1024
JWKS_TIMEOUT_SECONDS = 5


async def fetch_jwks(url: str) -> dict:
    """A node's JWKS over HTTP: exactly the URL configured, no redirects (a
    redirect would let whoever controls its target publish the keys), a
    JSON body of bounded size, within a few seconds."""
    import json

    import aiohttp

    timeout = aiohttp.ClientTimeout(total=JWKS_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, allow_redirects=False, headers={"Accept": "application/json"}) as r:
            if r.status != 200:
                raise LookupError(f"jwks answered {r.status}")
            if r.content_type not in ("application/json", "application/jwk-set+json"):
                raise LookupError("jwks is not json")
            # `read(n)` returns what has ARRIVED, not n bytes: a body sent in
            # chunks came back cut short. Read to the end, never past the cap.
            body = b""
            async for chunk in r.content.iter_chunked(8192):
                body += chunk
                if len(body) > JWKS_MAX_BYTES:
                    raise LookupError("jwks too large")
    jwks = json.loads(body)
    if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
        raise LookupError("not a jwks")
    return jwks
