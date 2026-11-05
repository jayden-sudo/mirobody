"""POST /api/v1/ingest/batch: a hub pushes signed device readings.

Mounted only when INGEST_PUBLIC_KEY is configured (server.py). Authentication
is the signature, not a user token. Order matters: nothing touches the database
until the signature and the one-time nonce are good.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time

from fastapi import APIRouter, Request
from pydantic import ValidationError

from mirobody.collect import hub_ingest
from mirobody.user.auth.hub_signature import PATH, Rejected, verify
from mirobody.utils.http import json_response_with_code

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_BODY_BYTES = 2 * 1024 * 1024
NONCE_TIMEOUT_S = 5
NONCE_TTL_S = 660  # 2 x skew + 60 s of clock difference


def _reply(status: int, code: str, data: dict | None = None):
    return json_response_with_code(0 if status == 200 else status, "ok" if status == 200 else code,
                                   data, disable_log=True, status=status)


def _reject_constant(name: str):
    raise ValueError(name)


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


async def _read_capped(request: Request) -> bytes | None:
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(PATH)
async def ingest(request: Request):
    cfg = request.app.state.hub
    started = time.monotonic()
    raw = await _read_capped(request)
    if raw is None:
        return _reply(413, "payload_too_large")

    try:
        nonce = verify(cfg["public_key"], cfg["audience"], request.headers, raw, time.time())
    except Rejected as exc:
        logger.info("hub request refused: signature, timestamp or header")
        return _reply(401, str(exc))
    try:
        fresh = await asyncio.wait_for(
            request.app.state.ephemeral.set(f"ingest:nonce:{nonce}", "1", ex=NONCE_TTL_S, nx=True), NONCE_TIMEOUT_S)
    except Exception as exc:
        logger.warning("hub nonce store failed: %s", type(exc).__name__)
        return _reply(503, "nonce_store_unavailable")
    if not fresh:
        logger.info("hub request refused: nonce already used")
        return _reply(401, str(Rejected()))

    try:
        doc = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        return _reply(422, "invalid_json")
    try:
        batch = hub_ingest.Batch.model_validate(doc)
    except ValidationError:
        return _reply(422, "invalid_request")

    try:
        result = await hub_ingest.ingest_batch(
            batch, registry=cfg["registry"], backfill_days=cfg["backfill_days"], auto_create=cfg["auto_create"])
    except hub_ingest.RequestError as exc:
        short_uid = batch.uid[:8]
        logger.info("hub batch refused: batch=%s uid=%s code=%s", batch.batch_id, short_uid, exc.code)
        return _reply(exc.status, exc.code, {"errors": exc.errors} if exc.errors else None)
    short_uid = batch.uid[:8]
    received_count, inserted, skipped = result["received"], result["inserted"], result["skipped"]
    rejected_count = sum(result["rejected"].values())
    duration_ms = int((time.monotonic() - started) * 1000)
    logger.info("hub batch: batch=%s uid=%s devices=%d received=%d inserted=%d skipped=%d rejected=%d in %dms",
                batch.batch_id, short_uid, len(batch.devices), received_count, inserted, skipped, rejected_count, duration_ms)
    return _reply(200, "ok", result)
