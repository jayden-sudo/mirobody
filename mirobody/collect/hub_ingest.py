"""A signed batch of device readings, written for an existing person.

The caller (the route) has already checked the signature. This module checks
the batch, finds or opens the subject's account, and hands drafts to
`observations.ingest` one device and at most CHUNK drafts at a time.
"""

from __future__ import annotations

import logging

from collections import Counter
from datetime import datetime, UTC
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mirobody.collect import observations as obs
from mirobody.collect.hub_registry import Registry, Reject, check_range, parse_time, render
from mirobody.kernel.series import SOURCE_MEASURER
from mirobody.user.user import ensure_sibian_user, find_external_user

logger = logging.getLogger(__name__)

MAX_SAMPLES = 2000
CHUNK = 500
MAX_ERRORS = 100


class RequestError(Exception):
    """The whole request is refused: an HTTP status and a snake_case code."""

    def __init__(self, status: int, code: str, errors: list[dict] | None = None) -> None:
        super().__init__(code)
        self.status, self.code, self.errors = status, code, errors or []


_strict = ConfigDict(extra="forbid", strict=True)


class Sample(BaseModel):
    model_config = _strict
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]
    metric: Annotated[str, Field(min_length=1, max_length=64)]
    value: int | float | str | bool
    unit: Annotated[str, Field(max_length=16)]
    t: str
    t_end: str | None = None


class Device(BaseModel):
    model_config = _strict
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
    serial: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")] | None = None
    samples: list[Sample]

    @field_validator("serial")
    @classmethod
    def _not_dash(cls, v: str | None) -> str | None:
        if v == "-":  # "-" stands for "no serial" in source ids
            raise ValueError("serial may not be '-'")
        return v


class Batch(BaseModel):
    model_config = _strict
    v: int = Field(ge=1, le=1)
    batch_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")] | None = None
    uid: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tz: str | None = None
    devices: Annotated[list[Device], Field(min_length=1, max_length=50)]


async def _subject(uid: str, auto_create: bool) -> str:
    found = await find_external_user("sibian", uid)
    if found.state == "deleted":
        raise RequestError(422, "subject_deleted")
    if found.state == "found":
        return str(found.user_id)
    if not auto_create:
        raise RequestError(422, "subject_not_found")
    account = await ensure_sibian_user(subject=uid, name="", iss="hub")
    if account is None:
        raise RequestError(503, "subject_unavailable")
    return str(account.user_id)


async def ingest_batch(batch: Batch, *, registry: Registry, backfill_days: int,
                       auto_create: bool, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    if sum(len(d.samples) for d in batch.devices) > MAX_SAMPLES:
        raise RequestError(422, "too_many_samples")
    if len({(d.name, d.serial) for d in batch.devices}) != len(batch.devices):
        raise RequestError(422, "duplicate_device")
    if any(len({s.id for s in d.samples}) != len(d.samples) for d in batch.devices):
        raise RequestError(422, "duplicate_sample_id")
    if batch.tz is not None:
        try:
            ZoneInfo(batch.tz)
        except Exception:
            raise RequestError(422, "invalid_tz") from None

    rejected: Counter[str] = Counter()
    errors: list[dict] = []

    def refuse(d: Device, s: Sample, reason: str) -> None:
        rejected[reason] += 1
        errors.append({"device": d.name, "serial": d.serial, "id": s.id, "reason": reason})

    prepared: list[tuple[Device, list[tuple[Sample, obs.Draft]]]] = []
    for d in batch.devices:
        ok: list[tuple[Sample, obs.Draft]] = []
        for s in d.samples:
            try:
                m = registry.find(s.metric)
                text = render(m, s.value, s.unit)
                start, tz = parse_time(s.t, batch.tz)
                end = parse_time(s.t_end, batch.tz)[0] if s.t_end is not None else start
                check_range(start, end, now, backfill_days)
            except Reject as exc:
                refuse(d, s, str(exc))
                continue
            ok.append((s, obs.Draft(
                name_text=m.name, value_text=text, unit_text=m.unit,
                observed_start=start, observed_end=end, tz=tz, window=m.window,
                source_record_id=f"{d.name}:{d.serial or '-'}:{s.id}")))
        prepared.append((d, ok))

    total = sum(len(d.samples) for d in batch.devices)
    if not any(ok for _, ok in prepared):
        raise RequestError(422, "all_samples_rejected", errors[:MAX_ERRORS])

    try:
        user_id = await _subject(batch.uid, auto_create)
    except RequestError:
        raise
    except Exception as exc:
        logger.warning("hub subject lookup failed: %s", type(exc).__name__)
        raise RequestError(503, "subject_unavailable") from None
    inserted = skipped = 0
    for d, ok in prepared:
        prov = obs.Provenance(
            modality=obs.MODALITY_DEVICE, source_kind=obs.SOURCE_DEVICE,
            source_ref=f"hub:{d.name}:{d.serial or '-'}", source_class=SOURCE_MEASURER,
            vendor=f"hub.{d.name}", extractor="")
        for i in range(0, len(ok), CHUNK):
            part = ok[i:i + CHUNK]
            try:
                report = await obs.ingest(user_id, [dr for _, dr in part], prov, user_tz="UTC",
                                          on_conflict=obs.ON_CONFLICT_AMEND)
            except Exception as exc:
                logger.warning("hub write failed: %s", type(exc).__name__)
                raise RequestError(503, "write_failed") from None
            inserted += report.inserted
            skipped += report.skipped
            if len(report.rejected) == 1:  # one reason: every refused draft can be named
                reason = next(iter(report.rejected))
                for (s, _), outcome in zip(part, report.outcomes):  # noqa: B905  one outcome per draft
                    if outcome == obs.OUTCOME_REJECTED:
                        refuse(d, s, reason)
            else:  # several reasons: counts only, as the lower layer gave them
                rejected.update(report.rejected)
    return {"batch_id": batch.batch_id, "status": "partial" if rejected else "ok",
            "received": total, "inserted": inserted, "skipped": skipped,
            "rejected": dict(rejected), "errors": errors[:MAX_ERRORS]}
