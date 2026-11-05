"""The metrics a hub may push, and the rules that turn its values into rows.

Pure functions: no database, no network. A `Registry` is loaded once at start
(a bad file is a start-up error); `render` and `parse_time` raise `Reject` with
the per-sample reason code the caller reports.
"""

from __future__ import annotations

import re
import unicodedata

from datetime import datetime, timedelta, UTC
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, model_validator
from ruamel.yaml import YAML


class Reject(Exception):
    """One sample is refused; `str(exc)` is the reason code."""


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: list[str] = []
    kind: Literal["quantity", "ordinal", "nominal", "boolean"]
    unit: str
    accept_units: dict[str, float] = {}
    decimals: int | None = None
    bounds: tuple[float, float] | None = None
    window: str = ""
    allowed: list[str] = []
    censor: dict[str, str] = {}

    @model_validator(mode="after")
    def _check(self) -> Metric:
        if self.kind == "quantity" and (self.decimals is None or not 0 <= self.decimals <= 9):
            raise ValueError("a quantity needs decimals between 0 and 9")
        if self.kind in ("ordinal", "nominal") and not self.allowed:
            raise ValueError(f"{self.kind} needs allowed")
        if self.kind == "quantity" and not self.unit:
            raise ValueError("a quantity needs a unit")
        if self.bounds and self.bounds[0] > self.bounds[1]:
            raise ValueError("bounds are reversed")
        if any(f <= 0 for f in self.accept_units.values()):
            raise ValueError("accept_units factors must be positive")
        if self.window and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", self.window):
            raise ValueError("window must be HH:MM")
        self.accept_units = {self.unit: 1.0, **self.accept_units} if self.unit not in self.accept_units else self.accept_units
        self.allowed = [_nfc(a) for a in self.allowed]
        return self


class Registry:
    def __init__(self, metrics: dict[str, Metric]) -> None:
        self._by_name: dict[str, Metric] = {}
        for key, metric in metrics.items():
            for n in (key, *metric.aliases):
                n = _nfc(n)
                if n in self._by_name:
                    raise ValueError(f"metric name or alias used twice: {n}")
                self._by_name[n] = metric

    @classmethod
    def load(cls, path: str) -> Registry:
        with open(path, encoding="utf-8") as fh:
            doc = YAML(typ="safe").load(fh)
        if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("metrics"), dict):
            raise ValueError("registry must be a mapping with version: 1 and metrics")
        return cls({str(k): Metric.model_validate(v) for k, v in doc["metrics"].items()})

    def find(self, metric: str) -> Metric:
        try:
            return self._by_name[_nfc(metric)]
        except KeyError:
            raise Reject("unknown-metric") from None


def render(m: Metric, value: object, unit: str) -> str:
    """The text to store, identical for identical input."""
    factor = m.accept_units.get(unit)
    if factor is None:
        raise Reject("unknown-unit")
    if m.kind == "boolean":
        if not isinstance(value, bool):
            raise Reject("bad-value-type")
        return "true" if value else "false"
    if m.kind in ("ordinal", "nominal"):
        if not isinstance(value, str):
            raise Reject("bad-value-type")
        label = _nfc(value)
        if label not in m.allowed:
            raise Reject("bad-label")
        return label
    if isinstance(value, str) and _nfc(value) in m.censor:
        return m.censor[_nfc(value)]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Reject("bad-value-type")
    try:
        q = (Decimal(str(value)) * Decimal(str(factor))).quantize(
            Decimal(1).scaleb(-m.decimals), rounding=ROUND_HALF_EVEN)
    except InvalidOperation:
        raise Reject("out-of-range") from None
    if m.bounds and not Decimal(str(m.bounds[0])) <= q <= Decimal(str(m.bounds[1])):
        raise Reject("out-of-range")
    return format(abs(q) if q == 0 else q, "f")


_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})?")


def parse_time(text: str, batch_tz: str | None) -> tuple[datetime, str]:
    """An aware datetime and the tz name to store with it (batch tz, else `UTC±HH:MM`)."""
    if not isinstance(text, str) or not _ISO.fullmatch(text):
        raise Reject("bad-time")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise Reject("bad-time") from None
    if dt.tzinfo is not None:
        off = dt.utcoffset()
        if off is None or abs(off) > timedelta(hours=14):
            raise Reject("bad-time")
        minutes = int(off.total_seconds() // 60)
        sign = "-" if minutes < 0 else "+"
        return dt, batch_tz or f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"
    if not batch_tz:
        raise Reject("missing-tz")
    zone = ZoneInfo(batch_tz)
    first, second = dt.replace(tzinfo=zone, fold=0), dt.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        raise Reject("ambiguous-or-nonexistent-local-time")
    if first.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != dt:
        raise Reject("ambiguous-or-nonexistent-local-time")
    return first, batch_tz


def check_range(start: datetime, end: datetime, now: datetime, backfill_days: int) -> None:
    if end < start or end - start > timedelta(hours=36):
        raise Reject("bad-interval")
    if end > now + timedelta(hours=24) or start > now + timedelta(hours=24):
        raise Reject("future")
    if start < now - timedelta(days=backfill_days):
        raise Reject("too-old")
