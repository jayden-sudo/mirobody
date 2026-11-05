"""POST /api/v1/ingest/batch through the ASGI app, with the database replaced by stubs."""

import base64
import hashlib
import json
import logging
import time

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI

from mirobody.collect import hub_ingest, observations
from mirobody.collect.hub_registry import Registry
from mirobody.server.routers import hub_router
from mirobody.user.auth.hub_signature import PATH, load_public_key
from mirobody.user.user import ExternalUser, SibianAccount

AUD = "mirobody-dev"
UID = "ab" * 32
KEY = Ed25519PrivateKey.generate()
b64u = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()  # noqa: E731

REGISTRY = """
version: 1
metrics:
  heart_rate: {name: heart rate, kind: quantity, unit: /min, accept_units: {"/min": 1, bpm: 1}, decimals: 0, bounds: [20, 250]}
  sleep: {name: sleep duration, kind: quantity, unit: min, decimals: 0, window: "18:00"}
"""


class Ephemeral:
    def __init__(self):
        self.seen, self.calls, self.boom = set(), [], None

    async def set(self, key, value, *, ex=None, nx=False):
        self.calls.append((key, value, ex, nx))
        if self.boom:
            raise self.boom
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


class Stubs:
    def __init__(self):
        self.lookups, self.created, self.writes = 0, 0, []
        self.state = ExternalUser("found", 7)
        self.fail_on_write = None


@pytest.fixture
def env(tmp_path, monkeypatch):
    p = tmp_path / "r.yaml"
    p.write_text(REGISTRY, encoding="utf-8")
    eph, st = Ephemeral(), Stubs()

    async def find(provider, subject):
        st.lookups += 1
        return st.state

    async def ensure(subject, name, iss):
        st.created += 1
        return SibianAccount(9, True, "x@sibian.invalid")

    async def ingest(user_id, drafts, prov, **kw):
        if st.fail_on_write is not None and len(st.writes) == st.fail_on_write:
            raise RuntimeError("db down")
        st.writes.append((user_id, drafts, prov, kw))
        return observations.Report(inserted=len(drafts), outcomes=["inserted"] * len(drafts))

    monkeypatch.setattr(hub_ingest, "find_external_user", find)
    monkeypatch.setattr(hub_ingest, "ensure_sibian_user", ensure)
    monkeypatch.setattr(hub_ingest.obs, "ingest", ingest)
    app = FastAPI()
    app.state.ephemeral = eph
    app.state.hub = {
        "public_key": load_public_key(base64.b64encode(KEY.public_key().public_bytes_raw()).decode()),
        "audience": AUD, "registry": Registry.load(str(p)), "auto_create": True, "backfill_days": 4000,
    }
    app.include_router(hub_router.router)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    return client, eph, st, app


def body(*samples, **top):
    doc = {"v": 1, "uid": UID, "devices": [{"name": "dev1", "samples": list(samples)}]} | top
    return json.dumps(doc, ensure_ascii=False).encode()


def sample(i="s1", **kw):
    return {"id": i, "metric": "heart_rate", "value": 58, "unit": "/min", "t": "2026-11-05T02:00:00+00:00"} | kw


def headers(raw, nonce=None, ts=None, key=KEY):
    ts = str(int(time.time())) if ts is None else str(ts)
    nonce = nonce or b64u(__import__("os").urandom(16))
    msg = "\n".join(["mirobody-hub-ingest-v1", AUD, "POST", PATH, ts, nonce, hashlib.sha256(raw).hexdigest()])
    return {"X-Ingest-Timestamp": ts, "X-Ingest-Nonce": nonce, "X-Ingest-Signature": b64u(key.sign(msg.encode())),
            "Content-Type": "application/json"}


async def post(client, raw, **kw):
    return await client.post(PATH, content=raw, headers=headers(raw, **kw))


async def test_valid_batch_is_written_with_device_provenance(env):
    client, eph, st, _ = env
    raw = body(sample(), sample("s2", t="2026-11-05T03:00:00+00:00"))
    r = await post(client, raw)
    assert r.status_code == 200
    data = r.json()["data"]
    assert (data["status"], data["received"], data["inserted"]) == ("ok", 2, 2)
    user_id, drafts, prov, kw = st.writes[0]
    assert user_id == "7" and kw["on_conflict"] == observations.ON_CONFLICT_AMEND
    assert (prov.modality, prov.source_kind, prov.source_class) == ("device-sensed", "device", "measurer")
    assert (prov.source_ref, prov.vendor, prov.extractor) == ("hub:dev1:-", "hub.dev1", "")
    d = drafts[0]
    assert (d.name_text, d.value_text, d.unit_text, d.tz) == ("heart rate", "58", "/min", "UTC+00:00")
    assert d.source_record_id == "dev1:-:s1" and d.observed_end == d.observed_start
    key, value, ex, nx = eph.calls[0]
    assert key.startswith("ingest:nonce:") and (value, ex, nx) == ("1", 660, True)


async def test_replay_is_refused_and_writes_once(env):
    client, _, st, _ = env
    raw = body(sample())
    h = headers(raw)
    assert (await client.post(PATH, content=raw, headers=h)).status_code == 200
    assert (await client.post(PATH, content=raw, headers=h)).status_code == 401
    assert len(st.writes) == 1


async def test_every_401_leaves_stores_and_database_untouched(env):
    client, eph, st, _ = env
    raw = body(sample())
    cases = [headers(raw, key=Ed25519PrivateKey.generate()), headers(raw, ts=1),
             {k: v for k, v in headers(raw).items() if k != "X-Ingest-Signature"}]
    for h in cases:
        assert (await client.post(PATH, content=raw, headers=h)).status_code == 401
    assert (await client.post(PATH, content=raw + b" ", headers=headers(raw))).status_code == 401
    assert eph.calls == [] and st.lookups == 0 and st.created == 0 and st.writes == []


async def test_nonce_store_failure_is_503_never_a_pass(env):
    client, eph, st, _ = env
    eph.boom = TimeoutError()
    assert (await post(client, body(sample()))).status_code == 503
    assert st.writes == []


async def test_oversized_body_is_413_even_without_content_length(env):
    client, *_ = env
    big = b"x" * (2 * 1024 * 1024 + 1)
    assert (await client.post(PATH, content=big, headers=headers(big))).status_code == 413

    async def gen():
        yield big
    assert (await client.post(PATH, content=gen(), headers=headers(big))).status_code == 413


@pytest.mark.parametrize("raw", [
    b'{"v":1,"uid":"%b","devices":[{"name":"d","samples":[{"id":"a","metric":"heart_rate","value":NaN,"unit":"/min","t":"x"}]}]}' % UID.encode(),
    b'{"value": Infinity}', b'{"value": -Infinity}', b'{"value": 1e999}', b'{"value": -1e999}',
    b"[" * 200000, b'{"value": ' + b"9" * 6000 + b"}", b"\xff\xfe{}", b"not json",
], ids=lambda raw: repr(raw)[:24])
async def test_hostile_json_is_422_invalid_json(env, raw):
    r = await post(env[0], raw)
    assert (r.status_code, r.json()["msg"]) == (422, "invalid_json")


@pytest.mark.parametrize("patch", [{"user_id": "1"}, {"theta_user_id": "1"}, {"v": 2}, {"uid": "short"}, {"devices": []}])
async def test_unknown_or_malformed_structure_is_invalid_request(env, patch):
    doc = json.loads(body(sample())) | patch
    r = await post(env[0], json.dumps(doc).encode())
    assert (r.status_code, r.json()["msg"]) == (422, "invalid_request")


async def test_request_level_codes(env):
    client, _, st, _ = env
    dup_id = body(sample("a"), sample("a"))
    two_devs = json.dumps({"v": 1, "uid": UID, "devices": [
        {"name": "d", "samples": [sample()]}, {"name": "d", "samples": [sample()]}]}).encode()
    two_serials = json.dumps({"v": 1, "uid": UID, "devices": [
        {"name": "d", "serial": "1", "samples": [sample()]}, {"name": "d", "serial": "2", "samples": [sample()]}]}).encode()
    many = body(*[sample(f"s{i}") for i in range(2001)])
    for raw, code in [(dup_id, "duplicate_sample_id"), (two_devs, "duplicate_device"),
                      (body(sample(), tz="Mars/Base"), "invalid_tz"), (many, "too_many_samples")]:
        r = await post(client, raw)
        assert (r.status_code, r.json()["msg"]) == (422, code)
    assert (await post(client, two_serials)).status_code == 200


async def test_all_rejected_is_422_without_database_calls(env):
    client, _, st, _ = env
    r = await post(client, body(sample(metric="nope"), sample("s2", unit="kg")))
    assert (r.status_code, r.json()["msg"]) == (422, "all_samples_rejected")
    assert {e["reason"] for e in r.json()["data"]["errors"]} == {"unknown-metric", "unknown-unit"}
    assert st.lookups == 0 and st.writes == []


async def test_some_rejected_is_200_partial(env):
    client, _, st, _ = env
    r = await post(client, body(sample(), sample("s2", value=999)))
    data = r.json()["data"]
    assert (r.status_code, data["status"], data["inserted"], data["rejected"]) == (200, "partial", 1, {"out-of-range": 1})
    assert data["errors"] == [{"device": "dev1", "serial": None, "id": "s2", "reason": "out-of-range"}]


async def test_subject_states(env):
    client, _, st, app = env
    st.state = ExternalUser("deleted", 5)
    r = await post(client, body(sample()))
    assert (r.status_code, r.json()["msg"], st.created) == (422, "subject_deleted", 0)
    st.state = ExternalUser("missing")
    r = await post(client, body(sample()))
    assert (r.status_code, st.created, st.writes[-1][0]) == (200, 1, "9")
    app.state.hub["auto_create"] = False
    r = await post(client, body(sample("s9")))
    assert (r.status_code, r.json()["msg"], st.created) == (422, "subject_not_found", 1)


async def test_writes_are_chunked_and_a_failed_chunk_is_503(env):
    client, _, st, _ = env
    raw = body(*[sample(f"s{i}") for i in range(1200)])
    assert (await post(client, raw)).status_code == 200
    assert [len(w[1]) for w in st.writes] == [500, 500, 200]
    st.writes.clear()
    st.fail_on_write = 1
    assert (await post(client, body(*[sample(f"t{i}") for i in range(600)]))).status_code == 503
    assert len(st.writes) == 1  # the first chunk is committed; a whole resend is idempotent


async def test_logs_never_hold_values_full_uid_or_signature(env, caplog):
    client, *_ = env
    caplog.set_level(logging.DEBUG)
    raw = body(sample(value=58))
    h = headers(raw)
    await client.post(PATH, content=raw, headers=h)
    await client.post(PATH, content=raw, headers=h)  # 401
    await post(client, b"not json")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert UID not in text and h["X-Ingest-Signature"] not in text and "heart" not in text
    assert UID[:8] in text


async def test_stalled_nonce_store_is_503(env, monkeypatch):
    import asyncio
    client, eph, st, _ = env
    monkeypatch.setattr(hub_router, "NONCE_TIMEOUT_S", 0.05)

    async def hang(*a, **k):
        await asyncio.sleep(10)
    eph.set = hang
    assert (await post(client, body(sample()))).status_code == 503
    assert st.writes == []


async def test_lower_layer_reasons_pass_through_unrenamed(env, monkeypatch):
    client, _, st, _ = env

    async def mixed(user_id, drafts, prov, **kw):
        return observations.Report(inserted=1, rejected={"impossible_value": 1, "write-error": 1},
                                   outcomes=["inserted", "rejected", "rejected"])
    monkeypatch.setattr(hub_ingest.obs, "ingest", mixed)
    r = await post(client, body(sample("a"), sample("b"), sample("c")))
    data = r.json()["data"]
    assert (r.status_code, data["status"], data["rejected"]) == (200, "partial", {"impossible_value": 1, "write-error": 1})

    async def single(user_id, drafts, prov, **kw):
        return observations.Report(rejected={"impossible_value": 1}, outcomes=["rejected"])
    monkeypatch.setattr(hub_ingest.obs, "ingest", single)
    r = await post(client, body(sample("d")))
    assert r.json()["data"]["errors"] == [{"device": "dev1", "serial": None, "id": "d", "reason": "impossible_value"}]


async def test_database_error_while_finding_the_subject_is_503(env, monkeypatch):
    client, *_ = env

    async def boom(provider, subject):
        raise RuntimeError("db down")
    monkeypatch.setattr(hub_ingest, "find_external_user", boom)
    assert (await post(client, body(sample()))).status_code == 503


async def test_dash_serial_is_refused(env):
    doc = {"v": 1, "uid": UID, "devices": [{"name": "d", "serial": "-", "samples": [sample()]}]}
    assert (await post(env[0], json.dumps(doc).encode())).json()["msg"] == "invalid_request"


def test_offset_tz_is_understood_by_the_writer_and_keeps_the_sleep_night_together():
    """§17: `Draft.tz = "UTC+08:00"` is a zone the writer places, not a silent UTC fallback."""
    from datetime import datetime

    from mirobody import standardize
    tz, source = standardize.resolve_tz("UTC+08:00", "UTC")
    assert (tz, source) == ("UTC+08:00", "offset-only")
    late = datetime.fromisoformat("2026-11-04T23:30:00+08:00")
    early = datetime.fromisoformat("2026-11-05T02:00:00+08:00")
    assert standardize.local_day(late, tz, "18:00") == standardize.local_day(early, tz, "18:00")
