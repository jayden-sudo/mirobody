"""The hub route against a real Postgres (spec 14.3). Skipped without MIROBODY_TEST_POSTGRES_DSN."""

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI

DSN = os.environ.get("MIROBODY_TEST_POSTGRES_DSN", "")
pytestmark = [pytest.mark.needs_db, pytest.mark.skipif(not DSN, reason="MIROBODY_TEST_POSTGRES_DSN is not set")]

AUD = "mirobody-dev"
UID = "cd" * 32
KEY = Ed25519PrivateKey.generate()
PATH = "/api/v1/ingest/batch"

REGISTRY = """
version: 1
metrics:
  heart_rate: {name: heart rate, kind: quantity, unit: /min, accept_units: {"/min": 1, bpm: 1}, decimals: 0, bounds: [20, 250]}
  sleep_stage: {name: sleep stage duration, kind: quantity, unit: min, decimals: 0, window: "18:00"}
"""


def b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


@pytest.fixture
async def world(monkeypatch, tmp_path):
    import psycopg
    from cryptography.fernet import Fernet
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    from mirobody.collect import observations
    from mirobody.collect.hub_registry import Registry
    from mirobody.server.routers import hub_router
    from mirobody.user.auth.hub_signature import load_public_key
    from mirobody.utils import db
    from mirobody.utils.ephemeral import EphemeralStore

    schema = "hub_" + uuid4().hex
    conninfo = make_url(DSN).set(drivername="postgresql").render_as_string(hide_password=False)
    admin = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    engine = None
    try:
        await admin.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        await admin.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        await admin.execute("CREATE FUNCTION encrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT nullif($1,'')$$")
        await admin.execute("CREATE FUNCTION decrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT $1$$")
        root = Path(__file__).resolve().parents[1] / "schema"
        for f in ("10_accounts.sql", "20_files.sql", "30_observations.sql", "31_medications.sql",
                  "32_genomics.sql", "40_devices.sql", "41_device_rules.sql", "61_ephemeral.sql", "62_personal_mcp.sql"):
            await admin.execute((root / f).read_text())
        engine = create_async_engine(make_url(DSN).set(drivername="postgresql+psycopg"),
                                     connect_args={"options": f"-c search_path={schema},public -c timezone=UTC"})
        monkeypatch.setattr(db, "_provider", lambda _: engine)
        monkeypatch.setattr(db, "_engines", {})
        monkeypatch.setattr(observations, "_range_rules", None)

        class _Pg:
            async def get_async_client(self):
                return await psycopg.AsyncConnection.connect(conninfo, options=f"-c search_path={schema}")

        reg = tmp_path / "r.yaml"
        reg.write_text(REGISTRY, encoding="utf-8")
        app = FastAPI()
        app.state.ephemeral = EphemeralStore(_Pg(), Fernet.generate_key().decode())
        app.state.hub = {"public_key": load_public_key(base64.b64encode(KEY.public_key().public_bytes_raw()).decode()),
                         "audience": AUD, "registry": Registry.load(str(reg)), "auto_create": True, "backfill_days": 4000}
        app.include_router(hub_router.router)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=60)

        class W:
            pass
        w = W()
        w.client, w.app, w.q = client, app, db.execute_query
        yield w
    finally:
        if engine is not None:
            await engine.dispose()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


def sample(i, metric="heart_rate", value=58, unit="/min", t="2026-11-04T02:00:00+08:00", **kw):
    return {"id": i, "metric": metric, "value": value, "unit": unit, "t": t} | kw


async def post(w, samples, device="dev1", serial=None, uid=UID, tz=None):
    dev = {"name": device, "samples": samples} | ({"serial": serial} if serial else {})
    doc = {"v": 1, "uid": uid, "devices": [dev]} | ({"tz": tz} if tz else {})
    raw = json.dumps(doc, ensure_ascii=False).encode()
    ts, nonce = str(int(time.time())), b64u(os.urandom(16))
    msg = "\n".join(["mirobody-hub-ingest-v1", AUD, "POST", PATH, ts, nonce, hashlib.sha256(raw).hexdigest()])
    h = {"X-Ingest-Timestamp": ts, "X-Ingest-Nonce": nonce, "X-Ingest-Signature": b64u(KEY.sign(msg.encode()))}
    return await w.client.post(PATH, content=raw, headers=h)


async def count(w, table, where="true"):
    return (await w.q(f"SELECT count(*) AS n FROM {table} WHERE {where}"))[0]["n"]


async def make_account(w, uid=UID, deleted=False):
    row = await w.q("INSERT INTO health_app_user (is_del, email, name) VALUES (:d, :e, '') RETURNING id",
                    {"d": deleted, "e": uuid4().hex + "@x.invalid"})
    await w.q("INSERT INTO external_identity (provider, subject, user_id, first_iss) VALUES ('sibian', :s, :u, 'test')",
              {"s": uid, "u": row[0]["id"]})
    return row[0]["id"]


async def test_rows_and_provenance(world):
    w = world
    uid = await make_account(w)
    r = await post(w, [sample("a"), sample("b", t="2026-11-04T03:00:00+08:00", value=61)], serial="S1")
    assert (r.status_code, r.json()["data"]["inserted"]) == (200, 2), r.text
    rows = await w.q("SELECT * FROM th_observation WHERE user_id = :u ORDER BY observed_start", {"u": str(uid)})
    assert len(rows) == 2
    r0 = rows[0]
    assert (r0["modality"], r0["source_kind"], r0["source_class"]) == ("device-sensed", "device", "measurer")
    assert (r0["vendor"], r0["source_ref"], r0["source_record_id"]) == ("hub.dev1", "hub:dev1:S1", "dev1:S1:a")
    assert (r0["name_text"], r0["value_text"], r0["unit_text"], r0["tz"]) == ("heart rate", "58", "/min", "UTC+08:00")
    assert float(r0["value_num"]) == 58.0 and r0["extraction_id"] is None
    assert await count(w, "th_extraction") == 0


async def test_auto_create_makes_one_account_and_reuses_it(world):
    w = world
    assert (await post(w, [sample("a")])).status_code == 200
    assert (await post(w, [sample("b", t="2026-11-04T04:00:00+08:00")])).status_code == 200
    assert await count(w, "health_app_user") == 1
    assert await count(w, "external_identity", f"subject = '{UID}'") == 1


async def test_auto_create_off_is_422_and_creates_nothing(world):
    w = world
    w.app.state.hub["auto_create"] = False
    r = await post(w, [sample("a")])
    assert (r.status_code, r.json()["msg"]) == (422, "subject_not_found")
    assert await count(w, "health_app_user") == 0 and await count(w, "th_observation") == 0


async def test_deleted_account_is_never_reopened_by_the_hub_but_still_is_by_sign_in(world):
    from mirobody.user.user import ensure_sibian_user
    w = world
    old = await make_account(w, deleted=True)
    r = await post(w, [sample("a")])
    assert (r.status_code, r.json()["msg"]) == (422, "subject_deleted")
    assert await count(w, "health_app_user") == 1 and await count(w, "th_observation") == 0
    mapped = (await w.q("SELECT user_id FROM external_identity WHERE subject = :s", {"s": UID}))[0]["user_id"]
    assert mapped == old
    acct = await ensure_sibian_user(UID, "", "test")   # the sign-in flow keeps its existing behavior
    assert acct is not None and acct.is_new and acct.user_id != old


async def test_concurrent_first_batches_make_one_account(world):
    import asyncio
    w = world
    rs = await asyncio.gather(*[post(w, [sample(f"c{i}")]) for i in range(4)])
    assert all(r.status_code in (200, 503) for r in rs)
    assert any(r.status_code == 200 for r in rs)
    live = await count(w, "health_app_user", "is_del = false")
    assert live == 1 and await count(w, "external_identity", f"subject = '{UID}'") == 1


async def test_resend_is_skipped_and_a_changed_value_is_an_amendment(world):
    w = world
    await make_account(w)
    batch = [sample("a"), sample("b", t="2026-11-04T03:00:00+08:00")]
    first = (await post(w, batch)).json()["data"]
    again = (await post(w, batch)).json()["data"]
    assert (first["inserted"], again["inserted"], again["skipped"]) == (2, 0, 2)
    changed = (await post(w, [sample("a", value=70), batch[1]])).json()["data"]
    assert (changed["inserted"], changed["skipped"]) == (1, 1)
    latest = await w.q("SELECT value_text, amends FROM th_observation WHERE source_record_id = :r ORDER BY id",
                         {"r": "dev1:-:a"})
    assert [x["value_text"] for x in latest] == ["58", "70"] and latest[1]["amends"] is not None
    assert await count(w, "th_observation") == 3


async def test_sleep_night_around_midnight_is_one_local_day_with_the_registry_window(world):
    w = world
    await make_account(w)
    r = await post(w, [sample("n1", "sleep_stage", 30, "min", "2026-11-04T23:30:00+08:00"),
                       sample("n2", "sleep_stage", 40, "min", "2026-11-05T02:00:00+08:00")])
    assert r.status_code == 200, r.text
    days = {str(x["local_date"]) for x in await w.q("SELECT local_date FROM th_observation")}
    assert days == {"2026-11-04"}


async def test_naive_local_times_use_the_batch_zone(world):
    w = world
    await make_account(w)
    r = await post(w, [sample("a", t="2026-11-04T02:00:00")], tz="Asia/Taipei")
    assert r.status_code == 200, r.text
    row = (await w.q("SELECT observed_start, tz FROM th_observation"))[0]
    assert row["tz"] == "Asia/Taipei" and row["observed_start"].isoformat().startswith("2026-11-03T18:00:00")


async def test_failed_second_chunk_keeps_the_first_and_a_full_resend_completes(world, monkeypatch):
    from mirobody.collect import hub_ingest
    w = world
    await make_account(w)
    real, calls = hub_ingest.obs.ingest, {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("db down")
        return await real(*a, **k)
    monkeypatch.setattr(hub_ingest.obs, "ingest", flaky)
    batch = [sample(f"s{i}", t=f"2026-11-04T{i // 60:02d}:{i % 60:02d}:00+08:00", value=40 + i % 100) for i in range(700)]
    assert (await post(w, batch)).status_code == 503
    assert await count(w, "th_observation") == 500
    r = (await post(w, batch)).json()["data"]
    assert (r["skipped"], r["inserted"]) == (500, 200)
    assert await count(w, "th_observation") == 700


async def test_out_of_range_and_unknown_are_reported_per_sample_and_not_stored(world):
    w = world
    await make_account(w)
    r = await post(w, [sample("a"), sample("b", value=999), sample("c", metric="nope")])
    d = r.json()["data"]
    assert (r.status_code, d["status"], d["inserted"], d["rejected"]) == (200, "partial", 1, {"out-of-range": 1, "unknown-metric": 1})
    assert await count(w, "th_observation") == 1
