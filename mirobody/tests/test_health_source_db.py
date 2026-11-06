"""Rows of /health-indicators say WHERE a reading came from and WHEN the newest one was taken (needs Postgres)."""

import os
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")

DSN = os.environ.get("MIROBODY_TEST_POSTGRES_DSN", "")
pytestmark = [pytest.mark.needs_db, pytest.mark.skipif(not DSN, reason="MIROBODY_TEST_POSTGRES_DSN is not set")]

USER = "501"
OTHER = "502"


@pytest.fixture
async def db(monkeypatch, request):
    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    from mirobody.collect import observations
    from mirobody.utils import db as dbmod

    schema = "src_" + uuid4().hex
    conninfo = make_url(DSN).set(drivername="postgresql").render_as_string(hide_password=False)
    admin = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    engine = None
    try:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        await admin.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        await admin.execute("CREATE FUNCTION encrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT nullif($1,'')$$")
        await admin.execute("CREATE FUNCTION decrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT $1$$")
        root = Path(__file__).resolve().parents[1] / "schema"
        for f in ("10_accounts.sql", "20_files.sql", "30_observations.sql", "31_medications.sql", "32_genomics.sql",
                  "40_devices.sql", "41_device_rules.sql", "62_personal_mcp.sql"):
            await admin.execute((root / f).read_text())
        engine = create_async_engine(make_url(DSN).set(drivername="postgresql+psycopg"),
                                     connect_args={"options": f"-c search_path={schema},public -c timezone={getattr(request, 'param', 'UTC')}"})
        monkeypatch.setattr(dbmod, "_provider", lambda _: engine)
        monkeypatch.setattr(dbmod, "_engines", {})
        monkeypatch.setattr(observations, "_range_rules", None)
        yield dbmod.execute_query
    finally:
        if engine is not None:
            await engine.dispose()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


def at(days_ago: int, hour: int = 8, minute: int = 0) -> datetime:
    d = datetime.now(UTC).date() - timedelta(days=days_ago)
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=UTC)


async def write(user, name, value, unit, when, *, device=None, serial=None, manual=False, uid=USER):
    from mirobody.collect import observations as o
    from mirobody.kernel.series import SOURCE_MANUAL, SOURCE_MEASURER
    if manual:
        prov = o.Provenance(modality=o.MODALITY_MANUAL, source_kind=o.SOURCE_API, source_ref="records:hygieia",
                            source_class=SOURCE_MANUAL)
    else:
        prov = o.Provenance(modality=o.MODALITY_DEVICE, source_kind=o.SOURCE_DEVICE,
                            source_ref=f"hub:{device}:{serial or '-'}", source_class=SOURCE_MEASURER, vendor=f"hub.{device}")
    draft = o.Draft(name_text=name, value_text=str(value), unit_text=unit, observed_start=when, tz="UTC",
                    source_record_id=f"{name}:{when.isoformat()}")
    rep = await o.ingest(uid or user, [draft], prov, user_tz="UTC")
    assert rep.inserted == 1, rep


def hq():
    from mirobody.collect import PostgresHealthQuery
    return PostgresHealthQuery(reported=False)


def win(days=30):
    from mirobody.kernel import query
    now = datetime.now(UTC)
    return query.Window(start_ms=int((now - timedelta(days=days)).timestamp() * 1000),
                        end_ms=int((now + timedelta(days=1)).timestamp() * 1000), tz="UTC")


async def test_catalog_rows_carry_the_last_time_and_where_it_came_from(db):
    await write(USER, "restingHeartRates", 60, "/min", at(3), device="watch_1", serial="A1")
    await write(USER, "restingHeartRates", 58, "/min", at(1, 9, 30), device="watch_1", serial="A1")
    await write(USER, "restingHeartRates", 70, "/min", at(2), device="band_2")
    rows = await hq().catalog(USER, None)
    (row,) = [r for r in rows if r["code"] == "40443-4"]
    want = at(1, 9, 30).strftime("%Y-%m-%d %H:%M:%S")
    assert row["last_time"] == want and row["last_date"] == want[:10]
    assert row["last_source"] == {"source_kind": "device", "modality": "device-sensed",
                                  "vendor": "hub.watch_1", "source_ref": "hub:watch_1:A1"}
    assert row["source_count"] == 2          # watch_1 and band_2


async def test_a_manual_row_counts_its_source_kind_when_it_has_no_vendor(db):
    await write(USER, "restingHeartRates", 60, "/min", at(2), device="watch_1")
    # A manual write is coded by the resolver, which knows the English term (not the camelCase device name).
    await write(USER, "resting heart rate", 62, "/min", at(1), manual=True)
    (row,) = [r for r in await hq().catalog(USER, None) if r["code"] == "40443-4"]
    assert row["last_source"]["modality"] == "manual" and row["last_source"]["vendor"] == ""
    assert row["last_source"]["source_kind"] == "api"
    assert row["source_count"] == 2


async def test_reading_rows_name_the_device_they_came_from(db):
    from mirobody.kernel import query
    await write(USER, "restingHeartRates", 58, "/min", at(1), device="watch_1", serial="A1")
    sel = query.Selection(indicators=("40443-4",))
    for rows in (await hq().readings(USER, sel, win(), limit=10), await hq().latest(USER, sel, win())):
        (row,) = rows
        assert (row["vendor"], row["source_ref"], row["source_kind"]) == ("hub.watch_1", "hub:watch_1:A1", "device")
    page = await hq().records(USER, start_time=date.today() - timedelta(days=30))
    (rec,) = page["rows"]
    assert (rec["vendor"], rec["source_ref"], rec["source_kind"]) == ("hub.watch_1", "hub:watch_1:A1", "device")


async def test_one_person_never_sees_anothers_sources(db):
    await write(OTHER, "restingHeartRates", 50, "/min", at(1), device="secret_watch", uid=OTHER)
    assert [r for r in await hq().catalog(USER, None) if r["code"] == "40443-4"] == []


# ---- the timeline: one chronological list, device data collapsed to one entry per metric/day/source ------------


async def timeline(uid=USER, days=30, **kw):
    return await hq().timeline(uid, start=date.today() - timedelta(days=days), end=date.today(), **kw)


async def test_device_readings_collapse_to_one_entry_per_metric_day_and_source(db):
    for minute, bpm in ((0, 60), (1, 70), (2, 80)):
        await write(USER, "restingHeartRates", bpm, "/min", at(1, 3, minute), device="watch_1", serial="A1")
    await write(USER, "restingHeartRates", 50, "/min", at(1, 4), device="band_2")      # another source, same day
    page = await timeline()
    days = [r for r in page["rows"] if r["kind"] == "day"]
    assert len(days) == 2 and page["has_more"] is False
    watch = next(r for r in days if r["vendor"] == "hub.watch_1")
    assert (watch["n"], watch["avg"], watch["min"], watch["max"]) == (3, 70.0, 60.0, 80.0)
    assert (watch["value"], watch["unit"], watch["code"]) == ("80", "/min", "40443-4")       # the newest reading's value
    assert watch["time"].startswith(at(1, 3, 2).strftime("%Y-%m-%d %H:%M"))
    assert watch["source_ref"] == "hub:watch_1:A1" and watch["source_kind"] == "device"
    assert watch["local_date"] == at(1).strftime("%Y-%m-%d")
    band = next(r for r in days if r["vendor"] == "hub.band_2")
    assert (band["n"], band["avg"]) == (1, 50.0)                      # one reading still has the summary shape


async def test_manual_and_uncoded_rows(db):
    await write(USER, "resting heart rate", 62, "/min", at(1), manual=True)
    await write(USER, "totally_made_up_metric", 9, "u", at(1), device="watch_1")      # no standard code: not shown
    rows = (await timeline())["rows"]
    assert [(r["kind"], r["code"]) for r in rows] == [("reading", "40443-4")]
    assert rows[0]["modality"] == "manual" and rows[0]["row_id"] and rows[0]["value"] == "62"


async def test_order_and_cursor_paging_never_lose_or_split_entries(db):
    for d in (1, 2, 3):
        for minute in (0, 1):
            await write(USER, "restingHeartRates", 60 + minute, "/min", at(d, 3, minute), device="watch_1")
        await write(USER, "resting heart rate", 55 + d, "/min", at(d, 9), manual=True)
        await write(USER, "body weight", 70 + d, "kg", at(d, 10), manual=True)
    everything = (await timeline(limit=200))["rows"]
    assert len(everything) == 9                                      # per day: 1 summary + 2 manual readings
    dates = [r["local_date"] for r in everything]
    assert dates == sorted(dates, reverse=True)
    first_day = [r for r in everything if r["local_date"] == dates[0]]
    assert [r["kind"] for r in first_day] == ["day", "reading", "reading"]          # summary first, then readings newest first
    assert first_day[1]["time"] > first_day[2]["time"]
    seen, cursor = [], None
    for _ in range(20):
        page = await timeline(limit=2, cursor=cursor)
        seen += page["rows"]
        cursor = page["next_cursor"]
        if not page["has_more"]:
            break
    assert [r["entry_id"] for r in seen] == [r["entry_id"] for r in everything]      # same list, same order, no repeats


async def test_a_summary_keeps_its_position_when_new_readings_arrive(db):
    await write(USER, "restingHeartRates", 60, "/min", at(1, 3), device="watch_1")
    await write(USER, "resting heart rate", 61, "/min", at(1, 9), manual=True)
    before = (await timeline())["rows"]
    await write(USER, "restingHeartRates", 90, "/min", at(1, 23, 59), device="watch_1")          # a later reading syncs
    after = (await timeline())["rows"]
    assert [r["entry_id"] for r in before] == [r["entry_id"] for r in after]
    assert next(r for r in after if r["kind"] == "day")["n"] == 2


async def test_a_reading_belongs_to_its_own_local_day(db):
    from mirobody.collect import observations as o
    from mirobody.kernel.series import SOURCE_MANUAL
    prov = o.Provenance(modality=o.MODALITY_MANUAL, source_kind=o.SOURCE_API, source_ref="r", source_class=SOURCE_MANUAL)
    when = at(1, 15, 30)                                    # 15:30 UTC is 23:30 the same day in Taipei
    draft = o.Draft(name_text="resting heart rate", value_text="61", unit_text="/min", observed_start=when,
                    tz="Asia/Taipei", source_record_id="tz1")
    await o.ingest(USER, [draft], prov, user_tz="UTC")
    (row,) = (await timeline())["rows"]
    assert row["local_date"] == when.strftime("%Y-%m-%d") and row["time"].startswith(when.strftime("%Y-%m-%d") + " 23:30")


async def test_timeline_is_scoped_to_one_person_and_rejects_a_bad_cursor(db):
    await write(OTHER, "resting heart rate", 50, "/min", at(1), manual=True, uid=OTHER)
    assert (await timeline())["rows"] == []
    assert len((await timeline(uid=OTHER))["rows"]) == 1
    with pytest.raises(ValueError):
        await timeline(cursor="not-a-cursor")


async def test_timeline_route(db, monkeypatch):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI

    import importlib
    ir = importlib.import_module("mirobody.server.routers.indicator_router")      # the package re-exports the router object
    app = FastAPI()
    app.include_router(ir.router)
    app.dependency_overrides[ir.verify_token] = lambda: USER
    for d in (1, 2, 3):
        await write(USER, "resting heart rate", 60 + d, "/min", at(d, 9), manual=True)
        await write(USER, "restingHeartRates", 50 + d, "/min", at(d, 3), device="watch_1")
    await write(OTHER, "resting heart rate", 99, "/min", at(1), manual=True, uid=OTHER)
    today, since = date.today().isoformat(), (date.today() - timedelta(days=10)).isoformat()
    base = f"/api/v1/health-indicators/timeline?start_time={since}&end_time={today}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        body = (await client.get(base + "&limit=4")).json()
        assert body["code"] == 0 and len(body["data"]["rows"]) == 4 and body["data"]["has_more"] is True
        rest = (await client.get(base + "&limit=4&cursor=" + body["data"]["next_cursor"])).json()["data"]
        assert len(rest["rows"]) == 2 and rest["has_more"] is False and rest["next_cursor"] is None
        ids = [r["entry_id"] for r in body["data"]["rows"] + rest["rows"]]
        assert len(ids) == len(set(ids)) == 6 and all("99" != r["value"] for r in body["data"]["rows"] + rest["rows"])
        assert (await client.get(base + "&start_time=nope")).json()["code"] == 400
        assert (await client.get(base + "&cursor=garbage")).json()["code"] == 400
        assert (await client.get(base + "&limit=201")).status_code == 422
        async def refuse(user_id, target):
            return None
        monkeypatch.setattr(ir, "subject_for", refuse)                      # not allowed to read this member
        assert (await client.get(base + "&target_user_id=999")).json()["code"] == 403


@pytest.mark.parametrize("db", ["Asia/Shanghai", "America/Los_Angeles"], indirect=True)
async def test_paging_does_not_depend_on_the_database_session_zone(db):
    # Two sources on the same day (two summaries share a date), plus typed readings: a page boundary falls between them.
    await write(USER, "restingHeartRates", 60, "/min", at(1, 3), device="watch_1")
    await write(USER, "restingHeartRates", 61, "/min", at(1, 4), device="band_2")
    await write(USER, "resting heart rate", 62, "/min", at(1, 9), manual=True)
    await write(USER, "resting heart rate", 63, "/min", at(1, 10), manual=True)
    await write(USER, "restingHeartRates", 64, "/min", at(2, 3), device="watch_1")
    everything = (await timeline(limit=200))["rows"]
    assert len(everything) == 5
    for size in (1, 2, 3):
        seen, cursor = [], None
        for _ in range(10):
            page = await timeline(limit=size, cursor=cursor)
            seen += page["rows"]
            cursor = page["next_cursor"]
            if not page["has_more"]:
                break
        assert [r["entry_id"] for r in seen] == [r["entry_id"] for r in everything], size


async def test_the_timeline_span_is_capped(db):
    from mirobody.collect import PostgresHealthQuery  # noqa: F401  (the route owns the cap)
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    import importlib

    from fastapi import FastAPI
    ir = importlib.import_module("mirobody.server.routers.indicator_router")
    app = FastAPI()
    app.include_router(ir.router)
    app.dependency_overrides[ir.verify_token] = lambda: USER
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        ok = f"start_time={(date.today() - timedelta(days=ir.TIMELINE_MAX_DAYS)).isoformat()}&end_time={date.today().isoformat()}"
        too_long = f"start_time={(date.today() - timedelta(days=ir.TIMELINE_MAX_DAYS + 1)).isoformat()}&end_time={date.today().isoformat()}"
        backwards = f"start_time={date.today().isoformat()}&end_time={(date.today() - timedelta(days=1)).isoformat()}"
        assert (await client.get("/api/v1/health-indicators/timeline?" + ok)).json()["code"] == 0
        assert (await client.get("/api/v1/health-indicators/timeline?" + too_long)).json()["code"] == 400
        assert (await client.get("/api/v1/health-indicators/timeline?" + backwards)).json()["code"] == 400


async def test_a_day_entry_says_how_its_readings_combine_and_carries_the_sum(db):
    # heart rate points are samples (average them); step counts are increments (add them up)
    for minute, bpm in ((0, 60), (1, 80)):
        await write(USER, "restingHeartRates", bpm, "/min", at(1, 3, minute), device="watch_1")
    for minute, n in ((0, 100), (1, 250), (2, 50)):
        await write(USER, "steps", n, "{steps}", at(1, 12, minute), device="watch_1")
    rows = {r["code"]: r for r in (await timeline())["rows"] if r["kind"] == "day"}
    assert (rows["40443-4"]["policy"], rows["40443-4"]["avg"], rows["40443-4"]["sum"]) == ("mean_min_max", 70.0, 140.0)
    assert (rows["55423-8"]["policy"], rows["55423-8"]["sum"]) == ("sum_delta", 400.0)


async def test_catalog_rows_carry_the_policy(db):
    await write(USER, "steps", 100, "{steps}", at(1), device="watch_1")
    (row,) = [r for r in await hq().catalog(USER, None) if r["code"] == "55423-8"]
    assert row["policy"] == "sum_delta"


async def test_what_the_journal_wrote_is_not_listed_twice(db):
    from mirobody.collect import observations as o
    from mirobody.kernel.series import SOURCE_MANUAL
    prov = o.Provenance(modality=o.MODALITY_SELF, source_kind=o.SOURCE_MANUAL, source_ref="journal", source_class=SOURCE_MANUAL)
    draft = o.Draft(name_text="resting heart rate", value_text="61", unit_text="/min", observed_start=at(1, 9), tz="UTC", source_record_id="j1")
    await o.ingest(USER, [draft], prov, user_tz="UTC")
    await write(USER, "resting heart rate", 62, "/min", at(1, 10), manual=True)
    rows = (await timeline())["rows"]
    assert [r["value"] for r in rows] == ["62"]                     # the memo already shows what the journal wrote
