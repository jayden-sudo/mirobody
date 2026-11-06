"""POST /api/data with a chosen standard `code`: the reading is filed under exactly that code (needs Postgres)."""

import os
from datetime import datetime, timedelta, UTC
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

DSN = os.environ.get("MIROBODY_TEST_POSTGRES_DSN", "")
pytestmark = [pytest.mark.needs_db, pytest.mark.skipif(not DSN, reason="MIROBODY_TEST_POSTGRES_DSN is not set")]

USER, OTHER = "601", "602"


@pytest.fixture
async def world(monkeypatch):
    import importlib

    import psycopg
    from fastapi import FastAPI
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    from mirobody.collect import observations
    from mirobody.utils import db

    rr = importlib.import_module("mirobody.server.routers.records_router")
    schema = "rec_" + uuid4().hex
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
                                     connect_args={"options": f"-c search_path={schema},public -c timezone=UTC"})
        monkeypatch.setattr(db, "_provider", lambda _: engine)
        monkeypatch.setattr(db, "_engines", {})
        monkeypatch.setattr(observations, "_range_rules", None)
        app = FastAPI()
        app.include_router(rr.router)
        who = {"id": USER}
        app.dependency_overrides[rr.verify_token] = lambda: who["id"]
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")

        class W:
            pass
        w = W()
        w.client, w.who, w.q = client, who, db.execute_query
        yield w
        await client.aclose()
    finally:
        if engine is not None:
            await engine.dispose()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


def when(hours=2):
    return (datetime.now(UTC) - timedelta(hours=hours)).replace(microsecond=0).isoformat()


async def post(w, **rec):
    return (await w.client.post("/api/data", json={"records": [{"time": when(), **rec}]})).json()


async def rows(w, user=USER):
    return await w.q("SELECT o.name_text, o.value_text, o.unit_text, k.value_canonical, k.unit_canonical, o.modality, "
                     "o.source_kind, k.code_system, k.code, k.outcome FROM th_observation o JOIN th_coding_current k "
                     "ON k.observation_id = o.id WHERE o.user_id = :u ORDER BY o.id", {"u": user})


async def test_what_free_text_and_catalog_names_do_today(world):
    """Characterization: a typed English term is coded by the vocabulary; a camelCase catalog name is not."""
    await post(world, indicator="resting heart rate", value=60, unit="/min")
    await post(world, indicator="restingHeartRates", value=61, unit="/min", time=when(3))
    first, second = await rows(world)
    assert first["code"] == "40443-4"
    assert second["code"] != "40443-4"


async def test_a_chosen_code_is_filed_under_exactly_that_code_whatever_the_name(world):
    body = await post(world, indicator="whatever the user typed", code="40443-4", value=60, unit="/min")
    assert (body["ingested"], body["rejected"]) == (1, {})
    (row,) = await rows(world)
    assert (row["code_system"], row["code"], row["outcome"]) == ("http://loinc.org", "40443-4", "coded")
    assert row["name_text"] == "Resting Heart Rate"                         # the catalogue name for that code
    assert (row["value_text"], row["unit_text"], row["modality"]) == ("60", "/min", "manual")   # as typed, a manual entry


async def test_the_indicator_text_is_optional_when_a_code_is_given(world):
    body = await post(world, code="29463-7", value=70.5, unit="kg")
    assert body["ingested"] == 1
    (row,) = await rows(world)
    assert row["code"] == "29463-7" and row["unit_canonical"] == "g" and row["value_canonical"] == 70500.0


async def test_a_unit_that_cannot_become_the_catalogue_unit_is_refused(world):
    body = await post(world, code="29463-7", value=70, unit="mmol/L")
    assert body["ingested"] == 0 and body["rejected"] == {"unit-mismatch": 1}
    assert await rows(world) == []


async def test_an_unknown_code_is_refused_and_nothing_is_written(world):
    body = await post(world, code="0000-0", value=1, unit="kg")
    assert body["ingested"] == 0 and body["rejected"] == {"unknown-code": 1}
    assert await rows(world) == []


async def test_a_known_range_still_guards_a_chosen_code(world):
    body = await post(world, code="29463-7", value=900, unit="kg")           # weight is bounded to [2, 300] kg
    assert body["ingested"] == 0 and body["rejected"] == {"out-of-range": 1}


async def test_a_metric_without_a_known_range_accepts_any_number(world):
    body = await post(world, code="2276-4", value=987654, unit="ng/mL")      # ferritin: no range rule, so no guard
    assert (body["ingested"], body["rejected"]) == (1, {})
    (row,) = await rows(world)
    assert (row["code"], row["value_text"]) == ("2276-4", "987654")


async def test_a_resting_heart_rate_that_cannot_be_real_is_refused_by_the_known_range(world):
    body = await post(world, code="40443-4", value=9999, unit="/min")
    assert body["ingested"] == 0 and body["rejected"] == {"out-of-range": 1}


async def test_sending_the_same_record_again_writes_nothing_twice_and_users_stay_apart(world):
    t = when(5)
    assert (await post(world, code="29463-7", value=70, unit="kg", time=t))["ingested"] == 1
    assert (await post(world, code="29463-7", value=70, unit="kg", time=t))["ingested"] == 0
    world.who["id"] = OTHER
    assert (await post(world, code="29463-7", value=80, unit="kg", time=t))["ingested"] == 1
    assert [r["value_text"] for r in await rows(world, USER)] == ["70"]
    assert [r["value_text"] for r in await rows(world, OTHER)] == ["80"]


async def test_the_chosen_code_survives_a_correction_and_a_recode(world):
    from mirobody.collect import observations
    body = await post(world, code="29463-7", value=70, unit="kg")
    assert body["ingested"] == 1
    (first,) = await world.q("SELECT id FROM th_observation WHERE user_id = :u", {"u": USER})
    new_id = await observations.amend(USER, int(first["id"]), value_text="71")
    assert new_id and new_id != first["id"]
    await observations.recode(USER)                       # a replay under the same release must not move it either
    live = await world.q("SELECT o.value_text, k.code, k.outcome FROM v_observation o JOIN th_coding_current k "
                         "ON k.observation_id = o.id WHERE o.user_id = :u", {"u": USER})
    assert [(r["value_text"], r["code"], r["outcome"]) for r in live] == [("71", "29463-7", "coded")]


async def test_an_unverified_catalogue_code_is_not_offered_or_accepted(world):
    from mirobody.kernel import metrics
    unverified = next(m for m in metrics.ROWS if m.loinc and m.confidence == "unverified"
                      and not any(c.loinc == m.loinc and c.confidence == "confident" for c in metrics.ROWS))
    body = await post(world, code=unverified.loinc, value=1, unit=unverified.standard_unit)
    assert body["ingested"] == 0 and body["rejected"] == {"unknown-code": 1}
    assert unverified.loinc not in [r["code"] for r in metrics.search(unverified.loinc)]


async def test_a_unit_of_another_dimension_is_filed_under_the_code_of_that_unit(world):
    """Cholesterol is 2093-3 in mass units and 14647-2 in molar ones: the coder keeps to the unit, so the chosen
    analyte is filed under its sibling code for the unit typed (still coded, never left to be confirmed)."""
    body = await post(world, code="2093-3", value=5.2, unit="mmol/L")
    assert body["ingested"] == 1, body
    (row,) = await rows(world)
    assert (row["code"], row["outcome"]) == ("14647-2", "coded")
    same_unit = await post(world, code="2093-3", value=200, unit="mg/dL", time=when(7))
    assert same_unit["ingested"] == 1
    assert [(r["code"], r["outcome"]) for r in await rows(world)] == [("14647-2", "coded"), ("2093-3", "coded")]


async def test_a_chosen_code_with_a_device_source_is_still_a_manual_entry_held_to_the_range(world):
    ok = await post(world, code="40443-4", value=58, unit="/min", source="apple")
    assert (ok["ingested"], ok["rejected"]) == (1, {})
    (row,) = await rows(world)
    assert (row["code"], row["name_text"], row["modality"], row["source_kind"]) == ("40443-4", "Resting Heart Rate", "manual", "api")
    bad = await post(world, code="40443-4", value=9999, unit="/min", source="apple", time=when(9))
    assert bad["ingested"] == 0 and bad["rejected"] == {"out-of-range": 1}


async def test_a_stored_code_prefix_is_trusted_only_for_a_catalogue_code(world):
    from mirobody.collect import observations as o
    from mirobody.kernel.series import SOURCE_MEASURER
    prov = o.Provenance(modality=o.MODALITY_DEVICE, source_kind=o.SOURCE_DEVICE, source_ref="x:y", source_class=SOURCE_MEASURER, vendor="x")
    start = datetime.now(UTC) - timedelta(hours=3)
    draft = o.Draft(name_text="mystery", value_text="1", unit_text="u", observed_start=start, vendor_field="code:0000-0")
    await o.ingest(USER, [draft], prov, user_tz="UTC")
    (row,) = await rows(world)
    assert row["code"] != "0000-0"
