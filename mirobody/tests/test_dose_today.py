"""Today's doses and the tick that answers one, over HTTP.

`GET /api/v1/medications/today` is every slot an active plan schedules on the
person's local today with its derived state (`meds.slot_state`: nothing
derived is stored); `POST /api/v1/medications/doses` answers one slot taken or
skipped; `DELETE /api/v1/medications/doses/{event_id}` takes the answer back,
so the slot is what the clock says again. Needs PostgreSQL.
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")

DSN = os.environ.get("MIROBODY_TEST_POSTGRES_DSN", "")
pytestmark = [pytest.mark.needs_db, pytest.mark.skipif(not DSN, reason="MIROBODY_TEST_POSTGRES_DSN is not set")]
TZ = "Asia/Taipei"


@pytest.fixture
async def api(monkeypatch):
    """A schema with the accounts and medication tables, two people in it
    (1 and 2, both in Asia/Taipei), and the medications router answering as
    whoever `api.as_user` says."""
    import httpx
    import psycopg
    from fastapi import FastAPI
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine
    from mirobody.server.auth import verify_token
    from mirobody.server.routers.medication_router import router
    from mirobody.utils import db

    schema = "doses_" + uuid4().hex
    conninfo = make_url(DSN).set(drivername="postgresql").render_as_string(hide_password=False)
    admin = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    engine = None
    try:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        await admin.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        await admin.execute("CREATE FUNCTION encrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT nullif($1,'')$$")
        await admin.execute("CREATE FUNCTION decrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT $1$$")
        root = Path(__file__).resolve().parents[1] / "schema"
        for file in ("10_accounts.sql", "31_medications.sql"):
            await admin.execute((root / file).read_text())
        for email in ("one@example.invalid", "two@example.invalid"):
            await admin.execute("INSERT INTO health_app_user (is_del, email, tz) VALUES (false, %s, %s)", (email, TZ))
        engine = create_async_engine(make_url(DSN).set(drivername="postgresql+psycopg"),
                                     connect_args={"options": f"-c search_path={schema} -c timezone=UTC"})
        monkeypatch.setattr(db, "_provider", lambda _: engine)
        monkeypatch.setattr(db, "_engines", {})

        app = FastAPI()
        app.include_router(router)
        state = {"user": "1"}
        app.dependency_overrides[verify_token] = lambda: state["user"]

        class Api:
            def as_user(self, uid: str):
                state["user"] = uid
                return self

            async def __call__(self, method: str, path: str, body=None):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as c:
                    r = await c.request(method, f"/api/v1/medications{path}", json=body)
                return r.json()

        yield Api()
    finally:
        if engine is not None:
            await engine.dispose()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


def _today():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(TZ)).date()


async def _plan(api, name="Amlodipine", times=("00:00", "23:59")):
    """A plan the person keeps today: one slot already due (00:00), one not
    yet (23:59, upcoming for all but the last minute of the day)."""
    answer = await api("POST", "", {
        "name": name, "strength": "5 mg", "start_date": _today().isoformat(),
        "schedule": [{"dose": {"value": 1, "unit": "tablet"}, "times": list(times)}],
    })
    assert answer["code"] == 0, answer
    return answer["data"]["plan_id"]


async def test_today_lists_every_slot_with_its_state(api):
    plan_id = await _plan(api)
    answer = await api("GET", "/today")
    assert answer["code"] == 0, answer
    slots = answer["data"]["slots"]
    assert [(s["plan_id"], s["name"], s["slot"], s["state"]) for s in slots] == [
        (plan_id, "Amlodipine", "00:00", "due"),
        (plan_id, "Amlodipine", "23:59", "upcoming"),
    ]
    assert slots[0]["date"] == _today().isoformat()
    assert slots[0]["dose"] == {"value": 1, "unit": "{tablet}"}
    assert slots[0]["event_id"] is None
    assert slots[0]["answered_at"] is None


async def test_a_tick_answers_the_slot_and_an_undo_gives_it_back_to_the_clock(api):
    plan_id = await _plan(api)
    tick = await api("POST", "/doses", {"plan_id": plan_id, "date": _today().isoformat(), "slot": "00:00", "status": "taken"})
    assert tick["code"] == 0, tick
    first = (await api("GET", "/today"))["data"]["slots"][0]
    assert (first["state"], first["event_id"]) == ("taken", tick["data"]["event_id"])
    # When it was answered, in the person's zone: a slot with no clock time
    # ("once a day") has only that to show on a timeline.
    from datetime import datetime
    assert datetime.fromisoformat(first["answered_at"]).date() == _today()
    assert first["answered_at"].endswith("+08:00")

    undo = await api("DELETE", f"/doses/{tick['data']['event_id']}")
    assert undo["code"] == 0, undo
    first = (await api("GET", "/today"))["data"]["slots"][0]
    assert (first["state"], first["event_id"]) == ("due", None)


async def test_someone_else_s_plan_or_dose_answers_as_a_missing_one(api):
    plan_id = await _plan(api)
    tick = await api("POST", "/doses", {"plan_id": plan_id, "date": _today().isoformat(), "slot": "00:00", "status": "taken"})
    api.as_user("2")
    assert (await api("GET", "/today"))["data"]["slots"] == []
    stranger = await api("POST", "/doses", {"plan_id": plan_id, "date": _today().isoformat(), "slot": "00:00", "status": "skipped"})
    assert stranger["code"] == 404
    assert (await api("DELETE", f"/doses/{tick['data']['event_id']}"))["code"] == 404
    api.as_user("1")
    assert (await api("GET", "/today"))["data"]["slots"][0]["state"] == "taken"


async def test_a_slot_the_plan_does_not_schedule_is_refused(api):
    plan_id = await _plan(api)
    answer = await api("POST", "/doses", {"plan_id": plan_id, "date": _today().isoformat(), "slot": "12:00", "status": "taken"})
    assert answer["code"] == 400
