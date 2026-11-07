"""`user/erase.py` and `POST /api/v1/user/erase` without a database: the plan's
shape, that the route needs the literal confirmation and is caller-only, how it
answers each outcome. The rows themselves are in `test_erase_db.py`."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from fastapi import FastAPI

from mirobody.server.auth import verify_token
from mirobody.server.routers import user_erase_router
from mirobody.user import erase


def test_every_statement_is_scoped_to_the_person():
    for step in erase.PLAN:
        assert any(p in step.sql for p in (":u", ":ui", ":sessions")), step.sql
        assert "WHERE" in step.sql, "a delete without a WHERE would erase everyone"


def test_a_table_a_deployment_lacks_cannot_abort_the_plan():
    # every step names the tables it needs; the runner skips the step when one is missing
    assert all(step.tables for step in erase.PLAN)
    assert "th_series_data_retired_15" in erase.TABLES and "th_storage_purge" in erase.TABLES


def test_children_are_deleted_before_their_parents():
    order = [t for s in erase.PLAN for t in s.tables]
    assert order.index("th_observation") < order.index("th_extraction")
    assert order.index("care_circle_members") < order.index("care_circles")
    # chat sessions/messages, files and genotype sets are deleted with RETURNING inside _erase_rows, not by a plan step
    assert not {"th_messages", "th_sessions", "th_files", "th_genotype_set"} & set(order)
    assert "th_files" in erase.TABLES and "th_sessions" in erase.TABLES


def test_the_account_is_closed_and_scrubbed_by_the_same_transaction():
    sql = erase.CLOSE_ACCOUNT
    for part in ("is_del = TRUE", "tokens_valid_after = now()", "password_hash = NULL", "name = ''", "birth = NULL", "@deleted.invalid"):
        assert part in sql
    assert sql.rstrip().endswith("WHERE id = :ui")


def app(user_id="7"):
    a = FastAPI()
    a.include_router(user_erase_router.router)
    a.dependency_overrides[verify_token] = lambda: user_id
    return a


async def post(body, user_id="7"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(user_id)), base_url="http://t") as c:
        return (await c.post("/api/v1/user/erase", json=body)).json()


async def test_it_needs_the_exact_confirmation(monkeypatch):
    async def boom(uid):
        raise AssertionError("must not run")

    monkeypatch.setattr(user_erase_router, "erase_user_data", boom)
    for body in ({}, {"confirm": "yes"}, {"confirm": "ERASE-ALL-MY-DATA"}, {"confirm": True}):
        out = await post(body)
        assert out.get("code") == 400 or out.get("detail"), out


async def test_it_erases_the_caller_and_nobody_else(monkeypatch):
    seen = []

    async def fake(uid):
        seen.append(uid)
        return {"erased": {"records": 3}, "objects_pending": 0, "threads_failed": 0}

    monkeypatch.setattr(user_erase_router, "erase_user_data", fake)
    out = await post({"confirm": user_erase_router.CONFIRM, "target_user_id": "99", "user_id": "99"}, user_id="7")
    assert seen == ["7"]
    assert out["code"] == 0 and out["data"]["erased"] == {"records": 3}


async def test_managed_accounts_and_closed_accounts_have_their_own_answers(monkeypatch):
    async def managed(uid):
        raise erase.AccountManagesOthers

    async def gone(uid):
        raise erase.AccountNotFound

    monkeypatch.setattr(user_erase_router, "erase_user_data", managed)
    assert (await post({"confirm": user_erase_router.CONFIRM}))["code"] == 409
    monkeypatch.setattr(user_erase_router, "erase_user_data", gone)
    assert (await post({"confirm": user_erase_router.CONFIRM}))["code"] == 404


async def test_a_failure_says_nothing_about_why(monkeypatch):
    async def boom(uid):
        raise RuntimeError("secret detail Metformin")

    monkeypatch.setattr(user_erase_router, "erase_user_data", boom)
    out = await post({"confirm": user_erase_router.CONFIRM})
    assert out["code"] == 500 and "Metformin" not in str(out)
