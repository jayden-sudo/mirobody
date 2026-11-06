"""Searching the catalogue of coded metrics by Chinese or English name, LOINC code or category."""

from mirobody.kernel import metrics


def codes(q, **kw):
    return [r["code"] for r in metrics.search(q, **kw)]


def test_finds_by_chinese_name_english_name_and_loinc_code():
    assert "40443-4" in codes("静息心率")
    assert "40443-4" in codes("resting heart")
    assert codes("40443-4")[0] == "40443-4"
    assert "29463-7" in codes("体重")
    assert "29463-7" in codes("weight")


def test_each_result_is_one_loinc_code_with_what_a_form_needs():
    row = next(r for r in metrics.search("静息心率") if r["code"] == "40443-4")
    assert row["display_zh"] == "静息心率" and row["zh_machine"] is False
    assert row["display"] and row["category"] and row["standard_unit"] and row["unit_ucum"] and row["data_type"]
    assert row["system"] == "http://loinc.org"
    both = codes("心率")
    assert len(both) == len(set(both))                          # a code shared by several catalogue members appears once


def test_a_machine_translated_name_is_marked_and_searchable():
    row = next(r for r in metrics.search("谷丙") + metrics.search("丙氨酸氨基转移酶") if "丙氨酸" in r["display_zh"])
    assert row["zh_machine"] is True


def test_ranking_limit_and_edge_cases():
    first = metrics.search("心率", limit=3)
    assert len(first) <= 3
    assert metrics.search("") == [] and metrics.search("   ") == []
    assert metrics.search("zzzz-no-such-thing") == []
    assert len(metrics.search("e", limit=500)) <= 50                 # the cap holds
    assert codes("resting heart rate")[0] == "40443-4"                # exact display beats partial hits
    assert metrics.search("HEART RATE") == metrics.search("heart rate")      # case does not matter


def test_uncoded_catalogue_rows_are_never_offered():
    offered = {r["code"] for r in metrics.search("a", limit=50)}
    coded = {m.loinc for m in metrics.ROWS if m.loinc and m.confidence in ("confident", "unverified")}
    assert offered <= coded


async def test_search_route_needs_a_session_and_returns_only_catalogue_data():
    import importlib

    import pytest
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi import FastAPI
    ir = importlib.import_module("mirobody.server.routers.indicator_router")
    app = FastAPI()
    app.include_router(ir.router)
    app.dependency_overrides[ir.verify_token] = lambda: "501"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        body = (await client.get("/api/v1/health-indicators/catalog", params={"q": "静息心率"})).json()
        assert body["code"] == 0 and body["data"]["rows"][0]["code"] == "40443-4"
        assert (await client.get("/api/v1/health-indicators/catalog", params={"q": ""})).status_code == 422
        assert (await client.get("/api/v1/health-indicators/catalog", params={"q": "x", "limit": 51})).status_code == 422
        assert (await client.get("/api/v1/health-indicators/catalog")).status_code == 422
    app.dependency_overrides.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        assert (await client.get("/api/v1/health-indicators/catalog", params={"q": "heart"})).status_code != 200
