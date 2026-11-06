"""Simplified-Chinese names for every metric that carries a standard (LOINC) code."""

import re

from mirobody.kernel import metrics

HAN = re.compile(r"[一-鿿]")
CODED = [m for m in metrics.ROWS if m.loinc and m.confidence in ("confident", "unverified")]


def _table(locale: str) -> dict[str, str]:
    return {r["key"]: r["label"] for r in metrics._read_tsv(f"catalog/labels/{locale}.tsv") if r["kind"] == "metric"}


def test_every_coded_metric_has_a_chinese_name():
    curated, auto = _table("zh"), _table("zh-auto")
    missing = [m.member for m in CODED if not (curated.get(m.member) or auto.get(m.member))]
    assert missing == []


def test_machine_names_are_well_formed_and_never_shadow_a_curated_one():
    curated, auto = _table("zh"), _table("zh-auto")
    members = {m.member for m in metrics.ROWS}
    assert auto and set(auto) <= members
    assert set(auto).isdisjoint(curated)                     # a hand-written label always wins
    for key, text in auto.items():
        assert text == text.strip() and "\t" not in text and "\n" not in text, key
        assert HAN.search(text), (key, text)                 # an abbreviation alone is not a Chinese name
        assert len(text) <= 60, (key, text)


def test_a_loinc_code_has_one_name_and_curated_comes_first():
    assert metrics.zh_for_loinc("40443-4") == ("静息心率", False)          # curated (hand written)
    machine = next(m for m in CODED if m.member in _table("zh-auto") and m.loinc not in
                   {c.loinc for c in CODED if c.member in _table("zh")})
    label, is_machine = metrics.zh_for_loinc(machine.loinc)
    assert label == _table("zh-auto")[machine.member] and is_machine is True
    assert metrics.zh_for_loinc("0000-0") is None and metrics.zh_for_loinc("") is None


def test_every_coded_loinc_resolves():
    assert [m.loinc for m in CODED if metrics.zh_for_loinc(m.loinc) is None] == []


def test_rows_carry_the_chinese_name_and_say_when_it_is_a_machine_translation():
    from mirobody.collect import query as q
    only_machine = next(m for m in CODED if m.member in _table("zh-auto") and m.loinc not in
                        {c.loinc for c in CODED if c.member in _table("zh")})
    loinc = "http://loinc.org"
    hand = q._catalog_row({"code_system": loinc, "code": "40443-4", "series_id": "s", "display": "x"})
    assert (hand["indicator_zh"], hand["zh_machine"]) == ("静息心率", False)
    auto = q._catalog_row({"code_system": loinc, "code": only_machine.loinc, "series_id": "s", "display": "x"})
    assert (auto["indicator_zh"], auto["zh_machine"]) == (_table("zh-auto")[only_machine.member], True)
    none = q._catalog_row({"code_system": "mirobody-device", "code": "sleep_duration", "series_id": "s", "display": "x"})
    assert (none["indicator_zh"], none["zh_machine"]) == ("", False)
    day = q._timeline_row({"entry_kind": "day", "code_system": loinc, "code": "40443-4", "display": "Resting heart rate",
                           "local_date": "2026-11-04", "entry_id": "e"})
    assert (day["indicator_zh"], day["zh_machine"]) == ("静息心率", False)
