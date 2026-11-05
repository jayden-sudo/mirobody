"""The metric registry: validation, unit conversion, rendering, local-time rules."""

from datetime import datetime, timedelta, UTC

import pytest

from mirobody.collect.hub_registry import Registry, Reject, check_range, parse_time, render

YAML = """
version: 1
metrics:
  sleep_duration:
    name: sleep duration
    aliases: ["睡眠时长"]
    kind: quantity
    unit: min
    accept_units: {min: 1, h: 60}
    decimals: 0
    bounds: [0, 1440]
    window: "18:00"
  glucose:
    name: blood glucose
    kind: quantity
    unit: mmol/L
    accept_units: {mmol/L: 1, mg/dL: 0.0555}
    decimals: 2
    bounds: [1.0, 35.0]
    censor: {LOW: "<2.2"}
  stress:
    name: stress level
    kind: ordinal
    unit: ""
    allowed: [低, 中, 高]
  on_wrist:
    name: on wrist
    kind: boolean
    unit: ""
"""


@pytest.fixture
def reg(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(YAML, encoding="utf-8")
    return Registry.load(str(p))


def test_alias_and_key_lookup(reg):
    assert reg.find("sleep_duration") is reg.find(" 睡眠时长 ")
    with pytest.raises(Reject, match="unknown-metric"):
        reg.find("nope")


@pytest.mark.parametrize("patch", [
    "    decimals: 0\n", "    bounds: [0, 1440]\n", "    accept_units: {min: 1, h: 60}\n",
])
def test_invalid_registries_fail_at_load(tmp_path, patch):
    bad = {
        "decimals": YAML.replace(patch, "", 1),                          # quantity needs decimals
        "bounds": YAML.replace(patch, "    bounds: [5, 1]\n", 1),         # reversed
        "accept_units": YAML.replace(patch, "    accept_units: {min: 0}\n", 1),  # factor <= 0
    }
    key = "decimals" if "decimals" in patch else "bounds" if "bounds" in patch else "accept_units"
    p = tmp_path / "bad.yaml"
    p.write_text(bad[key], encoding="utf-8")
    with pytest.raises(ValueError):
        Registry.load(str(p))


def test_duplicate_alias_and_missing_allowed_fail(tmp_path):
    for text in (YAML.replace('aliases: ["睡眠时长"]', 'aliases: [glucose]'),
                 YAML.replace("    allowed: [低, 中, 高]\n", "")):
        p = tmp_path / "bad.yaml"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            Registry.load(str(p))


def test_quantity_conversion_and_fixed_rendering(reg):
    m = reg.find("glucose")
    assert render(m, 100, "mg/dL") == "5.55"
    assert render(m, 0.1 + 0.2 + 5, "mmol/L") == "5.30"
    assert render(reg.find("sleep_duration"), 5.5, "h") == "330"
    assert render(reg.find("sleep_duration"), 0.4, "min") == "0"   # no "-0", no sci notation


def test_quantity_rejections(reg):
    m = reg.find("sleep_duration")
    for value, unit, why in [(10, "kg", "unknown-unit"), ("10", "min", "bad-value-type"),
                             (True, "min", "bad-value-type"), (1441, "min", "out-of-range"),
                             (-1, "min", "out-of-range"), (10**40, "min", "out-of-range")]:
        with pytest.raises(Reject, match=why):
            render(m, value, unit)
    assert render(m, 0, "min") == "0" and render(m, 1440, "min") == "1440"   # bounds are inclusive


def test_censor_marker_is_stored_as_given_in_registry(reg):
    m = reg.find("glucose")
    assert render(m, "LOW", "mmol/L") == "<2.2"
    with pytest.raises(Reject, match="bad-value-type"):
        render(m, "HIGH", "mmol/L")


def test_labels_and_booleans(reg):
    assert render(reg.find("stress"), " 高 ", "") == "高"
    with pytest.raises(Reject, match="bad-label"):
        render(reg.find("stress"), "極高", "")
    with pytest.raises(Reject, match="bad-value-type"):
        render(reg.find("stress"), 3, "")
    assert render(reg.find("on_wrist"), True, "") == "true"
    assert render(reg.find("on_wrist"), False, "") == "false"
    with pytest.raises(Reject, match="bad-value-type"):
        render(reg.find("on_wrist"), 1, "")


def test_window_is_exposed(reg):
    assert reg.find("sleep_duration").window == "18:00"
    assert reg.find("glucose").window == ""


# ---- time -------------------------------------------------------------------------


def test_offset_and_z_times():
    dt, tz = parse_time("2026-11-05T02:00:00+08:00", None)
    assert dt.utcoffset() == timedelta(hours=8) and tz == "UTC+08:00"
    dt, tz = parse_time("2026-11-05T02:00:00Z", None)
    assert dt.utcoffset() == timedelta(0) and tz == "UTC+00:00"
    assert parse_time("2026-11-05T02:00:00+08:00", "Asia/Taipei")[1] == "Asia/Taipei"


def test_naive_time_needs_batch_tz():
    dt, tz = parse_time("2026-11-05T02:00:00", "Asia/Taipei")
    assert dt.utcoffset() == timedelta(hours=8) and tz == "Asia/Taipei"
    with pytest.raises(Reject, match="missing-tz"):
        parse_time("2026-11-05T02:00:00", None)


@pytest.mark.parametrize("text", ["2016/1/1 00:00", "2026-11-05", "yesterday", "2026-11-05T25:00:00", ""])
def test_non_iso_is_bad_time(text):
    with pytest.raises(Reject, match="bad-time"):
        parse_time(text, "Asia/Taipei")


@pytest.mark.parametrize("zone,text", [
    ("America/New_York", "2026-03-08T02:30:00"), ("America/New_York", "2026-11-01T01:30:00"),
    ("Europe/Berlin", "2026-03-29T02:30:00"), ("Europe/Berlin", "2026-10-25T02:30:00"),
])
def test_dst_gaps_and_overlaps_are_refused(zone, text):
    with pytest.raises(Reject, match="ambiguous-or-nonexistent-local-time"):
        parse_time(text, zone)


@pytest.mark.parametrize("zone,text", [
    ("America/New_York", "2026-03-08T03:30:00"), ("America/New_York", "2026-03-08T01:59:00"),
    ("Europe/Berlin", "2026-03-29T03:30:00"), ("Europe/Berlin", "2026-10-25T03:30:00"),
])
def test_times_next_to_a_transition_pass(zone, text):
    parse_time(text, zone)


NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def test_range_rules():
    t = NOW - timedelta(hours=40)
    check_range(t, t, NOW, 4000)
    check_range(t, t + timedelta(hours=36), NOW, 4000)
    for start, end, why in [
        (NOW + timedelta(hours=25), NOW + timedelta(hours=25), "future"),
        (NOW - timedelta(days=4001), NOW - timedelta(days=4001), "too-old"),
        (t, t - timedelta(seconds=1), "bad-interval"),
        (t, t + timedelta(hours=36, seconds=1), "bad-interval"),
    ]:
        with pytest.raises(Reject, match=why):
            check_range(start, end, NOW, 4000)


def test_censor_marker_matches_after_nfc_and_trim(reg):
    assert render(reg.find("glucose"), " LOW ", "mmol/L") == "<2.2"
