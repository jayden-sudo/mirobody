"""Fixes from the code review that need no database: a missing label file, the range-rule cache after a failed
load. (The SQL-level ones are covered in the _db tests.)"""

from __future__ import annotations

import pytest

from mirobody.collect import observations
from mirobody.kernel import metrics


def test_a_missing_label_file_leaves_those_names_out_instead_of_failing_every_read(monkeypatch):
    real = metrics._read_tsv

    def missing(name):
        if name.endswith("zh-auto.tsv"):
            raise FileNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metrics, "_read_tsv", missing)
    monkeypatch.setattr(metrics, "_zh_by_loinc", None)
    assert metrics.zh_for_loinc("8302-2") is None or isinstance(metrics.zh_for_loinc("8302-2"), tuple)   # no exception
    assert isinstance(metrics._zh_by_loinc, dict)                                                       # and it was built, once


async def test_a_range_load_that_came_back_empty_is_asked_again_not_kept_for_good(monkeypatch):
    calls = []

    class Rules:
        async def load(self, db_config=None):
            calls.append(1)
            return 0 if len(calls) == 1 else 7

    monkeypatch.setattr(observations.standardize, "ValueRangeValidator", Rules)
    monkeypatch.setattr(observations, "_range_rules", None)
    monkeypatch.setattr(observations, "_range_rules_retry_at", 0.0)
    clock = [1000.0]
    monkeypatch.setattr(observations.time, "monotonic", lambda: clock[0])

    first = await observations.value_ranges()
    assert await observations.value_ranges() is first and len(calls) == 1      # inside the retry window: not asked again
    clock[0] += observations._RANGE_RETRY + 1
    second = await observations.value_ranges()
    assert len(calls) == 2 and second is not first                              # outage over: rules are loaded now
    clock[0] += 10_000
    assert await observations.value_ranges() is second and len(calls) == 2      # a real load is kept for good


async def test_an_empty_result_from_the_first_load_does_not_block_a_second_writer(monkeypatch):
    class Rules:
        async def load(self, db_config=None):
            return 0

    monkeypatch.setattr(observations.standardize, "ValueRangeValidator", Rules)
    monkeypatch.setattr(observations, "_range_rules", None)
    monkeypatch.setattr(observations, "_range_rules_retry_at", 0.0)
    assert await observations.value_ranges() is not None


@pytest.mark.parametrize("name", ["zh.tsv", "zh-auto.tsv"])
def test_both_label_files_still_ship_in_the_package(name):
    assert metrics._label_file(f"catalog/labels/{name}")
