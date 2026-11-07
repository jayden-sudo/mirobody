"""Erasing a person against a real Postgres: only the caller's rows go, across every
table; another person's rows, and an object another person's file still uses, stay;
the account is closed and scrubbed and the next sign-in of the same identity gets a
new one. Skipped without MIROBODY_TEST_POSTGRES_DSN (a disposable database: it
creates and drops one schema)."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")
pytest.importorskip("cryptography")

DSN = os.environ.get("MIROBODY_TEST_POSTGRES_DSN", "")
pytestmark = [pytest.mark.needs_db, pytest.mark.skipif(not DSN, reason="MIROBODY_TEST_POSTGRES_DSN is not set")]

SCHEMAS = ("10_accounts.sql", "20_files.sql", "30_observations.sql", "31_medications.sql", "32_genomics.sql",
           "40_devices.sql", "41_device_rules.sql", "50_chat.sql", "60_tasks.sql", "62_personal_mcp.sql")

_FILL = {"text": "x", "character varying": "x", "character": "x", "varchar": "x", "integer": 0, "bigint": 0, "smallint": 0,
         "boolean": False, "numeric": 0, "jsonb": "{}", "json": "{}", "uuid": None, "ARRAY": None}


@pytest.fixture
async def world(monkeypatch):
    from pathlib import Path

    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    from mirobody.user import erase
    from mirobody.utils import db

    schema = "erase_" + uuid4().hex
    conninfo = make_url(DSN).set(drivername="postgresql").render_as_string(hide_password=False)
    admin = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    engine = None
    try:
        await admin.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        await admin.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
        await admin.execute("CREATE FUNCTION encrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT nullif($1,'')$$")
        await admin.execute("CREATE FUNCTION decrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT $1$$")
        root = Path(erase.__file__).resolve().parents[1] / "schema"
        for f in SCHEMAS:
            await admin.execute((root / f).read_text())
        engine = create_async_engine(make_url(DSN).set(drivername="postgresql+psycopg"),
                                     connect_args={"options": f"-c search_path={schema},public -c timezone=UTC"})
        monkeypatch.setattr(db, "_provider", lambda _: engine)
        monkeypatch.setattr(db, "_engines", {})

        from mirobody.user import user as account

        account._closed.clear()
        account._active.clear()

        class W:
            pass

        w = W()
        w.admin, w.schema, w.q = admin, schema, db.execute_query
        w.deleted_keys, w.fail_keys, w.threads = [], set(), []

        async def delete_file(key):
            if key in w.fail_keys:
                return False
            w.deleted_keys.append(key)
            return True

        async def delete_thread(thread):
            w.threads.append(thread)

        monkeypatch.setattr("mirobody.collect.files.services.file_processing_service.delete_file_from_storage", delete_file)
        monkeypatch.setattr("mirobody.agent.checkpointer.delete_thread", delete_thread)

        async def seed(table, **cols):
            """Insert one row, filling every other NOT NULL column that has no default."""
            info = await db.execute_query(
                "SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_schema = :s AND table_name = :t"
                " AND is_nullable = 'NO' AND column_default IS NULL AND is_identity = 'NO' AND is_generated = 'NEVER'",
                {"s": schema, "t": table}, log_sql=False)
            row = dict(cols)
            for c in info:
                if c["column_name"] not in row:
                    kind = c["data_type"]
                    row[c["column_name"]] = (
                        date(2026, 1, 1) if kind == "date"
                        else datetime(2026, 1, 1, tzinfo=UTC) if kind.startswith("timestamp")
                        else uuid4().hex[: c["character_maximum_length"] or 32] if kind in ("text", "character varying", "character", "varchar")
                        else _FILL.get(kind, "x")
                    )
            names = ", ".join(row)
            marks = ", ".join(f":{k}" if not isinstance(v, dict) else f"CAST(:{k} AS jsonb)" for k, v in row.items())
            params = {k: (str(v) if isinstance(v, dict) else v) for k, v in row.items()}
            await db.execute_query(f"INSERT INTO {table} ({names}) VALUES ({marks})", params, log_sql=False)

        async def user(email, **kw):
            rows = await db.execute_query(
                "INSERT INTO health_app_user (is_del, email, name, birth, blood, gender, password_hash, managed_by)"
                " VALUES (false, :e, 'Ada', '1990-01-01', 'O+', 2, 'hash', :m) RETURNING id",
                {"e": email, "m": kw.get("managed_by")}, log_sql=False)
            return rows[0]["id"] if isinstance(rows, list) else rows["id"]

        async def count(table, where="true", **params):
            rows = await db.execute_query(f"SELECT count(*) AS n FROM {table} WHERE {where}", params, log_sql=False)
            return int((rows[0] if isinstance(rows, list) else rows)["n"])

        w.seed, w.user, w.count = seed, user, count
        yield w
    finally:
        from mirobody.user import user as account

        account._closed.clear()
        account._active.clear()
        if engine is not None:
            await engine.dispose()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


async def fill(w, uid):
    """One row of every kind for `uid` (a string id)."""
    await w.seed("th_extraction", user_id=uid, source_ref="ref", source_kind="manual", extractor="manual", status="ok", digest=f"d{uid}")
    ext = await w.q("SELECT id FROM th_extraction WHERE user_id = :u", {"u": uid}, log_sql=False)
    await w.seed(
        "th_observation", user_id=uid, kind="measurement", modality="manual", source_kind="manual", tz="UTC", tz_source="iana",
        grain="instant", value_kind="narrative", extraction_id=ext[0]["id"],
    )
    await w.seed("th_check_result", user_id=uid)
    await w.seed("th_series_data_genetic", user_id=uid)
    await w.seed("series_data", user_id=uid)
    await w.seed("th_medication_plan", user_id=uid)
    await w.seed("th_medication_course", user_id=uid)
    await w.seed("th_dose_event", user_id=uid)
    await w.seed("th_override", user_id=uid)
    await w.seed("health_user_profile_by_system", user_id=uid)
    await w.seed("health_user_provider", user_id=uid)
    await w.seed("health_vital_user", app_user_id=uid)
    await w.seed("health_data_garmin", theta_user_id=uid)
    await w.seed("th_task_queue", queue_name="profile_refresh_queue", payload=uid)


TABLES_BY_USER = {
    "th_observation": "user_id", "th_extraction": "user_id", "th_check_result": "user_id", "th_series_data_genetic": "user_id",
    "series_data": "user_id", "th_medication_plan": "user_id", "th_medication_course": "user_id", "th_dose_event": "user_id",
    "th_override": "user_id", "health_user_profile_by_system": "user_id", "health_user_provider": "user_id",
    "health_vital_user": "app_user_id", "health_data_garmin": "theta_user_id",
}


async def test_only_the_callers_rows_go_and_the_account_is_closed(world):
    from mirobody.user import erase, user as account

    a = await world.user("a@x.test")
    b = await world.user("b@x.test")
    await fill(world, str(a))
    await fill(world, str(b))
    await world.seed("th_task_queue", queue_name="other_queue", payload=str(a))  # another queue is not ours to touch
    for t, col in TABLES_BY_USER.items():
        assert await world.count(t, f"{col} = :u", u=str(a)) == 1, t

    out = await erase.erase_user_data(str(a))

    for t, col in TABLES_BY_USER.items():
        assert await world.count(t, f"{col} = :u", u=str(a)) == 0, f"{t} still has the caller's rows"
        assert await world.count(t, f"{col} = :u", u=str(b)) == 1, f"{t} lost another person's rows"
    assert await world.count("th_task_queue", "payload = :u AND queue_name = 'profile_refresh_queue'", u=str(a)) == 0
    assert await world.count("th_task_queue", "payload = :u AND queue_name = 'profile_refresh_queue'", u=str(b)) == 1
    assert await world.count("th_task_queue", "queue_name = 'other_queue'") == 1
    assert out["erased"]["records"] >= 1 and out["objects_pending"] == 0

    row = (await world.q("SELECT * FROM health_app_user WHERE id = :i", {"i": a}, log_sql=False))[0]
    assert row["is_del"] is True and row["name"] == "" and row["birth"] is None and row["blood"] is None
    assert row["password_hash"] is None and row["email"].endswith("@deleted.invalid") and row["tokens_valid_after"] is not None
    other = (await world.q("SELECT * FROM health_app_user WHERE id = :i", {"i": b}, log_sql=False))[0]
    assert other["is_del"] is False and other["name"] == "Ada" and other["birth"] == "1990-01-01"
    assert await account.get_user(user_id=a) is None and await account.is_active_account(a) is False


async def test_files_chat_and_access_grants(world):
    from mirobody.user import erase

    a, b = await world.user("a@x.test"), await world.user("b@x.test")
    A, B = str(a), str(b)
    # files: A's own, one a caregiver uploaded about A, one A uploaded about B (B's), one B's own; one object shared by A and B
    await world.seed("th_files", user_id=A, file_key="a/own.pdf")
    await world.seed("th_files", user_id=B, query_user_id=A, file_key="a/by-caregiver.pdf")
    await world.seed("th_files", user_id=A, query_user_id=B, file_key="b/uploaded-by-a.pdf")
    await world.seed("th_files", user_id=B, file_key="b/own.pdf")
    # one object A's file row and B's genotype set both point at (keys are unique per file row, but a genotype set names its own)
    await world.seed("th_files", user_id=A, file_key="shared/object.pdf")
    await world.seed("th_genotype_set", user_id=B, file_key="shared/object.pdf")
    # chat: A asked about A, B asked about A, A asked about B, B asked about B
    for sid, asker, subject in (("s-aa", A, A), ("s-ba", B, A), ("s-ab", A, B), ("s-bb", B, B)):
        await world.seed("th_sessions", session_id=sid, user_id=asker, query_user_id=subject)
        await world.seed("th_messages", session_id=sid, user_id=asker, query_user_id=subject)
        await world.seed("th_chat_citation", session_id=sid, subject_id=subject)
    # access: MCP links in both directions, passkey, care circles
    await world.seed("th_personal_mcp_url", creator_id=B, subject_id=A, secret_hash=uuid4().bytes)
    await world.seed("th_personal_mcp_url", creator_id=B, subject_id=B, secret_hash=uuid4().bytes)
    await world.seed("webauthn_credentials", user_id=a, credential_id=b"\x01", public_key=b"k")
    await world.seed("webauthn_credentials", user_id=b, credential_id=b"\x02", public_key=b"k")
    mine = (await world.q("INSERT INTO care_circles (owner_user_id, name) VALUES (:o, 'a') RETURNING id", {"o": a}, log_sql=False))[0]["id"]
    theirs = (await world.q("INSERT INTO care_circles (owner_user_id, name) VALUES (:o, 'b') RETURNING id", {"o": b}, log_sql=False))[0]["id"]
    for circle, member in ((mine, a), (mine, b), (theirs, b), (theirs, a)):
        await world.q("INSERT INTO care_circle_members (care_circle_id, user_id, status) VALUES (:c, :u, 2)", {"c": circle, "u": member}, log_sql=False)

    out = await erase.erase_user_data(A)

    left = {r["file_key"] for r in await world.q("SELECT file_key FROM th_files", log_sql=False)}
    assert left == {"b/uploaded-by-a.pdf", "b/own.pdf"}
    assert sorted(world.deleted_keys) == ["a/by-caregiver.pdf", "a/own.pdf"]  # the object B's genotype set still names is not deleted
    assert await world.count("th_genotype_set", "user_id = :b", b=B) == 1
    assert await world.count("th_storage_purge") == 0 and out["objects_pending"] == 0
    sessions = {r["session_id"] for r in await world.q("SELECT session_id FROM th_sessions", log_sql=False)}
    assert sessions == {"s-bb"}, "a session A asked, or that was about A, goes; B's about B stays"
    assert {r["session_id"] for r in await world.q("SELECT session_id FROM th_messages", log_sql=False)} == {"s-bb"}
    assert {r["session_id"] for r in await world.q("SELECT session_id FROM th_chat_citation", log_sql=False)} == {"s-bb"}
    assert len(world.threads) == 3, "one conversation copy per erased session"
    assert await world.count("th_personal_mcp_url", "subject_id = :a", a=A) == 0
    assert await world.count("th_personal_mcp_url", "subject_id = :b", b=B) == 1
    assert await world.count("webauthn_credentials", "user_id = :u", u=a) == 0 and await world.count("webauthn_credentials", "user_id = :u", u=b) == 1
    circles = {r["name"] for r in await world.q("SELECT name FROM care_circles", log_sql=False)}
    assert circles == {"b"}
    members = {(r["care_circle_id"], r["user_id"]) for r in await world.q("SELECT care_circle_id, user_id FROM care_circle_members", log_sql=False)}
    assert members == {(theirs, b)}, "A's circle goes with its members; A's membership in B's circle goes"


async def test_a_failed_object_delete_is_remembered_and_retried_by_the_next_erasure(world):
    from mirobody.user import erase

    a, b = await world.user("a@x.test"), await world.user("b@x.test")
    await world.seed("th_files", user_id=str(a), file_key="a/stuck.pdf")
    await world.seed("th_files", user_id=str(b), file_key="b/own.pdf")
    world.fail_keys.add("a/stuck.pdf")
    first = await erase.erase_user_data(str(a))
    assert first["objects_pending"] == 1
    assert [r["file_key"] for r in await world.q("SELECT file_key FROM th_storage_purge", log_sql=False)] == ["a/stuck.pdf"]
    world.fail_keys.clear()
    second = await erase.erase_user_data(str(b))
    assert second["objects_pending"] == 0
    assert sorted(world.deleted_keys) == ["a/stuck.pdf", "b/own.pdf"]


async def test_an_account_that_manages_others_is_refused_and_nothing_changes(world):
    from mirobody.user import erase

    a = await world.user("a@x.test")
    await world.user("kid@x.test", managed_by=a)
    await fill(world, str(a))
    with pytest.raises(erase.AccountManagesOthers):
        await erase.erase_user_data(str(a))
    assert await world.count("th_observation", "user_id = :u", u=str(a)) == 1
    assert (await world.q("SELECT is_del FROM health_app_user WHERE id = :i", {"i": a}, log_sql=False))[0]["is_del"] is False


async def test_a_closed_account_is_not_found_and_a_second_call_changes_nothing(world):
    from mirobody.user import erase

    a = await world.user("a@x.test")
    await erase.erase_user_data(str(a))
    with pytest.raises(erase.AccountNotFound):
        await erase.erase_user_data(str(a))


async def test_the_next_sign_in_of_the_same_identity_gets_a_new_empty_account(world):
    from mirobody.user import erase, user as account

    a = await world.user("a@x.test")
    await world.q("INSERT INTO external_identity (provider, subject, user_id) VALUES ('sibian', 'did:x', :u)", {"u": a}, log_sql=False)
    await erase.erase_user_data(str(a))
    new = await account.ensure_sibian_user("did:x", "Ada", "iss")
    assert new is not None and new.user_id != a and new.is_new


async def test_an_extraction_another_person_also_uses_does_not_block_the_erasure(world):
    from mirobody.user import erase

    a, b = await world.user("a@x.test"), await world.user("b@x.test")
    A, B = str(a), str(b)
    await world.seed("th_extraction", user_id=A, source_ref="shared-batch", source_kind="manual", extractor="manual", status="ok", digest="same")
    await world.seed("th_extraction", user_id=A, source_ref="only-mine", source_kind="manual", extractor="manual", status="ok", digest="mine")
    shared = (await world.q("SELECT id FROM th_extraction WHERE source_ref = 'shared-batch'", log_sql=False))[0]["id"]
    kw = {"kind": "measurement", "modality": "manual", "source_kind": "manual", "tz": "UTC", "tz_source": "iana", "grain": "instant", "value_kind": "narrative"}
    await world.seed("th_observation", user_id=A, extraction_id=shared, **kw)
    await world.seed("th_observation", user_id=B, extraction_id=shared, **kw)   # B ingested the same batch: the same row
    out = await erase.erase_user_data(A)                                         # used to abort on the foreign key
    assert await world.count("th_observation", "user_id = :u", u=B) == 1
    assert await world.count("th_observation", "user_id = :u", u=A) == 0
    assert await world.count("th_extraction", "source_ref = 'shared-batch'") == 1        # B still points at it
    assert await world.count("th_extraction", "source_ref = 'only-mine'") == 0
    assert out["erased"]["records"] >= 2


async def test_conversation_copies_are_deleted_by_thread_id_even_with_no_saver(world, monkeypatch):
    from mirobody.user import erase

    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        await world.admin.execute(f"CREATE TABLE {table} (thread_id text NOT NULL, note text)")
    a, b = await world.user("a@x.test"), await world.user("b@x.test")
    A, B = str(a), str(b)
    await world.seed("th_sessions", session_id="s-aa", user_id=A, query_user_id=A)
    await world.seed("th_sessions", session_id="s-ba", user_id=B, query_user_id=A)      # B asked about A
    await world.seed("th_sessions", session_id="s-bb", user_id=B, query_user_id=B)
    for thread in (f"{A}:s-aa", f"{B}:s-ba", f"{A}:never-saved-a-session-row", f"{B}:s-bb"):
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            await world.admin.execute(f"INSERT INTO {table} (thread_id) VALUES ('{thread}')")

    async def no_saver(thread):                    # delete_thread swallows everything and does nothing without a saver
        return None

    monkeypatch.setattr("mirobody.agent.checkpointer.delete_thread", no_saver)
    out = await erase.erase_user_data(A)
    left = {r["thread_id"] for r in await world.q("SELECT thread_id FROM checkpoints", log_sql=False)}
    assert left == {f"{B}:s-bb"}
    for table in ("checkpoint_blobs", "checkpoint_writes"):
        assert {r["thread_id"] for r in await world.q(f"SELECT thread_id FROM {table}", log_sql=False)} == {f"{B}:s-bb"}
    assert out["threads_failed"] == 0


async def test_a_deployment_without_the_purge_table_still_deletes_the_objects(world):
    from mirobody.user import erase

    await world.admin.execute("DROP TABLE th_storage_purge")
    a = await world.user("a@x.test")
    await world.seed("th_files", user_id=str(a), file_key="a/own.pdf")
    out = await erase.erase_user_data(str(a))
    assert world.deleted_keys == ["a/own.pdf"]                    # not orphaned, though not retryable
    assert out["objects_pending"] == 0


async def test_failed_objects_are_retried_least_failed_first(world):
    from mirobody.user import erase

    for i in range(erase.RETRY_BATCH + 5):                         # keys that can never be deleted, many times over
        await world.seed("th_storage_purge", file_key=f"stuck/{i}", attempts=50)
    await world.seed("th_storage_purge", file_key="fresh/leftover", attempts=0)
    world.fail_keys.update(f"stuck/{i}" for i in range(erase.RETRY_BATCH + 5))
    a = await world.user("a@x.test")
    await erase.erase_user_data(str(a))
    assert "fresh/leftover" in world.deleted_keys                  # not starved by the stuck ones


async def test_a_queued_object_someone_started_using_since_is_dropped_from_the_queue_not_deleted(world):
    from mirobody.user import erase

    a, b = await world.user("a@x.test"), await world.user("b@x.test")
    await world.seed("th_storage_purge", file_key="queued/shared.pdf")
    await world.seed("th_storage_purge", file_key="queued/free.pdf")
    await world.seed("th_files", user_id=str(b), file_key="queued/shared.pdf")     # B's file row names it now
    await erase.erase_user_data(str(a))
    assert "queued/shared.pdf" not in world.deleted_keys                           # B's file is still there
    assert "queued/free.pdf" in world.deleted_keys
    assert await world.count("th_storage_purge") == 0                              # off the queue either way


async def test_a_key_from_this_very_erasure_that_someone_names_before_the_purge_runs_is_kept(world):
    from mirobody.user import erase

    b = await world.user("b@x.test")
    await world.seed("th_storage_purge", file_key="just/queued.pdf")             # queued by the transaction that just committed
    await world.seed("th_files", user_id=str(b), file_key="just/queued.pdf")     # and a new row names it before the purge runs
    await erase._purge_objects(["just/queued.pdf"])
    assert "just/queued.pdf" not in world.deleted_keys
    assert await world.count("th_storage_purge") == 0


async def test_when_sharing_cannot_be_checked_nothing_is_deleted_and_the_keys_stay_queued(world, monkeypatch):
    from mirobody.user import erase
    from mirobody.utils import db

    await world.seed("th_storage_purge", file_key="queued/a.pdf")
    real = db.execute_query

    async def flaky(query, params=None, *a, **kw):
        if "FROM th_files WHERE file_key" in query:
            raise TimeoutError("connection dropped")
        return await real(query, params, *a, **kw)

    monkeypatch.setattr(db, "execute_query", flaky)
    pending = await erase._purge_objects(["queued/a.pdf"])
    assert pending == -1                                      # "could not check" is not "not shared"
    assert world.deleted_keys == []
    monkeypatch.setattr(db, "execute_query", real)
    assert await world.count("th_storage_purge") == 1         # still queued: the next erasure retries it
