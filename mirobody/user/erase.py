"""Erase everything the server holds about one person (`POST /api/v1/user/erase`).

What "everything" is, and whose it is. Every table is listed in `PLAN`, with the
column that makes a row this person's. The rules, one line each:

* A row whose SUBJECT is the person goes: observations and their coding,
  medications, genotypes, the generated profile, device credentials and data,
  files (a file's subject is `query_user_id` when it has one, else `user_id`),
  and the chat sessions they asked or that were asked about them.
* An access grant that names the person goes, either way round: the personal
  MCP links they made or that point at their record, their care-circle
  memberships, the circles they own (with everyone's membership in them) and the
  account activations they made or received.
* The `health_app_user` row is kept, scrubbed (address, name, birth, blood,
  password) and closed, in the SAME transaction as the data: a crash cannot
  leave a live account without its data, and every token is refused from the
  next check on (`tokens_valid_after`; another worker may take up to the
  account cache's 30 s to see it). A write that was already past that check
  when the transaction committed can still land under the closed id: the
  client waits for its own uploads before it asks, and the profile refresh
  looks the account up uncached before it starts. `external_identity` stays, so the next sign-in of the same sibian
  identity gets a new, empty account (`user.ensure_sibian_user`).
* An account that manages other accounts is refused: their records are theirs.
* A stored object another person's row still points at is not deleted (keys
  carry no owner, and an upload can be shared); the rest are queued in
  `th_storage_purge` inside the transaction and deleted after it, so a delete
  that fails or is interrupted is retried by the next erasure, never forgotten.

Not covered: backups (they keep copies until `RETENTION_DAYS`), and tables
that exist only in other deployments (`account_merge.py` names a few). A table
this deployment does not have is skipped; a missing table cannot abort the
erasure. Counts are returned, names and values never are, and nothing here logs
either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mirobody.kernel.ops import is_driver_exception

logger = logging.getLogger(__name__)

#: The only queue whose payload is a person's id.
PROFILE_QUEUE = "profile_refresh_queue"

#: Retry this many left-over objects from earlier erasures on every erasure.
RETRY_BATCH = 100


class AccountManagesOthers(Exception):
    """The account manages other accounts: remove or hand them over first."""


class AccountNotFound(Exception):
    """No live account: nothing to erase (it may already have been)."""


@dataclass(frozen=True)
class Step:
    """One delete: the group it counts under, the tables it needs, and the SQL.
    `:u` is the id as text, `:ui` as an integer, `:sessions` the session ids."""

    group: str
    tables: tuple[str, ...]
    sql: str


#: Order matters where a foreign key points the other way (observations before
#: their extraction, a circle's members before the circle).
PLAN: tuple[Step, ...] = (
    Step("chat", ("th_chat_citation",), "DELETE FROM th_chat_citation WHERE session_id = ANY(:sessions)"),
    Step("chat", ("th_chat_citation_seq",), "DELETE FROM th_chat_citation_seq WHERE session_id = ANY(:sessions)"),
    Step("chat", ("th_session_share",), "DELETE FROM th_session_share WHERE user_id = :u OR session_id = ANY(:sessions)"),
    Step("chat", ("th_messages",), "DELETE FROM th_messages WHERE user_id = :u OR query_user_id = :u OR session_id = ANY(:sessions)"),
    Step("chat", ("th_sessions",), "DELETE FROM th_sessions WHERE user_id = :u OR query_user_id = :u"),
    Step("records", ("th_check_result",), "DELETE FROM th_check_result WHERE user_id = :u"),
    Step("records", ("th_day_authority",), "DELETE FROM th_day_authority WHERE user_id = :u"),
    Step("records", ("th_observation",), "DELETE FROM th_observation WHERE user_id = :u"),
    Step("records", ("th_extraction",), "DELETE FROM th_extraction WHERE user_id = :u"),
    Step("records", ("th_coding_alias",), "DELETE FROM th_coding_alias WHERE scope = 'user:' || :u"),
    Step("records", ("series_data",), "DELETE FROM series_data WHERE user_id = :u"),
    Step("records", ("th_series_data_retired_15",), "DELETE FROM th_series_data_retired_15 WHERE user_id = :u"),
    Step("genetics", ("th_series_data_genetic",), "DELETE FROM th_series_data_genetic WHERE user_id = :u"),
    Step("genetics", ("th_genotype_set",), "DELETE FROM th_genotype_set WHERE user_id = :u"),
    Step("medications", ("th_medication_plan",), "DELETE FROM th_medication_plan WHERE user_id = :u"),
    Step("medications", ("th_medication_course",), "DELETE FROM th_medication_course WHERE user_id = :u"),
    Step("medications", ("th_dose_event",), "DELETE FROM th_dose_event WHERE user_id = :u"),
    Step("medications", ("th_override",), "DELETE FROM th_override WHERE user_id = :u"),
    Step("files", ("th_files",), "DELETE FROM th_files WHERE COALESCE(NULLIF(query_user_id, ''), user_id) = :u"),
    Step("profile", ("health_user_profile_by_system",), "DELETE FROM health_user_profile_by_system WHERE user_id = :u"),
    Step("devices", ("health_user_provider",), "DELETE FROM health_user_provider WHERE user_id = :u"),
    Step("devices", ("health_vital_user",), "DELETE FROM health_vital_user WHERE app_user_id = :u"),
    Step("devices", ("health_data_garmin",), "DELETE FROM health_data_garmin WHERE theta_user_id = :u"),
    Step("devices", ("health_data_whoop",), "DELETE FROM health_data_whoop WHERE theta_user_id = :u"),
    Step("devices", ("health_data_oura",), "DELETE FROM health_data_oura WHERE theta_user_id = :u"),
    Step("access", ("th_personal_mcp_url",), "DELETE FROM th_personal_mcp_url WHERE creator_id = :u OR subject_id = :u"),
    Step("access", ("webauthn_credentials",), "DELETE FROM webauthn_credentials WHERE user_id = :ui"),
    Step("access", ("th_account_activation",), "DELETE FROM th_account_activation WHERE member_id = :ui OR created_by = :ui"),
    Step(
        "access",
        ("care_circle_members", "care_circles"),
        "DELETE FROM care_circle_members WHERE user_id = :ui"
        " OR care_circle_id IN (SELECT id FROM care_circles WHERE owner_user_id = :ui)",
    ),
    Step("access", ("care_circles",), "DELETE FROM care_circles WHERE owner_user_id = :ui"),
    Step("queue", ("th_task_queue",), "DELETE FROM th_task_queue WHERE queue_name = '" + PROFILE_QUEUE + "' AND payload = :u"),
)

#: Every table the plan or the key collection reads or writes.
TABLES: tuple[str, ...] = tuple(
    dict.fromkeys(t for s in PLAN for t in s.tables)
    | dict.fromkeys(("th_files", "th_genotype_set", "th_sessions", "th_messages", "th_storage_purge"))
)

_FILE_SUBJECT = "COALESCE(NULLIF(query_user_id, ''), user_id)"

#: The scrub that closes the account row, run in the erasing transaction.
CLOSE_ACCOUNT = (
    "UPDATE health_app_user SET"
    " email = 'erased-' || md5(random()::text || id::text) || '@deleted.invalid',"
    " name = '', gender = NULL, birth = NULL, blood = NULL, ethnicity = NULL, apple_sub = NULL,"
    " password_hash = NULL, mfa_enabled = FALSE, is_del = TRUE,"
    " tokens_valid_after = now(), update_at = CURRENT_TIMESTAMP"
    " WHERE id = :ui"
)


async def _present(tx: Any) -> frozenset[str]:
    """Which of `TABLES` this deployment has (one query, before any delete)."""
    rows = await tx.execute(
        "SELECT n FROM unnest(CAST(:names AS text[])) AS n WHERE to_regclass(n) IS NOT NULL",
        {"names": list(TABLES)},
    )
    return frozenset(str(r["n"]) for r in rows or [])


async def _count(tx: Any, sql: str, params: dict[str, Any]) -> int:
    rows = await tx.execute(f"WITH d AS ({sql} RETURNING 1) SELECT count(*) AS n FROM d", params)
    return int((rows[0] if rows else {}).get("n") or 0)


async def _erase_rows(user_id: str) -> tuple[dict[str, int], list[str], list[tuple[str, str]]]:
    """The one transaction: lock and check the account, queue the storage keys,
    delete every row, close the account. Returns counts, the object keys to
    delete after the commit and the checkpoint threads' (owner, session) pairs."""
    from mirobody.utils import db

    uid, ui = str(user_id), int(user_id)
    counts: dict[str, int] = {}
    async with db.transaction() as tx:
        live = await tx.execute("SELECT id FROM health_app_user WHERE id = :ui AND is_del = false FOR UPDATE", {"ui": ui})
        if not live:
            raise AccountNotFound
        if await tx.execute("SELECT 1 FROM health_app_user WHERE managed_by = :ui AND is_del = false LIMIT 1", {"ui": ui}):
            raise AccountManagesOthers
        present = await _present(tx)
        params: dict[str, Any] = {"u": uid, "ui": ui}

        keys: list[str] = []
        if "th_files" in present:
            keys += [r["file_key"] for r in await tx.execute(
                f"SELECT file_key FROM th_files WHERE {_FILE_SUBJECT} = :u AND file_key <> ''", params) or []]
        if "th_genotype_set" in present:
            keys += [r["file_key"] for r in await tx.execute(
                "SELECT file_key FROM th_genotype_set WHERE user_id = :u AND file_key IS NOT NULL AND file_key <> ''", params) or []]
        keys = sorted(set(keys))
        if keys and "th_storage_purge" in present:
            shared: set[str] = set()
            if "th_files" in present:
                shared |= {r["file_key"] for r in await tx.execute(
                    f"SELECT DISTINCT file_key FROM th_files WHERE file_key = ANY(:keys) AND {_FILE_SUBJECT} <> :u",
                    {**params, "keys": keys}) or []}
            if "th_genotype_set" in present:
                shared |= {r["file_key"] for r in await tx.execute(
                    "SELECT DISTINCT file_key FROM th_genotype_set WHERE file_key = ANY(:keys) AND user_id <> :u",
                    {**params, "keys": keys}) or []}
            keys = [k for k in keys if k not in shared]
            if keys:
                await tx.execute(
                    "INSERT INTO th_storage_purge (file_key) SELECT unnest(CAST(:keys AS text[])) ON CONFLICT DO NOTHING",
                    {"keys": keys})
        else:
            if keys:
                logger.warning("erase: th_storage_purge is missing; %d stored objects cannot be queued for deletion", len(keys))
            keys = []

        threads: list[tuple[str, str]] = []
        sessions: list[str] = []
        if "th_sessions" in present:
            rows = await tx.execute(
                "SELECT session_id, user_id FROM th_sessions WHERE user_id = :u OR query_user_id = :u", params) or []
            sessions = [r["session_id"] for r in rows]
            threads = [(str(r["user_id"] or uid), str(r["session_id"])) for r in rows]
        if "th_messages" in present:
            more = await tx.execute(
                "SELECT DISTINCT session_id, user_id FROM th_messages WHERE user_id = :u OR query_user_id = :u", params) or []
            sessions = sorted(set(sessions) | {r["session_id"] for r in more})
            threads += [(str(r["user_id"] or uid), str(r["session_id"])) for r in more]
        params["sessions"] = sessions

        for step in PLAN:
            if all(t in present for t in step.tables):
                counts[step.group] = counts.get(step.group, 0) + await _count(tx, step.sql, params)
        await tx.execute(CLOSE_ACCOUNT, {"ui": ui})
    return counts, keys, sorted(set(threads))


async def _purge_objects(keys: list[str]) -> int:
    """Delete the objects, forgetting each key only once it is gone. Returns how
    many are still pending (this call's and earlier ones')."""
    from mirobody.collect.files.services.file_processing_service import delete_file_from_storage
    from mirobody.utils import db

    try:
        left = await db.execute_query(
            "SELECT file_key FROM th_storage_purge ORDER BY created_at LIMIT :n", {"n": RETRY_BATCH + len(keys)}, log_sql=False) or []
        for key in dict.fromkeys([*keys, *[r["file_key"] for r in left]]):
            if await delete_file_from_storage(key):
                await db.execute_query("DELETE FROM th_storage_purge WHERE file_key = :k", {"k": key}, log_sql=False)
            else:
                await db.execute_query("UPDATE th_storage_purge SET attempts = attempts + 1 WHERE file_key = :k", {"k": key}, log_sql=False)
        remaining = await db.execute_query("SELECT count(*) AS n FROM th_storage_purge", log_sql=False)
        return int((remaining[0] if remaining else {}).get("n") or 0)
    except Exception as e:
        logger.warning("erase: stored objects could not all be deleted: error_type=%s", type(e).__name__,
                       exc_info=not is_driver_exception(e))
        return -1


async def _purge_threads(user_id: str, threads: list[tuple[str, str]]) -> int:
    """The agent's own copy of each conversation, by thread id and then by the
    person's prefix (a thread whose session row is already gone). Returns how
    many threads could not be deleted."""
    from mirobody.agent.checkpointer import delete_thread, thread_for

    failed = 0
    for owner, session in threads:
        try:
            await delete_thread(thread_for(owner, session))
        except Exception as e:
            failed += 1
            logger.warning("erase: a conversation copy could not be deleted: error_type=%s", type(e).__name__)
    try:
        from mirobody.utils import db

        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await db.execute_query(
                f"DELETE FROM {table} WHERE thread_id LIKE :p", {"p": f"{user_id}:%"}, log_sql=False)
    except Exception as e:
        # The saver's tables are made at run time; without them there is nothing to sweep.
        logger.info("erase: no checkpoint tables swept: error_type=%s", type(e).__name__)
    return failed


async def erase_user_data(user_id: str) -> dict[str, Any]:
    """Erase everything about `user_id` and close the account.

    Raises `AccountManagesOthers` or `AccountNotFound`. On success returns
    `{"erased": {group: rows}, "objects_pending": n, "threads_failed": n}`;
    `objects_pending` is -1 when the object store could not be reached (the keys
    are queued and the next erasure retries them)."""
    from mirobody.user import user as account

    counts, keys, threads = await _erase_rows(user_id)
    uid = int(user_id)
    account._active.pop(uid, None)
    account._closed.add(uid)
    pending = await _purge_objects(keys)
    threads_failed = await _purge_threads(str(user_id), threads)
    logger.info("erase: account closed: groups=%s objects_pending=%s threads_failed=%s",
                sorted(counts), pending, threads_failed)
    return {"erased": counts, "objects_pending": pending, "threads_failed": threads_failed}
