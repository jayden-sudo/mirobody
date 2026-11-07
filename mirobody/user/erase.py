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
    # (chat sessions and messages, files and genotype sets are deleted in `_erase_rows` with RETURNING: it needs the
    # session ids and object keys of exactly the rows it deleted, not of rows seen by an earlier SELECT.)
    Step("records", ("th_check_result",), "DELETE FROM th_check_result WHERE user_id = :u"),
    Step("records", ("th_day_authority",), "DELETE FROM th_day_authority WHERE user_id = :u"),
    Step("records", ("th_observation",), "DELETE FROM th_observation WHERE user_id = :u"),
    # An extraction row is shared by whoever ingested the same (source_ref, extractor, payload): the unique key has no
    # user in it, so another person's observations can point at this person's row. Those rows stay; they hold what the
    # other person also ingested.
    Step(
        "records",
        ("th_extraction", "th_observation"),
        "DELETE FROM th_extraction e WHERE e.user_id = :u AND NOT EXISTS (SELECT 1 FROM th_observation o WHERE o.extraction_id = e.id)",
    ),
    Step("records", ("th_coding_alias",), "DELETE FROM th_coding_alias WHERE scope = 'user:' || :u"),
    Step("records", ("series_data",), "DELETE FROM series_data WHERE user_id = :u"),
    Step("records", ("th_series_data_retired_15",), "DELETE FROM th_series_data_retired_15 WHERE user_id = :u"),
    Step("genetics", ("th_series_data_genetic",), "DELETE FROM th_series_data_genetic WHERE user_id = :u"),
    Step("medications", ("th_medication_plan",), "DELETE FROM th_medication_plan WHERE user_id = :u"),
    Step("medications", ("th_medication_course",), "DELETE FROM th_medication_course WHERE user_id = :u"),
    Step("medications", ("th_dose_event",), "DELETE FROM th_dose_event WHERE user_id = :u"),
    Step("medications", ("th_override",), "DELETE FROM th_override WHERE user_id = :u"),
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
    | dict.fromkeys(
        (
            "th_files", "th_genotype_set", "th_sessions", "th_messages", "th_session_share", "th_chat_citation",
            "th_chat_citation_seq", "th_storage_purge",
        )
    )
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
    """The one transaction: lock and check the account, delete every row, queue the storage keys, close the
    account. Returns counts, the object keys to delete after the commit and the checkpoint threads'
    (owner, session) pairs. Keys and session ids come from DELETE ... RETURNING, so they are exactly the rows this
    transaction removed (a row that appeared after an earlier SELECT cannot slip past)."""
    from mirobody.utils import db

    uid, ui = str(user_id), int(user_id)
    counts: dict[str, int] = {}

    def add(group: str, n: int) -> None:
        counts[group] = counts.get(group, 0) + n

    async with db.transaction() as tx:
        live = await tx.execute("SELECT id FROM health_app_user WHERE id = :ui AND is_del = false FOR UPDATE", {"ui": ui})
        if not live:
            raise AccountNotFound
        if await tx.execute("SELECT 1 FROM health_app_user WHERE managed_by = :ui AND is_del = false LIMIT 1", {"ui": ui}):
            raise AccountManagesOthers
        present = await _present(tx)
        params: dict[str, Any] = {"u": uid, "ui": ui}

        # Chat first: sessions the person asked or that were about them, then the messages of those and any other
        # message about them, then what hangs off the session ids (citations, shares).
        threads: set[tuple[str, str]] = set()
        sessions: set[str] = set()
        if "th_sessions" in present:
            rows = await tx.execute(
                "DELETE FROM th_sessions WHERE user_id = :u OR query_user_id = :u RETURNING session_id, user_id", params) or []
            add("chat", len(rows))
            sessions |= {str(r["session_id"]) for r in rows}
            threads |= {(str(r["user_id"] or uid), str(r["session_id"])) for r in rows}
        if "th_messages" in present:
            rows = await tx.execute(
                "DELETE FROM th_messages WHERE user_id = :u OR query_user_id = :u OR session_id = ANY(:sessions)"
                " RETURNING session_id, user_id", {**params, "sessions": sorted(sessions)}) or []
            add("chat", len(rows))
            sessions |= {str(r["session_id"]) for r in rows}
            threads |= {(str(r["user_id"] or uid), str(r["session_id"])) for r in rows}
        by_session = {**params, "sessions": sorted(sessions)}
        for table, where in (
            ("th_chat_citation", "session_id = ANY(:sessions)"),
            ("th_chat_citation_seq", "session_id = ANY(:sessions)"),
            ("th_session_share", "user_id = :u OR session_id = ANY(:sessions)"),
        ):
            if table in present:
                add("chat", await _count(tx, f"DELETE FROM {table} WHERE {where}", by_session))

        # Files and genotype sets: the object keys of exactly the rows deleted.
        keys: set[str] = set()
        if "th_files" in present:
            rows = await tx.execute(f"DELETE FROM th_files WHERE {_FILE_SUBJECT} = :u RETURNING file_key", params) or []
            add("files", len(rows))
            keys |= {str(r["file_key"]) for r in rows if r.get("file_key")}
        if "th_genotype_set" in present:
            rows = await tx.execute("DELETE FROM th_genotype_set WHERE user_id = :u RETURNING file_key", params) or []
            add("genetics", len(rows))
            keys |= {str(r["file_key"]) for r in rows if r.get("file_key")}
        # An object another person's row still names stays (keys carry no owner; an upload can be shared).
        still_used: set[str] = set()
        if keys:
            for table in ("th_files", "th_genotype_set"):
                if table in present:
                    got = await tx.execute(f"SELECT DISTINCT file_key FROM {table} WHERE file_key = ANY(:keys)", {"keys": sorted(keys)}) or []
                    still_used |= {str(r["file_key"]) for r in got}
        deletable = sorted(keys - still_used)
        if deletable and "th_storage_purge" in present:
            await tx.execute(
                "INSERT INTO th_storage_purge (file_key) SELECT unnest(CAST(:keys AS text[])) ON CONFLICT DO NOTHING",
                {"keys": deletable})
        elif deletable:
            # No queue table in this deployment: the keys are still deleted right after the commit, just not retried.
            logger.warning("erase: th_storage_purge is missing; %d stored objects are deleted without a retry queue", len(deletable))

        for step in PLAN:
            if all(t in present for t in step.tables):
                add(step.group, await _count(tx, step.sql, params))
        await tx.execute(CLOSE_ACCOUNT, {"ui": ui})
    return counts, deletable, sorted(threads)


async def _still_named(keys: list[str]) -> set[str]:
    """The keys among `keys` that a file row or a genotype set (anyone's) names now. A table this deployment lacks
    names nothing; any OTHER failure (a timeout, a dropped connection) raises, so the caller deletes nothing this
    run instead of treating "could not check" as "not shared"."""
    from mirobody.utils import db

    named: set[str] = set()
    for table in ("th_files", "th_genotype_set"):
        exists = await db.execute_query("SELECT to_regclass(:t) AS t", {"t": table}, log_sql=False)
        if not (exists and exists[0].get("t")):
            continue
        got = await db.execute_query(f"SELECT DISTINCT file_key FROM {table} WHERE file_key = ANY(:keys)", {"keys": keys}, log_sql=False) or []
        named |= {str(r["file_key"]) for r in got}
    return named


async def _purge_objects(keys: list[str], queued: bool = True) -> int:
    """Delete the objects, forgetting each queued key only once it is gone. Returns how many are still pending
    (this call's and earlier ones'), -1 when the queue could not be read at all."""
    from mirobody.collect.files.services.file_processing_service import delete_file_from_storage
    from mirobody.utils import db

    pending = -1
    try:
        left: list[dict] = []
        try:
            # Oldest-failed-least first, so a handful of keys that can never be deleted cannot starve the rest.
            left = await db.execute_query(
                "SELECT file_key FROM th_storage_purge ORDER BY attempts, created_at LIMIT :n", {"n": RETRY_BATCH}, log_sql=False) or []
        except Exception as e:                                   # no queue table in this deployment
            queued = False
            logger.info("erase: no purge queue to retry from: error_type=%s", type(e).__name__)
        failed = 0
        # 这一轮要删的（刚排进队的）和以前留下的重试：排队之后有别人的行用上了这个对象的，都不删对象，只从队列里拿掉。
        # 查不出来（_still_named 抛）就整轮什么都不删——「没查成」不能当成「没人用」；它们还留在队列里，下次再来
        retried = [r["file_key"] for r in left]
        candidates = list(dict.fromkeys([*keys, *retried]))
        taken = await _still_named(candidates) if candidates else set()
        if taken and queued:
            for key in taken:
                await db.execute_query("DELETE FROM th_storage_purge WHERE file_key = :k", {"k": key}, log_sql=False)
        for key in (k for k in candidates if k not in taken):
            if await delete_file_from_storage(key):
                if queued:
                    await db.execute_query("DELETE FROM th_storage_purge WHERE file_key = :k", {"k": key}, log_sql=False)
            else:
                failed += 1
                if queued:
                    await db.execute_query("UPDATE th_storage_purge SET attempts = attempts + 1 WHERE file_key = :k", {"k": key}, log_sql=False)
        if queued:
            remaining = await db.execute_query("SELECT count(*) AS n FROM th_storage_purge", log_sql=False)
            pending = int((remaining[0] if remaining else {}).get("n") or 0)
        else:
            pending = failed
    except Exception as e:
        logger.warning("erase: stored objects could not all be deleted: error_type=%s", type(e).__name__,
                       exc_info=not is_driver_exception(e))
    return pending


async def _purge_threads(user_id: str, threads: list[tuple[str, str]]) -> int:
    """The agent's own copy of each conversation. The checkpoint rows are deleted by exact thread id, straight from
    the tables (that works with no saver loaded, and it is not swallowed the way `delete_thread` swallows), plus
    every thread this person owns whose session row is already gone. The saver is asked too, best effort. Returns
    how many threads could not be deleted."""
    from mirobody.agent.checkpointer import delete_thread, thread_for
    from mirobody.utils import db

    ids = sorted({thread_for(owner, session) for owner, session in threads})
    failed = 0
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        try:
            exists = await db.execute_query("SELECT to_regclass(:t) AS t", {"t": table}, log_sql=False)
            if not (exists and exists[0].get("t")):
                continue                                     # the saver's tables are made at run time; none, nothing to delete
            if ids:
                await db.execute_query(f"DELETE FROM {table} WHERE thread_id = ANY(:ids)", {"ids": ids}, log_sql=False)
            await db.execute_query(f"DELETE FROM {table} WHERE thread_id LIKE :p", {"p": f"{user_id}:%"}, log_sql=False)
        except Exception as e:
            failed = max(failed, len(ids) or 1)
            logger.warning("erase: conversation copies could not be deleted from %s: error_type=%s", table, type(e).__name__)
    for thread in ids:
        try:
            await delete_thread(thread)
        except Exception as e:                               # delete_thread logs and swallows its own; this is the import/setup failing
            logger.info("erase: the checkpoint saver was not asked to delete a thread: error_type=%s", type(e).__name__)
    return failed


async def erase_user_data(user_id: str) -> dict[str, Any]:
    """Erase everything about `user_id` and close the account.

    Raises `AccountManagesOthers` or `AccountNotFound`, both before anything is changed. Once the transaction has
    committed, nothing else raises: the rest (objects, conversation copies) is reported in the result, so a failure
    there is never presented as "nothing was changed". Returns
    `{"erased": {group: rows}, "objects_pending": n, "threads_failed": n}`; `objects_pending` is -1 when the object
    store or its queue could not be reached (queued keys are retried by the next erasure)."""
    from mirobody.user import user as account

    counts, keys, threads = await _erase_rows(user_id)
    uid = int(user_id)
    account._active.pop(uid, None)
    account._closed.add(uid)
    pending, threads_failed = -1, -1
    try:
        pending = await _purge_objects(keys)
    except Exception as e:
        logger.warning("erase: object deletion step failed: error_type=%s", type(e).__name__)
    try:
        threads_failed = await _purge_threads(str(user_id), threads)
    except Exception as e:
        logger.warning("erase: conversation copy step failed: error_type=%s", type(e).__name__)
    logger.info("erase: account closed: groups=%s objects_pending=%s threads_failed=%s", sorted(counts), pending, threads_failed)
    return {"erased": counts, "objects_pending": pending, "threads_failed": threads_failed}
