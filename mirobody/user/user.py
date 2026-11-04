"""User records, and the rule for which database layer to use.

**Two layers, and the boundary is atomicity.** `utils/db.execute_query` runs ONE
statement inside its own `engine.begin()`, so anything that must commit or roll
back together cannot use it: two calls are two transactions. That is the whole
rule, and in this package it applies to `del_user` (two tables marked deleted
together) and `account_merge` (an explicit `conn.transaction()` over the whole
merge). Those take an injected
`AsyncConnectionPool` and hand-roll `cur.execute(..., %s)`.

Everything else (every single-statement read, wherever it lives) goes through
`execute_query` with `:named` params. Choose by need, never by file.

**And one query, not twenty.** A hand-rolled `SELECT ... FROM health_app_user
WHERE ... AND is_del = false` at every call site is a per-call-site chance to
drop the `is_del` filter, and a lookup without it answers for deleted accounts
name, language, timezone and all. :func:`get_user` is the single lookup;
`test_user_lookup.py` fails if a new hand-rolled one appears.
"""

from __future__ import annotations

import logging
import secrets

from datetime import date, datetime
from typing import TYPE_CHECKING, NamedTuple

# Type-checking only: `AsyncConnectionPool` appears in two parameter
# annotations, and psycopg_pool lives in the [app] extra. A module-scope
# import here made `import mirobody.user.care_circle` (the pure authorization
# rules examples/06 demonstrates) require the server extra.
if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

from mirobody.kernel.ops import is_driver_exception
from mirobody.utils.db import execute_query

logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------

# Every column a caller has asked for across those 20 sites, so the one accessor
# can serve all of them. The table is 15 narrow columns; naming them beats `*`
# because a dropped column then fails here instead of at the first KeyError.
_USER_COLUMNS = ("id, email, name, lang, tz, gender, birth, blood, mfa_enabled, managed_by,"
                 " (password_hash IS NOT NULL) AS has_password,"
                 " (password_hash IS NOT NULL AND email_verified_at IS NULL) AS unproven,"
                 " EXTRACT(EPOCH FROM tokens_valid_after) AS tokens_valid_after")


async def get_user(
    *,
    user_id     : int | str | None = None,
    email       : str | None = None,
) -> dict | None:
    """The one `health_app_user` lookup. Returns the live row, or None.

    Exactly one selector. `is_del = false` is not optional and not a parameter:
    a soft-deleted account must not be findable by either key, which is the
    bug this function exists to make unrepeatable.
    """
    keys = {"id": user_id, "email": email}
    given = {k: v for k, v in keys.items() if v not in (None, "")}
    if len(given) != 1:
        raise ValueError(f"get_user needs exactly one of {list(keys)}, got {list(given)}")

    column, value = next(iter(given.items()))
    if column == "id":
        value = int(value)
    elif column == "email":
        value = str(value).strip().lower()

    rows = await execute_query(
        f"SELECT {_USER_COLUMNS} FROM health_app_user "
        f"WHERE {column} = :value AND is_del = false LIMIT 1;",
        params={"value": value},
    )
    return rows[0] if rows else None

#-----------------------------------------------------------------------------

async def ensure_user(email: str) -> int | None:
    """The live account for an address, created on first sight; None for a
    malformed address, and for a managed account's address: nobody signs in
    to those or is invited as them.

    One statement, so two sign-ins racing on a new address both get the same
    row. This replaced two copies: `add_or_get_user` (sign-in), which read then
    inserted and returned the second racer an IntegrityError as its message,
    and `care_circle.resolve_email_to_user` (invitations), which was this.
    """
    clean = (email or "").strip().lower()
    if not clean or "@" not in clean:
        return None
    rows = await execute_query(
        """
        WITH ins AS (
            INSERT INTO health_app_user (is_del, email, name)
            VALUES (false, :email, :name)
            ON CONFLICT (email) WHERE (is_del = false) DO NOTHING
            RETURNING id
        )
        SELECT id FROM ins
        UNION ALL
        SELECT id FROM health_app_user WHERE email = :email AND is_del = false AND managed_by IS NULL
        LIMIT 1
        """,
        {"email": clean, "name": clean.split("@")[0]},
    )
    return int(rows[0]["id"]) if rows else None

#-----------------------------------------------------------------------------

#: The longest display name kept from an identity provider; the column has no
#: limit, but a name is shown in a header, not stored as a document.
EXTERNAL_NAME_MAX = 64


class SibianAccount(NamedTuple):
    user_id: int
    is_new: bool
    email: str


def _sibian_address() -> str:
    """A new sibian account's address. `health_app_user.email` is NOT NULL, so
    the account needs one, and it is RANDOM on purpose: an address derived
    from the identity number (which is public) could be registered first by
    anyone through a password or invitation route, and that account would
    then be the one the person's first sign-in landed in. `.invalid`
    (RFC 2606) is never delivered, so no code sign-in can claim it either."""
    return secrets.token_hex(16) + "@sibian.invalid"


async def ensure_sibian_user(subject: str, name: str, iss: str) -> SibianAccount | None:
    """The live account for a sibian identity, opened on first sight.

    The identity is found only through `external_identity`; no existing
    account row is ever adopted (see `_sibian_address`). Opening one is a
    single statement that creates the account and claims the identity for it
    together, and the claim is a compare-and-set against what was read: it
    takes the row only while it is still absent, or still points at the
    deleted account seen. A concurrent first sign-in that loses retires the
    account it just made and returns the winner's. After a deletion the
    identity gets a new account; the deleted one never comes back.
    """
    clean = " ".join((name or "").split())[:EXTERNAL_NAME_MAX]
    for _attempt in range(2):
        rows = await execute_query(
            "SELECT user_id FROM external_identity WHERE provider = 'sibian' AND subject = :subject",
            {"subject": subject},
        )
        seen = int(rows[0]["user_id"]) if rows else None
        live = await get_user(user_id=seen) if seen else None
        if live:
            await execute_query(
                "UPDATE external_identity SET last_seen = now() WHERE provider = 'sibian' AND subject = :subject",
                {"subject": subject},
            )
            # The name is the person's own, kept current from the node that
            # vouched for them; an empty one keeps what is there.
            if clean and live.get("name") != clean:
                await execute_query(
                    "UPDATE health_app_user SET name = :name, update_at = CURRENT_TIMESTAMP"
                    " WHERE id = :id AND is_del = false",
                    {"name": clean, "id": seen},
                )
            return SibianAccount(seen, False, str(live["email"]))

        email = _sibian_address()
        rows = await execute_query(
            """
            WITH account AS (
                INSERT INTO health_app_user (is_del, email, name)
                VALUES (false, :email, :name)
                RETURNING id
            ), claimed AS (
                INSERT INTO external_identity (provider, subject, user_id, first_iss)
                SELECT 'sibian', :subject, id, :iss FROM account
                ON CONFLICT (provider, subject) DO UPDATE
                    SET user_id = EXCLUDED.user_id, last_seen = now()
                    WHERE external_identity.user_id = CAST(:seen AS integer)
                RETURNING user_id
            )
            SELECT (SELECT id FROM account) AS created, (SELECT user_id FROM claimed) AS claimed
            """,
            {"email": email, "name": clean, "subject": subject, "iss": iss, "seen": seen},
        )
        created, claimed = rows[0]["created"], rows[0]["claimed"]
        if claimed is not None:
            return SibianAccount(int(claimed), True, email)
        # Another sign-in claimed the identity between the read and the write.
        await execute_query(
            "UPDATE health_app_user SET is_del = TRUE, update_at = CURRENT_TIMESTAMP WHERE id = :id",
            {"id": created},
        )
    return None

#-----------------------------------------------------------------------------

async def update_user_name(
    db_pool : AsyncConnectionPool,
    user_id : int,
    name    : str,
) -> str | None:
    """Update health_app_user.name for the given user. Returns a sentence for
    the caller on failure, None on success. Caller is responsible for
    trimming / length validation."""
    if user_id <= 0:
        return "Invalid user ID."

    if not isinstance(name, str) or not name:
        return "Invalid name."

    if not db_pool:
        return "Invalid database connection."

    try:
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE health_app_user"
                    "   SET name=%s, update_at=CURRENT_TIMESTAMP"
                    " WHERE id=%s AND is_del=FALSE;",
                    [name, user_id]
                )
                await conn.commit()
    except Exception as e:
        logger.error("name update failed: user_id=%s error_type=%s", user_id, type(e).__name__,
                     exc_info=not is_driver_exception(e))
        return "The name could not be saved. Try again."

    return None

#-----------------------------------------------------------------------------

async def del_user(
    db_pool : AsyncConnectionPool,
    user_id : int
) -> str | None:
    """Close the account. Returns a sentence for the caller on failure, None
    on success."""
    if user_id <= 0:
        return "Invalid user ID."

    if not db_pool:
        return "Invalid database connection."

    # Both statements are awaited. They were not, so `/user/del` returned before
    # either ran and no account was ever deleted. `app_user_id` is a varchar: an
    # int there fails the statement and rolls the deletion back.
    try:
        async with db_pool.connection() as conn:
            await conn.execute(
                "UPDATE health_app_user SET is_del=TRUE WHERE id=%s;",
                [user_id]
            )

            await conn.execute(
                "UPDATE health_vital_user SET is_del=TRUE WHERE app_user_id=%s;",
                [str(user_id)]
            )

            await conn.commit()
    except Exception as e:
        logger.error("account deletion failed: user_id=%s error_type=%s", user_id, type(e).__name__,
                     exc_info=not is_driver_exception(e))
        return "The account could not be deleted. Try again."

    _active.pop(user_id, None)
    _closed.add(user_id)
    return None

#-----------------------------------------------------------------------------

#: How long a live account is taken on trust before it is looked up again. A
#: token outlives the account, or a revocation, by at most this long; a closed
#: account is final and is remembered for good.
_ACTIVE_TTL = 30.0
_active: dict[int, float] = {}
_closed: set[int] = set()
#: `tokens_valid_after` as epoch seconds, cached alongside `_active`.
_valid_after: dict[int, float] = {}


async def is_active_account(user_id: int | str, minted_at: float | None = None) -> bool:
    """Whether a token's subject is an account that still exists, and, given
    the token's `minted_at`, whether the account has not revoked it since. A
    JWT is valid for 30 days and carries nothing a deletion can revoke, so
    every token check asks this. A lookup that fails answers True: an outage
    must not sign everyone out."""
    import time

    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return False
    if uid <= 0 or uid in _closed:
        return False
    now = time.monotonic()
    if _active.get(uid, 0.0) <= now:
        try:
            row = await get_user(user_id=uid)
        except Exception as e:
            logger.warning("account lookup failed; token accepted: error_type=%s", type(e).__name__)
            return True
        if row is None:
            _closed.add(uid)
            return False
        _active[uid] = now + _ACTIVE_TTL
        _valid_after[uid] = float(row.get("tokens_valid_after") or 0)
    return minted_at is None or minted_at >= _valid_after.get(uid, 0.0)


#: Whether this deployment can prove an address, which is whether it sends
#: mail. Without mail an address is only a username. Set by UserService.
_addresses_provable = False


def addresses_provable() -> bool:
    return _addresses_provable


def set_addresses_provable(value: bool) -> None:
    global _addresses_provable
    _addresses_provable = bool(value)


async def prove_address(user_id: int) -> bool:
    """Record that a code sent to this account's address came back.

    If the account had a password nobody had proven (it was registered with
    this address by whoever typed it first), the password is cleared and every
    token minted before now stops working. Returns whether that happened.
    The cut-off is this process's clock, the one that stamps `iat`.
    """
    import time

    cutoff = int(time.time())
    rows = await execute_query(
        """
        UPDATE health_app_user
           SET password_hash      = CASE WHEN email_verified_at IS NULL THEN NULL ELSE password_hash END,
               tokens_valid_after = CASE WHEN email_verified_at IS NULL AND password_hash IS NOT NULL
                                         THEN to_timestamp(:cutoff) ELSE tokens_valid_after END,
               email_verified_at  = now()
         WHERE id = :id AND is_del = false
        RETURNING EXTRACT(EPOCH FROM tokens_valid_after) AS tokens_valid_after
        """,
        {"id": int(user_id), "cutoff": cutoff},
    )
    if not rows:
        return False
    after = float(rows[0]["tokens_valid_after"] or 0)
    _valid_after[int(user_id)] = after
    return after == cutoff


#: The shapes `health_app_user.birth` arrives in. The column is free text
#: (`character varying`), written by every client that ever set it.
_BIRTH_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d")


def age_from_birth(birth: date | str | None) -> int | None:
    """Whole years since `birth` (`health_app_user.birth`), or None when it
    is not a date. A time after the date is ignored."""
    if isinstance(birth, datetime):
        birth = birth.date()
    if not isinstance(birth, date):
        text = str(birth or "").strip()[:10]
        for fmt in _BIRTH_FORMATS:
            try:
                birth = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
        else:
            return None
    today = date.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

#-----------------------------------------------------------------------------

class UserInfo:
    def __init__(
        self,
        name: str,
        language: str,
        timezone: str
    ):
        self.name       = name
        self.language   = language
        self.timezone   = timezone


async def get_user_info(
    user_id : int
) -> tuple[UserInfo | None, str | None]:
    if user_id <= 0:
        return None, "Invalid user ID."

    try:
        row = await get_user(user_id=user_id)
        if row:
            return UserInfo(row["name"], row["lang"], row["tz"]), None

    except Exception as e:
        logger.error("account lookup failed: user_id=%s error_type=%s", user_id, type(e).__name__,
                     exc_info=not is_driver_exception(e))
        return None, "Lookup failed."

    return None, "Not found."

#-----------------------------------------------------------------------------
