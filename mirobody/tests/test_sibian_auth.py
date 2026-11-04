"""Signing in with a sibian-os identity: the assertion a sibian node issues,
and what this server makes of it.

A sibian node verifies its user's own Ed25519 signature and answers with a
short EdDSA JWT (`aud` = this server, `sub` = the identity number). These pin
which of those this server accepts. Keys are generated per test; no network.
"""
import time

import pytest

jwt = pytest.importorskip("jwt")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mirobody.user.auth.sibian import SibianAssertionRejected, SibianAssertionVerifier, issuers_from_config

ISS = "https://sibian-os.exe.xyz"
JWKS_URL = ISS + "/.well-known/sibian-health/jwks.json"
SUBJECT = "b3f1c0ffee5e1d00b3f1c0ffee5e1d00"


class Node:
    """One sibian node: a signing key and the JWKS it publishes."""

    def __init__(self, iss: str = ISS, kid: str = "k1"):
        self.iss, self.kid = iss, kid
        self.key = Ed25519PrivateKey.generate()

    def jwks(self) -> dict:
        jwk = jwt.algorithms.OKPAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        return {"keys": [{**jwk, "kid": self.kid, "alg": "EdDSA", "use": "sig"}]}

    def assertion(self, **over) -> str:
        now = int(time.time())
        claims = {"iss": self.iss, "aud": "mirobody", "sub": SUBJECT, "name": "Ada",
                  "iat": now, "exp": now + 300, "jti": "ask-" + str(now)}
        claims.update(over)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, self.key, algorithm="EdDSA", headers={"kid": self.kid})


def verifier(*nodes: Node, fetched: list | None = None) -> SibianAssertionVerifier:
    by_url = {n.iss + "/.well-known/sibian-health/jwks.json": n for n in nodes}

    async def fetch(url: str) -> dict:
        if fetched is not None:
            fetched.append(url)
        return by_url[url].jwks()

    return SibianAssertionVerifier(
        issuers={n.iss: n.iss + "/.well-known/sibian-health/jwks.json" for n in nodes},
        audience="mirobody",
        fetch_jwks=fetch,
    )


async def test_an_assertion_from_a_trusted_node_names_the_identity():
    node = Node()
    claims = await verifier(node).verify(node.assertion())
    assert claims.subject == SUBJECT
    assert claims.name == "Ada"
    assert claims.issuer == ISS


def _forged_hs256(node: Node) -> str:
    # The classic confusion: sign HS256 with the PUBLIC key's bytes as the
    # secret, hoping the verifier lets the token pick the algorithm.
    now = int(time.time())
    claims = {"iss": node.iss, "aud": "mirobody", "sub": SUBJECT, "iat": now, "exp": now + 300, "jti": "x"}
    secret = node.jwks()["keys"][0]["x"].encode()
    return jwt.encode(claims, secret, algorithm="HS256", headers={"kid": node.kid})


@pytest.mark.parametrize("make", [
    pytest.param(lambda n: Node(iss="https://evil.example").assertion(), id="untrusted-issuer"),
    pytest.param(lambda n: n.assertion(aud="someone-else"), id="wrong-audience"),
    pytest.param(lambda n: n.assertion(iat=int(time.time()) - 900, exp=int(time.time()) - 600), id="expired"),
    pytest.param(lambda n: n.assertion(exp=int(time.time()) + 3600), id="lives-too-long"),
    pytest.param(lambda n: Node(kid=n.kid).assertion(), id="another-key-same-kid"),
    pytest.param(_forged_hs256, id="hs256-with-public-key"),
    pytest.param(lambda n: n.assertion(sub=None), id="no-subject"),
    pytest.param(lambda n: n.assertion(jti=None), id="no-jti"),
    pytest.param(lambda n: "not-a-jwt", id="garbage"),
])
async def test_an_assertion_that_speaks_for_no_one_is_refused(make):
    node = Node()
    with pytest.raises(SibianAssertionRejected):
        await verifier(node).verify(make(node))


async def test_a_node_s_keys_are_fetched_once_for_many_sign_ins():
    node, fetched = Node(), []
    v = verifier(node, fetched=fetched)
    for _ in range(3):
        await v.verify(node.assertion())
    assert fetched == [JWKS_URL]


async def test_a_node_that_rotated_its_key_is_believed_after_one_refetch():
    node, fetched = Node(), []
    v = verifier(node, fetched=fetched)
    await v.verify(node.assertion())
    node.key, node.kid = Ed25519PrivateKey.generate(), "k2"
    claims = await v.verify(node.assertion())
    assert claims.subject == SUBJECT
    assert fetched == [JWKS_URL, JWKS_URL]


async def test_made_up_key_ids_do_not_make_this_server_hammer_the_node():
    node, fetched = Node(), []
    v = verifier(node, fetched=fetched)
    await v.verify(node.assertion())
    for kid in ("nope-1", "nope-2", "nope-3"):
        stranger = Node(kid=kid)
        with pytest.raises(SibianAssertionRejected):
            await v.verify(stranger.assertion())
    assert len(fetched) == 2


# -- the exchange, against PostgreSQL ------------------------------------------
#
# Set MIROBODY_TEST_POSTGRES_DSN to a disposable database (docs/testing.md).
# Each test builds the accounts and ephemeral tables in a schema of its own.

import os
from pathlib import Path
from uuid import uuid4

DSN = os.environ.get("MIROBODY_TEST_POSTGRES_DSN", "")


def db_test(fn):
    return pytest.mark.needs_db(pytest.mark.skipif(not DSN, reason="MIROBODY_TEST_POSTGRES_DSN is not set")(fn))


JWT_KEY = "test-only-" + "k" * 40


@pytest.fixture
async def accounts(monkeypatch):
    """A schema holding the accounts and ephemeral tables; `db.execute_query`
    and an `EphemeralStore` both pointed at it. Yields that store."""
    import psycopg
    from cryptography.fernet import Fernet
    from psycopg import sql
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine
    from mirobody.utils import db
    from mirobody.utils.ephemeral import EphemeralStore

    schema = "sibian_" + uuid4().hex
    conninfo = make_url(DSN).set(drivername="postgresql").render_as_string(hide_password=False)
    admin = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    engine = None
    try:
        # pgcrypto for password sign-up (`crypt`, `gen_salt`), in `public`:
        # an extension lives once per database, so the schema reaches it there.
        await admin.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        await admin.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        await admin.execute("CREATE FUNCTION encrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT nullif($1,'')$$")
        await admin.execute("CREATE FUNCTION decrypt_content(text) RETURNS text LANGUAGE SQL IMMUTABLE AS $$SELECT $1$$")
        root = Path(__file__).resolve().parents[1] / "schema"
        for file in ("10_accounts.sql", "61_ephemeral.sql"):
            await admin.execute((root / file).read_text())
        engine = create_async_engine(make_url(DSN).set(drivername="postgresql+psycopg"),
                                     connect_args={"options": f"-c search_path={schema},public -c timezone=UTC"})
        monkeypatch.setattr(db, "_provider", lambda _: engine)
        monkeypatch.setattr(db, "_engines", {})

        class _Pg:
            async def get_async_client(self):
                return await psycopg.AsyncConnection.connect(conninfo, options=f"-c search_path={schema}")

        store = EphemeralStore(_Pg(), Fernet.generate_key().decode())
        store.test_conninfo, store.test_schema = conninfo, schema
        yield store
    finally:
        if engine is not None:
            await engine.dispose()
        await admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        await admin.close()


def exchange_app(ephemeral, *nodes: Node, auth_methods=("sibian",), **service):
    from starlette.applications import Starlette
    from mirobody.user.auth.jwt import JwtTokenValidator
    from mirobody.user.user_service import UserService

    service = UserService(
        token_validator=JwtTokenValidator(JWT_KEY),
        routes=[],
        ephemeral=ephemeral,
        auth_methods=auth_methods,
        sibian_verifier=verifier(*nodes) if nodes else None,
        **service,
    )
    return Starlette(routes=service.routes)


async def post_exchange(app, assertion: str) -> dict:
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        r = await client.post("/auth/sibian/exchange", json={"assertion": assertion})
    return {"status": r.status_code, **r.json()}


@db_test
async def test_the_first_exchange_opens_an_account_and_answers_with_a_one_hour_session(accounts):
    from mirobody.user.auth.bearer import bearer_subject
    from mirobody.user.auth.jwt import JwtTokenValidator

    node = Node()
    answer = await post_exchange(exchange_app(accounts, node), node.assertion())

    assert answer["code"] == 0, answer
    data = answer["data"]
    assert data["is_new_user"] is True
    assert data["expires_in"] == 3600
    assert "refresh_token" not in data
    payload, err = JwtTokenValidator(JWT_KEY).verify_token(data["access_token"])
    assert err is None
    assert payload["exp"] - payload["iat"] == 3600
    assert str(await bearer_subject(payload)) == str(data["user_id"])



@db_test
async def test_the_same_person_through_another_node_is_the_same_account(accounts):
    from mirobody.user.user import get_user

    first, second = Node(), Node(iss="https://localfirst-city.exe.xyz")
    app = exchange_app(accounts, first, second)
    a = (await post_exchange(app, first.assertion(name="Ada")))["data"]
    b = (await post_exchange(app, second.assertion(name="Ada Lovelace", jti="other")))["data"]

    assert b["user_id"] == a["user_id"]
    assert b["is_new_user"] is False
    assert (await get_user(user_id=a["user_id"]))["name"] == "Ada Lovelace"


@db_test
async def test_an_assertion_signs_in_once(accounts):
    node = Node()
    app = exchange_app(accounts, node)
    assertion = node.assertion()
    assert (await post_exchange(app, assertion))["code"] == 0
    again = await post_exchange(app, assertion)
    assert again["status"] == 401
    assert again["code"] != 0


@db_test
async def test_after_the_account_is_deleted_the_identity_gets_a_new_one(accounts):
    from mirobody.utils.db import execute_query

    node = Node()
    app = exchange_app(accounts, node)
    old = (await post_exchange(app, node.assertion(jti="before")))["data"]
    # What `/user/del` does to the account row (user.del_user).
    await execute_query("UPDATE health_app_user SET is_del = TRUE WHERE id = :id", {"id": int(old["user_id"])})

    new = (await post_exchange(app, node.assertion(jti="after")))["data"]
    assert new["user_id"] != old["user_id"]
    assert new["is_new_user"] is True


# -- which sign-in routes exist --------------------------------------------------

SIGN_IN_PATHS = ("/email/login", "/email/verify", "/email/bind", "/password/register", "/password/login",
                 "/account/activation/complete", "/auth/sibian/exchange")


async def _status_by_path(app) -> dict[str, int]:
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        # OPTIONS: every sign-in handler answers it before reading anything, so
        # the status says only whether the route is mounted.
        return {p: (await client.options(p)).status_code for p in SIGN_IN_PATHS}


async def test_a_sibian_only_deployment_mounts_no_address_or_password_sign_in():
    status = await _status_by_path(exchange_app(object(), Node(), auth_methods=("sibian",)))
    assert {p for p, s in status.items() if s != 404} == {"/auth/sibian/exchange"}


async def test_by_default_the_sign_in_routes_are_the_ones_there_were_before():
    status = await _status_by_path(exchange_app(None, auth_methods=None))
    assert status["/auth/sibian/exchange"] == 404
    assert all(s != 404 for p, s in status.items() if p != "/auth/sibian/exchange")


@pytest.mark.parametrize("methods, nodes, ephemeral", [
    pytest.param(("sms",), (), None, id="unknown-method"),
    pytest.param(("sibian",), (), object(), id="sibian-without-issuers"),
    pytest.param(("sibian",), (Node(),), None, id="sibian-without-replay-store"),
])
def test_a_sign_in_configuration_that_cannot_work_stops_the_boot(methods, nodes, ephemeral):
    with pytest.raises(ValueError):
        exchange_app(ephemeral, *nodes, auth_methods=methods)


# -- SIBIAN_ISSUERS ----------------------------------------------------------------


def test_a_node_listed_by_address_publishes_its_keys_at_the_well_known_path():
    assert issuers_from_config(["https://sibian-os.exe.xyz/", "https://localfirst-city.exe.xyz"]) == {
        "https://sibian-os.exe.xyz": "https://sibian-os.exe.xyz/.well-known/sibian-health/jwks.json",
        "https://localfirst-city.exe.xyz": "https://localfirst-city.exe.xyz/.well-known/sibian-health/jwks.json",
    }


def test_a_node_may_name_its_own_jwks_address():
    assert issuers_from_config({"https://pol.4seas.xyz": "https://keys.4seas.xyz/jwks.json"}) == {
        "https://pol.4seas.xyz": "https://keys.4seas.xyz/jwks.json",
    }


def test_no_nodes_configured_means_none_trusted():
    assert issuers_from_config(None) == {}
    assert issuers_from_config("") == {}


@pytest.mark.parametrize("value", [
    pytest.param(["http://sibian-os.exe.xyz"], id="plain-http-issuer"),
    pytest.param({"https://pol.4seas.xyz": "http://keys.4seas.xyz/jwks.json"}, id="plain-http-jwks"),
    pytest.param(["sibian-os.exe.xyz"], id="no-scheme"),
    pytest.param(42, id="not-a-list"),
])
def test_a_node_whose_keys_could_be_swapped_in_transit_is_refused(value):
    with pytest.raises(ValueError):
        issuers_from_config(value)


def test_a_node_on_this_machine_may_use_plain_http_for_development():
    assert issuers_from_config(["http://localhost:8000"]) == {
        "http://localhost:8000": "http://localhost:8000/.well-known/sibian-health/jwks.json",
    }


# -- reading a node's JWKS over HTTP ---------------------------------------------


@pytest.fixture
async def node_server():
    """A real HTTP server on localhost standing in for a sibian node."""
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    node, routes = Node(), {}

    async def handle(request):
        return await routes[request.path](request)

    app = web.Application()
    app.router.add_route("GET", "/{tail:.*}", handle)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    node.iss = str(server.make_url("")).rstrip("/")
    routes["/.well-known/sibian-health/jwks.json"] = lambda r: _json(node.jwks())
    try:
        yield node, routes, server
    finally:
        await server.close()


async def _json(body):
    from aiohttp import web
    return web.json_response(body)


async def test_a_node_s_published_keys_are_read_over_http(node_server):
    from mirobody.user.auth.sibian import fetch_jwks

    node, _, _ = node_server
    v = SibianAssertionVerifier(issuers_from_config([node.iss]), "mirobody", fetch_jwks)
    assert (await v.verify(node.assertion())).subject == SUBJECT


@pytest.mark.parametrize("answer", ["redirect", "huge", "not-json"])
async def test_a_node_answer_that_is_not_a_plain_jwks_is_not_trusted(node_server, answer):
    from aiohttp import web
    from mirobody.user.auth.sibian import fetch_jwks

    node, routes, server = node_server

    async def odd(request):
        if answer == "redirect":
            raise web.HTTPFound(str(server.make_url("/elsewhere")))
        if answer == "huge":
            return web.json_response({"keys": node.jwks()["keys"], "pad": "x" * 200_000})
        return web.Response(text="<html>", content_type="text/html")

    routes["/.well-known/sibian-health/jwks.json"] = odd
    routes["/elsewhere"] = lambda r: _json(node.jwks())
    v = SibianAssertionVerifier(issuers_from_config([node.iss]), "mirobody", fetch_jwks)
    with pytest.raises(SibianAssertionRejected):
        await v.verify(node.assertion())


# -- the settings, as a deployment writes them -----------------------------------


def _config(text: str):
    import io
    from mirobody.utils.config.config import Config
    return Config(io.StringIO(text))


def test_a_sibian_only_deployment_is_three_lines_of_config():
    options = _config(
        "AUTH_METHODS: [sibian]\n"
        "SIBIAN_ISSUERS: ['https://sibian-os.exe.xyz', 'https://localfirst-city.exe.xyz']\n"
    ).get_sign_in_options()
    assert options["auth_methods"] == ["sibian"]
    assert set(issuers_from_config(options["sibian_issuers"])) == {
        "https://sibian-os.exe.xyz", "https://localfirst-city.exe.xyz"}
    assert options["sibian_audience"] == "mirobody"
    assert options["sibian_session_seconds"] == 3600


def test_without_sign_in_settings_nothing_changes():
    options = _config("LOG_LEVEL: INFO\n").get_sign_in_options()
    assert options["auth_methods"] == ["email", "password"]
    assert issuers_from_config(options["sibian_issuers"]) == {}


@pytest.mark.parametrize("env", [
    "https://sibian-os.exe.xyz, https://localfirst-city.exe.xyz",
    '["https://sibian-os.exe.xyz", "https://localfirst-city.exe.xyz"]',
])
def test_the_nodes_can_be_set_in_the_environment(monkeypatch, env):
    monkeypatch.setenv("SIBIAN_ISSUERS", env)
    monkeypatch.setenv("AUTH_METHODS", "sibian")
    options = _config("").get_sign_in_options()
    assert options["auth_methods"] == ["sibian"]
    assert set(issuers_from_config(options["sibian_issuers"])) == {
        "https://sibian-os.exe.xyz", "https://localfirst-city.exe.xyz"}


# -- the browser's side: CORS and caching -----------------------------------------

CITIES = "https://sibian-os.exe.xyz, https://localfirst-city.exe.xyz"


def _app_with_middlewares(headers: list[tuple[str, str]], routes=None):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from mirobody.server.middleware_stack import build_middlewares

    async def ping(request):
        return JSONResponse({"code": 0})

    return Starlette(routes=routes or [Route("/api/ping", ping, methods=["GET", "POST"])],
                     middleware=build_middlewares(http_headers=headers, jwt_key=JWT_KEY))


async def _preflight(app, origin: str):
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        return await client.options("/api/ping", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type"})


@pytest.mark.parametrize("origin", ["https://sibian-os.exe.xyz", "https://localfirst-city.exe.xyz"])
async def test_every_listed_city_may_call_the_api_from_its_pages(origin):
    r = await _preflight(_app_with_middlewares([("Access-Control-Allow-Origin", CITIES)]), origin)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin


async def test_a_page_on_an_unlisted_origin_may_not():
    r = await _preflight(_app_with_middlewares([("Access-Control-Allow-Origin", CITIES)]), "https://evil.example")
    assert r.headers.get("access-control-allow-origin") is None


async def test_an_api_answer_tells_every_cache_on_the_way_not_to_keep_it():
    import httpx

    app = _app_with_middlewares([])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        r = await client.get("/api/ping")
    assert r.headers["cache-control"] == "private, no-store"


async def test_a_response_that_chose_its_own_caching_keeps_it():
    import httpx
    from starlette.responses import Response
    from starlette.routing import Route

    async def asset(request):
        return Response(b"x", headers={"Cache-Control": "public, max-age=31536000, immutable"})

    app = _app_with_middlewares([], routes=[Route("/assets/a.js", asset)])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        r = await client.get("/assets/a.js")
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"


async def test_a_node_briefly_unreachable_when_its_keys_expire_does_not_lock_everyone_out():
    node, now = Node(), [1000.0]
    down, attempts = [False], []

    async def fetch(url):
        attempts.append(now[0])
        if down[0]:
            raise ConnectionError("node is down")
        return node.jwks()

    v = SibianAssertionVerifier({ISS: JWKS_URL}, "mirobody", fetch, clock=lambda: now[0])
    await v.verify(node.assertion())
    down[0], now[0] = True, now[0] + 2 * 3600
    assert (await v.verify(node.assertion())).subject == SUBJECT
    # And while it stays down, sign-ins do not each wait on another attempt.
    for _ in range(5):
        await v.verify(node.assertion())
    assert len(attempts) == 2


async def test_the_web_client_keeps_its_own_caching_behind_the_no_store_default(tmp_path):
    """The server as `Server.start` assembles it: the middleware stack first,
    the web client mounted last (`add_htdoc_routes` adds its cache headers as
    the OUTERMOST layer)."""
    import httpx
    from fastapi import FastAPI
    from mirobody.server.htdoc import add_htdoc_routes
    from mirobody.server.middleware_stack import build_middlewares

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>")
    (tmp_path / "assets" / "app-1a2b3c.js").write_text("console.log(1)")

    app = FastAPI(middleware=build_middlewares(http_headers=[], jwt_key=JWT_KEY))

    @app.get("/api/ping")
    async def ping():
        return {"code": 0}

    add_htdoc_routes(app, str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        asset = await client.get("/assets/app-1a2b3c.js")
        page = await client.get("/")
        api = await client.get("/api/ping")
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert page.headers["cache-control"] == "no-cache"
    assert api.headers["cache-control"] == "private, no-store"


# -- review findings (2026-11-04) -------------------------------------------------


@db_test
async def test_an_account_registered_in_advance_never_captures_a_sibian_identity(accounts):
    """The attack: on a deployment that also allows password sign-up without
    mail, register the address a sibian identity's account WOULD get (it was
    derivable from the public identity number), then wait for the victim."""
    import hashlib
    import httpx

    node = Node()
    app = exchange_app(accounts, node, auth_methods=("password", "sibian"))
    guessed = hashlib.sha256(SUBJECT.encode()).hexdigest()[:32] + "@sibian.invalid"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        r = await client.post("/password/register", json={"email": guessed, "password": "attacker-pw-123"})
        attacker = r.json()["data"]["access_token"]
    from mirobody.user.auth.jwt import JwtTokenValidator
    attacker_id = JwtTokenValidator(JWT_KEY).verify_token(attacker)[0]["sub"]

    victim = (await post_exchange(app, node.assertion()))["data"]
    assert str(victim["user_id"]) != str(attacker_id)
    assert victim["is_new_user"] is True


@db_test
async def test_two_first_sign_ins_at_once_open_one_account(accounts):
    import asyncio
    from mirobody.user.user import ensure_sibian_user
    from mirobody.utils.db import execute_query

    results = await asyncio.gather(*(ensure_sibian_user("racer", "Ada", ISS) for _ in range(6)))
    assert len({r.user_id for r in results}) == 1
    live = await execute_query("SELECT count(*) AS n FROM health_app_user WHERE is_del = false")
    assert live[0]["n"] == 1


@db_test
async def test_after_an_account_merge_the_identity_signs_in_to_the_surviving_account(accounts):
    """`/email/bind` onto an address that already has an account merges this
    account into that one (account_merge.merge_accounts). The sibian identity
    must follow the data, not stay on the closed account."""
    from psycopg_pool import AsyncConnectionPool
    from mirobody.user.account_merge import merge_accounts
    from mirobody.user.user import ensure_user

    node = Node()
    app = exchange_app(accounts, node)
    mine = (await post_exchange(app, node.assertion(jti="before")))["data"]
    surviving = await ensure_user("ada@example.org")

    pool = AsyncConnectionPool(accounts.test_conninfo, open=False,
                               kwargs={"options": f"-c search_path={accounts.test_schema},public"})
    await pool.open()
    try:
        _, err = await merge_accounts(pool, int(mine["user_id"]), surviving)
    finally:
        await pool.close()
    assert err is None

    after = (await post_exchange(app, node.assertion(jti="after")))["data"]
    assert str(after["user_id"]) == str(surviving)
    assert after["is_new_user"] is False


@db_test
async def test_an_account_that_turned_on_a_second_factor_is_asked_for_it(accounts):
    """Every other sign-in asks an MFA account for its passkey
    (`_generate_auth_response`). The exchange must too: a token without `aal`
    would be refused on every route for such an account."""
    from psycopg_pool import AsyncConnectionPool
    from mirobody.user.auth.jwt import JwtTokenValidator
    from mirobody.utils.db import execute_query

    pool = AsyncConnectionPool(accounts.test_conninfo, open=False,
                               kwargs={"options": f"-c search_path={accounts.test_schema},public"})
    await pool.open()
    try:
        node = Node()
        app = exchange_app(accounts, node, db_pool=pool, webauthn_rp_id="mirobody.test",
                           webauthn_origin="https://mirobody.test")
        first = (await post_exchange(app, node.assertion(jti="before")))["data"]
        uid = int(first["user_id"])
        await execute_query("UPDATE health_app_user SET mfa_enabled = TRUE WHERE id = :id", {"id": uid})
        await execute_query("INSERT INTO webauthn_credentials (user_id, credential_id, public_key)"
                            " VALUES (:id, 'cred', 'key')", {"id": uid})

        second = (await post_exchange(app, node.assertion(jti="after")))["data"]
    finally:
        await pool.close()

    assert second["status"] == "mfa_required"
    assert second["mfa_ticket"]
    fallback = JwtTokenValidator(JWT_KEY).verify_token(second["fallback_token"])[0]
    assert fallback["aal"] == 1
    assert fallback["exp"] - fallback["iat"] == 3600


async def test_a_jwks_that_arrives_in_pieces_is_read_whole(node_server):
    import asyncio
    import json
    from aiohttp import web
    from mirobody.user.auth.sibian import fetch_jwks

    node, routes, _ = node_server

    async def chunked(request):
        body = json.dumps(node.jwks()).encode()
        r = web.StreamResponse(headers={"Content-Type": "application/json"})
        r.enable_chunked_encoding()
        await r.prepare(request)
        for i in range(0, len(body), 16):
            await r.write(body[i:i + 16])
            await asyncio.sleep(0.005)
        await r.write_eof()
        return r

    routes["/.well-known/sibian-health/jwks.json"] = chunked
    v = SibianAssertionVerifier(issuers_from_config([node.iss]), "mirobody", fetch_jwks)
    assert (await v.verify(node.assertion())).subject == SUBJECT


@pytest.mark.parametrize("city", ["https://sibian-os.exe.xyz", "https://localfirst-city.exe.xyz"])
def test_a_device_link_returns_the_person_to_any_listed_city(monkeypatch, city):
    """The CORS origins are also where a vendor's OAuth callback may send the
    person back to (public_router._return_origins): a list there must be a
    list here too."""
    from types import SimpleNamespace
    import importlib
    from starlette.requests import Request

    # The package exports the APIRouter under this name; the module is what we need.
    public_router = importlib.import_module("mirobody.server.routers.public_router")

    cfg = SimpleNamespace(get_list=lambda key: [],
                          http=SimpleNamespace(headers=[("Access-Control-Allow-Origin", CITIES)]))
    monkeypatch.setattr(public_router, "global_config", lambda: cfg)
    request = Request({"type": "http", "method": "GET", "path": "/cb", "scheme": "https",
                       "server": ("api.example", 443), "headers": [(b"host", b"api.example")]})

    answer = public_router.handle_redirect(request, city + "/health/sources", "oura", "oura")
    assert answer.status_code == 302
    assert answer.headers["location"].startswith(city + "/health/sources?")


async def test_a_page_the_api_renders_is_not_kept_by_a_cache(tmp_path):
    """The OAuth completion page is HTML, but it is the API's answer to one
    person's link, not the web client's shell: it gets the no-store default."""
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from mirobody.server.htdoc import add_htdoc_routes
    from mirobody.server.middleware_stack import build_middlewares

    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>")
    app = FastAPI(middleware=build_middlewares(http_headers=[], jwt_key=JWT_KEY))

    @app.get("/api/v1/pulse/oura/oura/callback")
    async def completed():
        return HTMLResponse("<p>Linked</p>")

    add_htdoc_routes(app, str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mirobody") as client:
        page = await client.get("/api/v1/pulse/oura/oura/callback")
        # A navigation, as a browser makes it: the SPA fallback answers only those.
        shell = await client.get("/timeline", headers={"Accept": "text/html"})
    assert page.headers["cache-control"] == "private, no-store"
    assert shell.headers["cache-control"] == "no-cache"


@db_test
async def test_a_sign_in_that_failed_on_our_side_can_be_retried_with_the_same_assertion(accounts, monkeypatch):
    """503 says "try again": the assertion must still be good for that."""
    from mirobody.user import user_service

    real, calls = user_service.ensure_sibian_user, []

    async def flaky(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionError("database restarted")
        return await real(*args)

    monkeypatch.setattr(user_service, "ensure_sibian_user", flaky)
    node = Node()
    app = exchange_app(accounts, node)
    assertion = node.assertion()
    assert (await post_exchange(app, assertion))["status"] == 503
    retried = await post_exchange(app, assertion)
    assert retried["status"] == 200 and retried["code"] == 0
    # And once it has signed someone in, it is spent.
    assert (await post_exchange(app, assertion))["status"] == 401


async def test_an_unknown_key_on_a_cold_cache_costs_one_read_not_two():
    node, fetched = Node(), []
    v = verifier(node, fetched=fetched)
    with pytest.raises(SibianAssertionRejected):
        await v.verify(Node(kid="never-published").assertion())
    assert fetched == [JWKS_URL]


async def test_sign_ins_arriving_together_share_one_read_of_the_node_s_keys():
    import asyncio

    node, fetched = Node(), []

    async def slow_fetch(url):
        fetched.append(url)
        await asyncio.sleep(0.05)
        return node.jwks()

    v = SibianAssertionVerifier({ISS: JWKS_URL}, "mirobody", slow_fetch)
    results = await asyncio.gather(*(v.verify(node.assertion(jti=f"j{i}")) for i in range(8)))
    assert {r.subject for r in results} == {SUBJECT}
    assert fetched == [JWKS_URL]


def test_a_stale_issuer_setting_does_not_stop_a_server_that_does_not_use_it():
    from mirobody.server.server import _sibian_verifier

    assert _sibian_verifier(["email", "password"], ["http://old-node.example"], "mirobody") is None
    with pytest.raises(ValueError):
        _sibian_verifier(["sibian"], ["http://old-node.example"], "mirobody")


@pytest.mark.parametrize("configured, signed", [
    ("https://sibian-os.exe.xyz", "https://sibian-os.exe.xyz/"),
    ("https://sibian-os.exe.xyz/", "https://sibian-os.exe.xyz"),
])
async def test_a_trailing_slash_does_not_make_a_trusted_node_a_stranger(configured, signed):
    node = Node(iss=signed)

    async def fetch(url):
        return node.jwks()

    v = SibianAssertionVerifier(issuers_from_config([configured]), "mirobody", fetch)
    assert (await v.verify(node.assertion())).subject == SUBJECT
