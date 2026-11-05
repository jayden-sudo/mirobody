# Device push: `POST /api/v1/ingest/batch`

A trusted hub pushes device readings for sibian users. Off unless
`INGEST_PUBLIC_KEY` is set (the route then answers 404).

## Config
| Key | Meaning |
|---|---|
| `INGEST_PUBLIC_KEY` | base64 of the hub's 32 raw Ed25519 public-key bytes |
| `INGEST_AUDIENCE` | string both sides sign in (1-64 chars, no control characters) |
| `INGEST_REGISTRY_FILE` | YAML metric registry (below). Optional. Unset, **every metric is accepted exactly as sent**: its name, unit and value are stored as given, with no unit conversion, range check, label check or day window. Set it for a real deployment so typos and wrong units are refused. With a file, only the metrics it lists are accepted |
| `INGEST_AUTO_CREATE_USERS` | default `true`: open an account for an unseen sibian identity |
| `INGEST_BACKFILL_DAYS` | default 4000: older readings are refused (`too-old`) |

A set key with a bad audience, day count, or an unreadable/invalid registry file stops the server at start.

## Request
Headers `X-Ingest-Timestamp` (unix s, ±300 s), `X-Ingest-Nonce` (base64url, no padding, 16-64 bytes, one use),
`X-Ingest-Signature` (base64url, no padding, 64 bytes). The signed message is these seven lines joined by `\n`
(no trailing newline): `mirobody-hub-ingest-v1`, audience, `POST`, `/api/v1/ingest/batch`, timestamp, nonce,
lowercase hex sha256 of the **exact body bytes sent**.

Body (strict; unknown fields are refused):
```json
{"v":1,"batch_id":"b1","uid":"<64 hex sibian sub>","tz":"Asia/Taipei",
 "devices":[{"name":"device_1","serial":"A1","samples":[
   {"id":"d1-0001","metric":"heart_rate","value":58,"unit":"/min","t":"2026-11-05T02:00:00+08:00"}]}]}
```
Limits: 1-50 devices, ≤ 2000 samples, ≤ 2 MiB. `(name, serial)` unique per request; `id` unique per device.
Times: ISO 8601. An offset or `Z` is used as is; a time without one needs the batch `tz` and is refused if the
local time does not exist or happens twice (DST). Never UTC by default.

## Answers
`200` (`status` ok/partial, counts, `rejected`, `errors` ≤ 100), `401` (any authentication failure, one message),
`404` (off), `413`, `422` (`invalid_request`, `invalid_json`, `invalid_tz`, `duplicate_device`, `duplicate_sample_id`,
`too_many_samples`, `all_samples_rejected`, `subject_not_found`, `subject_deleted`), `503` (nonce store or a write
failed: resend the whole batch with a fresh timestamp, nonce and signature). Per-sample reasons: `unknown-metric`,
`unknown-unit`, `bad-value-type`, `out-of-range`, `bad-label`, `too-old`, `future`, `bad-interval`, `bad-time`,
`missing-tz`, `ambiguous-or-nonexistent-local-time`, `write-error`.

## Registry
```yaml
version: 1
metrics:
  heart_rate:
    name: heart rate        # stored name
    aliases: ["心率"]
    kind: quantity          # quantity | ordinal | nominal | boolean
    unit: /min              # stored unit
    accept_units: {"/min": 1, bpm: 1}   # input unit -> factor into `unit`
    decimals: 0             # required for quantity
    bounds: [20, 250]       # optional, inclusive
    window: "18:00"         # optional local-day boundary (use for sleep)
    # allowed: [...]        # required for ordinal/nominal
    # censor: {LOW: "<2.2"} # quantity only: marker -> stored text
```
Values are rendered deterministically (without a registry file: numbers as plain decimals with trailing zeros dropped, so `58` and `58.0` are the same; strings trimmed; booleans `true`/`false`) (fixed decimals, `true`/`false`, NFC labels). Unknown metrics are refused;
devices need no registration. Edit the file and restart.

## Idempotency
A row's identity is (subject, metric, start, end, `hub:{device}:{serial|-}`, `{device}:{serial|-}:{id}`). Resend
the same batch with the same ids, times, device name/serial and `tz` form: identical rows are `skipped`, a changed
value is written as an amendment. Changing any identity field creates a new row.

## Accepted risks
One key, full power: whoever holds the private key can write (or forge) data for any user and can learn whether an
identity has an account. No rate limit (use `limit_req` in the proxy), no audit log, auto-created accounts for
people who may never have signed in. Rotate by changing `INGEST_PUBLIC_KEY` and restarting. Use HTTPS.
Reference clients: `examples/hub_client.py` (Python) and a TypeScript project in the ThetaWellness repo.
