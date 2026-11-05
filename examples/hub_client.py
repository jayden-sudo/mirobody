"""Reference hub client: sign one batch and POST it. Usage:
HUB_PRIVATE_KEY=<base64 seed> INGEST_AUDIENCE=... SUBJECT_UID=<64 hex> python examples/hub_client.py [url]
503 -> resend the whole batch (fresh timestamp/nonce/signature, same ids); 401/422 -> do not retry."""
import base64
import hashlib
import json
import os
import sys
import time

import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PATH = "/api/v1/ingest/batch"
def b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()



def send(base_url, key, audience, payload, attempts=5):
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")  # sign THESE bytes
    for n in range(attempts):
        ts, nonce = str(int(time.time())), b64u(os.urandom(16))
        msg = "\n".join(["mirobody-hub-ingest-v1", audience, "POST", PATH, ts, nonce, hashlib.sha256(body).hexdigest()])
        headers = {"Content-Type": "application/json", "X-Ingest-Timestamp": ts, "X-Ingest-Nonce": nonce,
                   "X-Ingest-Signature": b64u(key.sign(msg.encode()))}
        r = requests.post(base_url + PATH, data=body, headers=headers, timeout=30)
        if r.status_code != 503:
            return r
        time.sleep(min(30, 2 ** n))
    return r


if __name__ == "__main__":
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ["HUB_PRIVATE_KEY"]))
    payload = {"v": 1, "uid": os.environ["SUBJECT_UID"], "devices": [{"name": "device_1", "samples": [
        {"id": "d1-0001", "metric": "heart_rate", "value": 58, "unit": "/min", "t": "2026-11-05T02:00:00+08:00"}]}]}
    r = send(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18060", key, os.environ["INGEST_AUDIENCE"], payload)
    print(r.status_code, r.text)
