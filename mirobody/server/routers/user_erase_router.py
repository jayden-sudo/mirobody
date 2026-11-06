"""`POST /api/v1/user/erase`: erase everything the server holds about the caller.

Caller only: there is no `target_user_id`, and a care-circle grant, even a
read-write one, does not reach it. The body must say `"confirm":
"erase-all-my-data"`, so a stray request cannot do it. What goes, and what
does not, is written at the top of `user/erase.py`. The account is closed in the
same transaction, so this token (and every other) stops working the moment it
returns; a retry that arrives after it gets the standard rejection of a closed
account, which a client takes as "already erased".
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from mirobody.server.auth import verify_token
from mirobody.server.envelope import ErrorResponse, StandardResponse, failed, ok
from mirobody.user.erase import AccountManagesOthers, AccountNotFound, erase_user_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/user", tags=["user"])

CONFIRM = "erase-all-my-data"


class EraseRequest(BaseModel):
    confirm: str = ""


@router.post("/erase", response_model=StandardResponse | ErrorResponse)
async def erase(body: EraseRequest, user_id: str = Depends(verify_token)):
    if body.confirm != CONFIRM:
        return ErrorResponse(code=400, msg=f'Send {{"confirm": "{CONFIRM}"}} to erase everything.')
    try:
        return ok(await erase_user_data(str(user_id)))
    except AccountManagesOthers:
        return ErrorResponse(code=409, msg="This account manages other accounts. Remove or hand them over first.")
    except AccountNotFound:
        return ErrorResponse(code=404, msg="This account is already closed.")
    except Exception as exc:
        return failed("erase", exc, "The erasure could not complete. Nothing was changed.")
