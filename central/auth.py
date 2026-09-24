"""
central/auth.py — API key authentication (Phase 7).

Reads the key from the Authorization: Bearer <key> header (REST)
or from the ?key= query param (WebSocket).

If api_key is empty string, auth is DISABLED entirely.

Usage
─────
    from central.auth import require_key

    @app.get("/shoppers")
    async def list_shoppers(_: None = Depends(require_key)):
        ...
"""
from __future__ import annotations
from fastapi import Depends, HTTPException, Query, Request, status


def _get_configured_key() -> str:
    from config import CONFIG
    return CONFIG.api_key.strip()


async def require_key(
    request: Request,
    key_param: str = Query(default="", alias="key"),
) -> None:
    """
    FastAPI dependency. Raises HTTP 401 if the key is wrong.
    Accepts key from:
      - Authorization: Bearer <key>  header
      - ?key=<key>                   query param (needed for WebSocket)
    Skips check entirely when api_key is not configured (empty string).
    """
    configured = _get_configured_key()
    if not configured:
        return   # auth disabled

    # Try Authorization header first
    auth_header = request.headers.get("Authorization", "")
    provided = ""
    if auth_header.lower().startswith("bearer "):
        provided = auth_header[7:].strip()
    if not provided:
        provided = key_param

    if provided != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
