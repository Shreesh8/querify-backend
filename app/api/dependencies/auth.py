"""
api/dependencies/auth.py

FastAPI dependency for JWT authentication.

Usage in any route:
    async def my_route(current_user_id: uuid.UUID = Depends(get_current_user)):

Since the frontend (Lovable) handles auth, we only verify the JWT
and return the user_id. No user lookup from DB needed unless you
want to check if the user still exists.
"""

import uuid
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> uuid.UUID:
    """
    Decode JWT from Authorization: Bearer <token> header.
    Raises 401 if token is missing or invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = decode_access_token(credentials.credentials)
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is not a valid UUID.",
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[uuid.UUID]:
    """Non-raising variant — returns None if unauthenticated."""
    if not credentials:
        return None
    user_id_str = decode_access_token(credentials.credentials)
    if not user_id_str:
        return None
    try:
        return uuid.UUID(user_id_str)
    except ValueError:
        return None
