"""
api/dependencies/auth.py
Firebase token verification.
"""

import uuid
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Initialize Firebase Admin SDK once
if not firebase_admin._apps:
    cred = credentials.Certificate("/home/ec2-user/querify-backend/firebase-service-account.json")
    firebase_admin.initialize_app(cred)

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> uuid.UUID:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        decoded = firebase_auth.verify_id_token(credentials.credentials)
        uid = decoded["uid"]
        # Convert Firebase UID to UUID5 (deterministic, consistent)
        return uuid.uuid5(uuid.NAMESPACE_URL, uid)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[uuid.UUID]:
    if not credentials:
        return None
    try:
        decoded = firebase_auth.verify_id_token(credentials.credentials)
        uid = decoded["uid"]
        return uuid.uuid5(uuid.NAMESPACE_URL, uid)
    except Exception:
        return None
