"""Shared JWT issuance/verification, configured through the project Settings."""
from datetime import datetime, timedelta, timezone

import jwt
from fastapi.security import OAuth2PasswordBearer

from src.core.config import get_settings

ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def _secret() -> str:
    secret = get_settings().JWT_SECRET_KEY
    if len(secret) < 32:
        raise ValueError("必须配置至少 32 字符的 JWT_SECRET_KEY")
    return secret


def encode_jwt(payload: dict) -> str:
    payload_copy = payload.copy()
    payload_copy["exp"] = datetime.now(timezone.utc) + timedelta(minutes=30)
    payload_copy["iat"] = datetime.now(timezone.utc)
    return jwt.encode(payload_copy, key=_secret(), algorithm=ALGORITHM)


def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, key=_secret(), algorithms=[ALGORITHM],
                          options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError:
        raise ValueError("token已经过期") from None
    except jwt.InvalidTokenError:
        raise ValueError("非法token") from None
