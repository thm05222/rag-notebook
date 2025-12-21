"""
Security utilities for JWT authentication and password hashing.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration from environment variables
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# Admin credentials from environment
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")

# Turnstile configuration
TURNSTILE_SECRET_KEY = os.environ.get("TURNSTILE_SECRET_KEY", "")


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    username: Optional[str] = None
    exp: Optional[datetime] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against a bcrypt hash.
    
    Args:
        plain_password: The plain text password to verify
        hashed_password: The bcrypt hash to verify against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """
    Generate a bcrypt hash for a password.
    
    Args:
        password: The plain text password to hash
        
    Returns:
        The bcrypt hash of the password
    """
    return pwd_context.hash(password)


def create_access_token(
    data: dict, 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary of claims to encode in the token
        expires_delta: Optional expiration time delta (defaults to ACCESS_TOKEN_EXPIRE_MINUTES)
        
    Returns:
        Encoded JWT token string
        
    Raises:
        ValueError: If JWT_SECRET_KEY is not configured
    """
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY environment variable is not set")
    
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT access token.
    
    Args:
        token: The JWT token string to decode
        
    Returns:
        TokenData if valid, None if invalid or expired
    """
    if not JWT_SECRET_KEY:
        return None
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        exp = payload.get("exp")
        
        if username is None:
            return None
        
        # Convert exp to datetime if it's a timestamp
        exp_datetime = None
        if exp:
            exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        
        return TokenData(username=username, exp=exp_datetime)
    except JWTError:
        return None


def is_jwt_auth_enabled() -> bool:
    """
    Check if JWT authentication is properly configured.
    
    Returns:
        True if JWT auth is enabled (JWT_SECRET_KEY is set), False otherwise
    """
    return bool(JWT_SECRET_KEY)


def is_legacy_auth_enabled() -> bool:
    """
    Check if legacy password authentication is enabled.
    This is for backward compatibility with OPEN_NOTEBOOK_PASSWORD.
    
    Returns:
        True if legacy auth is enabled, False otherwise
    """
    return bool(os.environ.get("OPEN_NOTEBOOK_PASSWORD"))


def get_auth_mode() -> str:
    """
    Determine the current authentication mode.
    
    Returns:
        "jwt" if JWT auth is configured,
        "legacy" if only OPEN_NOTEBOOK_PASSWORD is set,
        "none" if no authentication is configured
    """
    if is_jwt_auth_enabled():
        return "jwt"
    elif is_legacy_auth_enabled():
        return "legacy"
    else:
        return "none"
