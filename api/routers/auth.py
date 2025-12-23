"""
Authentication router for Open Notebook API.
Provides endpoints for login and authentication status.
"""

import os
from datetime import timedelta

import httpx
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from api.core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ADMIN_PASSWORD_HASH,
    ADMIN_USERNAME,
    TURNSTILE_SECRET_KEY,
    TURNSTILE_SITE_KEY,
    create_access_token,
    get_auth_mode,
    get_password_hash,
    is_jwt_auth_enabled,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Login request body."""
    username: str
    password: str
    captcha_token: str = ""  # Optional, only required if Turnstile is configured


class LoginResponse(BaseModel):
    """Login response body."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class AuthStatusResponse(BaseModel):
    """Authentication status response."""
    auth_enabled: bool
    auth_mode: str  # "jwt", "legacy", or "none"
    turnstile_enabled: bool
    turnstile_site_key: str  # Public site key for Turnstile (safe to expose)
    message: str


async def verify_turnstile_token(token: str) -> bool:
    """
    Verify a Cloudflare Turnstile token.
    
    Args:
        token: The Turnstile token from the frontend
        
    Returns:
        True if verification succeeded, False otherwise
    """
    if not TURNSTILE_SECRET_KEY:
        # Turnstile not configured, skip verification
        return True
    
    if not token:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={
                    "secret": TURNSTILE_SECRET_KEY,
                    "response": token,
                },
                timeout=10.0,
            )
            
            if response.status_code != 200:
                logger.warning(f"Turnstile API returned status {response.status_code}")
                return False
            
            result = response.json()
            success = result.get("success", False)
            
            if not success:
                error_codes = result.get("error-codes", [])
                logger.warning(f"Turnstile verification failed: {error_codes}")
            
            return success
    except httpx.TimeoutException:
        logger.error("Turnstile verification timed out")
        return False
    except Exception as e:
        logger.error(f"Turnstile verification error: {e}")
        return False


@router.get("/status", response_model=AuthStatusResponse)
async def get_auth_status():
    """
    Check authentication configuration status.
    Returns whether authentication is enabled and what mode is active.
    """
    auth_mode = get_auth_mode()
    auth_enabled = auth_mode != "none"
    turnstile_enabled = bool(TURNSTILE_SECRET_KEY)
    
    if auth_mode == "jwt":
        message = "JWT authentication is enabled"
    elif auth_mode == "legacy":
        message = "Legacy password authentication is enabled"
    else:
        message = "Authentication is disabled"
    
    return AuthStatusResponse(
        auth_enabled=auth_enabled,
        auth_mode=auth_mode,
        turnstile_enabled=turnstile_enabled,
        turnstile_site_key=TURNSTILE_SITE_KEY if turnstile_enabled else "",
        message=message,
    )


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT token.
    
    This endpoint performs the following steps:
    1. Verify Turnstile captcha (if configured)
    2. Validate username and password
    3. Generate and return JWT access token
    """
    # Step 1: Verify Turnstile captcha if configured
    if TURNSTILE_SECRET_KEY:
        if not request.captcha_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Captcha verification required",
            )
        
        captcha_valid = await verify_turnstile_token(request.captcha_token)
        if not captcha_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Captcha verification failed",
            )
    
    # Step 2: Check if JWT auth is properly configured
    if not is_jwt_auth_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not configured",
        )
    
    # Step 3: Validate credentials
    # Check username
    if request.username != ADMIN_USERNAME:
        # Use generic error message to prevent username enumeration
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check password
    password_valid = False
    
    if ADMIN_PASSWORD_HASH:
        # Use bcrypt hash verification
        password_valid = verify_password(request.password, ADMIN_PASSWORD_HASH)
    else:
        # Fallback: Check if plain password is set in ADMIN_PASSWORD env var
        # This is less secure but allows easier initial setup
        admin_password = os.environ.get("ADMIN_PASSWORD", "")
        if admin_password:
            password_valid = request.password == admin_password
            if password_valid:
                logger.warning(
                    "Using plain text ADMIN_PASSWORD. "
                    "Consider using ADMIN_PASSWORD_HASH for better security."
                )
    
    if not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Step 4: Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.username},
        expires_delta=access_token_expires,
    )
    
    logger.info(f"User '{request.username}' logged in successfully")
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
    )


@router.post("/hash-password")
async def hash_password(password: str):
    """
    Utility endpoint to generate a bcrypt hash for a password.
    This is useful for generating ADMIN_PASSWORD_HASH.
    
    Note: This endpoint should be disabled in production or
    protected with additional security measures.
    """
    # Only allow in development or when explicitly enabled
    allow_hash_endpoint = os.environ.get("ALLOW_PASSWORD_HASH_ENDPOINT", "false").lower() == "true"
    
    if not allow_hash_endpoint:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is disabled. Set ALLOW_PASSWORD_HASH_ENDPOINT=true to enable.",
        )
    
    return {"hash": get_password_hash(password)}
