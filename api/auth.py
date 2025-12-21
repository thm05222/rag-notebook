"""
Authentication middleware for Open Notebook API.
Supports both JWT authentication and legacy password authentication for backward compatibility.
"""

import os
from typing import Optional

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.core.security import decode_access_token, get_auth_mode


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Unified authentication middleware that supports both JWT and legacy password modes.
    
    Authentication Mode Selection:
    - JWT Mode: If JWT_SECRET_KEY is set, JWT tokens are validated
    - Legacy Mode: If only OPEN_NOTEBOOK_PASSWORD is set, simple password matching is used
    - No Auth: If neither is set, all requests are allowed
    
    The middleware automatically detects which mode to use based on environment variables.
    """
    
    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.legacy_password = os.environ.get("OPEN_NOTEBOOK_PASSWORD")
        self.excluded_paths = excluded_paths or [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/auth/status",
            "/api/auth/login",
            "/api/config",
        ]
    
    async def dispatch(self, request, call_next):
        auth_mode = get_auth_mode()
        
        # Skip authentication if no auth is configured
        if auth_mode == "none":
            return await call_next(request)
        
        # Skip authentication for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Skip authentication for CORS preflight requests (OPTIONS)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing authorization header"},
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Parse authorization header
        try:
            scheme, credentials = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authentication scheme")
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization header format"},
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Validate credentials based on auth mode
        if auth_mode == "jwt":
            # JWT validation
            token_data = decode_access_token(credentials)
            if token_data is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"},
                    headers={"WWW-Authenticate": "Bearer"}
                )
        else:
            # Legacy password validation
            if credentials != self.legacy_password:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid password"},
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        # Authentication successful, proceed with the request
        response = await call_next(request)
        return response


# Backward compatibility alias
PasswordAuthMiddleware = AuthMiddleware


# Optional: HTTPBearer security scheme for OpenAPI documentation
security = HTTPBearer(auto_error=False)


def check_api_password(credentials: Optional[HTTPAuthorizationCredentials] = None) -> bool:
    """
    Utility function to check API password.
    Can be used as a dependency in individual routes if needed.
    
    Note: This function is deprecated. Use api.deps.get_current_user instead.
    """
    auth_mode = get_auth_mode()
    
    # No auth configured, allow access
    if auth_mode == "none":
        return True
    
    # No credentials provided
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if auth_mode == "jwt":
        # JWT validation
        token_data = decode_access_token(credentials.credentials)
        if token_data is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        # Legacy password validation
        password = os.environ.get("OPEN_NOTEBOOK_PASSWORD")
        if credentials.credentials != password:
            raise HTTPException(
                status_code=401,
                detail="Invalid password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    return True
