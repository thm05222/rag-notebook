"""
FastAPI dependencies for authentication and authorization.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.core.security import decode_access_token, get_auth_mode, TokenData

# HTTP Bearer token security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenData:
    """
    Dependency to get the current authenticated user from JWT token.
    
    This dependency extracts and validates the JWT token from the
    Authorization header.
    
    Args:
        credentials: HTTP Bearer credentials from the Authorization header
        
    Returns:
        TokenData containing the authenticated user information
        
    Raises:
        HTTPException 401: If token is missing, invalid, or expired
    """
    auth_mode = get_auth_mode()
    
    # If no authentication is configured, return a dummy user
    if auth_mode == "none":
        return TokenData(username="anonymous")
    
    # For legacy auth mode, we don't validate JWT (middleware handles it)
    if auth_mode == "legacy":
        return TokenData(username="legacy_user")
    
    # JWT auth mode - validate the token
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if credentials is None:
        raise credentials_exception
    
    token = credentials.credentials
    token_data = decode_access_token(token)
    
    if token_data is None:
        raise credentials_exception
    
    return token_data


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """
    Dependency to optionally get the current user.
    
    Unlike get_current_user, this dependency doesn't raise an exception
    if no valid token is provided. Useful for endpoints that work
    differently for authenticated vs. anonymous users.
    
    Args:
        credentials: HTTP Bearer credentials from the Authorization header
        
    Returns:
        TokenData if authenticated, None otherwise
    """
    auth_mode = get_auth_mode()
    
    if auth_mode == "none":
        return TokenData(username="anonymous")
    
    if credentials is None:
        return None
    
    token = credentials.credentials
    return decode_access_token(token)
