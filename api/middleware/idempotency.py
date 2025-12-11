from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from loguru import logger
import json
from typing import Callable
from api.idempotency_service import IdempotencyService, IdempotencyConflictError


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle Idempotency-Key header for non-idempotent requests
    """

    # Methods that require idempotency handling
    NON_IDEMPOTENT_METHODS = {"POST", "PUT", "PATCH"}

    # Paths to exclude from idempotency handling
    EXCLUDED_PATHS = {
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/api/chat",  # Streaming responses
        "/api/ask",  # Streaming responses
    }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with idempotency handling"""

        # Skip if method doesn't need idempotency
        if request.method not in self.NON_IDEMPOTENT_METHODS:
            return await call_next(request)

        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.EXCLUDED_PATHS):
            return await call_next(request)

        # Get Idempotency-Key header (case-insensitive)
        idempotency_key = request.headers.get("Idempotency-Key") or request.headers.get(
            "idempotency-key"
        )

        # If no key provided, proceed without idempotency
        if not idempotency_key:
            logger.debug(
                f"No Idempotency-Key for {request.method} {request.url.path}"
            )
            return await call_next(request)

        # Validate key format (should be UUID or similar)
        if len(idempotency_key) < 16 or len(idempotency_key) > 255:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Invalid Idempotency-Key format. Must be 16-255 characters."
                },
            )

        try:
            # Read request body
            body_bytes = await request.body()
            request_body = None

            if body_bytes:
                try:
                    request_body = json.loads(body_bytes)
                except json.JSONDecodeError:
                    # If body is not JSON (e.g., multipart/form-data), use None
                    # The fingerprint will still work with None
                    request_body = None

            # Check if this request was already processed
            is_duplicate, cached_response = await IdempotencyService.check_idempotency(
                idempotency_key=idempotency_key,
                endpoint=request.url.path,
                method=request.method,
                request_body=request_body,
            )

            if is_duplicate and cached_response:
                # Return cached response
                if cached_response.get("status") == "processing":
                    return JSONResponse(
                        status_code=409,
                        content=cached_response,
                        headers={
                            "Retry-After": str(cached_response.get("retry_after", 60))
                        },
                    )

                # Return successful cached response
                logger.info(
                    f"Returning cached response for idempotency key: {idempotency_key}"
                )
                return JSONResponse(
                    status_code=cached_response.get("status", 200),
                    content=cached_response.get("body", {}),
                    headers={"X-Idempotent-Replayed": "true"},
                )

            # Create processing lock
            await IdempotencyService.create_processing_lock(
                idempotency_key=idempotency_key,
                endpoint=request.url.path,
                method=request.method,
                request_body=request_body,
            )

            # Reconstruct request with body (it was consumed)
            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request._receive = receive

            # Store key in request state for later use
            request.state.idempotency_key = idempotency_key
            request.state.idempotency_endpoint = request.url.path
            request.state.idempotency_method = request.method
            request.state.idempotency_request_body = request_body

            # Process request
            response = await call_next(request)

            # Store response if successful (2xx status)
            if 200 <= response.status_code < 300:
                # Read response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                response_json = None
                command_id = None

                try:
                    response_json = json.loads(response_body)
                    # Try to extract command_id if present
                    if isinstance(response_json, dict):
                        command_id = response_json.get("command_id")
                except json.JSONDecodeError:
                    # Response is not JSON
                    response_json = None

                # Store idempotency record
                await IdempotencyService.store_idempotency_record(
                    idempotency_key=idempotency_key,
                    endpoint=request.url.path,
                    method=request.method,
                    request_body=request_body,
                    response_status=response.status_code,
                    response_body=response_json,
                    command_id=command_id,
                )

                # Reconstruct response
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            elif response.status_code >= 500:
                # Mark as failed for server errors
                await IdempotencyService.mark_as_failed(
                    idempotency_key=idempotency_key,
                    error_message=f"Server error: {response.status_code}",
                )

            return response

        except IdempotencyConflictError as e:
            logger.warning(f"Idempotency conflict: {e}")
            return JSONResponse(
                status_code=422,
                content={"detail": str(e), "error_code": "idempotency_conflict"},
            )
        except Exception as e:
            logger.error(f"Error in idempotency middleware: {e}")
            logger.exception(e)
            # On error, allow request through (fail open)
            # But mark the idempotency record as failed if we have the key
            try:
                if idempotency_key:
                    await IdempotencyService.mark_as_failed(
                        idempotency_key=idempotency_key, error_message=str(e)
                    )
            except:
                pass
            return await call_next(request)

