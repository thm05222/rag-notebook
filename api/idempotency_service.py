from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
import hashlib
import json
from loguru import logger
from open_notebook.database.repository import repo_query, repo_create, ensure_record_id


class IdempotencyService:
    """Service for handling idempotent API requests"""

    DEFAULT_TTL_HOURS = 24
    LOCK_TIMEOUT_SECONDS = 60

    @staticmethod
    def generate_request_fingerprint(
        endpoint: str, method: str, body: Optional[Dict[str, Any]]
    ) -> str:
        """Generate hash fingerprint of request"""
        content = f"{method}:{endpoint}:{json.dumps(body, sort_keys=True) if body else ''}"
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    async def check_idempotency(
        idempotency_key: str,
        endpoint: str,
        method: str,
        request_body: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if request with idempotency key exists

        Returns:
            (is_duplicate, cached_response)
            - (False, None): New request, proceed
            - (True, response): Duplicate, return cached response
            - Raises conflict if key exists with different fingerprint
        """
        try:
            # Generate fingerprint
            fingerprint = IdempotencyService.generate_request_fingerprint(
                endpoint, method, request_body
            )

            # Query existing record
            result = await repo_query(
                """
                SELECT * FROM idempotency_record 
                WHERE idempotency_key = $key 
                AND expires_at > time::now()
                LIMIT 1
                """,
                {"key": idempotency_key},
            )

            if not result:
                # No existing record
                return False, None

            record = result[0]

            # Check if still locked (request in progress)
            if record.get("locked_until"):
                locked_until_str = record["locked_until"]
                # Parse datetime string
                if isinstance(locked_until_str, str):
                    # SurrealDB returns ISO format datetime strings
                    locked_until = datetime.fromisoformat(
                        locked_until_str.replace("Z", "+00:00")
                    )
                else:
                    locked_until = locked_until_str

                if locked_until > datetime.now(timezone.utc):
                    # Request still processing, return 409 with retry-after
                    retry_seconds = int(
                        (locked_until - datetime.now(timezone.utc)).total_seconds()
                    )
                    return True, {
                        "status": "processing",
                        "message": "Request is being processed",
                        "retry_after": max(retry_seconds, 1),
                    }

            # Check fingerprint match
            if record.get("request_fingerprint") != fingerprint:
                # Same key, different request - CONFLICT
                raise IdempotencyConflictError(
                    f"Idempotency key '{idempotency_key}' already used for different request"
                )

            # Check status
            if record.get("status") == "completed":
                # Return cached response
                return True, {
                    "status": record.get("response_status"),
                    "body": record.get("response_body"),
                    "from_cache": True,
                }
            elif record.get("status") == "failed":
                # Previous attempt failed, allow retry
                logger.info(
                    f"Previous attempt with key {idempotency_key} failed, allowing retry"
                )
                return False, None

            # Processing or unknown status
            return True, {
                "status": "processing",
                "message": "Request is being processed",
            }

        except IdempotencyConflictError:
            raise
        except Exception as e:
            logger.error(f"Error checking idempotency: {e}")
            logger.exception(e)
            # On error, allow request to proceed (fail open)
            return False, None

    @staticmethod
    async def store_idempotency_record(
        idempotency_key: str,
        endpoint: str,
        method: str,
        request_body: Optional[Dict[str, Any]],
        response_status: int,
        response_body: Optional[Dict[str, Any]],
        command_id: Optional[str] = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        """Store successful request/response for idempotency"""
        try:
            fingerprint = IdempotencyService.generate_request_fingerprint(
                endpoint, method, request_body
            )

            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

            # Check if record exists
            existing = await repo_query(
                "SELECT * FROM idempotency_record WHERE idempotency_key = $key",
                {"key": idempotency_key},
            )

            if existing:
                # Update existing record
                await repo_query(
                    """
                    UPDATE idempotency_record 
                    SET 
                        request_fingerprint = $fingerprint,
                        endpoint = $endpoint,
                        http_method = $method,
                        request_body = $request_body,
                        response_status = $response_status,
                        response_body = $response_body,
                        command_id = $command_id,
                        expires_at = $expires_at,
                        locked_until = NONE,
                        status = "completed"
                    WHERE idempotency_key = $key
                    """,
                    {
                        "key": idempotency_key,
                        "fingerprint": fingerprint,
                        "endpoint": endpoint,
                        "method": method,
                        "request_body": request_body,
                        "response_status": response_status,
                        "response_body": response_body,
                        "command_id": ensure_record_id(command_id)
                        if command_id
                        else None,
                        "expires_at": expires_at.isoformat(),
                    },
                )
            else:
                # Create new record
                await repo_create(
                    "idempotency_record",
                    {
                        "idempotency_key": idempotency_key,
                        "request_fingerprint": fingerprint,
                        "endpoint": endpoint,
                        "http_method": method,
                        "request_body": request_body,
                        "response_status": response_status,
                        "response_body": response_body,
                        "command_id": ensure_record_id(command_id)
                        if command_id
                        else None,
                        "expires_at": expires_at.isoformat(),
                        "status": "completed",
                    },
                )

            logger.info(f"Stored idempotency record for key: {idempotency_key}")

        except Exception as e:
            logger.error(f"Error storing idempotency record: {e}")
            logger.exception(e)
            # Don't raise - storing cache should not break the request

    @staticmethod
    async def create_processing_lock(
        idempotency_key: str,
        endpoint: str,
        method: str,
        request_body: Optional[Dict[str, Any]],
        lock_timeout_seconds: int = LOCK_TIMEOUT_SECONDS,
    ) -> None:
        """Create a lock record to indicate request is being processed"""
        try:
            fingerprint = IdempotencyService.generate_request_fingerprint(
                endpoint, method, request_body
            )

            locked_until = datetime.now(timezone.utc) + timedelta(
                seconds=lock_timeout_seconds
            )
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=IdempotencyService.DEFAULT_TTL_HOURS
            )

            await repo_create(
                "idempotency_record",
                {
                    "idempotency_key": idempotency_key,
                    "request_fingerprint": fingerprint,
                    "endpoint": endpoint,
                    "http_method": method,
                    "request_body": request_body,
                    "response_status": 0,  # Not completed yet
                    "expires_at": expires_at.isoformat(),
                    "locked_until": locked_until.isoformat(),
                    "status": "processing",
                },
            )

            logger.debug(
                f"Created processing lock for idempotency key: {idempotency_key}"
            )

        except Exception as e:
            # If record already exists (unique constraint violation), that's ok
            if "already contains" in str(e).lower() or "unique" in str(e).lower():
                logger.debug(
                    f"Processing lock already exists for key: {idempotency_key}"
                )
            else:
                logger.error(f"Error creating processing lock: {e}")
                logger.exception(e)

    @staticmethod
    async def mark_as_failed(
        idempotency_key: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark an idempotency record as failed"""
        try:
            await repo_query(
                """
                UPDATE idempotency_record 
                SET 
                    status = "failed",
                    locked_until = NONE,
                    response_body = $error
                WHERE idempotency_key = $key
                """,
                {
                    "key": idempotency_key,
                    "error": {"error": error_message} if error_message else None,
                },
            )
            logger.info(f"Marked idempotency record as failed: {idempotency_key}")
        except Exception as e:
            logger.error(f"Error marking record as failed: {e}")

    @staticmethod
    async def cleanup_expired_records() -> int:
        """Clean up expired idempotency records"""
        try:
            result = await repo_query(
                """
                DELETE FROM idempotency_record 
                WHERE expires_at < time::now()
                RETURN BEFORE
                """
            )
            count = len(result) if result else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired idempotency records")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired records: {e}")
            logger.exception(e)
            return 0


class IdempotencyConflictError(Exception):
    """Raised when idempotency key is reused with different request"""

    pass

