"""
Message utilities for Open Notebook.
Handles message truncation and token limit management.
"""

from typing import List

from langchain_core.messages import BaseMessage

from open_notebook.utils.token_utils import token_count
from loguru import logger

# Safe token limit for requests (well below OpenAI's 500K TPM limit)
# Reserve space for output tokens and system prompts
DEFAULT_MAX_INPUT_TOKENS = 300000  # 300K tokens, leaving 200K buffer for output and overhead


def truncate_messages(
    messages: List[BaseMessage],
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    keep_recent: bool = True,
) -> List[BaseMessage]:
    """
    Truncate messages to fit within token limits.
    
    This function ensures that the total token count of messages stays within
    the specified limit, preventing rate limit errors from OpenAI API.
    
    Args:
        messages: List of messages to truncate
        max_tokens: Maximum allowed tokens (default: 300K)
        keep_recent: If True, keeps the most recent messages and removes older ones.
                     If False, keeps the first messages and removes later ones.
    
    Returns:
        Truncated list of messages that fits within token limit
    """
    if not messages:
        return messages
    
    # Calculate total tokens
    total_tokens = sum(token_count(str(msg.content)) for msg in messages)
    
    if total_tokens <= max_tokens:
        logger.debug(f"Message token count {total_tokens} within limit {max_tokens}")
        return messages
    
    logger.warning(
        f"Message token count {total_tokens} exceeds limit {max_tokens}, truncating..."
    )
    
    if keep_recent:
        # Keep the most recent messages (typically the most important)
        # Start from the end and work backwards
        truncated = []
        current_tokens = 0
        
        # Always keep the last message (the user's current question)
        for msg in reversed(messages):
            msg_tokens = token_count(str(msg.content))
            
            if current_tokens + msg_tokens <= max_tokens:
                truncated.insert(0, msg)  # Insert at beginning to maintain order
                current_tokens += msg_tokens
            else:
                # If adding this message would exceed limit, stop
                # But ensure we have at least the last message
                if not truncated:
                    # If we can't even fit the last message, truncate its content
                    truncated.insert(0, msg)
                    logger.warning(
                        f"Even the last message exceeds token limit, keeping it anyway"
                    )
                break
        
        removed_count = len(messages) - len(truncated)
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} older messages, "
                f"kept {len(truncated)} recent messages "
                f"({current_tokens} tokens)"
            )
        
        return truncated
    else:
        # Keep the first messages (less common, but available if needed)
        truncated = []
        current_tokens = 0
        
        for msg in messages:
            msg_tokens = token_count(str(msg.content))
            
            if current_tokens + msg_tokens <= max_tokens:
                truncated.append(msg)
                current_tokens += msg_tokens
            else:
                break
        
        removed_count = len(messages) - len(truncated)
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} later messages, "
                f"kept {len(truncated)} early messages "
                f"({current_tokens} tokens)"
            )
        
        return truncated


def truncate_payload_with_system_message(
    system_message: BaseMessage,
    messages: List[BaseMessage],
    max_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> List[BaseMessage]:
    """
    Truncate messages while ensuring system message is always included.
    
    Args:
        system_message: System message (always kept)
        messages: List of conversation messages to truncate
        max_tokens: Maximum allowed tokens (default: 300K)
    
    Returns:
        Payload with system message + truncated messages
    """
    system_tokens = token_count(str(system_message.content))
    available_tokens = max_tokens - system_tokens
    
    if available_tokens <= 0:
        logger.warning(
            f"System message uses {system_tokens} tokens, "
            f"exceeding limit {max_tokens}. Keeping only system message."
        )
        return [system_message]
    
    # Truncate conversation messages to fit in remaining space
    truncated_messages = truncate_messages(messages, max_tokens=available_tokens)
    
    return [system_message] + truncated_messages

