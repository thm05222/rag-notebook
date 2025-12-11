"""
Text utilities for Open Notebook.
Extracted from main utils to avoid circular imports.
"""

import re
import unicodedata
from typing import Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .token_utils import token_count

# Pattern for matching thinking content in AI responses
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_text(txt: str, chunk_size=500):
    """
    Split the input text into chunks.

    Args:
        txt (str): The input text to be split.
        chunk_size (int): The size of each chunk. Default is 500.

    Returns:
        list: A list of text chunks.
    """
    overlap = int(chunk_size * 0.15)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=token_count,
        separators=[
            "\n\n",
            "\n",
            ".",
            ",",
            " ",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )
    return text_splitter.split_text(txt)


def remove_non_ascii(text: str) -> str:
    """Remove non-ASCII characters from text."""
    return re.sub(r"[^\x00-\x7F]+", "", text)


def remove_non_printable(text: str) -> str:
    """Remove non-printable characters from text."""
    # Replace any special Unicode whitespace characters with a regular space
    text = re.sub(r"[\u2000-\u200B\u202F\u205F\u3000]", " ", text)

    # Replace unusual line terminators with a single newline
    text = re.sub(r"[\u2028\u2029\r]", "\n", text)

    # Remove control characters, except newlines and tabs
    text = "".join(
        char for char in text if unicodedata.category(char)[0] != "C" or char in "\n\t"
    )

    # Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ").strip()

    # Keep letters (including accented ones), numbers, spaces, newlines, tabs, and basic punctuation
    return re.sub(r"[^\w\s.,!?\-\n\t]", "", text, flags=re.UNICODE)


def parse_thinking_content(content: str) -> Tuple[str, str]:
    """
    Parse message content to extract thinking content from <think> tags.

    Args:
        content (str): The original message content

    Returns:
        Tuple[str, str]: (thinking_content, cleaned_content)
            - thinking_content: Content from within <think> tags
            - cleaned_content: Original content with <think> blocks removed

    Example:
        >>> content = "<think>Let me analyze this</think>Here's my answer"
        >>> thinking, cleaned = parse_thinking_content(content)
        >>> print(thinking)
        "Let me analyze this"
        >>> print(cleaned)
        "Here's my answer"
    """
    # Input validation
    if not isinstance(content, str):
        return "", str(content) if content is not None else ""

    # Limit processing for very large content (100KB limit)
    if len(content) > 100000:
        return "", content

    # Find all thinking blocks
    thinking_matches = THINK_PATTERN.findall(content)

    if not thinking_matches:
        return "", content

    # Join all thinking content with double newlines
    thinking_content = "\n\n".join(match.strip() for match in thinking_matches)

    # Remove all <think>...</think> blocks from the original content
    cleaned_content = THINK_PATTERN.sub("", content)

    # Clean up extra whitespace
    cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content).strip()

    return thinking_content, cleaned_content


def clean_thinking_content(content: str) -> str:
    """
    Remove thinking content from AI responses, returning only the cleaned content.

    This is a convenience function for cases where you only need the cleaned
    content and don't need access to the thinking process.

    Args:
        content (str): The original message content with potential <think> tags

    Returns:
        str: Content with <think> blocks removed and whitespace cleaned.
             If cleaned content is empty but original content has thinking tags,
             returns the thinking content as the answer (since thinking process
             will be extracted separately from LangGraph state).

    Example:
        >>> content = "<think>Let me think...</think>Here's the answer"
        >>> clean_thinking_content(content)
        "Here's the answer"
        
        >>> content = "<think>Here's the answer</think>"
        >>> clean_thinking_content(content)
        "Here's the answer"  # Uses thinking content when cleaned is empty
    """
    from loguru import logger
    
    # 關鍵修復：記錄清理過程的詳細信息
    original_length = len(content) if content else 0
    logger.debug(f"clean_thinking_content: Input length: {original_length} chars")
    
    thinking_content, cleaned_content = parse_thinking_content(content)
    
    thinking_length = len(thinking_content) if thinking_content else 0
    cleaned_length = len(cleaned_content) if cleaned_content else 0
    
    logger.debug(f"clean_thinking_content: Extracted thinking: {thinking_length} chars, Cleaned: {cleaned_length} chars")
    
    # 關鍵修復：如果清理後內容為空，但原始內容有思考標籤，則使用思考標籤內的內容作為答案
    # 因為思考過程會從 LangGraph 狀態中單獨提取，所以這裡可以使用思考內容作為答案
    if not cleaned_content or len(cleaned_content.strip()) == 0:
        if thinking_content and len(thinking_content.strip()) > 0:
            # 使用思考標籤內的內容作為答案
            logger.debug(f"clean_thinking_content: Using thinking content as answer ({thinking_length} chars)")
            return thinking_content.strip()
        # 如果思考內容也是空的，返回原始內容（以防萬一）
        logger.debug(f"clean_thinking_content: Both cleaned and thinking are empty, returning original content ({original_length} chars)")
        return content.strip() if content else ""
    
    logger.debug(f"clean_thinking_content: Returning cleaned content ({cleaned_length} chars)")
    return cleaned_content
