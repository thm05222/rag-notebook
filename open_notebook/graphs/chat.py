"""
Agentic RAG graph for Chat functionality.
Supports iterative search, multi-tool usage, self-evaluation, and hallucination detection.
"""

import asyncio
import operator
import os
import time
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiosqlite
from ai_prompter import Prompter
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

# Try new import path first (langgraph-checkpoint-sqlite 2.0+), fallback to old path
try:
    from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
except ImportError:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.domain.notebook import Notebook
from open_notebook.exceptions import ToolNotFoundError
from open_notebook.graphs.utils import provision_langchain_model
from open_notebook.services.evaluation_service import evaluation_service
from open_notebook.services.tool_service import tool_registry
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.token_utils import token_count


class ChatAgenticState(TypedDict):
    """State for Agentic RAG Chat workflow."""

    # 對話相關
    messages: Annotated[list[BaseMessage], add_messages]
    notebook: Optional[Notebook]
    context_config: Optional[dict]  # 用戶選擇的 sources
    conversation_context: Optional[List[Dict[str, str]]]  # 格式化的對話歷史

    # Agentic RAG 核心狀態
    question: str  # 從最新訊息提取的問題
    iteration_count: int
    search_history: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Search history - 關鍵修復：改為自動累積
    collected_results: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Accumulated search results - 關鍵修復：改為自動累積
    current_tool_calls: Annotated[
        List[Dict[str, Any]], operator.add
    ]  # Current tool calls - 關鍵修復：改為自動累積
    evaluation_result: Optional[Dict[str, Any]]  # Evaluation result
    hallucination_check: Optional[Dict[str, Any]]  # Hallucination detection result
    partial_answer: str  # Partial answer
    final_answer: Optional[str]  # Final answer
    reasoning_trace: Annotated[List[str], operator.add]  # Reasoning trace

    # 限制和配置
    max_iterations: int  # Maximum iterations (default 5 for chat)
    token_count: int  # Token usage
    max_tokens: int  # Token limit (default 300000)
    start_time: float  # Start timestamp
    max_duration: float  # Maximum duration (seconds, default 60)
    decision_history: Annotated[
        List[str], operator.add
    ]  # Decision history for cycle detection
    error_history: Annotated[List[Dict[str, Any]], operator.add]  # Error history
    unavailable_tools: Annotated[List[str], operator.add]  # Tools that are unavailable (not found or failed)
    model_override: Optional[str]  # Model override for this session
    current_decision: Optional[Dict[str, Any]]  # 當前決策（關鍵修復：添加缺失的字段）
    refinement_feedback: Optional[Dict[str, Any]]  # Feedback from Refiner when rejecting answer


class Decision(BaseModel):
    """Decision model for agent actions."""

    action: Literal["use_tool", "evaluate", "synthesize", "finish"] = Field(
        description="Action to take: 'use_tool', 'evaluate', 'synthesize', or 'finish'"
    )
    tool_name: Optional[str] = Field(
        None, description="Tool name if action is 'use_tool'"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Tool parameters if action is 'use_tool'"
    )
    reasoning: str = Field(description="Reasoning for this decision")


def extract_question_and_context(
    messages: List[BaseMessage],
) -> tuple[str, List[Dict[str, str]]]:
    """
    從對話歷史中提取當前問題和相關上下文。

    Returns:
        - current_question: 最新的用戶問題
        - conversation_context: 相關的對話歷史（格式化的摘要）
    """
    from langchain_core.messages import AIMessage

    # 獲取最新的用戶訊息
    user_messages = [
        msg
        for msg in messages
        if isinstance(msg, HumanMessage)
        or (hasattr(msg, "type") and msg.type == "human")
    ]
    if not user_messages:
        # 如果沒有用戶訊息，嘗試從最後一條訊息提取
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                return str(last_msg.content), []
        return "", []

    current_question = (
        user_messages[-1].content
        if hasattr(user_messages[-1], "content")
        else str(user_messages[-1])
    )

    # 提取對話上下文（最近 2-3 輪對話）
    conversation_context = []
    ai_messages = [
        msg
        for msg in messages
        if isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai")
    ]

    # 如果對話歷史較短，包含所有歷史
    if len(user_messages) <= 3:
        for i in range(len(user_messages) - 1):  # 不包括當前問題
            if i < len(ai_messages):
                user_content = (
                    user_messages[i].content
                    if hasattr(user_messages[i], "content")
                    else str(user_messages[i])
                )
                ai_content = (
                    ai_messages[i].content
                    if hasattr(ai_messages[i], "content")
                    else str(ai_messages[i])
                )
                # 截斷避免過長
                conversation_context.append(
                    {
                        "user": user_content[:200],
                        "assistant": ai_content[:300],
                    }
                )
    else:
        # 只包含最近 2 輪對話作為上下文
        for i in range(max(0, len(user_messages) - 3), len(user_messages) - 1):
            if i < len(ai_messages):
                user_content = (
                    user_messages[i].content
                    if hasattr(user_messages[i], "content")
                    else str(user_messages[i])
                )
                ai_content = (
                    ai_messages[i].content
                    if hasattr(ai_messages[i], "content")
                    else str(ai_messages[i])
                )
                conversation_context.append(
                    {
                        "user": user_content[:200],
                        "assistant": ai_content[:300],
                    }
                )

    return current_question, conversation_context


def should_include_conversation_history(
    current_question: str, conversation_context: List[Dict[str, str]]
) -> bool:
    """
    判斷是否需要包含對話歷史。

    需要包含的情況：
    1. 當前問題包含代詞或指代
    2. 當前問題很短（< 20 字符）
    3. 當前問題是後續問題
    """
    if not conversation_context:
        return False

    # 檢查代詞
    pronouns = [
        "這個",
        "那個",
        "它",
        "它們",
        "上面",
        "下面",
        "之前",
        "剛才",
        "還有",
        "然後",
    ]
    has_pronoun = any(pronoun in current_question for pronoun in pronouns)

    # 檢查問題長度
    is_short = len(current_question.strip()) < 20

    # 檢查是否為後續問題
    is_followup = any(
        keyword in current_question
        for keyword in ["還有", "然後", "另外", "還有呢", "然後呢"]
    )

    return has_pronoun or is_short or is_followup


async def initialize_chat_state(
    state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
    """Initialize ChatAgenticState with default values and extract question from messages."""
    import os

    # 從 messages 中提取問題和對話歷史
    messages = state.get("messages", [])
    question, conversation_context = extract_question_and_context(messages)

    # 判斷是否需要包含對話歷史
    include_history = should_include_conversation_history(
        question, conversation_context
    )

    # 允許配置的值，fallback 到 env vars 或默認值（Chat 使用更保守的限制）
    max_iterations = state.get("max_iterations") or int(
        os.getenv("CHAT_MAX_ITERATIONS", "10")
    )  # 從 5 改為 10
    max_tokens = state.get("max_tokens") or int(os.getenv("CHAT_MAX_TOKENS", "300000"))
    max_duration = state.get("max_duration") or float(
        os.getenv("CHAT_MAX_DURATION", "120")
    )  # 從 60 改為 120 秒

    # 關鍵修復：強制重置所有 Agentic RAG 相關狀態，確保不會受到 checkpoint 中舊狀態的影響
    logger.info(
        f"Initializing chat state for question: {question[:100] if question else 'None'}..."
    )

    initialized_state = {
        # 對話相關
        "messages": messages,
        "notebook": state.get("notebook"),
        "context_config": state.get("context_config"),
        "conversation_context": conversation_context if include_history else [],
        # Agentic RAG 核心狀態 - 強制重置
        "question": question,
        "iteration_count": 0,  # 強制重置為 0
        "search_history": [],  # 強制重置
        "collected_results": [],  # 強制重置
        "current_tool_calls": [],  # 強制重置
        "evaluation_result": None,  # 強制重置
        "hallucination_check": None,  # 強制重置
        "partial_answer": "",  # 強制重置
        "final_answer": None,  # 強制重置
        "reasoning_trace": [],  # 強制重置
        # 限制和配置
        "max_iterations": max_iterations,
        "token_count": 0,  # 強制重置
        "max_tokens": max_tokens,
        "start_time": time.time(),  # 強制重置為當前時間
        "max_duration": max_duration,
        "decision_history": [],  # 關鍵修復：強制重置，避免舊狀態影響
        "error_history": [],  # 強制重置
        "unavailable_tools": [],  # 類型一致性（不會物理清空，因為 operator.add 的行為是 舊列表 + [] = 舊列表）
        "model_override": state.get("model_override"),
        "current_decision": None,  # 關鍵修復：強制重置當前決策為 None
        "refinement_feedback": None,  # 強制重置 feedback
    }

    logger.info(
        f"State initialized: iteration_count=0, decision_history=[], search_history=[], collected_results=[]"
    )
    return initialized_state


def check_limits(state: ChatAgenticState) -> str:
    """Check if we've exceeded any limits."""
    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        return "limit_exceeded"

    # Check timeout
    elapsed = time.time() - state["start_time"]
    if elapsed > state["max_duration"]:
        return "timeout"

    # Check token limit
    if state["token_count"] >= state["max_tokens"]:
        return "token_limit"

    # Check for circular reasoning
    if detect_circular_reasoning(state):
        return "circular_reasoning"
    
    # 新增：檢查連續工具錯誤
    error_history = state.get("error_history", [])
    if len(error_history) >= 5:
        recent_tool_errors = [
            e for e in error_history[-5:] 
            if e.get("step") == "execute_tool"
        ]
        if len(recent_tool_errors) >= 5:
            # 記錄失敗的工具名稱
            failed_tools = [e.get("tool", "unknown") for e in recent_tool_errors]
            tool_counts = {}
            for tool in failed_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # 詳細記錄所有錯誤信息
            logger.error(
                f"[Iteration {state.get('iteration_count', 0)}] Too many tool errors detected. "
                f"Total error_history length: {len(error_history)}, "
                f"Recent tool errors: {len(recent_tool_errors)}. "
                f"Failed tools: {tool_counts}"
            )
            # 記錄每個失敗工具的詳細信息
            for error in recent_tool_errors:
                logger.error(
                    f"  - Tool: {error.get('tool', 'unknown')}, "
                    f"Error: {error.get('error', 'N/A')[:200]}, "
                    f"Iteration: {error.get('iteration', 'N/A')}"
                )
            return "too_many_tool_errors"

    return "continue"


def detect_circular_reasoning(state: ChatAgenticState) -> bool:
    """Detect if we're stuck in circular reasoning."""
    decision_history = state.get("decision_history", [])
    iteration_count = state.get("iteration_count", 0)

    # 關鍵修復：如果 iteration_count 為 0，說明這是第一次決策，不可能是循環
    if iteration_count == 0:
        return False

    # 至少需要 3 個決策才能檢測循環
    if len(decision_history) < 3:
        return False

    # 關鍵修復：檢查是否有重複的決策模式
    # decision_history 現在記錄的是 "action:tool_name" 格式（例如 "use_tool:vector_search"）
    # 這樣可以區分不同的工具調用，避免誤判為循環
    recent = decision_history[-5:] if len(decision_history) >= 5 else decision_history

    if len(recent) >= 3:
        # 1. 檢查最後 3 次是否完全相同（包括工具名稱）
        if len(set(recent[-3:])) == 1:
            logger.warning(
                f"Circular reasoning detected: last 3 decisions are identical: {recent[-3:]}"
            )
            return True
        
        # 2. 關鍵修復：檢查是否連續 3 次都是 "use_tool" 但工具執行都失敗或沒有結果
        # 這種情況下，即使工具不同，也應該視為循環（因為沒有取得有效結果）
        recent_three = recent[-3:]
        all_use_tool = all(entry.startswith("use_tool:") for entry in recent_three)
        if all_use_tool:
            # 檢查最近的搜索歷史，看是否這些工具調用都失敗或沒有結果
            search_history = state.get("search_history", [])
            recent_searches = search_history[-3:] if search_history else []
            
            # 如果最近 3 次工具調用都失敗或沒有結果，視為循環
            if len(recent_searches) >= 3:
                all_failed_or_empty = all(
                    not s.get("success", False) or s.get("result_count", 0) == 0
                    for s in recent_searches
                )
                if all_failed_or_empty:
                    logger.warning(
                        f"Circular reasoning detected: last 3 tool calls all failed or returned no results: {recent_three}"
                    )
                    return True
        
        # 3. 檢查在 5 次中有 4 次相同（更寬鬆的標準）
        if len(recent) >= 5:
            from collections import Counter

            counter = Counter(recent)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 4:
                logger.warning(
                    f"Circular reasoning detected: {most_common[0][0]} appears {most_common[0][1]} times in last 5 decisions"
                )
                return True

    return False



def _categorize_tool(tool_name: str) -> str:
    """將工具分類為：local_search, external_api, pageindex, calculation, etc."""
    tool_name_lower = tool_name.lower()
    if "pageindex" in tool_name_lower or "page_index" in tool_name_lower:
        return "pageindex"
    elif "vector" in tool_name_lower or "text" in tool_name_lower:
        return "local_search"
    elif "::" in tool_name or "mcp" in tool_name_lower:
        return "external_api"
    elif "calculation" in tool_name_lower or "calc" in tool_name_lower:
        return "calculation"
    elif "internet" in tool_name_lower or "search" in tool_name_lower:
        return "internet_search"
    else:
        return "other"

async def agent_decision(
    state: ChatAgenticState, config: RunnableConfig
) -> Dict[str, Any]:
    """Agent makes decision on next action."""
    iteration = state.get("iteration_count", 0)
    logger.info(f"[Iteration {iteration}] Agent making decision...")

    # 關鍵修復：檢測循環決策
    # 只有在 iteration_count >= 3 時才檢查循環（至少需要 3 次決策才能形成循環）
    decision_history = state.get("decision_history", [])
    if iteration >= 3 and len(decision_history) >= 3:
        # 檢查最近 3 次是否都是 evaluate
        recent_decisions = decision_history[-3:]
        if len(set(recent_decisions)) == 1 and recent_decisions[0] == "evaluate":
            logger.warning(
                f"[Iteration {iteration}] Detected circular evaluate decisions, forcing tool execution or synthesis"
            )
            # 如果有 collected_results，強制 synthesize
            if state.get("collected_results"):
                return {
                    "current_decision": {"action": "synthesize"},
                    "reasoning_trace": [
                        "Circular evaluate detected, forcing synthesis"
                    ],
                    "iteration_count": state["iteration_count"] + 1,
                }
            # 否則強制執行工具
            else:
                return {
                    "current_decision": {
                        "action": "use_tool",
                        "tool_name": "vector_search",
                        "parameters": {"query": state["question"], "limit": 10},
                    },
                    "reasoning_trace": [
                        "Circular evaluate detected, forcing tool execution"
                    ],
                    "iteration_count": state["iteration_count"] + 1,
                }

    # Check limits first
    limit_check = check_limits(state)
    if limit_check != "continue":
        logger.warning(f"[Iteration {iteration}] Limit reached: {limit_check}")
        logger.warning(
            f"State details: decision_history={len(state.get('decision_history', []))}, search_history={len(state.get('search_history', []))}, collected_results={len(state.get('collected_results', []))}"
        )
        
        # 如果有限制，記錄詳細的錯誤信息
        if limit_check == "too_many_tool_errors":
            error_history = state.get("error_history", [])
            recent_tool_errors = [
                e for e in error_history[-5:] 
                if e.get("step") == "execute_tool"
            ]
            failed_tools = [e.get("tool", "unknown") for e in recent_tool_errors]
            tool_counts = {}
            for tool in failed_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            # 構建詳細的錯誤訊息
            error_details = []
            for tool, count in tool_counts.items():
                error_details.append(f"{tool} ({count}次)")
            error_summary = "、".join(error_details)
            
            logger.error(
                f"[Iteration {iteration}] Tool execution failures detected: {error_summary}. "
                f"Recent errors: {recent_tool_errors[-3:]}"
            )
            
            final_answer = (
                state.get("partial_answer")
                or state.get("final_answer")
                or f"無法完成查詢：工具執行連續失敗多次。失敗的工具：{error_summary}。請檢查工具配置或嘗試重新表述問題。"
            )
        else:
            final_answer = (
                state.get("partial_answer")
                or state.get("final_answer")
                or f"Unable to complete due to: {limit_check}"
            )
        return {
            "reasoning_trace": [f"Stopped due to: {limit_check}"],
            "final_answer": final_answer,
        }

    # 新增：檢查 collected_results 的長度和 token 數量，防止無限累積
    collected_results = state.get("collected_results", [])
    collected_results_count = len(collected_results)
    
    # 估算 collected_results 的 token 數量
    collected_results_tokens = 0
    MAX_COLLECTED_RESULTS = 50  # 最大結果數量
    MAX_COLLECTED_TOKENS = 100000  # 最大 token 數量（100k）
    
    if collected_results:
        # 估算每個結果的平均 token 數（基於 content 長度）
        for result in collected_results:
            content = result.get("content", "") or result.get("text", "") or result.get("summary", "")
            if isinstance(content, str):
                collected_results_tokens += token_count(content)
            elif isinstance(content, list):
                # 如果是列表（如 matches），計算所有元素的 token
                for item in content:
                    if isinstance(item, str):
                        collected_results_tokens += token_count(item)
    
    # 如果超過限制，強制選擇 synthesize 或 summarize
    if collected_results_count > MAX_COLLECTED_RESULTS:
        logger.warning(
            f"[Iteration {iteration}] Collected results count ({collected_results_count}) exceeds limit ({MAX_COLLECTED_RESULTS}), forcing synthesize"
        )
        return {
            "current_decision": {"action": "synthesize"},
            "reasoning_trace": [
                f"Collected results count ({collected_results_count}) exceeds limit ({MAX_COLLECTED_RESULTS}), forcing synthesis"
            ],
            "iteration_count": state["iteration_count"] + 1,
        }
    
    if collected_results_tokens > MAX_COLLECTED_TOKENS:
        logger.warning(
            f"[Iteration {iteration}] Collected results tokens ({collected_results_tokens}) exceeds limit ({MAX_COLLECTED_TOKENS}), forcing synthesize"
        )
        return {
            "current_decision": {"action": "synthesize"},
            "reasoning_trace": [
                f"Collected results tokens ({collected_results_tokens}) exceeds limit ({MAX_COLLECTED_TOKENS}), forcing synthesis"
            ],
            "iteration_count": state["iteration_count"] + 1,
        }
    
    # 記錄當前狀態
    if collected_results_count > 0:
        logger.info(
            f"[Iteration {iteration}] Collected results: {collected_results_count} items, ~{collected_results_tokens} tokens"
        )

    # Get available tools
    available_tools = await tool_registry.list_tools()

    # 過濾不可用的工具
    unavailable_tools = state.get("unavailable_tools", [])
    if unavailable_tools:
        logger.warning(f"[Iteration {iteration}] Filtering unavailable tools: {unavailable_tools}")
        available_tools = [
            tool for tool in available_tools 
            if tool.get("name") not in unavailable_tools
        ]

    # 關鍵修復：記錄所有可用工具，特別標記 MCP 工具和 PageIndex
    tool_names = [tool.get("name", "unknown") for tool in available_tools]
    mcp_tools = [
        name
        for name in tool_names
        if "::" in name
        or "mcp" in name.lower()
        or "yahoo" in name.lower()
        or "finance" in name.lower()
    ]
    pageindex_tools = [
        name
        for name in tool_names
        if "pageindex" in name.lower() or "page_index" in name.lower()
    ]
    
    logger.info(
        f"[Iteration {iteration}] Available tools ({len(available_tools)} total): {', '.join(tool_names)}"
    )
    if mcp_tools:
        logger.info(
            f"[Iteration {iteration}] MCP tools detected: {', '.join(mcp_tools)}"
        )
    else:
        logger.warning(
            f"[Iteration {iteration}] No MCP tools detected in available tools list"
        )
    
    if pageindex_tools:
        logger.info(
            f"[Iteration {iteration}] PageIndex tools detected: {', '.join(pageindex_tools)}"
        )
    
    # 增強工具信息：確保每個工具都包含完整的描述和參數 schema
    enhanced_tools = []
    for tool in available_tools:
        enhanced_tool = {
            "name": tool.get("name", "unknown"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
            "category": _categorize_tool(tool.get("name", "")),
        }
        # 特別標記 PageIndex 工具
        if "pageindex" in tool.get("name", "").lower():
            enhanced_tool["special_note"] = "PageIndex tool: Use for structured document search and hierarchical content retrieval"
        enhanced_tools.append(enhanced_tool)
    
    available_tools = enhanced_tools

    # Prepare decision prompt data
    # 包含錯誤歷史，讓 Agent 知道哪些工具失敗了（關鍵修復）
    error_history = state.get("error_history", [])
    recent_errors = error_history[-5:] if error_history else []  # 最近 5 個錯誤

    # 檢查是否有強制停止標記
    for error in recent_errors:
        if error.get("error_details", {}).get("force_stop"):
            logger.error(f"[Iteration {iteration}] Force stop detected due to tool errors")
            return {
                "final_answer": error.get("error_details", {}).get("suggestion", "無法完成查詢"),
                "reasoning_trace": [f"強制停止：{error.get('error', '未知錯誤')}"],
            }

    # 獲取 refinement_feedback（如果存在）
    refinement_feedback = state.get("refinement_feedback")
    
    # Build knowledge base overview (Mental Map) from notebook sources
    knowledge_base_overview = []
    notebook = state.get("notebook")
    if notebook:
        try:
            sources = await notebook.get_sources()
            for source in sources:
                source_info = {
                    "id": source.id,
                    "title": source.title or "Untitled",
                    "summary": None,
                    "topics": source.topics or [],
                    "supports_pageindex": source.pageindex_structure is not None,  # 新增：標記是否支援 PageIndex
                }
                
                # Extract summary from pageindex_structure
                if source.pageindex_structure:
                    # Priority 1: Use doc_description if exists
                    if "doc_description" in source.pageindex_structure:
                        source_info["summary"] = source.pageindex_structure["doc_description"]
                    # Priority 2: Extract summary from root node
                    elif isinstance(source.pageindex_structure, dict):
                        # PageIndex structure format: {"structure": [...]}
                        structure = source.pageindex_structure.get("structure", [])
                        if structure and isinstance(structure, list) and len(structure) > 0:
                            # Get summary from first root node
                            first_node = structure[0]
                            if isinstance(first_node, dict) and "summary" in first_node:
                                source_info["summary"] = first_node["summary"]
                
                knowledge_base_overview.append(source_info)
        except Exception as e:
            logger.warning(f"Failed to build knowledge base overview: {e}")
            knowledge_base_overview = []
    
    prompt_data = {
        "question": state["question"],
        "iteration_count": state["iteration_count"],
        "max_iterations": state["max_iterations"],
        "token_count": state["token_count"],
        "max_tokens": state["max_tokens"],
        "search_history": state["search_history"][-3:],
        "collected_results": collected_results,  # 使用檢查過的 collected_results
        "collected_results_count": collected_results_count,  # 添加計數信息
        "collected_results_tokens": collected_results_tokens,  # 添加 token 信息
        "partial_answer": state.get("partial_answer", ""),
        "reasoning_trace": state["reasoning_trace"][-3:],
        "available_tools": available_tools,
        # 移除 conversation_context，現在使用原生 Messages 格式
        "error_history": recent_errors,  # 添加錯誤歷史，讓 Agent 知道工具失敗
        "decision_history": state.get("decision_history", [])[
            -5:
        ],  # 最近 5 個決策，幫助避免循環
        "refinement_feedback": refinement_feedback,  # 添加 Refiner 的 feedback
        "knowledge_base_overview": knowledge_base_overview,  # Knowledge base overview (Mental Map)
    }

    parser = PydanticOutputParser(pydantic_object=Decision)
    prompt_data["format_instructions"] = parser.get_format_instructions()
    
    # 渲染 System Prompt（純指令，不含對話歷史）
    system_content = Prompter(
        prompt_template="chat_agentic/orchestrator", parser=parser
    ).render(data=prompt_data)
    system_msg = SystemMessage(content=system_content)

    # 構建 Messages 列表：SystemMessage + 歷史對話
    messages_payload = [system_msg] + state["messages"]

    # Track tokens
    tokens = token_count(system_content)
    new_token_count = state["token_count"] + tokens

    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )
        model = await provision_langchain_model(
            str(messages_payload),  # 僅用於日誌
            model_id,
            "orchestrator",
            max_tokens=2000,
            structured=dict(type="json"),
        )

        response = await model.ainvoke(messages_payload)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        cleaned_content = clean_thinking_content(content)

        # Parse decision with retry logic
        max_retries = 3
        decision = None
        parse_error = None
        
        for attempt in range(max_retries):
            try:
                decision = parser.parse(cleaned_content)
                break
            except Exception as e:
                parse_error = e
                logger.warning(
                    f"[Iteration {iteration}] Parser failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                
                if attempt < max_retries - 1:
                    # 將錯誤訊息加入 prompt，讓 LLM 修正
                    prompt_data["parse_error"] = str(e)
                    prompt_data["previous_attempt"] = cleaned_content[:500]  # 保留前500字符作為參考
                    system_content = Prompter(
                        prompt_template="chat_agentic/orchestrator", parser=parser
                    ).render(data=prompt_data)
                    system_msg = SystemMessage(content=system_content)
                    messages_payload = [system_msg] + state["messages"]
                    
                    # 重新生成
                    logger.info(
                        f"[Iteration {iteration}] Retrying decision generation with parse error feedback..."
                    )
                    response = await model.ainvoke(messages_payload)
                    content = (
                        response.content
                        if isinstance(response.content, str)
                        else str(response.content)
                    )
                    cleaned_content = clean_thinking_content(content)
                else:
                    # 最後一次嘗試失敗，使用 fallback
                    logger.error(
                        f"[Iteration {iteration}] Parser failed after {max_retries} attempts, using fallback decision"
                    )
                    # 使用 fallback 決策（默認使用 vector_search）
                    decision = Decision(
                        action="use_tool",
                        tool_name="vector_search",
                        parameters={"query": state["question"], "limit": 10},
                        reasoning=f"Parser failed after {max_retries} attempts: {str(parse_error)}. Using fallback decision."
                    )
        
        if decision is None:
            # 如果所有嘗試都失敗且沒有 fallback，使用默認決策
            logger.error(
                f"[Iteration {iteration}] All parsing attempts failed, using default decision"
            )
            decision = Decision(
                action="use_tool",
                tool_name="vector_search",
                parameters={"query": state["question"], "limit": 10},
                reasoning="Parser failed, using default decision."
            )

        logger.info(
            f"[Iteration {iteration}] Decision: action={decision.action}, tool={decision.tool_name}, reasoning={decision.reasoning[:100] if decision.reasoning else 'None'}..."
        )

        # Update token count (estimate response tokens)
        new_token_count += 500  # Estimate

        # 關鍵修復：decision_history 使用 Annotated[List[str], operator.add]
        # 所以只需要返回要追加的元素，LangGraph 會自動累積
        # 不要手動累積，否則會導致重複累積

        # 追加推理追蹤（同樣使用 operator.add）

        # 關鍵修復：確保 current_decision 被正確設置
        current_decision_dict = {
            "action": decision.action,
            "tool_name": decision.tool_name,
            "parameters": decision.parameters or {},
            "reasoning": decision.reasoning,
        }

        logger.info(
            f"[Iteration {iteration}] Returning state update with current_decision: {current_decision_dict}"
        )

        # 關鍵修復：如果 action 是 finish，將 reasoning 設置為 final_answer
        # 這樣即使圖直接路由到 END，也能返回答案給用戶
        
        # 關鍵修復：decision_history 記錄更詳細的信息，包含 tool_name（如果有的話）
        # 這樣可以區分不同的工具調用，避免誤判為循環
        decision_entry = decision.action
        if decision.action == "use_tool" and decision.tool_name:
            decision_entry = f"{decision.action}:{decision.tool_name}"
        elif decision.action in ["synthesize", "evaluate", "finish"]:
            decision_entry = decision.action
        
        result = {
            "decision_history": [decision_entry],  # 記錄 action:tool_name 格式，LangGraph 會自動累積
            "reasoning_trace": [
                decision.reasoning
            ],  # 只返回新元素，LangGraph 會自動累積
            "token_count": new_token_count,
            "current_decision": current_decision_dict,  # 關鍵修復：設置 current_decision
        }

        if decision.action == "finish" and decision.reasoning:
            # 將 reasoning 作為答案
            reasoning_answer = decision.reasoning.strip()
            result["final_answer"] = reasoning_answer
            result["partial_answer"] = reasoning_answer

            # 創建 AIMessage 包含答案
            from langchain_core.messages import AIMessage

            ai_message = AIMessage(content=reasoning_answer)
            result["messages"] = [ai_message]

            logger.info(
                f"[Iteration {iteration}] Agent chose finish action, setting reasoning as final_answer (length: {len(reasoning_answer)} chars)"
            )

        return result
    except Exception as e:
        logger.error(f"Error in agent decision: {e}")
        # 關鍵修復：error_history 和 reasoning_trace 使用 operator.add
        # 只返回新元素，LangGraph 會自動累積

        return {
            "error_history": [
                {
                    "step": "agent_decision",
                    "error": str(e),
                    "iteration": state["iteration_count"],
                }
            ],  # 只返回新元素
            "reasoning_trace": [f"Decision failed: {str(e)}"],  # 只返回新元素
        }


def route_decision(state: ChatAgenticState) -> str:
    """Route based on agent decision."""
    current_decision = state.get("current_decision")
    iteration = state.get("iteration_count", 0)
    final_answer = state.get("final_answer")

    # 關鍵調試：記錄完整的狀態信息
    logger.info(
        f"[Iteration {iteration}] route_decision called. current_decision type: {type(current_decision)}, value: {current_decision}"
    )
    logger.info(f"[Iteration {iteration}] State keys: {list(state.keys())}")

    # 如果已經有 final_answer，直接路由到 finish
    if final_answer:
        logger.info(
            f"[Iteration {iteration}] final_answer exists, routing to finish"
        )
        return "finish"

    if not current_decision:
        logger.warning(
            f"[Iteration {iteration}] No current_decision found, routing to synthesize"
        )
        logger.warning(
            f"[Iteration {iteration}] decision_history length: {len(state.get('decision_history', []))}"
        )
        return "synthesize"

    action = current_decision.get("action", "synthesize")
    tool_name = current_decision.get("tool_name", "no tool")
    logger.info(f"[Iteration {iteration}] Routing to: {action} (tool: {tool_name})")
    return action


def _flatten_pageindex_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    將 PageIndex 的樹狀 JSON 結構扁平化為 Markdown 格式。
    
    Args:
        result: PageIndex 返回的結果字典
        
    Returns:
        扁平化後的結果字典，包含格式化的 content 欄位
    """
    title = result.get("title", "")
    summary = result.get("summary", "")
    text = result.get("text", "")
    metadata = result.get("metadata", {})
    
    # 構建 Markdown 格式的內容
    markdown_parts = []
    
    if title:
        markdown_parts.append(f"## {title}")
    
    if summary:
        markdown_parts.append(f"\n{summary}")
    
    if text:
        markdown_parts.append(f"\n{text}")
    
    if metadata:
        # 格式化 metadata（例如頁碼、章節等）
        metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if v])
        if metadata_str:
            markdown_parts.append(f"\n\n**Metadata:** {metadata_str}")
    
    # 合併所有部分
    formatted_content = "\n".join(markdown_parts)
    
    # 返回標準化的結果格式
    flattened = {
        "id": result.get("id", result.get("node_id", "")),
        "title": title,
        "content": formatted_content,
        "summary": summary,
        "text": text,
        "metadata": metadata,
        "type": "pageindex",
        "similarity": result.get("similarity", 0.0),
    }
    
    # 保留原始結果中的其他欄位
    for key, value in result.items():
        if key not in flattened:
            flattened[key] = value
    
    return flattened


async def execute_tool(
    state: ChatAgenticState, config: RunnableConfig
) -> Dict[str, Any]:
    """Execute the selected tool with context_config support."""
    current_decision = state.get("current_decision", {})
    tool_name = current_decision.get("tool_name")
    parameters = current_decision.get("parameters", {})

    # [新增] 自動降級邏輯：檢查 PageIndex 支援情況
    if tool_name == "pageindex_search":
        source_ids = parameters.get("source_ids")
        notebook_id = parameters.get("notebook_id")
        
        # 檢查指定的 sources 是否支援 PageIndex
        unsupported_sources = []
        if source_ids:
            notebook = state.get("notebook")
            if notebook:
                try:
                    sources = await notebook.get_sources()
                    source_map = {s.id: s for s in sources}
                    for source_id in source_ids:
                        source = source_map.get(source_id)
                        if source and not source.pageindex_structure:
                            unsupported_sources.append(source_id)
                except Exception as e:
                    logger.warning(f"Failed to check PageIndex support: {e}")
        
        # 如果有不支援的 sources，自動降級到 vector_search
        if unsupported_sources:
            iteration = state.get("iteration_count", 0)
            logger.info(
                f"[Iteration {iteration}] Auto-degrading to vector_search: Sources {unsupported_sources} do not support PageIndex. "
                f"Original tool: {tool_name}, query: {parameters.get('query', 'N/A')}"
            )
            tool_name = "vector_search"
            # 確保參數兼容（vector_search 使用 query 和 limit）
            parameters = {
                "query": parameters.get("query", ""),
                "limit": parameters.get("limit", 10),
            }
            # 更新 decision 以便後續邏輯使用
            current_decision = {
                **current_decision,
                "tool_name": tool_name,
                "parameters": parameters,
                "reasoning": f"Auto-degraded from pageindex_search to vector_search: Sources {unsupported_sources} do not support PageIndex. " + current_decision.get("reasoning", ""),
            }

    # 添加日誌以追蹤工具執行（關鍵修復）
    iteration = state.get("iteration_count", 0)
    logger.info(
        f"[Iteration {iteration}] Executing tool: {tool_name} with parameters: {list(parameters.keys())}"
    )

    if not tool_name:
        # 關鍵修復：也要記錄錯誤的工具調用
        error_tool_call = {
            "tool_name": "unknown",
            "success": False,
            "data": [],
            "error": "No tool name specified",
            "execution_time": 0.0,
            "metadata": {
                "error_type": "ValueError",
                "error_message": "No tool name specified",
            },
            "error_details": {
                "reason": "工具名稱未指定",
                "suggestion": "請檢查決策邏輯",
            },
        }
        return {
            "error_history": [
                {
                    "step": "execute_tool",
                    "error": "No tool name specified",
                    "tool": "unknown",
                    "iteration": state["iteration_count"],
                    "error_details": {
                        "reason": "工具名稱未指定",
                        "suggestion": "請檢查決策邏輯",
                    },
                }
            ],
            "current_tool_calls": [error_tool_call],  # 關鍵修復：記錄錯誤的工具調用
        }

    # 處理 context_config：限制本地搜尋工具的範圍
    context_config = state.get("context_config", {})
    sources_config = context_config.get("sources", {}) if context_config else {}

    # 關鍵修復：處理 sources_config 可能是 dict 或 list 的情況
    selected_source_ids = []

    if isinstance(sources_config, dict):
        # 格式：{source_id: status}
        selected_source_ids = [
            source_id
            for source_id, status in sources_config.items()
            if "not in" not in str(status)
        ]
    elif isinstance(sources_config, list):
        # 格式：[{'id': source_id, 'title': ..., ...}, ...]
        # 或者可能是其他格式，需要從錯誤訊息推斷
        # 從錯誤訊息看，格式是 [{'id': 'source:...', 'title': '...'}, ...]
        selected_source_ids = [
            item.get("id")
            for item in sources_config
            if isinstance(item, dict) and item.get("id")
        ]
        logger.info(
            f"[Iteration {iteration}] sources_config is a list, extracted {len(selected_source_ids)} source_ids"
        )
    else:
        logger.warning(
            f"[Iteration {iteration}] sources_config has unexpected type: {type(sources_config)}, value: {sources_config}"
        )

    # 對於本地搜尋工具，應用範圍限制
    if tool_name in ["vector_search", "text_search"]:
        if selected_source_ids:
            # 有選定的 sources：需要獲取對應的 notebook_ids
            # 從 source_id 提取 notebook_id（需要查詢數據庫）
            try:
                from open_notebook.domain.notebook import Source

                notebook_ids = []
                for source_id in selected_source_ids:
                    full_source_id = (
                        source_id
                        if source_id.startswith("source:")
                        else f"source:{source_id}"
                    )
                    try:
                        source = await Source.get(full_source_id)
                        if source and hasattr(source, "notebook_ids"):
                            # 如果 source 有 notebook_ids 屬性
                            notebook_ids.extend(getattr(source, "notebook_ids", []))
                        elif source:
                            # 否則從 reference 關係查找
                            from open_notebook.database.repository import (
                                ensure_record_id,
                                repo_query,
                            )

                            refs = await repo_query(
                                "SELECT out as notebook_id FROM reference WHERE in = $source_id",
                                {"source_id": ensure_record_id(full_source_id)},
                            )
                            if refs:
                                notebook_ids.extend(
                                    [
                                        ref.get("notebook_id")
                                        for ref in refs
                                        if ref.get("notebook_id")
                                    ]
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to get notebook_id for source {source_id}: {e}"
                        )
                        continue

                if notebook_ids:
                    # 去重
                    notebook_ids = list(set(notebook_ids))
                    parameters["notebook_ids"] = notebook_ids
                    logger.debug(f"Limiting search to notebook_ids: {notebook_ids}")
            except Exception as e:
                logger.warning(
                    f"Failed to process context_config for tool {tool_name}: {e}"
                )
                # 繼續執行，不限制範圍
        else:
            # 沒有選定的 sources：傳 None，搜尋所有
            parameters["notebook_ids"] = None
    
    # 關鍵修復：為 pageindex_search 處理 context_config
    elif tool_name == "pageindex_search":
        if selected_source_ids:
            # PageIndex 工具接受 source_ids 參數（複數）
            parameters["source_ids"] = selected_source_ids
            logger.info(
                f"[Iteration {iteration}] PageIndex search: Using source_ids from context_config ({len(selected_source_ids)} sources)"
            )
        else:
            # 如果沒有選定的 sources，嘗試從 state 中獲取 notebook_id
            # 優先使用 state 中的 notebook，因為這是最可靠的來源
            notebook = state.get("notebook")
            if notebook:
                if isinstance(notebook, dict) and notebook.get("id"):
                    parameters["notebook_id"] = notebook.get("id")
                    logger.info(
                        f"[Iteration {iteration}] PageIndex search: Using notebook_id from state (dict): {notebook.get('id')}"
                    )
                elif hasattr(notebook, 'id'):
                    parameters["notebook_id"] = notebook.id
                    logger.info(
                        f"[Iteration {iteration}] PageIndex search: Using notebook_id from state (object): {notebook.id}"
                    )
            else:
                # 如果 state 中沒有 notebook，記錄警告
                logger.warning(
                    f"[Iteration {iteration}] PageIndex search: No notebook in state and no source_ids selected. "
                    f"This may cause parameter validation to fail."
                )

    # 關鍵修復：清理參數，將字符串 'None' 轉換為 None
    # 這可能發生在 LLM 生成決策時將 None 序列化為字符串
    cleaned_parameters = {}
    for key, value in parameters.items():
        if isinstance(value, str) and value.lower() == 'none':
            # 跳過 None 值（不添加到參數中）
            continue
        elif isinstance(value, str) and value == 'None':
            # 跳過字符串 'None'
            continue
        else:
            cleaned_parameters[key] = value
    
    # 對於 pageindex_search，特別處理 notebook_ids 和 notebook_id 參數
    if tool_name == "pageindex_search":
        # 移除無效的 notebook_ids 參數（如果是字符串 'None'）
        if "notebook_ids" in cleaned_parameters:
            notebook_ids_value = cleaned_parameters.get("notebook_ids")
            if notebook_ids_value == 'None' or notebook_ids_value == 'none':
                del cleaned_parameters["notebook_ids"]
            elif notebook_ids_value is None:
                del cleaned_parameters["notebook_ids"]
        
        # 移除無效的 notebook_id 參數（如果是字符串 'None'）
        if "notebook_id" in cleaned_parameters:
            notebook_id_value = cleaned_parameters.get("notebook_id")
            if notebook_id_value == 'None' or notebook_id_value == 'none':
                del cleaned_parameters["notebook_id"]
                logger.warning(
                    f"[Iteration {iteration}] PageIndex search: Removed invalid notebook_id='None' string"
                )
            elif notebook_id_value is None:
                del cleaned_parameters["notebook_id"]
        
        # 確保至少有一個有效的參數：notebook_id, source_ids, 或 document_path
        has_valid_param = (
            cleaned_parameters.get("notebook_id") or
            cleaned_parameters.get("source_ids") or
            cleaned_parameters.get("document_path")
        )
        
        # 如果沒有有效參數，嘗試從 state 中獲取 notebook_id（作為最後的 fallback）
        if not has_valid_param:
            notebook = state.get("notebook")
            if notebook:
                if isinstance(notebook, dict) and notebook.get("id"):
                    cleaned_parameters["notebook_id"] = notebook.get("id")
                    logger.info(
                        f"[Iteration {iteration}] PageIndex search: Fallback - Using notebook_id from state (dict): {notebook.get('id')}"
                    )
                elif hasattr(notebook, 'id'):
                    cleaned_parameters["notebook_id"] = notebook.id
                    logger.info(
                        f"[Iteration {iteration}] PageIndex search: Fallback - Using notebook_id from state (object): {notebook.id}"
                    )
            else:
                logger.error(
                    f"[Iteration {iteration}] PageIndex search: No valid parameters found and no notebook in state. "
                    f"Parameters: {list(cleaned_parameters.keys())}, "
                    f"State notebook: {bool(state.get('notebook'))}"
                )
        else:
            # 記錄有效的參數
            logger.info(
                f"[Iteration {iteration}] PageIndex search: Valid parameters found - "
                f"notebook_id={bool(cleaned_parameters.get('notebook_id'))}, "
                f"source_ids={bool(cleaned_parameters.get('source_ids'))}, "
                f"document_path={bool(cleaned_parameters.get('document_path'))}"
            )
    
    parameters = cleaned_parameters

    try:
        # 關鍵修復：添加工具執行前後的日誌，追蹤執行時間
        tool_start_time = time.time()
        logger.info(
            f"[Iteration {iteration}] Tool {tool_name} execution started at {tool_start_time}"
        )
        logger.info(
            f"[Iteration {iteration}] Tool parameters: {list(parameters.keys())} = {dict((k, str(v)[:100] if not isinstance(v, (str, int, float, bool)) else v) for k, v in parameters.items())}"
        )

        # 關鍵修復：在調用 tool_registry.execute_tool 之前添加日誌
        logger.info(
            f"[Iteration {iteration}] About to call tool_registry.execute_tool({tool_name})"
        )

        # 添加超時保護（如果工具本身沒有超時保護）
        try:
            logger.info(
                f"[Iteration {iteration}] Creating asyncio.wait_for with timeout=60.0s for tool {tool_name}"
            )
            result = await asyncio.wait_for(
                tool_registry.execute_tool(tool_name, **parameters),
                timeout=60.0,  # 工具執行最多 60 秒
            )
            logger.info(
                f"[Iteration {iteration}] asyncio.wait_for completed for tool {tool_name}"
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[Iteration {iteration}] Tool {tool_name} execution timed out after 60s (asyncio.wait_for timeout)"
            )
            result = {
                "tool_name": tool_name,
                "success": False,
                "data": [],
                "error": "Tool execution timed out after 60 seconds",
                "execution_time": 60.0,
                "metadata": {
                    "error_type": "TimeoutError",
                    "error_message": "Tool execution timed out",
                    "query": parameters.get("query", ""),
                },
                "error_details": {
                    "reason": "工具執行超時",
                    "suggestion": "請稍後再試或簡化查詢",
                },
            }

        tool_end_time = time.time()
        execution_duration = tool_end_time - tool_start_time
        logger.info(
            f"[Iteration {iteration}] Tool {tool_name} execution completed in {execution_duration:.2f}s, success={result.get('success', False)}"
        )
        if not result.get("success"):
            logger.warning(
                f"[Iteration {iteration}] Tool {tool_name} failed: {result.get('error', 'Unknown error')}"
            )

        # Record search history
        search_entry = {
            "query": parameters.get("query", ""),
            "tool": tool_name,
            "result_count": len(result.get("data", [])) if result.get("success") else 0,
            "success": result.get("success", False),
            "timestamp": time.time(),
        }

        # 如果工具執行失敗，將錯誤信息作為結果返回
        if not result.get("success"):
            error_result = {
                "id": f"error_{tool_name}_{int(time.time())}",
                "type": "error",
                "title": f"工具執行失敗: {tool_name}",
                "content": result.get("error", "未知錯誤"),
                "error_details": result.get("error_details", {}),
                "tool_name": tool_name,
                "query": parameters.get("query", ""),
            }

            # 更新 search_entry 以包含錯誤信息
            search_entry["error"] = result.get("error")
            search_entry["error_details"] = result.get("error_details", {})

            # 關鍵修復：只返回錯誤結果，利用 operator.add 自動累積
            return {
                "search_history": [search_entry],
                "collected_results": [error_result],  # 只返回錯誤結果
                "current_tool_calls": [result],
                "iteration_count": state["iteration_count"] + 1,
                "error_history": [
                    {  # 只返回新元素，LangGraph 會自動累積
                        "step": "execute_tool",
                        "error": result.get("error", "工具執行失敗"),
                        "tool": tool_name,
                        "iteration": state["iteration_count"],
                        "error_details": result.get("error_details", {}),
                    }
                ],
            }

        # Collect results if successful
        # 關鍵修復：只返回新增的結果，利用 operator.add 自動累積
        tool_data = result.get("data", [])

        # 關鍵修復：詳細驗證工具返回的數據格式
        if tool_data:
            logger.info(
                f"[Iteration {iteration}] Tool {tool_name} returned {len(tool_data)} results"
            )

            # 驗證結果格式
            valid_results = []
            invalid_results = []
            missing_fields_count = {"id": 0, "content": 0, "matches": 0}
            
            # 檢查是否是 PageIndex 工具
            is_pageindex = "pageindex" in tool_name.lower() or "page_index" in tool_name.lower()

            for idx, item in enumerate(tool_data):
                if not isinstance(item, dict):
                    logger.warning(
                        f"[Iteration {iteration}] Tool {tool_name} result {idx} is not a dict: {type(item)}"
                    )
                    invalid_results.append(idx)
                    continue

                # 如果是 PageIndex 結果，使用適配器扁平化
                if is_pageindex:
                    try:
                        item = _flatten_pageindex_result(item)
                        logger.debug(
                            f"[Iteration {iteration}] Tool {tool_name} result {idx} flattened from PageIndex format"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Iteration {iteration}] Failed to flatten PageIndex result {idx}: {e}"
                        )
                        # 繼續處理，不中斷

                # 檢查必要字段
                has_id = bool(item.get("id"))
                has_content = bool(item.get("content"))
                has_matches = bool(item.get("matches"))

                if not has_id:
                    missing_fields_count["id"] += 1
                if not has_content and not has_matches:
                    missing_fields_count["content"] += 1
                    missing_fields_count["matches"] += 1

                # 記錄結果格式樣本（前3個）
                if idx < 3:
                    logger.debug(
                        f"[Iteration {iteration}] Tool {tool_name} result {idx} format: keys={list(item.keys())}, has_id={has_id}, has_content={has_content}, has_matches={has_matches}"
                    )

                valid_results.append(item)

            # 記錄驗證統計
            if missing_fields_count["id"] > 0 or missing_fields_count["content"] > 0:
                logger.warning(
                    f"[Iteration {iteration}] Tool {tool_name} result validation: {missing_fields_count['id']} missing id, {missing_fields_count['content']} missing content/matches"
                )

            if invalid_results:
                logger.warning(
                    f"[Iteration {iteration}] Tool {tool_name}: {len(invalid_results)} invalid results (not dict)"
                )

            # 過濾重複結果（基於 result.id）
            new_results = []
            # 使用 set 提高查找效率
            existing_ids = {
                r.get("id")
                for r in state["collected_results"]
                if r.get("id")
            }
            seen_ids_in_batch = set()  # 用於檢測同一批次內的重複
            
            for item in valid_results:
                item_id = item.get("id")
                if item_id:
                    # 檢查是否在已收集的結果中
                    if item_id not in existing_ids:
                        # 檢查是否在同一批次中重複
                        if item_id not in seen_ids_in_batch:
                            new_results.append(item)
                            seen_ids_in_batch.add(item_id)
                        else:
                            logger.debug(
                                f"[Iteration {iteration}] Tool {tool_name}: Skipping duplicate result in batch: {item_id[:50]}"
                            )
                    else:
                        logger.debug(
                            f"[Iteration {iteration}] Tool {tool_name}: Skipping duplicate result (already in collected_results): {item_id[:50]}"
                        )
                else:
                    # 如果沒有ID，直接添加（避免重複檢查）
                    # 但可以基於內容的 hash 進行去重（可選）
                    new_results.append(item)

            logger.info(
                f"[Iteration {iteration}] Tool {tool_name}: {len(new_results)} new results (after deduplication from {len(valid_results)} valid)"
            )

            if new_results:
                sample = new_results[0]
                logger.debug(
                    f"[Iteration {iteration}] Tool {tool_name} sample new result: keys={list(sample.keys())}, id={sample.get('id', 'N/A')[:50]}"
                )

            return {
                "search_history": [search_entry],
                "collected_results": new_results,  # 只返回新結果
                "current_tool_calls": [result],
                "iteration_count": state["iteration_count"] + 1,
            }
        else:
            # 關鍵修復：詳細記錄為什麼返回空列表
            logger.warning(f"[Iteration {iteration}] Tool {tool_name} returned no data")
            logger.warning(
                f"[Iteration {iteration}] Tool {tool_name} result structure: success={result.get('success')}, has_data={bool(result.get('data'))}, data_type={type(result.get('data')).__name__ if result.get('data') else 'None'}"
            )
            if result.get("error"):
                logger.warning(
                    f"[Iteration {iteration}] Tool {tool_name} error: {result.get('error')}"
                )
            if result.get("error_details"):
                logger.warning(
                    f"[Iteration {iteration}] Tool {tool_name} error_details: {result.get('error_details')}"
                )

            return {
                "search_history": [search_entry],
                "collected_results": [],  # 空列表，不會累積
                "current_tool_calls": [result],
                "iteration_count": state["iteration_count"] + 1,
            }
    except ToolNotFoundError as e:
        # 專門處理工具不存在的情況
        tool_end_time = time.time()
        execution_duration = (
            tool_end_time - tool_start_time if "tool_start_time" in locals() else 0
        )
        logger.error(
            f"[Iteration {iteration}] Tool {tool_name} not found after {execution_duration:.2f}s: {e}"
        )
        
        # 記錄不可用的工具
        unavailable_tools = state.get("unavailable_tools", [])
        new_unavailable_tool = None
        if tool_name not in unavailable_tools:
            new_unavailable_tool = tool_name  # 只返回新工具，LangGraph 會自動累積
        
        # 檢查是否連續失敗多次
        error_history = state.get("error_history", [])
        recent_errors = error_history[-3:] if len(error_history) >= 3 else error_history
        same_tool_errors = [err for err in recent_errors if err.get("tool") == tool_name]
        
        if len(same_tool_errors) >= 2:  # 連續 3 次失敗（包括這次）
            logger.error(f"[Iteration {iteration}] Tool {tool_name} failed 3 times consecutively, forcing stop")
            return {
                "unavailable_tools": [new_unavailable_tool] if new_unavailable_tool else [],
                "error_history": [{
                    "step": "execute_tool",
                    "error": f"Tool {tool_name} not found (failed 3 times)",
                    "tool": tool_name,
                    "iteration": state["iteration_count"],
                    "error_details": {
                        "reason": "工具不存在且連續失敗多次",
                        "suggestion": "請使用其他可用工具",
                        "force_stop": True
                    }
                }],
                "final_answer": f"無法完成查詢：工具 {tool_name} 不可用。請嘗試使用其他工具或重新表述問題。",
                "reasoning_trace": [f"工具 {tool_name} 不存在，已強制停止"],
                "iteration_count": state["iteration_count"] + 1,
            }
        
        # 正常錯誤處理
        error_result = {
            "id": f"error_{tool_name}_{int(time.time())}",
            "type": "error",
            "title": f"工具不可用: {tool_name}",
            "content": f"工具 {tool_name} 不存在或未配置",
            "error_details": {
                "reason": "工具不存在",
                "suggestion": "請使用其他可用工具",
            },
            "tool_name": tool_name,
            "query": parameters.get("query", ""),
        }
        
        return {
            "unavailable_tools": [new_unavailable_tool] if new_unavailable_tool else [],
            "search_history": [{
                "query": parameters.get("query", ""),
                "tool": tool_name,
                "result_count": 0,
                "success": False,
                "error": f"Tool {tool_name} not found",
                "timestamp": time.time(),
            }],
            "collected_results": [error_result],
            "current_tool_calls": [{
                "tool_name": tool_name,
                "success": False,
                "data": [],
                "error": f"Tool {tool_name} not found",
                "execution_time": execution_duration,
            }],
            "error_history": [{
                "step": "execute_tool",
                "error": f"Tool {tool_name} not found",
                "tool": tool_name,
                "iteration": state["iteration_count"],
                "error_details": {
                    "reason": "工具不存在",
                    "suggestion": "請使用其他可用工具",
                }
            }],
            "iteration_count": state["iteration_count"] + 1,
        }
    except Exception as e:
        tool_end_time = time.time()
        execution_duration = (
            tool_end_time - tool_start_time if "tool_start_time" in locals() else 0
        )
        logger.error(
            f"[Iteration {iteration}] Error executing tool {tool_name} after {execution_duration:.2f}s: {e}"
        )
        logger.exception(e)  # 記錄完整堆棧
        # 即使工具執行器本身出錯，也要返回錯誤結果
        error_result = {
            "id": f"error_{tool_name}_{int(time.time())}",
            "type": "error",
            "title": f"工具執行失敗: {tool_name}",
            "content": str(e),
            "error_details": {
                "reason": "工具執行器異常",
                "suggestion": "請檢查工具配置或嘗試其他工具",
            },
            "tool_name": tool_name,
            "query": parameters.get("query", ""),
        }

        # 關鍵修復：error_history 使用 operator.add，只返回新元素
        # 關鍵修復：也要記錄失敗的工具調用到 current_tool_calls
        error_tool_call = {
            "tool_name": tool_name,
            "success": False,
            "data": [],
            "error": str(e),
            "execution_time": execution_duration,
            "metadata": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query": parameters.get("query", ""),
            },
            "error_details": {
                "reason": "工具執行器異常",
                "suggestion": "請檢查工具配置或嘗試其他工具",
            },
        }

        return {
            "error_history": [
                {  # 只返回新元素，LangGraph 會自動累積
                    "step": "execute_tool",
                    "error": str(e),
                    "tool": tool_name,
                    "iteration": state["iteration_count"],
                    "error_details": {
                        "reason": "工具執行器異常",
                        "suggestion": "請檢查工具配置或嘗試其他工具",
                    },
                }
            ],
            "search_history": [
                {  # 只返回新元素，LangGraph 會自動累積
                    "query": parameters.get("query", ""),
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time(),
                }
            ],
            "current_tool_calls": [error_tool_call],  # 關鍵修復：記錄失敗的工具調用
            "collected_results": [
                error_result
            ],  # 關鍵修復：只返回錯誤結果，利用 operator.add 自動累積
            "iteration_count": state["iteration_count"] + 1,
        }


async def evaluate_results(
    state: ChatAgenticState, config: RunnableConfig
) -> Dict[str, Any]:
    """Evaluate current results and decide next step with hallucination detection."""
    try:
        # 關鍵修復：驗證 collected_results 的數據格式
        collected_results = state.get("collected_results", [])
        iteration = state.get("iteration_count", 0)
        logger.info(
            f"[Iteration {iteration}] Evaluating {len(collected_results)} results"
        )

        # 記錄數據格式樣本（用於調試）
        if collected_results:
            sample = collected_results[0]
            logger.debug(
                f"Sample result format: keys={list(sample.keys())}, has_content={bool(sample.get('content') or sample.get('matches'))}"
            )

        # 如果沒有結果，強制執行工具而不是繼續評估
        if not collected_results:
            logger.warning("No results to evaluate, forcing tool execution")
            return {
                "evaluation_result": {
                    "decision": "continue",
                    "reasoning": "No results available for evaluation, need to search first",
                    "combined_score": 0.0,
                    "rule_score": 0.0,
                    "llm_score": 0.0,
                },
                "current_decision": {
                    "action": "use_tool",
                    "tool_name": "vector_search",
                    "parameters": {"query": state["question"], "limit": 10},
                },
                "iteration_count": state["iteration_count"] + 1,
                "reasoning_trace": ["No results available, forcing tool execution"],
            }

        partial_answer = state.get("partial_answer", "")

        # 如果有 partial_answer，進行 hallucination 檢測
        hallucination_check = None
        if partial_answer:
            hallucination_check = await evaluation_service.detect_hallucination(
                answer=partial_answer, results=state["collected_results"]
            )

            # 如果檢測到高風險，調整決策
            if hallucination_check.get("has_hallucination_risk"):
                risk_score = hallucination_check.get("hallucination_risk_score", 0.0)
                if risk_score > 0.5:
                    # 高風險，需要繼續搜尋
                    return {
                        "hallucination_check": hallucination_check,
                        "evaluation_result": {
                            "decision": "continue",
                            "reasoning": f"High hallucination risk detected (score: {risk_score:.2f}), need more information",
                            "combined_score": 0.3,  # 低分數，觸發繼續搜尋
                        },
                        "current_decision": {"action": "continue"},
                        "reasoning_trace": [
                            f"Hallucination risk detected: {risk_score:.2f}. Continuing search for more reliable information."
                        ],
                        "iteration_count": state["iteration_count"]
                        + 1,  # 關鍵修復：增加迭代計數
                    }
                elif risk_score > 0.3:
                    # 中等風險，優化搜尋
                    return {
                        "hallucination_check": hallucination_check,
                        "evaluation_result": {
                            "decision": "refine_search",
                            "reasoning": f"Moderate hallucination risk (score: {risk_score:.2f}), refining search",
                            "combined_score": 0.5,
                        },
                        "current_decision": {"action": "refine_search"},
                        "reasoning_trace": [
                            f"Moderate hallucination risk: {risk_score:.2f}. Refining search strategy."
                        ],
                        "iteration_count": state["iteration_count"]
                        + 1,  # 關鍵修復：增加迭代計數
                    }

        # 正常評估流程
        eval_result = await evaluation_service.evaluate_results(
            question=state["question"],
            results=collected_results,  # 使用驗證過的 collected_results
            answer=partial_answer,
            model_id=config.get("configurable", {}).get("evaluation_model"),
        )

        decision = eval_result.get("decision", "continue")

        return {
            "evaluation_result": eval_result,
            "hallucination_check": hallucination_check,
            "current_decision": {"action": decision},
            "reasoning_trace": [f"Evaluation: {eval_result.get('reasoning', '')}"],
            "iteration_count": state["iteration_count"] + 1,  # 關鍵修復：增加迭代計數
        }
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        logger.exception(e)
        # Fallback to simple decision
        result_count = len(state.get("collected_results", []))
        decision = "synthesize" if result_count >= 3 else "continue"
        return {
            "evaluation_result": {
                "decision": decision,
                "reasoning": "Evaluation failed, using fallback",
                "combined_score": 0.5 if result_count >= 3 else 0.0,
                "rule_score": 0.5 if result_count >= 3 else 0.0,
                "llm_score": 0.5,
            },
            "current_decision": {"action": decision},
            "iteration_count": state["iteration_count"]
            + 1,  # 關鍵修復：即使錯誤也要增加
        }


async def refine_query(
    state: ChatAgenticState, config: RunnableConfig
) -> Dict[str, Any]:
    """Refine search query based on previous attempts."""
    prompt_data = {
        "question": state["question"],
        "search_history": state["search_history"][-5:],
        "collected_results": state["collected_results"],
        # 移除 conversation_context，現在使用原生 Messages 格式
    }

    parser = PydanticOutputParser(pydantic_object=Decision)
    system_content = Prompter(
        prompt_template="chat_agentic/self_correction", parser=parser
    ).render(data=prompt_data)
    system_msg = SystemMessage(content=system_content)

    # 構建 Messages 列表：SystemMessage + 歷史對話
    messages_payload = [system_msg] + state["messages"]

    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )
        model = await provision_langchain_model(
            str(messages_payload),  # 僅用於日誌
            model_id,
            "correction",
            max_tokens=1000,
            structured=dict(type="json"),
        )

        response = await model.ainvoke(messages_payload)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        cleaned_content = clean_thinking_content(content)

        refined_decision = parser.parse(cleaned_content)

        return {
            "current_decision": {
                "action": "use_tool",
                "tool_name": refined_decision.tool_name,
                "parameters": refined_decision.parameters or {},
                "reasoning": refined_decision.reasoning,
            },
            "reasoning_trace": [f"Refined query: {refined_decision.reasoning}"],
        }
    except Exception as e:
        logger.error(f"Error refining query: {e}")
        return {
            "current_decision": {"action": "evaluate"},
            "reasoning_trace": [f"Query refinement failed: {str(e)}"],
        }


def _build_fallback_answer(
    state: ChatAgenticState,
    iteration: int,
    error: Optional[Exception] = None,
) -> str:
    """
    構建 fallback 答案，當 LLM 返回空內容或發生錯誤時使用。

    Args:
        state: 當前狀態
        iteration: 當前迭代次數
        error: 可選的錯誤對象

    Returns:
        構建好的 fallback 答案字符串
    """
    collected_results = state.get("collected_results", [])
    question = state.get("question", "")

    # 優先使用 partial_answer
    if state.get("partial_answer") and len(state.get("partial_answer", "").strip()) > 0:
        fallback_answer = state.get("partial_answer").strip()
        logger.warning(
            f"[Iteration {iteration}] _build_fallback_answer: Using partial_answer as fallback (length: {len(fallback_answer)} chars)"
        )
        return fallback_answer

    # 如果沒有 partial_answer，嘗試從 collected_results 構建答案
    if collected_results and len(collected_results) > 0:
        logger.warning(
            f"[Iteration {iteration}] _build_fallback_answer: Attempting to build answer from {len(collected_results)} collected results"
        )
        try:
            # 構建一個簡單的答案摘要
            answer_parts = [f"根據收集到的 {len(collected_results)} 個相關結果："]
            for i, result in enumerate(collected_results[:3], 1):  # 最多使用前 3 個結果
                title = result.get("title", result.get("id", "未知來源"))
                content = result.get("content", "")
                if not content and result.get("matches"):
                    if isinstance(result["matches"], list) and result["matches"]:
                        content = (
                            result["matches"][0]
                            if isinstance(result["matches"][0], str)
                            else str(result["matches"][0])
                        )
                if content:
                    preview = content[:200] if len(content) > 200 else content
                    answer_parts.append(f"\n{i}. {title}: {preview}...")

            if len(collected_results) > 3:
                answer_parts.append(
                    f"\n... 還有 {len(collected_results) - 3} 個相關結果。"
                )

            answer_parts.append(
                f"\n\n抱歉，由於技術問題無法生成完整的答案。請嘗試重新發送請求。"
            )
            fallback_answer = "".join(answer_parts)
            logger.warning(
                f"[Iteration {iteration}] _build_fallback_answer: Built answer from collected_results (length: {len(fallback_answer)} chars)"
            )
            return fallback_answer
        except Exception as build_error:
            logger.error(
                f"[Iteration {iteration}] _build_fallback_answer: Failed to build answer from collected_results: {build_error}"
            )
            return f"抱歉，無法生成答案。已收集 {len(collected_results)} 個相關結果，但由於技術問題無法完成答案生成。請嘗試重新發送請求。"

    # 最後的 fallback：使用通用錯誤消息
    if error:
        fallback_answer = (
            f"抱歉，無法生成答案。錯誤：{str(error)}。請嘗試重新發送請求。"
        )
    else:
        fallback_answer = "抱歉，無法生成答案。請嘗試重新發送請求。"

    logger.error(
        f"[Iteration {iteration}] _build_fallback_answer: All fallback methods exhausted, using generic error message"
    )
    return fallback_answer


async def synthesize_answer(
    state: ChatAgenticState, config: RunnableConfig
) -> Dict[str, Any]:
    """Generate final answer from collected results with hallucination detection."""
    iteration = state["iteration_count"]
    max_iterations = state["max_iterations"]
    collected_results_count = len(state.get("collected_results", []))

    logger.info(
        f"[Iteration {iteration}] synthesize_answer: Starting synthesis. Collected results: {collected_results_count}, max_iterations: {max_iterations}"
    )

    prompt_data = {
        "question": state["question"],
        "collected_results": state["collected_results"],
        # 移除 conversation_context，現在使用原生 Messages 格式
    }

    system_content = Prompter(prompt_template="chat_agentic/refiner").render(
        data=prompt_data
    )
    system_msg = SystemMessage(content=system_content)

    # 構建 Messages 列表：SystemMessage + 歷史對話
    messages_payload = [system_msg] + state["messages"]

    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )
        model = await provision_langchain_model(
            str(messages_payload),  # 僅用於日誌
            model_id,
            "refiner",
            max_tokens=4000,
        )

        logger.info(
            f"[Iteration {iteration}] synthesize_answer: Invoking LLM for answer generation..."
        )
        logger.debug(
            f"[Iteration {iteration}] synthesize_answer: Prompt length: {len(system_content)} chars"
        )

        response = await model.ainvoke(messages_payload)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        # 關鍵修復：使用 ERROR 級別記錄原始 content，確保一定會顯示（即使日誌級別設置較高）
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: ===== RAW LLM RESPONSE START ====="
        )
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: Response type: {type(response.content)}, Content length: {len(content)} chars"
        )
        if content:
            # 檢查是否包含思考標籤（系統使用 <think> 標籤）
            has_redacted_tags = "<think>" in content
            logger.error(
                f"[Iteration {iteration}] synthesize_answer: Contains <think> tags: {has_redacted_tags}"
            )
            # 記錄完整的原始內容（至少前 1000 字元）
            preview_length = min(1000, len(content))
            logger.error(
                f"[Iteration {iteration}] synthesize_answer: Raw content preview (first {preview_length} chars):\n{content[:preview_length]}"
            )
            if len(content) > preview_length:
                logger.error(
                    f"[Iteration {iteration}] synthesize_answer: ... (truncated, total {len(content)} chars)"
                )
        else:
            logger.error(
                f"[Iteration {iteration}] synthesize_answer: LLM returned EMPTY or None content!"
            )
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: ===== RAW LLM RESPONSE END ====="
        )

        answer = clean_thinking_content(content)

        # 關鍵修復：使用 ERROR 級別記錄清理後的結果，確保一定會顯示
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: ===== AFTER clean_thinking_content ====="
        )
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: Cleaned answer length: {len(answer)} chars"
        )
        if answer:
            preview_length = min(500, len(answer))
            logger.error(
                f"[Iteration {iteration}] synthesize_answer: Cleaned answer preview (first {preview_length} chars):\n{answer[:preview_length]}"
            )
        else:
            logger.error(
                f"[Iteration {iteration}] synthesize_answer: Cleaned answer is EMPTY!"
            )
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: ===== END clean_thinking_content ====="
        )

        # 關鍵修復：如果清理後為空，記錄警告並嘗試提取思考內容
        if not answer or len(answer.strip()) == 0:
            logger.warning(
                f"[Iteration {iteration}] synthesize_answer: Cleaned answer is empty! Original content length: {len(content)}"
            )

            # 嘗試使用 parse_thinking_content 直接提取思考內容
            from open_notebook.utils import parse_thinking_content

            thinking_content, _ = parse_thinking_content(content)

            if thinking_content and len(thinking_content.strip()) > 0:
                logger.warning(
                    f"[Iteration {iteration}] synthesize_answer: Using thinking content as answer (length: {len(thinking_content)} chars)"
                )
                answer = thinking_content.strip()
            elif content and len(content.strip()) > 0:
                # 如果沒有思考標籤但原始內容有內容，使用原始內容
                logger.warning(
                    f"[Iteration {iteration}] synthesize_answer: Using original content as answer fallback (length: {len(content)} chars)"
                )
                answer = content.strip()
            else:
                # 最後的 fallback：使用 fallback 函數構建答案
                logger.error(
                    f"[Iteration {iteration}] synthesize_answer: Both original and cleaned content are empty, using fallback"
                )
                answer = _build_fallback_answer(state, iteration)
                logger.warning(
                    f"[Iteration {iteration}] synthesize_answer: Using fallback answer (length: {len(answer)} chars)"
                )

        logger.info(
            f"[Iteration {iteration}] synthesize_answer: LLM response received (length: {len(answer)} chars, preview: {answer[:100]}...)"
        )

        # 立即檢測 hallucination
        logger.info(
            f"[Iteration {iteration}] synthesize_answer: Running hallucination detection..."
        )
        hallucination_check = await evaluation_service.detect_hallucination(
            answer, state["collected_results"]
        )

        risk_score = hallucination_check.get("hallucination_risk_score", 0.0)
        has_risk = hallucination_check.get("has_hallucination_risk", False)
        logger.info(
            f"[Iteration {iteration}] synthesize_answer: Hallucination check completed - has_risk={has_risk}, risk_score={risk_score:.2f}"
        )

        # 關鍵修復：如果接近迭代限制（剩餘 <= 3次），即使風險高也強制設置 final_answer
        is_near_limit = iteration >= max_iterations - 3

        # 如果風險過高，拒絕答案並繼續搜尋
        # if has_risk and risk_score > 0.6:
        #     if is_near_limit:
        #         # 接近迭代限制時，強制接受（即使風險高）
        #         logger.warning(f"[Iteration {iteration}] synthesize_answer: High hallucination risk ({risk_score:.2f}) but near iteration limit, forcing final_answer")
        #         # 評估答案質量
        #         eval_result = await evaluation_service.evaluate_results(
        #             question=state["question"],
        #             results=state["collected_results"],
        #             answer=answer,
        #             model_id=config.get("configurable", {}).get("evaluation_model"),
        #         )
        #         return {
        #             "partial_answer": answer,
        #             "final_answer": answer,  # 強制設置 final_answer，避免無限循環
        #             "hallucination_check": hallucination_check,
        #             "evaluation_result": eval_result,
        #             "iteration_count": state["iteration_count"] + 1,
        #             "reasoning_trace": [
        #                 f"Generated answer with high hallucination risk ({risk_score:.2f}) but forced acceptance due to iteration limit",
        #                 f"Quality score: {eval_result.get('combined_score', 0):.2f}"
        #             ],
        #         }
        #     else:
        #         # 不在迭代限制附近，正常拒絕
        #         logger.warning(f"[Iteration {iteration}] synthesize_answer: High hallucination risk ({risk_score:.2f}), rejecting answer")
        #         return {
        #             "partial_answer": answer,
        #             "final_answer": None,  # 不設為最終答案
        #             "hallucination_check": hallucination_check,
        #             "evaluation_result": {
        #                 "decision": "reject",
        #                 "reasoning": f"High hallucination risk detected: {risk_score:.2f}",
        #                 "combined_score": 0.3,
        #             },
        #             "iteration_count": state["iteration_count"] + 1,
        #             "reasoning_trace": [
        #                 f"Generated answer rejected due to high hallucination risk: {risk_score:.2f}"
        #             ],
        #         }

        # 評估答案質量
        logger.info(
            f"[Iteration {iteration}] synthesize_answer: Running answer quality evaluation..."
        )
        eval_result = await evaluation_service.evaluate_results(
            question=state["question"],
            results=state["collected_results"],
            answer=answer,
            model_id=config.get("configurable", {}).get("evaluation_model"),
        )

        combined_score = eval_result.get("combined_score", 0.0)
        decision = eval_result.get("decision", "continue")
        logger.info(
            f"[Iteration {iteration}] synthesize_answer: Evaluation completed - decision={decision}, combined_score={combined_score:.2f}"
        )

        # 關鍵修復：如果評估結果為 reject 但接近迭代限制，也強制設置 final_answer
        if decision == "reject" and is_near_limit:
            logger.warning(
                f"[Iteration {iteration}] synthesize_answer: Evaluation rejected but near iteration limit, forcing final_answer"
            )
            return {
                "partial_answer": answer,
                "final_answer": answer,  # 強制設置 final_answer
                "hallucination_check": hallucination_check,
                "evaluation_result": eval_result,
                "iteration_count": state["iteration_count"] + 1,
                "reasoning_trace": [
                    f"Generated answer rejected by evaluation but forced acceptance due to iteration limit",
                    f"Quality score: {combined_score:.2f}, hallucination risk: {risk_score:.2f}",
                ],
            }
        
        # 如果評估結果為 reject，生成 feedback 供 Orchestrator 使用
        if decision == "reject":
            # 生成 feedback
            refinement_feedback = {
                "reason": eval_result.get("reasoning", "Answer quality below threshold"),
                "suggested_actions": [
                    "Search for more specific information",
                    "Refine search query based on missing information",
                    "Try different search tools (e.g., PageIndex if available)",
                ],
                "missing_info": [
                    "More detailed information needed",
                    "Higher quality sources required",
                ],
                "quality_score": combined_score,
                "hallucination_risk": risk_score,
                "evaluation_details": eval_result,
            }
            
            logger.info(
                f"[Iteration {iteration}] synthesize_answer: Answer rejected, generating feedback for Orchestrator"
            )
            
            return {
                "partial_answer": answer,
                "final_answer": None,  # 不設為最終答案
                "hallucination_check": hallucination_check,
                "evaluation_result": eval_result,
                "refinement_feedback": refinement_feedback,  # 添加 feedback
                "iteration_count": state["iteration_count"] + 1,
                "reasoning_trace": [
                    f"Generated answer rejected: quality score {combined_score:.2f}, hallucination risk {risk_score:.2f}",
                    f"Feedback generated for Orchestrator: {refinement_feedback['reason']}",
                ],
            }

        logger.info(
            f"[Iteration {iteration}] synthesize_answer: Setting final_answer (length: {len(answer)} chars)"
        )

        # 關鍵修復：將最終答案添加到 messages 狀態中
        # 這樣 execute_chat 就不需要手動添加消息了
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content=answer)
        current_messages = state.get("messages", [])

        return {
            "messages": [ai_message],  # 使用 add_messages 自動合併到現有消息列表
            "partial_answer": answer,
            "final_answer": answer,  # 關鍵修復：確保設置 final_answer
            "hallucination_check": hallucination_check,
            "evaluation_result": eval_result,
            "iteration_count": state["iteration_count"] + 1,
            "reasoning_trace": [
                f"Generated answer with quality score: {combined_score:.2f}, "
                f"hallucination risk: {risk_score:.2f}"
            ],
        }
    except Exception as e:
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: Error synthesizing answer: {e}"
        )
        logger.exception(e)

        # 關鍵修復：使用統一的 fallback 函數構建答案
        fallback_answer = _build_fallback_answer(state, iteration, error=e)
        logger.error(
            f"[Iteration {iteration}] synthesize_answer: Using fallback answer (length: {len(fallback_answer)} chars)"
        )

        # 關鍵修復：將 fallback 答案也添加到 messages 狀態中
        from langchain_core.messages import AIMessage

        ai_message = AIMessage(content=fallback_answer)

        return {
            "messages": [ai_message],  # 使用 add_messages 自動合併到現有消息列表
            "error_history": [
                {
                    "step": "synthesize_answer",
                    "error": str(e),
                    "iteration": state["iteration_count"],
                }
            ],
            "final_answer": fallback_answer,  # 關鍵修復：確保設置 final_answer
            "partial_answer": fallback_answer,
            "iteration_count": state["iteration_count"] + 1,
        }


def should_accept_answer(state: ChatAgenticState) -> str:
    """Determine if answer should be accepted."""
    iteration = state["iteration_count"]
    max_iterations = state["max_iterations"]

    # 關鍵修復：如果有 final_answer，直接接受
    if state.get("final_answer"):
        logger.info(
            f"[Iteration {iteration}] should_accept_answer: Final answer exists, accepting"
        )
        return "accept"

    # 如果達到最大迭代次數，強制接受（避免無限循環）
    if iteration >= max_iterations:
        logger.warning(
            f"[Iteration {iteration}] should_accept_answer: Reached max_iterations ({max_iterations}), forcing accept"
        )
        return "accept"

    eval_result = state.get("evaluation_result", {})
    combined_score = eval_result.get("combined_score", 0.0)
    decision_history = state.get("decision_history", [])

    # 關鍵修復：檢測循環（連續 3 次 synthesize）且有 partial_answer 時，強制接受
    if len(decision_history) >= 3:
        recent_decisions = decision_history[-3:]
        if len(set(recent_decisions)) == 1 and recent_decisions[0] == "synthesize":
            if state.get("partial_answer"):
                logger.warning(
                    f"[Iteration {iteration}] should_accept_answer: Detected circular synthesize decisions with partial_answer, forcing accept"
                )
                return "accept"

    # 關鍵修復：當有 collected_results 且 iteration_count >= 3 時，即使沒有 final_answer，也應該接受 partial_answer
    collected_results = state.get("collected_results", [])
    if iteration >= 3 and len(collected_results) > 0:
        if state.get("partial_answer"):
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: Has collected_results ({len(collected_results)}) and partial_answer, accepting"
            )
            return "accept"

    # 如果接近最大迭代次數（剩餘 3 次），大幅降低接受標準
    if iteration >= max_iterations - 3:
        logger.info(
            f"[Iteration {iteration}] should_accept_answer: Near iteration limit, lowering acceptance criteria"
        )
        # 如果有任何結果，就接受
        if collected_results and len(collected_results) > 0:
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: Has collected_results, accepting"
            )
            return "accept"
        # 或者降低分數要求到 0.3
        if combined_score >= 0.3:
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: Combined score ({combined_score:.2f}) >= 0.3, accepting"
            )
            return "accept"
        # 即使分數低，如果有部分答案也接受
        if state.get("partial_answer"):
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: Has partial_answer, accepting"
            )
            return "accept"

    # 如果已經執行了很多次搜索但分數仍低，也接受（避免無限循環）
    search_history = state.get("search_history", [])
    if iteration >= 5 and len(search_history) >= 3:
        if combined_score >= 0.5:
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: Multiple searches ({len(search_history)}) with score >= 0.5, accepting"
            )
            return "accept"
        # 如果有部分答案，即使分數低也接受
        if state.get("partial_answer"):
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: Multiple searches with partial_answer, accepting"
            )
            return "accept"

    # 檢查是否有太多工具錯誤（可能表示工具壞了）
    error_history = state.get("error_history", [])
    if len(error_history) >= 3:
        # 如果連續 3 個錯誤，強制接受（避免一直重試失敗的工具）
        recent_errors = error_history[-3:]
        if all(e.get("step") == "execute_tool" for e in recent_errors):
            logger.warning(
                f"[Iteration {iteration}] should_accept_answer: Too many tool errors ({len(error_history)}), forcing accept to avoid infinite loop"
            )
            return "accept"

    # 檢查 hallucination 風險
    hallucination_check = state.get("hallucination_check")
    if hallucination_check and hallucination_check.get("has_hallucination_risk"):
        risk_score = hallucination_check.get("hallucination_risk_score", 0.0)
        if risk_score > 0.6:
            # 如果已經接近最大迭代次數，即使風險高也要接受
            if iteration >= max_iterations - 1:
                logger.warning(
                    f"[Iteration {iteration}] should_accept_answer: High hallucination risk but near limit, accepting"
                )
                return "accept"
            logger.info(
                f"[Iteration {iteration}] should_accept_answer: High hallucination risk ({risk_score:.2f}), rejecting"
            )
            return "reject"  # 高風險，拒絕

    # 標準接受條件
    if combined_score >= 0.7:
        logger.info(
            f"[Iteration {iteration}] should_accept_answer: Combined score ({combined_score:.2f}) >= 0.7, accepting"
        )
        return "accept"
    elif iteration >= max_iterations - 1:
        logger.warning(
            f"[Iteration {iteration}] should_accept_answer: Near max_iterations, accepting even if imperfect"
        )
        return "accept"  # 接受即使不完美，如果達到限制
    else:
        logger.info(
            f"[Iteration {iteration}] should_accept_answer: Rejecting - score={combined_score:.2f}, iteration={iteration}/{max_iterations}"
        )
        return "reject"


class SynthesizeAnswer(BaseModel):
    answer: str


class ToolDecision(BaseModel):
    tool_name: Optional[str]
    parameters: Dict[str, Any]


async def ainvoke_for_answer(model_id: Any, system_prompt: str):
    model = await provision_langchain_model(
        system_prompt,
        model_id,
        "refiner",
        max_tokens=2000,
        structured=dict(type="json"),
    )

    response = await model.ainvoke(system_prompt)
    content = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    cleaned_content = clean_thinking_content(content)
    return cleaned_content


async def use_tools_node(
    state: ChatAgenticState, config: RunnableConfig
) -> dict[str, Any]:
    iteration = state.get("iteration_count", 0)
    logger.info(f"[Iteration {iteration}]: start use tool node")

    available_tools = await tool_registry.list_tools()
    question = state.get("question")

    parser = PydanticOutputParser(pydantic_object=ToolDecision)
    prompt_data = {
        "question": question,
        "available_tools": available_tools,
        "format_instructions": parser.get_format_instructions(),
    }
    system_prompt = Prompter(prompt_template="basic_chat/tool_decision").render(
        data=prompt_data
    )

    decision = ToolDecision(tool_name="", parameters={})
    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        cleaned_content = await ainvoke_for_answer(model_id, system_prompt)
        decision = parser.parse(cleaned_content)

        logger.info(
            f"[Iteration {iteration}] tool_node: tool_name={decision.tool_name}, parameters={decision.parameters}"
        )
    except Exception as e:
        logger.error(f"[Iteration {iteration} tool_node: err of {e}")

    logger.info(f"[Iteration {iteration}]: decision = {decision}")

    logger.info(f"[Iteration {iteration}]: finish use tool node")

    return {
        "current_decision": {
            "tool_name": decision.tool_name,
            "parameters": decision.parameters,
        }
    }


async def synthesize_answer_node(
    state: ChatAgenticState, config: RunnableConfig
) -> dict[str, Any]:
    iteration = state["iteration_count"]

    logger.info(f"[Iteration {iteration}]: start synthesize answer")

    parser = PydanticOutputParser(pydantic_object=SynthesizeAnswer)
    prompt_data = {
        "question": state.get("question"),
        # 移除 conversation_context，現在使用原生 Messages 格式
        "collected_results": state.get("collected_results"),
        "format_instructions": parser.get_format_instructions(),
    }
    system_content = Prompter(prompt_template="basic_chat/synthesize").render(
        data=prompt_data
    )
    system_msg = SystemMessage(content=system_content)

    # 構建 Messages 列表：SystemMessage + 歷史對話
    messages_payload = [system_msg] + state["messages"]

    response = SynthesizeAnswer(answer="Empty Answer")
    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        # 使用 Messages 格式調用模型
        model = await provision_langchain_model(
            str(messages_payload),  # 僅用於日誌
            model_id,
            "refiner",
            max_tokens=4000,
        )
        ai_response = await model.ainvoke(messages_payload)
        cleaned_content = (
            ai_response.content
            if isinstance(ai_response.content, str)
            else str(ai_response.content)
        )
        cleaned_content = clean_thinking_content(cleaned_content)
        logger.info(f"[Iteration {iteration}]: output is {cleaned_content[:100]}")
        response = parser.parse(cleaned_content)

    except Exception as e:
        logger.error(f"[Iteration {iteration} tool_node: err of {e}")

    from langchain_core.messages import AIMessage

    ai_message = AIMessage(content=response.answer)
    current_messages = state.get("messages", [])

    logger.info(f"[Iteration {iteration}]: finish synthesize answer")

    return {"messages": [ai_message], "finial_answer": response.answer}


# Create async SQLite checkpointer
# Use standard LangGraph approach: create aiosqlite connection in lifespan and pass to AsyncSqliteSaver

memory: Optional[AsyncSqliteSaver] = None  # Will be set to the actual checkpointer after initialization

# Build the graph structure first
agent_state = StateGraph(ChatAgenticState)

# Add nodes
agent_state.add_node("initialize", initialize_chat_state)
# agent_state.add_node("agent_decision", agent_decision)
# agent_state.add_node("execute_tool", execute_tool)
# agent_state.add_node("evaluate", evaluate_results)
# agent_state.add_node("refine_query", refine_query)
# agent_state.add_node("synthesize", synthesize_answer)

# # Add edges
# agent_state.add_edge(START, "initialize")
# agent_state.add_edge("initialize", "agent_decision")

# # Conditional routing from agent_decision
# agent_state.add_conditional_edges(
#     "agent_decision",
#     route_decision,
#     {
#         "use_tool": "execute_tool",
#         "evaluate": "evaluate",
#         "synthesize": "synthesize",
#         "finish": END,
#     },
# )

# agent_state.add_edge("execute_tool", "agent_decision")
# agent_state.add_edge("refine_query", "execute_tool")

# # Conditional routing from evaluate
# agent_state.add_conditional_edges(
#     "evaluate",
#     lambda state: state.get("evaluation_result", {}).get("decision", "continue"),
#     {
#         "continue": "agent_decision",
#         "refine_search": "refine_query",
#         "synthesize": "synthesize",
#     },
# )

# # Conditional routing from synthesize
# agent_state.add_conditional_edges(
#     "synthesize",
#     should_accept_answer,
#     {"accept": END, "reject": "agent_decision"},
# )
# 使用完整的 Multi-Agent 架構節點
agent_state.add_node("orchestrator", agent_decision)  # 重命名為 orchestrator
agent_state.add_node("executor", execute_tool)  # 重命名為 executor
agent_state.add_node("refiner", synthesize_answer)  # 使用完整的 synthesize_answer，重命名為 refiner

# 保持簡化節點以向後兼容（如果需要）
agent_state.add_node("synthesize", synthesize_answer_node)
agent_state.add_node("use_tools", use_tools_node)

agent_state.add_edge(START, "initialize")
agent_state.add_edge("initialize", "orchestrator")  # 使用完整的 Orchestrator

# Orchestrator 路由
agent_state.add_conditional_edges(
    "orchestrator",
    route_decision,  # 根據 decision.action 路由
    {
        "use_tool": "executor",
        "synthesize": "refiner",
        "finish": "refiner",
    },
)

# Executor -> Back to Orchestrator (Loop)
agent_state.add_edge("executor", "orchestrator")

# Refiner 決定是結束還是重來
agent_state.add_conditional_edges(
    "refiner",
    should_accept_answer,  # 評估函數
    {
        "accept": END,
        "reject": "orchestrator",  # 帶有 feedback 回到 Orchestrator
    },
)

# Graph will be compiled after checkpointer is initialized
_graph_instance = None


async def initialize_checkpointer(connection: aiosqlite.Connection):
    """Initialize the async checkpointer using an existing connection.
    
    Args:
        connection: An aiosqlite.Connection instance that will be managed by the application lifespan.
    
    Returns:
        The initialized AsyncSqliteSaver checkpointer instance.
    
    Raises:
        RuntimeError: If checkpointer initialization fails or required methods are missing.
    """
    global memory, graph, _graph_instance
    if memory is None:
        # Create AsyncSqliteSaver directly with the provided connection
        memory = AsyncSqliteSaver(connection)
        
        # Ensure database tables are set up
        # setup() creates the checkpoints table if it doesn't exist
        # It may raise an exception if tables already exist, which is safe to ignore
        try:
            await memory.setup()
            logger.info("Checkpointer database tables initialized")
        except Exception as e:
            # setup() may fail if tables already exist, which is acceptable
            error_msg = str(e).lower()
            if "already exists" in error_msg or "table" in error_msg:
                logger.debug(f"Checkpointer tables may already exist: {e}")
            else:
                logger.warning(f"Checkpointer setup() failed: {e}")
        
        # Verify checkpointer type and methods
        checkpointer_type = type(memory).__name__
        logger.info(f"Checkpointer type: {checkpointer_type}")

        # Verify required methods exist
        required_methods = ["aget_tuple", "aput", "aget", "alist"]
        missing_methods = [m for m in required_methods if not hasattr(memory, m)]
        if missing_methods:
            raise RuntimeError(
                f"Checkpointer missing required methods: {missing_methods}. "
                f"Type: {checkpointer_type}, Available methods: {[m for m in dir(memory) if not m.startswith('_')][:10]}"
            )

        logger.info("AsyncSqliteSaver checkpointer initialized and validated")

        # Compile graph with checkpointer
        try:
            # Try to set recursion_limit during compilation if supported
            try:
                _graph_instance = agent_state.compile(
                    checkpointer=memory,
                    **(
                        {"recursion_limit": 100}
                        if hasattr(agent_state, "compile")
                        else {}
                    ),
                )
            except TypeError:
                _graph_instance = agent_state.compile(checkpointer=memory)
                logger.warning(
                    "Could not set recursion_limit during compilation, will try at runtime"
                )
            graph = _graph_instance
            logger.info("Chat graph compiled with AsyncSqliteSaver checkpointer")
        except Exception as e:
            logger.error(f"Failed to compile graph with checkpointer: {e}")
            logger.exception(e)
            raise
    return memory


async def cleanup_checkpointer(connection: aiosqlite.Connection):
    """Close the shared database connection.
    
    Args:
        connection: The aiosqlite.Connection instance to close.
    """
    if connection:
        await connection.close()
        logger.info("Checkpointer connection closed")


def get_graph():
    """Get the compiled graph. Will be compiled after checkpointer is initialized."""
    global _graph_instance, memory
    if _graph_instance is None:
        if memory is None:
            raise RuntimeError(
                "Checkpointer not initialized. Call initialize_checkpointer() first. "
                "This usually happens in application lifespan startup."
            )
        _graph_instance = agent_state.compile(checkpointer=memory)
    return _graph_instance


# For backward compatibility, create graph variable
# It will be initialized when first accessed or after checkpointer is set
graph = None  # Will be set after checkpointer initialization in lifespan
