import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from open_notebook.database.repository import ensure_record_id, repo_query, repo_add_message, repo_get_chat_history
from open_notebook.domain.notebook import ChatSession, Notebook, Source
from open_notebook.exceptions import (
    NotFoundError,
)
from open_notebook.graphs.chat import get_graph

router = APIRouter()

# Request/Response models
class CreateSessionRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID to create session for")
    title: Optional[str] = Field(None, description="Optional session title")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this session"
    )


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="New session title")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class AgentThinkingStep(BaseModel):
    """Single step in agent thinking process."""
    step_type: str = Field(..., description="Step type: decision, tool_call, search, evaluation, refinement, synthesis")
    timestamp: float = Field(..., description="Step timestamp")
    content: str = Field(..., description="Step content/description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional step metadata")


class AgentThinkingProcess(BaseModel):
    """Complete agent thinking process for a message."""
    steps: List[AgentThinkingStep] = Field(default_factory=list, description="List of thinking steps")
    total_iterations: int = Field(0, description="Total number of iterations")
    total_tool_calls: int = Field(0, description="Total number of tool calls")
    search_count: int = Field(0, description="Number of searches performed")
    evaluation_scores: Optional[Dict[str, float]] = Field(None, description="Evaluation scores")
    reasoning_trace: List[str] = Field(default_factory=list, description="Reasoning trace")


class ChatMessage(BaseModel):
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type (human|ai)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    thinking_process: Optional[AgentThinkingProcess] = Field(None, description="Agent thinking process (for AI messages)")
    reasoning_content: Optional[str] = Field(None, description="Plain text reasoning content for simplified display")


class ChatSessionResponse(BaseModel):
    id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    notebook_id: Optional[str] = Field(None, description="Notebook ID")
    created: str = Field(..., description="Creation timestamp")
    updated: str = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(
        None, description="Number of messages in session"
    )
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class ChatSessionWithMessagesResponse(ChatSessionResponse):
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Session messages"
    )


class ExecuteChatRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message content")
    context: Dict[str, Any] = Field(
        ..., description="Chat context with sources"
    )
    model_override: Optional[str] = Field(
        None, description="Optional model override for this message"
    )
    notebook_id: Optional[str] = Field(
        None, description="Optional notebook ID for auto-creating session if not found"
    )


class ExecuteChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="Updated message list")


class BuildContextRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID")
    context_config: Dict[str, Any] = Field(..., description="Context configuration")


class BuildContextResponse(BaseModel):
    context: Dict[str, Any] = Field(..., description="Built context data")
    token_count: int = Field(..., description="Estimated token count")
    char_count: int = Field(..., description="Character count")


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")


@router.get("/chat/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(notebook_id: str = Query(..., description="Notebook ID")):
    """Get all chat sessions for a notebook."""
    try:
        # Get notebook to verify it exists
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Get sessions for this notebook
        sessions = await notebook.get_chat_sessions()

        return [
            ChatSessionResponse(
                id=session.id or "",
                title=session.title or "Untitled Session",
                notebook_id=notebook_id,
                created=str(session.created),
                updated=str(session.updated),
                message_count=0,  # TODO: Add message count if needed
                model_override=getattr(session, "model_override", None),
            )
            for session in sessions
        ]
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching chat sessions: {str(e)}"
        )


@router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    try:
        # Verify notebook exists
        notebook = await Notebook.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Create new session
        session = ChatSession(
            title=request.title or f"Chat Session {asyncio.get_event_loop().time():.0f}",
            model_override=request.model_override,
        )
        await session.save()

        # Relate session to notebook
        await session.relate_to_notebook(request.notebook_id)

        return ChatSessionResponse(
            id=session.id or "",
            title=session.title or "",
            notebook_id=request.notebook_id,
            created=str(session.created),
            updated=str(session.updated),
            message_count=0,
            model_override=session.model_override,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating chat session: {str(e)}"
        )


@router.get(
    "/chat/sessions/{session_id}/diagnostics"
)
async def get_session_diagnostics(session_id: str):
    """Get diagnostic information about a chat session, including tool failures and errors."""
    try:
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get session state from LangGraph
        # Note: thread_id should NOT have "chat_session:" prefix to match how messages are stored
        thread_id = session_id.replace("chat_session:", "") if session_id.startswith("chat_session:") else session_id
        chat_graph = get_graph()
        thread_state = await chat_graph.aget_state(
            config=RunnableConfig(configurable={"thread_id": thread_id})
        )
        
        if not thread_state or not hasattr(thread_state, 'values'):
            return {
                "session_id": session_id,
                "error": "No state found for this session",
                "tool_failures": [],
                "unavailable_tools": [],
                "error_history": [],
            }
        
        state_values = thread_state.values
        
        # Extract error history
        error_history = state_values.get("error_history", [])
        unavailable_tools = state_values.get("unavailable_tools", [])
        
        # Analyze tool failures
        tool_failures = []
        tool_failure_counts = {}
        
        for error in error_history:
            if error.get("step") == "execute_tool":
                tool_name = error.get("tool", "unknown")
                iteration = error.get("iteration", 0)
                error_msg = error.get("error", "Unknown error")
                error_details = error.get("error_details", {})
                
                tool_failures.append({
                    "tool": tool_name,
                    "iteration": iteration,
                    "error": error_msg,
                    "error_details": error_details,
                    "timestamp": error.get("timestamp"),
                })
                
                # Count failures per tool
                if tool_name not in tool_failure_counts:
                    tool_failure_counts[tool_name] = {
                        "count": 0,
                        "errors": []
                    }
                tool_failure_counts[tool_name]["count"] += 1
                tool_failure_counts[tool_name]["errors"].append({
                    "iteration": iteration,
                    "error": error_msg,
                    "error_details": error_details,
                })
        
        # Extract search history for context
        search_history = state_values.get("search_history", [])
        failed_searches = [
            search for search in search_history 
            if not search.get("success", True)
        ]
        
        # Extract decision history
        decision_history = state_values.get("decision_history", [])
        
        return {
            "session_id": session_id,
            "tool_failures": tool_failures,
            "tool_failure_summary": tool_failure_counts,
            "unavailable_tools": unavailable_tools,
            "error_history": error_history,
            "failed_searches": failed_searches,
            "total_iterations": state_values.get("iteration_count", 0),
            "total_errors": len(error_history),
            "total_tool_failures": len(tool_failures),
            "decision_history": decision_history[-10:],  # Last 10 decisions
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error fetching session diagnostics: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Error fetching session diagnostics: {str(e)}"
        )


@router.get(
    "/chat/sessions/{session_id}", response_model=ChatSessionWithMessagesResponse
)
async def get_session(session_id: str):
    """Get a specific session with its messages."""
    try:
        # Get session
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # === 從 SurrealDB 獲取歷史訊息 ===
        db_history = await repo_get_chat_history(full_session_id)
        messages: list[ChatMessage] = []
        
        logger.info(f"[PERSISTENCE get_session] Loaded {len(db_history)} messages from SurrealDB for session {full_session_id}")
        
        for record in db_history:
            # 還原 thinking_process（如果存在）
            thinking = None
            if record.get('thinking_process') and record.get('role') == 'ai':
                try:
                    thinking = AgentThinkingProcess(**record['thinking_process'])
                except Exception as e:
                    logger.warning(f"Failed to restore thinking_process: {e}")
            
            messages.append(ChatMessage(
                id=str(record.get('id', f"msg_{len(messages)}")),
                type="human" if record.get('role') == 'user' else "ai",
                content=record.get('content', ''),
                timestamp=str(record.get('created_at')) if record.get('created_at') else None,
                thinking_process=thinking,
                reasoning_content=record.get('reasoning_content'),
            ))

        # Find notebook_id (we need to query the relationship)
        notebook_query = await repo_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )

        notebook_id = notebook_query[0]["out"] if notebook_query else None

        if not notebook_id:
            # This might be an old session created before API migration
            logger.warning(
                f"No notebook relationship found for session {session_id} - may be an orphaned session"
            )

        session_model_override = getattr(session, "model_override", None)
        logger.info(f"[GET SESSION] Returning session {session_id} with model_override={session_model_override}, message_count={len(messages)}")
        
        return ChatSessionWithMessagesResponse(
            id=session.id or "",
            title=session.title or "Untitled Session",
            notebook_id=notebook_id,
            created=str(session.created),
            updated=str(session.updated),
            message_count=len(messages),
            messages=messages,
            model_override=session_model_override,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error fetching session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")


@router.put("/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session(session_id: str, request: UpdateSessionRequest):
    """Update session title."""
    try:
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        update_data = request.model_dump(exclude_unset=True)

        if "title" in update_data:
            session.title = update_data["title"]

        if "model_override" in update_data:
            logger.info(f"[UPDATE SESSION] Setting model_override={update_data['model_override']} for session {session_id}")
            session.model_override = update_data["model_override"]
        
        logger.info(f"[UPDATE SESSION] Saving session {session_id} with model_override={session.model_override}")
        await session.save()

        # Find notebook_id
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        notebook_query = await repo_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )
        notebook_id = notebook_query[0]["out"] if notebook_query else None

        return ChatSessionResponse(
            id=session.id or "",
            title=session.title or "",
            notebook_id=notebook_id,
            created=str(session.created),
            updated=str(session.updated),
            message_count=0,
            model_override=session.model_override,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")


@router.delete("/chat/sessions/{session_id}", response_model=SuccessResponse)
async def delete_session(session_id: str):
    """Delete a chat session."""
    try:
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        await session.delete()

        return SuccessResponse(success=True, message="Session deleted successfully")
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


async def generate_fallback_answer(
    question: str,
    state: Dict[str, Any],
    reason: str = "unknown"
) -> str:
    """
    生成 fallback 答案，解釋為什麼停止並整理現有結果。
    
    Args:
        question: 用戶問題
        state: 圖的當前狀態
        reason: 停止原因 (recursion_limit, iteration_limit, timeout, etc.)
    
    Returns:
        格式化的 fallback 答案
    """
    try:
        # 確保 state 是字典
        if not isinstance(state, dict):
            state = {}
        
        from ai_prompter import Prompter
        from open_notebook.graphs.utils import provision_langchain_model
        
        # 安全地收集狀態信息
        collected_results = state.get("collected_results", []) if isinstance(state, dict) else []
        evaluation_result = state.get("evaluation_result", {}) if isinstance(state, dict) else {}
        iteration_count = state.get("iteration_count", 0) if isinstance(state, dict) else 0
        reasoning_trace = state.get("reasoning_trace", []) if isinstance(state, dict) else []
        
        # 確保 evaluation_result 是字典
        if not isinstance(evaluation_result, dict):
            evaluation_result = {}
        
        # 構建解釋信息
        explanation_parts = []
        
        if reason == "recursion_limit":
            explanation_parts.append("已達到系統的最大遞歸限制（100次迭代）")
        elif reason == "iteration_limit":
            explanation_parts.append(f"已達到最大迭代次數（{iteration_count}次）")
        elif reason == "timeout":
            explanation_parts.append("已超過最大執行時間限制")
        
        # 安全地添加評估信息
        combined_score = evaluation_result.get("combined_score", 0.0) if isinstance(evaluation_result, dict) else 0.0
        if combined_score < 0.7:
            explanation_parts.append(f"當前答案的可信度較低（{combined_score:.2f}），未達到接受標準（0.7）")
        
        rule_score = evaluation_result.get("rule_score", 0.0) if isinstance(evaluation_result, dict) else 0.0
        llm_score = evaluation_result.get("llm_score", 0.0) if isinstance(evaluation_result, dict) else 0.0
        if rule_score < 0.5:
            explanation_parts.append("檢索結果數量或質量不足")
        if llm_score < 0.5:
            explanation_parts.append("答案質量評估不足")
        
        # 安全地添加 hallucination 風險信息
        hallucination_check = state.get("hallucination_check", {}) if isinstance(state, dict) else {}
        if isinstance(hallucination_check, dict) and hallucination_check.get("has_hallucination_risk"):
            risk_score = hallucination_check.get("hallucination_risk_score", 0.0)
            explanation_parts.append(f"檢測到幻覺風險（風險分數：{risk_score:.2f}）")
        
        # 準備 prompt 數據
        prompt_data = {
            "question": question,
            "explanation": "；".join(explanation_parts) if explanation_parts else "未知原因",
            "collected_results": collected_results,
            "result_count": len(collected_results),
            "iteration_count": iteration_count,
            "reasoning_trace": reasoning_trace[-5:] if isinstance(reasoning_trace, list) else [],
        }
        
        # 使用 fallback 模板生成答案
        try:
            logger.info(f"generate_fallback_answer: Attempting to generate fallback with LLM. Question: {question[:100]}..., results: {len(collected_results)}")
            system_prompt = Prompter(
                prompt_template="chat_agentic/fallback_answer"
            ).render(data=prompt_data)
            
            model = await provision_langchain_model(
                system_prompt,
                None,  # 使用默認模型
                "fallback",
                max_tokens=3000,
            )
            
            response = await model.ainvoke(system_prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            cleaned_content = content.replace("思考過程：", "").replace("思考：", "").strip()
            
            # 關鍵修復：確保返回的內容不是空的
            if not cleaned_content or len(cleaned_content) == 0:
                logger.warning("generate_fallback_answer: LLM returned empty content, falling back to simple fallback")
                return generate_simple_fallback(question, collected_results, explanation_parts)
            
            logger.info(f"generate_fallback_answer: Successfully generated fallback with LLM (length: {len(cleaned_content)} chars)")
            return cleaned_content
        except Exception as e:
            logger.error(f"Error generating fallback answer with LLM: {e}")
            logger.exception(e)
            # 關鍵修復：確保即使 LLM 失敗也返回有意義的內容
            simple_fallback = generate_simple_fallback(question, collected_results, explanation_parts)
            logger.info(f"generate_fallback_answer: Using simple fallback (length: {len(simple_fallback)} chars)")
            return simple_fallback
    except Exception as e:
        logger.error(f"Error in generate_fallback_answer: {e}")
        logger.exception(e)
        safe_state = state if isinstance(state, dict) else {}
        return generate_simple_fallback(question, safe_state.get("collected_results", []), [])


def generate_simple_fallback(
    question: str,
    collected_results: List[Dict[str, Any]],
    explanation_parts: List[str]
) -> str:
    """生成簡單的 fallback 答案（當 LLM 生成失敗時）"""
    logger.info(f"generate_simple_fallback: Generating fallback. Question length: {len(question)}, results count: {len(collected_results) if collected_results else 0}")
    
    explanation = "；".join(explanation_parts) if explanation_parts else "執行時間過長"
    
    # 關鍵修復：確保 collected_results 是列表
    if not isinstance(collected_results, list):
        logger.warning(f"generate_simple_fallback: collected_results is not a list, type: {type(collected_results)}")
        collected_results = []
    
    result_summary = ""
    if collected_results:
        result_summary = f"\n\n我已經收集了 {len(collected_results)} 個相關結果：\n\n"
        for i, result in enumerate(collected_results[:5], 1):  # 最多顯示5個
            if not isinstance(result, dict):
                logger.warning(f"generate_simple_fallback: Result {i} is not a dict, skipping")
                continue
                
            title = result.get("title", "無標題")
            # 關鍵修復：支持不同的內容格式（content 和 matches）
            content = result.get("content", "")
            if not content and result.get("matches"):
                # 支持 matches 數組格式（從 vector_search 返回）
                if isinstance(result["matches"], list) and result["matches"]:
                    content = result["matches"][0] if isinstance(result["matches"][0], str) else str(result["matches"][0])
            content_preview = content[:200] if content else "無內容"
            result_summary += f"{i}. **{title}**\n   {content_preview}...\n\n"
    
    # 關鍵修復：確保總是返回非空字符串
    fallback_text = f"""抱歉，我無法完成完整的回答。{explanation}。

基於我目前收集到的信息，我無法提供一個高質量的完整答案。{result_summary}

建議：
- 嘗試重新表述您的問題
- 提供更多上下文信息
- 檢查知識庫中是否有相關資料
"""
    
    # 如果 fallback_text 為空（理論上不應該發生），提供一個基本的消息
    if not fallback_text or len(fallback_text.strip()) == 0:
        logger.error("generate_simple_fallback: Generated fallback text is empty, using basic message")
        fallback_text = f"""抱歉，我無法完成完整的回答。{explanation}。

請嘗試重新表述您的問題或提供更多上下文信息。
"""
    
    logger.info(f"generate_simple_fallback: Generated fallback text (length: {len(fallback_text)} chars)")
    return fallback_text


def parse_token_from_trace(trace: str) -> tuple:
    """解析 reasoning_trace 中的 token 資訊
    
    Args:
        trace: 包含 token 資訊的 reasoning_trace 字串
               格式: "content |tokens:1234|in:800|out:434"
    
    Returns:
        tuple: (content, token_usage) 其中 token_usage 是 dict 或 None
    """
    import re
    
    if not isinstance(trace, str):
        return str(trace), None
    
    # 匹配 |tokens:xxx|in:xxx|out:xxx 格式
    token_pattern = r'\s*\|tokens:(\d+)\|in:(\d+)\|out:(\d+)\s*$'
    match = re.search(token_pattern, trace)
    
    if match:
        content = trace[:match.start()].strip()
        token_usage = {
            "total_tokens": int(match.group(1)),
            "input_tokens": int(match.group(2)),
            "output_tokens": int(match.group(3))
        }
        return content, token_usage
    
    # 嘗試匹配只有 total tokens 的格式 |tokens:xxx
    simple_pattern = r'\s*\|tokens:(\d+)\s*$'
    simple_match = re.search(simple_pattern, trace)
    if simple_match:
        content = trace[:simple_match.start()].strip()
        token_usage = {
            "total_tokens": int(simple_match.group(1)),
            "input_tokens": 0,
            "output_tokens": 0
        }
        return content, token_usage
    
    return trace, None


def build_thinking_process_from_state(state: Dict[str, Any]) -> Optional[AgentThinkingProcess]:
    """從狀態構建思考過程"""
    import time
    
    # 確保 state 是字典
    if not isinstance(state, dict):
        return None
    
    steps: List[AgentThinkingStep] = []
    
    # Collect reasoning trace and parse token info
    raw_reasoning_trace = state.get("reasoning_trace", []) if isinstance(state, dict) else []
    reasoning_trace = []  # 清理後的 reasoning trace（不含 token 標記）
    trace_token_map = {}  # 索引 -> token_usage 的映射
    
    for i, trace in enumerate(raw_reasoning_trace):
        content, token_usage = parse_token_from_trace(trace)
        reasoning_trace.append(content)
        if token_usage:
            trace_token_map[i] = token_usage
    
    # Collect decision history
    decision_history = state.get("decision_history", []) if isinstance(state, dict) else []
    current_decision = state.get("current_decision", {}) if isinstance(state, dict) else {}
    
    # Collect search history
    search_history = state.get("search_history", []) if isinstance(state, dict) else []
    
    # Collect tool calls
    current_tool_calls = state.get("current_tool_calls", []) if isinstance(state, dict) else []
    
    # Collect evaluation results
    evaluation_result = state.get("evaluation_result", {}) if isinstance(state, dict) else {}
    hallucination_check = state.get("hallucination_check", {}) if isinstance(state, dict) else {}
    
    # Collect error history (關鍵修復：顯示工具錯誤)
    error_history = state.get("error_history", []) if isinstance(state, dict) else []
    
    # Build steps chronologically
    start_time = state.get("start_time", time.time()) if isinstance(state, dict) else time.time()
    
    # Add decision steps
    for i, decision in enumerate(decision_history):
        steps.append(AgentThinkingStep(
            step_type="decision",
            timestamp=start_time + i * 0.1,
            content=f"Decision: {decision}",
            metadata={"decision": decision}
        ))
    
    # Add current decision if available
    if current_decision and isinstance(current_decision, dict):
        action = current_decision.get("action", "")
        reasoning = current_decision.get("reasoning", "")
        tool_name = current_decision.get("tool_name")
        
        # 嘗試從最後一個 reasoning_trace 獲取 token 資訊
        last_token_usage = None
        if trace_token_map and len(raw_reasoning_trace) > 0:
            last_idx = len(raw_reasoning_trace) - 1
            last_token_usage = trace_token_map.get(last_idx)
        
        decision_metadata = {
            "action": action,
            "tool_name": tool_name,
            "reasoning": reasoning,
            "parameters": current_decision.get("parameters", {})
        }
        
        # 添加 token_usage 到 metadata
        if last_token_usage:
            decision_metadata["token_usage"] = last_token_usage
        
        steps.append(AgentThinkingStep(
            step_type="decision",
            timestamp=start_time + len(decision_history) * 0.1,
            content=f"Action: {action}" + (f" | Tool: {tool_name}" if tool_name else ""),
            metadata=decision_metadata
        ))
    
    # Add tool call steps
    for tool_call in current_tool_calls:
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("tool_name", "unknown")
            success = tool_call.get("success", False)
            result_count = tool_call.get("metadata", {}).get("result_count", 0) if isinstance(tool_call.get("metadata"), dict) else 0
            steps.append(AgentThinkingStep(
                step_type="tool_call",
                timestamp=start_time + len(steps) * 0.1,
                content=f"Tool: {tool_name} | Success: {success} | Results: {result_count}",
                metadata={
                    "tool_name": tool_name,
                    "success": success,
                    "result_count": result_count,
                    "execution_time": tool_call.get("execution_time", 0.0)
                }
            ))
    
    # Add search steps
    for search in search_history:
        if isinstance(search, dict):
            query = search.get("query", "")
            tool = search.get("tool", "")
            result_count = search.get("result_count", 0)
            success = search.get("success", False)
            steps.append(AgentThinkingStep(
                step_type="search",
                timestamp=search.get("timestamp", start_time + len(steps) * 0.1),
                content=f"Search: {query} | Tool: {tool} | Results: {result_count} | Success: {success}",
                metadata={
                    "query": query,
                    "tool": tool,
                    "result_count": result_count,
                    "success": success
                }
            ))
    
    # Add evaluation steps
    if evaluation_result and isinstance(evaluation_result, dict):
        decision = evaluation_result.get("decision", "")
        combined_score = evaluation_result.get("combined_score", 0.0)
        reasoning = evaluation_result.get("reasoning", "")
        steps.append(AgentThinkingStep(
            step_type="evaluation",
            timestamp=start_time + len(steps) * 0.1,
            content=f"Evaluation: {decision} | Score: {combined_score:.2f}",
            metadata={
                "decision": decision,
                "combined_score": combined_score,
                "rule_score": evaluation_result.get("rule_score"),
                "llm_score": evaluation_result.get("llm_score"),
                "reasoning": reasoning
            }
        ))
    
    # Add hallucination check if available
    if hallucination_check and isinstance(hallucination_check, dict):
        risk_score = hallucination_check.get("hallucination_risk_score", 0.0)
        has_risk = hallucination_check.get("has_hallucination_risk", False)
        steps.append(AgentThinkingStep(
            step_type="evaluation",
            timestamp=start_time + len(steps) * 0.1,
            content=f"Hallucination Check: Risk Score {risk_score:.2f} | Has Risk: {has_risk}",
            metadata={
                "hallucination_risk_score": risk_score,
                "has_hallucination_risk": has_risk,
                "citation_ratio": hallucination_check.get("citation_ratio", 0.0)
            }
        ))
    
    # Add error history steps (關鍵修復：顯示工具錯誤)
    for error in error_history:
        if isinstance(error, dict):
            step = error.get("step", "unknown")
            error_msg = error.get("error", "未知錯誤")
            tool = error.get("tool", "")
            iteration = error.get("iteration", 0)
            error_details = error.get("error_details", {})
            steps.append(AgentThinkingStep(
                step_type="error",
                timestamp=start_time + len(steps) * 0.1,
                content=f"Error at {step}" + (f" | Tool: {tool}" if tool else "") + f" | Iteration: {iteration} | {error_msg}",
                metadata={
                    "step": step,
                    "error": error_msg,
                    "tool": tool,
                    "iteration": iteration,
                    "error_details": error_details
                }
            ))
    
    # 為帶有 token 資訊的 reasoning_trace 生成 synthesis 類型步驟
    # 這些通常對應 synthesize_answer 或 refine_query 的 LLM 調用
    for i, trace_content in enumerate(reasoning_trace):
        token_usage = trace_token_map.get(i)
        if token_usage:
            # 判斷步驟類型
            step_type = "synthesis"
            if "Refined query" in trace_content:
                step_type = "refinement"
            elif "Decision:" in trace_content or "Action:" in trace_content:
                step_type = "decision"
            elif "Generated answer" in trace_content or "quality score" in trace_content.lower():
                step_type = "synthesis"
            
            steps.append(AgentThinkingStep(
                step_type=step_type,
                timestamp=start_time + len(steps) * 0.1,
                content=trace_content,
                metadata={
                    "token_usage": token_usage,
                    "trace_index": i
                }
            ))
    
    # Build evaluation scores
    eval_scores = {}
    if evaluation_result and isinstance(evaluation_result, dict):
        eval_scores["combined_score"] = evaluation_result.get("combined_score", 0.0)
        eval_scores["rule_score"] = evaluation_result.get("rule_score", 0.0)
        eval_scores["llm_score"] = evaluation_result.get("llm_score", 0.0)
    if hallucination_check and isinstance(hallucination_check, dict):
        eval_scores["hallucination_risk"] = hallucination_check.get("hallucination_risk_score", 0.0)
    
    # Create thinking process
    return AgentThinkingProcess(
        steps=steps,
        total_iterations=state.get("iteration_count", 0) if isinstance(state, dict) else 0,
        total_tool_calls=len(current_tool_calls),
        search_count=len(search_history),
        evaluation_scores=eval_scores if eval_scores else None,
        reasoning_trace=reasoning_trace if isinstance(reasoning_trace, list) else []
    )


def build_thought_event(
    stage: str, content: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """構建思考事件（SSE 格式）"""
    event_data = {
        "type": "thought",
        "stage": stage,
        "content": content,
        "timestamp": time.time(),
    }
    if metadata:
        event_data["metadata"] = metadata
    return f"data: {json.dumps(event_data)}\n\n"


def build_content_event(content: str) -> str:
    """構建內容事件（SSE 格式）"""
    event_data = {
        "type": "content",
        "content": content,
    }
    return f"data: {json.dumps(event_data)}\n\n"


def build_error_event(message: str) -> str:
    """構建錯誤事件（SSE 格式）"""
    event_data = {
        "type": "error",
        "message": message,
    }
    return f"data: {json.dumps(event_data)}\n\n"


def build_complete_event(final_answer: Optional[str] = None, session_id: Optional[str] = None, thinking_process: Optional[AgentThinkingProcess] = None) -> str:
    """構建完成事件（SSE 格式）"""
    event_data = {
        "type": "complete",
    }
    if final_answer:
        event_data["final_answer"] = final_answer
    if session_id:
        event_data["session_id"] = session_id
    if thinking_process:
        # 將 AgentThinkingProcess 轉換為字典格式以便 JSON 序列化
        event_data["thinking_process"] = thinking_process.model_dump() if hasattr(thinking_process, "model_dump") else thinking_process.dict() if hasattr(thinking_process, "dict") else thinking_process
    return f"data: {json.dumps(event_data)}\n\n"


async def stream_chat_response(
    request: ExecuteChatRequest,
    session: ChatSession,
    model_override: Optional[str],
    graph_input: Dict[str, Any],
    config_dict: Dict[str, Any],
    actual_session_id: Optional[str] = None,
    session_thread_id: Optional[str] = None,  # 新增：用於保存最終答案的 session thread_id
) -> AsyncGenerator[str, None]:
    """串流聊天響應生成器"""
    chat_graph = get_graph()
    start_time = time.time()
    final_answer = None
    
    # === [修復] 使用唯一的 internal_thread_id 確保每個問題從乾淨狀態開始 ===
    # 不再需要清空狀態，因為每個問題使用獨立的 thread
    internal_thread_id = config_dict.get("configurable", {}).get("thread_id")
    logger.info(f"[STATE ISOLATION] Using internal_thread_id={internal_thread_id} for graph execution, session_thread_id={session_thread_id} for persistence")
    
    try:
        # 發送初始思考事件（確認前端能接收）
        yield build_thought_event(
            stage="planning",
            content="正在分析您的問題...",
            metadata={"phase": "init"},
        )
        logger.info("Stream: Sent initial thought event")
        
        # 移除 include_names 過濾器，以便捕獲所有事件（包括工具事件）
        # 在事件處理時自行過濾需要的事件類型
        event_count = 0
        async for event in chat_graph.astream_events(
            input=graph_input,
            config=config_dict,
            version="v2",
        ):
            event_type = event.get("event")
            name = event.get("name")
            tags = event.get("tags", [])
            data = event.get("data", {})
            
            # 調試日誌：記錄收到的事件
            event_count += 1
            if event_count <= 20:  # 只記錄前 20 個事件
                logger.debug(f"Stream event #{event_count}: type={event_type}, name={name}, tags={tags}")
            
            # 捕獲 Orchestrator 決策
            if event_type == "on_chain_end" and name == "orchestrator":
                output = data.get("output", {})
                if isinstance(output, dict):
                    decision = output.get("current_decision", {})
                    if decision:
                        reasoning = decision.get("reasoning", "")
                        action = decision.get("action", "")
                        tool_name = decision.get("tool_name")
                        
                        content = f"決策: {action}"
                        if tool_name:
                            content += f" | 工具: {tool_name}"
                        if reasoning:
                            content += f"\n推理: {reasoning[:200]}..." if len(reasoning) > 200 else f"\n推理: {reasoning}"
                        
                        yield build_thought_event(
                            stage="planning",
                            content=content,
                            metadata={
                                "action": action,
                                "tool_name": tool_name,
                                "reasoning": reasoning,
                                "parameters": decision.get("parameters", {}),
                            },
                        )
            
            # 捕獲工具執行開始
            elif event_type == "on_tool_start":
                tool_name = data.get("name", "unknown")
                input_data = data.get("input", {})
                query = input_data.get("query", "") if isinstance(input_data, dict) else ""
                
                content = f"正在調用工具: {tool_name}"
                if query:
                    content += f"\n查詢: {query[:100]}..." if len(query) > 100 else f"\n查詢: {query}"
                
                yield build_thought_event(
                    stage="executing",
                    content=content,
                    metadata={
                        "tool_name": tool_name,
                        "input": input_data,
                    },
                )
            
            # 捕獲工具執行結束
            elif event_type == "on_tool_end":
                tool_name = data.get("name", "unknown")
                output = data.get("output", {})
                
                # 嘗試從 output 中提取結果數量
                result_count = 0
                if isinstance(output, dict):
                    if "data" in output:
                        result_count = len(output["data"]) if isinstance(output["data"], list) else 0
                    elif "results" in output:
                        result_count = len(output["results"]) if isinstance(output["results"], list) else 0
                
                success = result_count > 0 or output is not None
                content = f"工具執行完成: {tool_name} | 成功: {success} | 結果數: {result_count}"
                
                yield build_thought_event(
                    stage="executing",
                    content=content,
                    metadata={
                        "tool_name": tool_name,
                        "success": success,
                        "result_count": result_count,
                    },
                )
            
            # 捕獲答案生成（token 流）
            elif event_type == "on_chat_model_stream":
                # 只捕獲 refiner 節點的輸出
                # 關鍵修復：檢查 name 和 tags，以及父節點名稱
                is_refiner = (
                    "refiner" in tags 
                    or name == "refiner" 
                    or name == "synthesize_answer"
                    or (isinstance(name, str) and "refiner" in name.lower())
                )
                if is_refiner:
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        logger.debug(f"stream_chat_response: Streaming content chunk from refiner (length: {len(chunk.content)} chars)")
                        yield build_content_event(chunk.content)
            
            # 捕獲 Refiner 完成
            elif event_type == "on_chain_end" and (name == "refiner" or "refiner" in tags or name == "synthesize_answer"):
                logger.info(f"stream_chat_response: Captured refiner on_chain_end event, name={name}, tags={tags}")
                output = data.get("output", {})
                logger.info(f"stream_chat_response: Refiner output type={type(output)}, keys={output.keys() if isinstance(output, dict) else 'not dict'}")
                if isinstance(output, dict):
                    # 嘗試從 output 中提取最終答案
                    if "final_answer" in output:
                        final_answer = output["final_answer"]
                        logger.info(f"stream_chat_response: Found final_answer in output (length: {len(final_answer) if final_answer else 0} chars)")
                    elif "messages" in output:
                        # 從 messages 中提取最後一個 AI 訊息
                        messages = output["messages"]
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content"):
                                final_answer = last_msg.content
                                logger.info(f"stream_chat_response: Found final_answer in messages (length: {len(final_answer) if final_answer else 0} chars)")
                    # 關鍵修復：檢查 output 中是否有 partial_answer
                    if not final_answer and "partial_answer" in output:
                        partial_answer = output["partial_answer"]
                        if partial_answer and len(partial_answer.strip()) > 0:
                            logger.info(f"stream_chat_response: Found partial_answer in output, using as final_answer (length: {len(partial_answer)} chars)")
                            yield build_content_event(partial_answer)
                            final_answer = partial_answer
                # 關鍵修復：如果 output 不是 dict，嘗試直接提取
                elif output:
                    logger.warning(f"stream_chat_response: Refiner output is not dict, type={type(output)}, trying to extract content")
                    if hasattr(output, "content"):
                        final_answer = output.content
                        logger.info(f"stream_chat_response: Extracted final_answer from output.content (length: {len(final_answer) if final_answer else 0} chars)")
            
            # 捕獲錯誤
            elif event_type == "on_chain_error":
                error = data.get("error", {})
                error_msg = str(error) if error else "未知錯誤"
                yield build_error_event(f"執行錯誤: {error_msg}")
        
        # 關鍵修復：在發送完成事件之前，檢查最終狀態，確保即使 final_answer 是 None，也使用 partial_answer
        if not final_answer:
            try:
                # 獲取最終狀態
                final_state = await chat_graph.aget_state(
                    config=config_dict
                )
                if final_state and hasattr(final_state, 'values'):
                    state_values = final_state.values
                    # 檢查 partial_answer
                    partial_answer = state_values.get("partial_answer")
                    if partial_answer and len(partial_answer.strip()) > 0:
                        # 關鍵修復：將 partial_answer 通過 SSE 發送給用戶（如果還沒有發送過）
                        # 因為可能沒有觸發 on_chat_model_stream 事件（LLM 返回空內容）
                        logger.info(f"stream_chat_response: Found partial_answer, sending via SSE (length: {len(partial_answer)} chars)")
                        # 將完整的 partial_answer 作為內容事件發送
                        yield build_content_event(partial_answer)
                        final_answer = partial_answer
                        logger.info(f"stream_chat_response: Using partial_answer as final_answer (length: {len(final_answer)} chars)")
                    # 如果還是沒有，檢查 messages 中的最後一個 AI 消息
                    elif not final_answer:
                        messages = state_values.get("messages", [])
                        if messages:
                            from langchain_core.messages import AIMessage
                            # 從後往前查找 AI 消息
                            for msg in reversed(messages):
                                if (hasattr(msg, 'type') and msg.type == 'ai') or isinstance(msg, AIMessage):
                                    if hasattr(msg, 'content') and msg.content:
                                        content = msg.content
                                        # 關鍵修復：如果找到了 AI 消息內容，也通過 SSE 發送
                                        logger.info(f"stream_chat_response: Found AI message content, sending via SSE (length: {len(content)} chars)")
                                        yield build_content_event(content)
                                        final_answer = content
                                        logger.info(f"stream_chat_response: Using last AI message content as final_answer (length: {len(final_answer)} chars)")
                                        break
            except Exception as state_error:
                logger.warning(f"stream_chat_response: Failed to get final state: {state_error}")
        
        # 構建 thinking_process 從最終狀態
        thinking_process = None
        try:
            final_state = await chat_graph.aget_state(config=config_dict)
            if final_state and hasattr(final_state, 'values'):
                state_values = final_state.values
                thinking_process = build_thinking_process_from_state(state_values)
        except Exception as thinking_error:
            logger.warning(f"stream_chat_response: Failed to build thinking_process: {thinking_error}")
        
        # === 儲存 AI 回應到 SurrealDB（包含思考過程）===
        if final_answer and actual_session_id:
            try:
                session_id_for_db = (
                    f"chat_session:{actual_session_id}" 
                    if not actual_session_id.startswith("chat_session:") 
                    else actual_session_id
                )
                
                # 將 thinking_process 轉換為 dict 格式存儲
                thinking_dict = None
                reasoning_content = None
                if thinking_process:
                    thinking_dict = (
                        thinking_process.model_dump() 
                        if hasattr(thinking_process, "model_dump") 
                        else thinking_process.dict() if hasattr(thinking_process, "dict") 
                        else None
                    )
                    # 從 thinking_process 提取 reasoning_trace 作為純文字版思考過程
                    if thinking_dict and thinking_dict.get('reasoning_trace'):
                        reasoning_content = "\n".join(thinking_dict['reasoning_trace'])
                
                await repo_add_message(
                    session_id_for_db, 
                    "ai", 
                    final_answer, 
                    thinking_process=thinking_dict,
                    reasoning_content=reasoning_content
                )
                logger.info(f"[PERSISTENCE] Saved AI message to SurrealDB (with thinking_process: {thinking_dict is not None}, reasoning_content: {reasoning_content is not None})")
            except Exception as e:
                logger.error(f"[PERSISTENCE] Failed to save AI message: {e}")
        
        # 發送完成事件（包含實際的 session_id 和 thinking_process，如果會話被自動創建）
        yield build_complete_event(final_answer, actual_session_id, thinking_process)
        
        # === [修復] 將最終答案和 thinking process 保存到 session_thread_id ===
        # 因為 graph 使用 internal_thread_id 執行，所以需要手動複製數據到 session_thread_id
        try:
            from langchain_core.messages import AIMessage, HumanMessage
            import time as time_module
            
            if final_answer and session_thread_id and internal_thread_id:
                # 從 internal_thread_id 獲取完整的執行狀態（包含 thinking process）
                internal_config = RunnableConfig(configurable={"thread_id": internal_thread_id})
                internal_state = await chat_graph.aget_state(config=internal_config)
                internal_values = internal_state.values if internal_state and hasattr(internal_state, 'values') else {}
                
                # 獲取 session 的當前消息
                session_config = RunnableConfig(configurable={"thread_id": session_thread_id})
                session_state = await chat_graph.aget_state(config=session_config)
                session_messages = session_state.values.get("messages", []) if session_state and hasattr(session_state, 'values') else []
                
                # 檢查是否已經有包含 final_answer 的 AI message
                final_answer_exists = any(
                    (hasattr(msg, 'type') and msg.type == 'ai') and 
                    (hasattr(msg, 'content') and msg.content and msg.content[:100] == final_answer[:100])
                    for msg in session_messages
                )
                
                if not final_answer_exists:
                    # 將 AI 回答保存到 session_thread_id
                    logger.info(f"[STATE ISOLATION] Saving final_answer to session_thread_id={session_thread_id} (length: {len(final_answer)} chars)")
                    
                    # 獲取用戶問題
                    user_question = request.message if request else None
                    
                    # 需要同時保存 HumanMessage 和 AIMessage
                    # 修復：每次對話都應該完整記錄問答對
                    # 之前的邏輯會跳過重複問題的 HumanMessage，導致對話歷史不完整
                    # 現在改為：檢查最後一條消息是否就是這個問題（避免同一請求重複保存）
                    last_message = session_messages[-1] if session_messages else None
                    is_duplicate_request = (
                        last_message and 
                        hasattr(last_message, 'type') and last_message.type == 'human' and
                        hasattr(last_message, 'content') and last_message.content == user_question
                    )
                    
                    messages_to_add = []
                    if not is_duplicate_request and user_question:
                        messages_to_add.append(HumanMessage(content=user_question))
                    messages_to_add.append(AIMessage(content=final_answer))
                    
                    # 構建完整的 messages 列表（現有 + 新增）
                    all_messages = list(session_messages) + messages_to_add
                    
                    # 從 internal state 獲取 thinking process 相關數據
                    # 這樣刷新頁面後仍然可以看到 thinking process
                    full_state_update = {
                        "messages": all_messages,
                        "notebook": internal_values.get("notebook"),
                        "context_config": internal_values.get("context_config"),
                        "conversation_context": internal_values.get("conversation_context"),
                        "question": user_question or internal_values.get("question", ""),
                        "iteration_count": internal_values.get("iteration_count", 0),
                        # 關鍵：保留 thinking process 相關數據
                        "search_history": internal_values.get("search_history", []),
                        "collected_results": internal_values.get("collected_results", []),
                        "current_tool_calls": internal_values.get("current_tool_calls", []),
                        "evaluation_result": internal_values.get("evaluation_result"),
                        "hallucination_check": internal_values.get("hallucination_check"),
                        "partial_answer": internal_values.get("partial_answer", ""),
                        "final_answer": final_answer,
                        "reasoning_trace": internal_values.get("reasoning_trace", []),
                        "max_iterations": internal_values.get("max_iterations", 20),
                        "token_count": internal_values.get("token_count", 0),
                        "max_tokens": internal_values.get("max_tokens", 300000),
                        "start_time": internal_values.get("start_time", time_module.time()),
                        "max_duration": internal_values.get("max_duration", 300),
                        "decision_history": internal_values.get("decision_history", []),
                        "error_history": internal_values.get("error_history", []),
                        "unavailable_tools": internal_values.get("unavailable_tools", []),
                        "model_override": internal_values.get("model_override"),
                        "current_decision": internal_values.get("current_decision"),
                        "refinement_feedback": internal_values.get("refinement_feedback"),
                    }
                    
                    await chat_graph.aupdate_state(
                        config=session_config,
                        values=full_state_update,
                        as_node="refiner"
                    )
                    logger.info(f"[STATE ISOLATION] Successfully saved state to session (messages: {len(all_messages)}, search_history: {len(full_state_update['search_history'])}, reasoning_trace: {len(full_state_update['reasoning_trace'])})")
                else:
                    logger.info("[STATE ISOLATION] Final answer already exists in session state")
        except Exception as fix_error:
            logger.warning(f"[STATE ISOLATION] Failed to save final answer to session: {fix_error}")
            import traceback
            logger.warning(f"[STATE ISOLATION] Traceback: {traceback.format_exc()}")
        
        # [調試] 驗證消息是否正確保存到 session state
        try:
            if session_thread_id:
                debug_config = RunnableConfig(configurable={"thread_id": session_thread_id})
                debug_state = await chat_graph.aget_state(config=debug_config)
                if debug_state and hasattr(debug_state, 'values'):
                    debug_messages = debug_state.values.get("messages", [])
                    logger.info(f"[DEBUG] Session state message count: {len(debug_messages)}")
                    for idx, msg in enumerate(debug_messages):
                        msg_type = msg.type if hasattr(msg, "type") else "unknown"
                        msg_content = msg.content[:50] if hasattr(msg, "content") and msg.content else "None"
                        logger.info(f"[DEBUG] Message {idx}: type={msg_type}, content_preview={msg_content}...")
                else:
                    logger.warning("[DEBUG] Could not retrieve session state or no values")
        except Exception as debug_error:
            logger.warning(f"[DEBUG] Failed to verify session state: {debug_error}")
        
    except Exception as e:
        logger.error(f"Error in stream_chat_response: {str(e)}")
        logger.exception(e)
        yield build_error_event(f"串流錯誤: {str(e)}")
        yield build_complete_event()


@router.post("/chat/execute")
async def execute_chat(request: ExecuteChatRequest):
    """Execute a chat request using Agentic RAG and stream the response."""
    try:
        # Verify session exists, auto-create if not found and notebook_id provided
        # Ensure session_id has proper table prefix
        full_session_id = (
            request.session_id
            if request.session_id.startswith("chat_session:")
            else f"chat_session:{request.session_id}"
        )
        try:
            session = await ChatSession.get(full_session_id)
        except NotFoundError:
            # Session not found - try to auto-create if notebook_id is provided
            if request.notebook_id:
                logger.info(f"Session {full_session_id} not found, auto-creating new session for notebook {request.notebook_id}")
                try:
                    # Verify notebook exists
                    notebook = await Notebook.get(request.notebook_id)
                    if not notebook:
                        raise HTTPException(status_code=404, detail=f"Notebook {request.notebook_id} not found")
                    
                    # Create new session with default title
                    import asyncio
                    default_title = request.message[:50] + "..." if len(request.message) > 50 else request.message
                    session = ChatSession(
                        title=default_title or f"Chat Session {asyncio.get_event_loop().time():.0f}",
                        model_override=request.model_override,
                    )
                    await session.save()
                    
                    # Relate session to notebook
                    await session.relate_to_notebook(request.notebook_id)
                    
                    # Update full_session_id to use the new session ID
                    full_session_id = session.id or full_session_id
                    logger.info(f"Auto-created session {full_session_id} for notebook {request.notebook_id}")
                except HTTPException:
                    raise
                except Exception as create_error:
                    logger.error(f"Failed to auto-create session: {create_error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Session not found and failed to auto-create: {str(create_error)}"
                    )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {full_session_id} not found. Please create a new session or provide notebook_id to auto-create."
                )

        # Determine model override (per-request override takes precedence over session-level)
        model_override = (
            request.model_override
            if request.model_override is not None
            else getattr(session, "model_override", None)
        )
        
        # [調試] 記錄模型選擇來源
        logger.info(f"[MODEL DEBUG] request.model_override={request.model_override}")
        logger.info(f"[MODEL DEBUG] session.model_override={getattr(session, 'model_override', None)}")
        logger.info(f"[MODEL DEBUG] Final model_override={model_override}")

        # Extract thread_id from full_session_id (remove prefix if present)
        # This ensures we use the correct session ID even if it was auto-created
        session_thread_id = full_session_id.replace("chat_session:", "") if full_session_id.startswith("chat_session:") else full_session_id
        
        # === [修復] 為每個新問題生成唯一的 internal_thread_id ===
        # 這樣可以確保每個問題從乾淨狀態開始，避免 operator.add 字段的殘留問題
        # session_thread_id 用於獲取消息歷史，internal_thread_id 用於 LangGraph 執行
        internal_thread_id = f"{session_thread_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"[STATE ISOLATION] session_thread_id={session_thread_id}, internal_thread_id={internal_thread_id}")
        
        # Get current state (for conversation history) - 使用 session_thread_id
        chat_graph = get_graph()
        current_state = await chat_graph.aget_state(
            config=RunnableConfig(
                configurable={"thread_id": session_thread_id}
            )
        )

        # === 從 SurrealDB 獲取歷史對話紀錄 ===
        from langchain_core.messages import HumanMessage, AIMessage
        
        db_history = await repo_get_chat_history(full_session_id)
        history_messages = []
        for record in db_history:
            if record.get('role') == 'user':
                history_messages.append(HumanMessage(content=record['content']))
            else:
                history_messages.append(AIMessage(content=record['content']))
        
        logger.info(f"[PERSISTENCE] Loaded {len(db_history)} messages from SurrealDB for session {full_session_id}")
        
        # 添加當前用戶訊息
        user_message = HumanMessage(content=request.message)
        messages = history_messages + [user_message]
        
        # === 立即儲存用戶訊息 ===
        try:
            await repo_add_message(full_session_id, "user", request.message)
            logger.info(f"[PERSISTENCE] Saved user message to SurrealDB")
        except Exception as e:
            logger.error(f"[PERSISTENCE] Failed to save user message: {e}")
        
        # Get notebook from session (or use provided notebook_id if session was auto-created)
        notebook_id_from_session = None
        notebook_id_query = await repo_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )
        notebook_id_from_session = notebook_id_query[0]["out"] if notebook_id_query else None
        
        # Use notebook_id from request if session was auto-created, otherwise use from session relationship
        notebook_id = request.notebook_id or notebook_id_from_session
        notebook = None
        if notebook_id:
            try:
                notebook = await Notebook.get(notebook_id)
            except Exception as e:
                logger.warning(f"Could not load notebook {notebook_id}: {e}")

        # Prepare input for Agentic RAG graph
        # === [修復] 只傳入必要的輸入，讓 initialize_chat_state 處理狀態初始化 ===
        # 不要在 graph_input 中傳入列表類型的字段，因為這會與 operator.add 產生衝突
        # 狀態重置由 initialize_chat_state 函數負責
        graph_input = {
            "messages": messages,
            "notebook": notebook,
            "context_config": request.context,  # context_config contains sources selection
            "model_override": model_override,
        }

        # Prepare config for graph execution
        # === [修復] 使用 internal_thread_id 確保每個問題從乾淨狀態開始 ===
        config_dict = {
            "recursion_limit": 100,
            "configurable": {
                "thread_id": internal_thread_id,  # 使用隔離的 thread_id
                "model_id": model_override,
            }
        }
        
        # Return streaming response
        # Pass actual_session_id in case session was auto-created
        actual_session_id_for_stream = full_session_id.replace("chat_session:", "") if full_session_id.startswith("chat_session:") else full_session_id
        return StreamingResponse(
            stream_chat_response(
                request=request,
                session=session,
                model_override=model_override,
                graph_input=graph_input,
                config_dict=config_dict,
                actual_session_id=actual_session_id_for_stream,
                session_thread_id=session_thread_id,  # 傳入 session_thread_id 用於保存最終答案
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing chat: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error executing chat: {str(e)}")
        
        # 以下代碼保留作為參考，但不會執行（因為已經返回 StreamingResponse）
        # 如果需要降級到非串流模式，可以取消註釋以下代碼
        """
        # Execute Agentic RAG chat graph
        chat_graph = get_graph()
        try:
            # 根據 LangGraph 錯誤訊息："You can increase the limit by setting the `recursion_limit` config key"
            # 使用 RunnableConfig，並添加整體執行超時保護（關鍵修復）
            import asyncio
            
            logger.info(f"Invoking graph with thread_id: {request.session_id}, model_override: {model_override}")
            
            # 根據 LangGraph 文檔和錯誤訊息，recursion_limit 應該作為 config 的頂層鍵
            # 而不是在 configurable 中
            config_dict = {
                "recursion_limit": 100,  # 在頂層設置（關鍵修復）
                "configurable": {
                    "thread_id": thread_id,
                    "model_id": model_override,
                }
            }
            logger.info(f"Config prepared: recursion_limit=100 (top-level), thread_id={thread_id}")
            
            # 嘗試使用字典形式的 config（LangGraph 可能接受）
            # 如果失敗，再嘗試 RunnableConfig
            try:
                config = config_dict
            except Exception:
                config = RunnableConfig(
                    configurable={
                        "thread_id": thread_id,
                        "model_id": model_override,
                    }
                )
                # 嘗試直接設置 recursion_limit 屬性（如果支持）
                if hasattr(config, 'recursion_limit'):
                    config.recursion_limit = 100
                logger.warning("Using RunnableConfig without recursion_limit in configurable, trying attribute")
            
            # 添加整體執行超時保護（120 秒，與 max_duration 一致）
            # 這可以防止圖在某個節點卡住時無限等待
            try:
                result = await asyncio.wait_for(
                    chat_graph.ainvoke(
                        input=graph_input,  # type: ignore[arg-type]
                        config=config,
                    ),
                    timeout=120.0  # 整體超時：120 秒
                )
                logger.info(f"Graph execution completed successfully")
            except asyncio.TimeoutError:
                logger.error("Graph execution timed out after 120 seconds")
                # 獲取當前狀態以生成 fallback
                try:
                    logger.info("Attempting to recover state from checkpointer after timeout...")
                    current_state = await chat_graph.aget_state(
                        config=RunnableConfig(
                            configurable={"thread_id": thread_id}
                        )
                    )
                    state_values = current_state.values if current_state and hasattr(current_state, 'values') else {}
                    if not isinstance(state_values, dict):
                        state_values = {}
                    
                    # 關鍵修復：記錄恢復的狀態信息
                    logger.info(f"Recovered state from checkpointer: iteration_count={state_values.get('iteration_count', 0)}, "
                               f"tool_calls={len(state_values.get('current_tool_calls', []))}, "
                               f"search_history={len(state_values.get('search_history', []))}, "
                               f"collected_results={len(state_values.get('collected_results', []))}")
                    
                    # 關鍵修復：優先使用 partial_answer 或 final_answer（如果存在）
                    final_answer = state_values.get("final_answer")
                    partial_answer = state_values.get("partial_answer")
                    collected_results = state_values.get("collected_results", [])
                    
                    logger.info(f"Timeout recovery: final_answer={final_answer is not None and final_answer != ''}, "
                               f"partial_answer={partial_answer is not None and partial_answer != ''}, "
                               f"collected_results={len(collected_results)}")
                    
                    answer_to_use = None
                    if final_answer:
                        answer_to_use = final_answer
                        logger.info(f"Timeout recovery: Using final_answer (length: {len(final_answer)} chars)")
                    elif partial_answer:
                        answer_to_use = partial_answer
                        logger.info(f"Timeout recovery: Using partial_answer (length: {len(partial_answer)} chars)")
                    
                    # 如果沒有答案，嘗試生成 fallback
                    if not answer_to_use:
                        logger.warning("Timeout recovery: No answer found in state, generating fallback")
                        try:
                            fallback_answer = await generate_fallback_answer(
                                question=request.message,
                                state=state_values,
                                reason="timeout"
                            )
                            # 關鍵修復：確保 fallback_answer 不是空的
                            if fallback_answer and len(fallback_answer.strip()) > 0:
                                answer_to_use = fallback_answer
                                logger.info(f"Timeout recovery: Generated fallback answer (length: {len(fallback_answer)} chars)")
                            else:
                                logger.error("Timeout recovery: Fallback answer is empty, using collected_results")
                                # 如果 fallback 也失敗，從 collected_results 構建基本答案
                                if collected_results:
                                    result_count = len(collected_results)
                                    answer_to_use = (
                                        "抱歉，執行時間過長，已自動停止。\n\n"
                                        "我已經收集了 " + str(result_count) + " 個相關結果，但無法在時間限制內完成完整的答案生成。\n\n"
                                        "請嘗試：\n"
                                        "- 簡化您的問題\n"
                                        "- 重新發送請求\n"
                                        "- 檢查知識庫中是否有相關資料"
                                    )
                                else:
                                    answer_to_use = "抱歉，執行時間過長，已自動停止。請嘗試重新表述問題或稍後再試。"
                        except Exception as fallback_error:
                            logger.error(f"Timeout recovery: Failed to generate fallback: {fallback_error}")
                            logger.exception(fallback_error)
                            # 最後的 fallback：從 collected_results 構建基本答案
                            if collected_results:
                                result_count = len(collected_results)
                                answer_to_use = (
                                    "抱歉，執行時間過長，已自動停止。\n\n"
                                    "我已經收集了 " + str(result_count) + " 個相關結果，但無法完成答案生成。\n\n"
                                    "請嘗試重新發送請求。"
                                )
                            else:
                                answer_to_use = "抱歉，執行時間過長，已自動停止。請嘗試重新表述問題或稍後再試。"
                    
                    # 關鍵修復：確保 answer_to_use 永遠不是空字符串
                    if not answer_to_use or len(answer_to_use.strip()) == 0:
                        logger.error("Timeout recovery: answer_to_use is empty after all attempts, using default message")
                        answer_to_use = "抱歉，執行時間過長，已自動停止。請嘗試重新發送請求。"
                    
                    from langchain_core.messages import AIMessage
                    import uuid
                    ai_message = AIMessage(content=answer_to_use, id=str(uuid.uuid4()))
                    messages.append(ai_message)
                    logger.info(f"Timeout recovery: Added AIMessage to messages (length: {len(answer_to_use)} chars)")
                    
                    # 構建思考過程（在清空之前，需要保存完整的 state 數據）
                    thinking_process = build_thinking_process_from_state(state_values)
                    
                    # 關鍵修復：保存 messages 到 state（如果需要）
                    try:
                        await chat_graph.aupdate_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id}),
                            values={"messages": messages},  # 只更新 messages
                            as_node="refiner"  # 指定作為 refiner 節點的輸出
                        )
                        logger.info(f"Timeout recovery: Saved messages to LangGraph state")
                    except Exception as state_update_error:
                        logger.warning(f"Timeout recovery: Failed to update state: {state_update_error}")
                    
                    # 關鍵修復：在構建 thinking_process 並保存 messages 後，清空 state 中的累積字段
                    # 確保下次新問題開始時是乾淨的狀態
                    try:
                        # 記錄清空前的情況
                        before_counts = {
                            "reasoning_trace": len(state_values.get("reasoning_trace", [])),
                            "decision_history": len(state_values.get("decision_history", [])),
                            "search_history": len(state_values.get("search_history", [])),
                            "current_tool_calls": len(state_values.get("current_tool_calls", [])),
                            "error_history": len(state_values.get("error_history", [])),
                            "unavailable_tools": len(state_values.get("unavailable_tools", [])),
                        }
                        logger.info(f"Timeout recovery: Before clearing - accumulated fields counts: {before_counts}")
                        
                        clear_values = {
                            "reasoning_trace": [],
                            "decision_history": [],
                            "search_history": [],
                            "current_tool_calls": [],
                            "error_history": [],
                            "unavailable_tools": [],  # 新增
                        }
                        await chat_graph.aupdate_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id}),
                            values=clear_values,
                            as_node="initialize"  # 作為初始化節點的輸出
                        )
                        
                        # 驗證清空是否成功
                        current_state_after = await chat_graph.aget_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id})
                        )
                        state_after = current_state_after.values if current_state_after else {}
                        
                        after_counts = {
                            "reasoning_trace": len(state_after.get("reasoning_trace", [])),
                            "decision_history": len(state_after.get("decision_history", [])),
                            "search_history": len(state_after.get("search_history", [])),
                            "current_tool_calls": len(state_after.get("current_tool_calls", [])),
                            "error_history": len(state_after.get("error_history", [])),
                            "unavailable_tools": len(state_after.get("unavailable_tools", [])),
                        }
                        logger.info(f"Timeout recovery: After clearing - accumulated fields counts: {after_counts}")
                        
                        all_cleared = all(count == 0 for count in after_counts.values())
                        if all_cleared:
                            logger.info(f"Timeout recovery: Successfully cleared all accumulated thinking process fields from state")
                        else:
                            logger.warning(f"Timeout recovery: Some fields were not fully cleared: before={before_counts}, after={after_counts}")
                            
                    except Exception as clear_error:
                        logger.warning(f"Timeout recovery: Failed to clear accumulated fields: {clear_error}")
                        logger.exception(clear_error)
                    
                    response_messages: list[ChatMessage] = []
                    logger.info(f"Timeout recovery: Building response_messages from {len(messages)} messages")
                    for idx, msg in enumerate(messages):
                        msg_type = msg.type if hasattr(msg, "type") else "unknown"
                        thinking = thinking_process if msg_type == "ai" and thinking_process else None
                        
                        msg_id = getattr(msg, "id", None)
                        if msg_id is None:
                            msg_id = str(uuid.uuid4())
                        elif not isinstance(msg_id, str):
                            msg_id = str(msg_id)
                        
                        msg_content = msg.content if hasattr(msg, "content") else str(msg)
                        logger.debug(f"Timeout recovery: Message {idx} - type={msg_type}, id={msg_id}, content_length={len(msg_content) if msg_content else 0}, has_thinking={thinking is not None}")
                        
                        if msg_type == "ai":
                            logger.info(f"Timeout recovery: AI message {idx} - content_length={len(msg_content)}, content_preview={msg_content[:100] if msg_content else 'None'}...")
                        
                        response_messages.append(
                            ChatMessage(
                                id=msg_id,
                                type=msg_type,
                                content=msg_content,
                                timestamp=None,
                                thinking_process=thinking,
                            )
                        )
                    
                    # 關鍵修復：記錄最終響應的詳細信息
                    ai_messages_in_response = [m for m in response_messages if m.type == "ai"]
                    logger.info(f"Timeout recovery: Response preparation - Total messages={len(response_messages)}, "
                               f"AI messages={len(ai_messages_in_response)}, "
                               f"Human messages={len([m for m in response_messages if m.type == 'human'])}")
                    
                    if ai_messages_in_response:
                        last_ai_msg = ai_messages_in_response[-1]
                        logger.info(f"Timeout recovery: Last AI message - content_length={len(last_ai_msg.content)}, "
                                   f"content_preview={last_ai_msg.content[:100] if last_ai_msg.content else 'None'}..., "
                                   f"has_thinking_process={last_ai_msg.thinking_process is not None}")
                    else:
                        logger.warning("Timeout recovery: No AI messages found in response_messages!")
                    
                    await session.save()
                    logger.info(f"Timeout recovery: Returning ExecuteChatResponse with {len(response_messages)} messages")
                    # Use actual session_id (may be different if session was auto-created)
                    actual_session_id = full_session_id.replace("chat_session:", "") if full_session_id.startswith("chat_session:") else full_session_id
                    return ExecuteChatResponse(session_id=actual_session_id, messages=response_messages)
                except Exception as fallback_error:
                    logger.error(f"Failed to generate timeout fallback: {fallback_error}")
                    logger.exception(fallback_error)
                    raise HTTPException(
                        status_code=500,
                        detail="查詢執行時間過長（超過 120 秒），已自動停止。請嘗試簡化問題或稍後再試。"
                    )
        except Exception as graph_error:
            error_msg = str(graph_error)
            error_str_lower = error_msg.lower()
            
            # 檢查是否是遞歸限制錯誤
            if "recursion limit" in error_str_lower or "GraphRecursionError" in str(type(graph_error)):
                logger.warning(f"Graph recursion limit reached: {error_msg}")
                
                try:
                    # 獲取當前狀態
                    current_state = await chat_graph.aget_state(
                        config=RunnableConfig(
                            configurable={"thread_id": thread_id}
                        )
                    )
                    state_values = current_state.values if current_state and hasattr(current_state, 'values') else {}
                    
                    # 確保 state_values 是字典
                    if not isinstance(state_values, dict):
                        state_values = {}
                    
                    # 關鍵修復：確保 messages 包含用戶消息
                    # 如果 messages 為空或沒有用戶消息，重新添加
                    if not messages or not any(msg.type == "human" if hasattr(msg, "type") else False for msg in messages):
                        logger.warning("Recursion limit fallback: messages is empty or missing user message, adding user message")
                        from langchain_core.messages import HumanMessage
                        user_message = HumanMessage(content=request.message)
                        messages = [user_message]
                    
                    logger.info(f"Recursion limit fallback: Current messages count: {len(messages)}, "
                               f"has user message: {any(msg.type == 'human' if hasattr(msg, 'type') else False for msg in messages)}")
                    
                    # 生成 fallback 答案
                    fallback_answer = await generate_fallback_answer(
                        question=request.message,
                        state=state_values,
                        reason="recursion_limit"
                    )
                    
                    # 使用 fallback 答案，確保有 ID
                    from langchain_core.messages import AIMessage
                    import uuid
                    ai_message = AIMessage(content=fallback_answer, id=str(uuid.uuid4()))
                    messages.append(ai_message)
                    logger.info(f"Recursion limit fallback: Added AIMessage to messages. Fallback length: {len(fallback_answer)} chars, Total messages: {len(messages)}")
                    
                    # 構建思考過程（在清空之前，需要保存完整的 state 數據）
                    thinking_process = build_thinking_process_from_state(state_values)
                    
                    # 關鍵修復：將 fallback answer 保存到 LangGraph state
                    # 這樣 refetchCurrentSession 時才能獲取到
                    # 只更新 messages，不更新其他字段，避免覆蓋累積字段
                    try:
                        await chat_graph.aupdate_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id}),
                            values={"messages": messages},  # 只更新 messages
                            as_node="refiner"  # 指定作為 refiner 節點的輸出
                        )
                        logger.info(f"Recursion limit fallback: Saved fallback answer to LangGraph state")
                    except Exception as state_update_error:
                        logger.warning(f"Recursion limit fallback: Failed to update state: {state_update_error}")
                        logger.exception(state_update_error)
                    
                    # 關鍵修復：在構建 thinking_process 並保存 messages 後，清空 state 中的累積字段
                    # 確保下次新問題開始時是乾淨的狀態
                    try:
                        # 記錄清空前的情況
                        before_counts = {
                            "reasoning_trace": len(state_values.get("reasoning_trace", [])),
                            "decision_history": len(state_values.get("decision_history", [])),
                            "search_history": len(state_values.get("search_history", [])),
                            "current_tool_calls": len(state_values.get("current_tool_calls", [])),
                            "error_history": len(state_values.get("error_history", [])),
                            "unavailable_tools": len(state_values.get("unavailable_tools", [])),
                        }
                        logger.info(f"Recursion limit fallback: Before clearing - accumulated fields counts: {before_counts}")
                        
                        clear_values = {
                            "reasoning_trace": [],
                            "decision_history": [],
                            "search_history": [],
                            "current_tool_calls": [],
                            "error_history": [],
                            "unavailable_tools": [],  # 新增
                        }
                        await chat_graph.aupdate_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id}),
                            values=clear_values,
                            as_node="initialize"  # 作為初始化節點的輸出
                        )
                        
                        # 驗證清空是否成功
                        current_state_after = await chat_graph.aget_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id})
                        )
                        state_after = current_state_after.values if current_state_after else {}
                        
                        after_counts = {
                            "reasoning_trace": len(state_after.get("reasoning_trace", [])),
                            "decision_history": len(state_after.get("decision_history", [])),
                            "search_history": len(state_after.get("search_history", [])),
                            "current_tool_calls": len(state_after.get("current_tool_calls", [])),
                            "error_history": len(state_after.get("error_history", [])),
                            "unavailable_tools": len(state_after.get("unavailable_tools", [])),
                        }
                        logger.info(f"Recursion limit fallback: After clearing - accumulated fields counts: {after_counts}")
                        
                        all_cleared = all(count == 0 for count in after_counts.values())
                        if all_cleared:
                            logger.info(f"Recursion limit fallback: Successfully cleared all accumulated thinking process fields from state")
                        else:
                            logger.warning(f"Recursion limit fallback: Some fields were not fully cleared: before={before_counts}, after={after_counts}")
                            
                    except Exception as clear_error:
                        logger.warning(f"Recursion limit fallback: Failed to clear accumulated fields: {clear_error}")
                        logger.exception(clear_error)
                    
                    # 繼續正常流程返回響應
                    response_messages: list[ChatMessage] = []
                    import uuid
                    logger.info(f"Recursion limit fallback: Building response_messages from {len(messages)} messages")
                    for msg in messages:
                        msg_type = msg.type if hasattr(msg, "type") else "unknown"
                        thinking = thinking_process if msg_type == "ai" and thinking_process else None
                        
                        # 確保 id 不是 None
                        msg_id = getattr(msg, "id", None)
                        if msg_id is None:
                            msg_id = str(uuid.uuid4())
                        elif not isinstance(msg_id, str):
                            msg_id = str(msg_id)
                        
                        response_messages.append(
                            ChatMessage(
                                id=msg_id,
                                type=msg_type,
                                content=msg.content if hasattr(msg, "content") else str(msg),
                                timestamp=None,
                                thinking_process=thinking,
                            )
                        )
                    
                    # 記錄最終響應信息
                    ai_messages_in_response = [m for m in response_messages if m.type == "ai"]
                    logger.info(f"Recursion limit fallback: Response preparation - Total messages={len(response_messages)}, "
                               f"AI messages={len(ai_messages_in_response)}")
                    if ai_messages_in_response:
                        last_ai_msg = ai_messages_in_response[-1]
                        logger.info(f"Recursion limit fallback: Last AI message - content length={len(last_ai_msg.content)}, "
                                   f"content preview={last_ai_msg.content[:100] if len(last_ai_msg.content) > 100 else last_ai_msg.content}")
                    
                    # Update session timestamp
                    await session.save()
                    logger.info(f"Recursion limit fallback: Returning ExecuteChatResponse with {len(response_messages)} messages")
                    # Use actual session_id (may be different if session was auto-created)
                    actual_session_id = full_session_id.replace("chat_session:", "") if full_session_id.startswith("chat_session:") else full_session_id
                    return ExecuteChatResponse(session_id=actual_session_id, messages=response_messages)
                except Exception as fallback_error:
                    logger.error(f"Failed to generate fallback answer: {fallback_error}")
                    logger.exception(fallback_error)
                    raise HTTPException(
                        status_code=500,
                        detail="查詢執行時間過長，已自動停止。請嘗試簡化問題或稍後再試。"
                    )
            else:
                # 其他錯誤，重新拋出
                raise

        # Extract final answer and add to messages
        # 關鍵修復：添加詳細日誌追蹤答案提取過程
        logger.info(f"Extracting answer from result. Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # 檢查 result 字典結構
        if isinstance(result, dict):
            logger.debug(f"Result structure sample: final_answer type={type(result.get('final_answer'))}, "
                        f"partial_answer type={type(result.get('partial_answer'))}, "
                        f"has messages={bool(result.get('messages'))}")
        
        final_answer = result.get("final_answer")
        partial_answer = result.get("partial_answer")
        
        # 記錄答案狀態
        logger.info(f"Answer extraction: final_answer={final_answer is not None and final_answer != ''}, "
                   f"partial_answer={partial_answer is not None and partial_answer != ''}, "
                   f"final_answer length={len(final_answer) if final_answer else 0}, "
                   f"partial_answer length={len(partial_answer) if partial_answer else 0}")
        
        if final_answer:
            logger.info(f"Using final_answer (first 100 chars): {final_answer[:100] if len(final_answer) > 100 else final_answer}")
        elif partial_answer:
            logger.info(f"Using partial_answer as fallback (first 100 chars): {partial_answer[:100] if len(partial_answer) > 100 else partial_answer}")
        
        thinking_process = None
        
        # 關鍵修復：即使沒有 final_answer，也構建思考過程
        # 這樣可以確保前端能看到已完成的工具調用，即使圖執行被中斷
        thinking_process = build_thinking_process_from_state(result)
        
        # 關鍵修復：在構建 thinking_process 後，清空 state 中的累積字段
        # 確保下次新問題開始時是乾淨的狀態
        try:
            # 先獲取當前狀態，記錄清空前的情況
            current_state_before = await chat_graph.aget_state(
                config=RunnableConfig(configurable={"thread_id": thread_id})
            )
            state_before = current_state_before.values if current_state_before else {}
            
            before_counts = {
                "reasoning_trace": len(state_before.get("reasoning_trace", [])),
                "decision_history": len(state_before.get("decision_history", [])),
                "search_history": len(state_before.get("search_history", [])),
                "current_tool_calls": len(state_before.get("current_tool_calls", [])),
                "error_history": len(state_before.get("error_history", [])),
                "unavailable_tools": len(state_before.get("unavailable_tools", [])),
            }
            logger.info(f"Before clearing - accumulated fields counts: {before_counts}")
            
            # 清空累積字段，確保不會影響下次問題
            # 注意：即使字段使用 operator.add，aupdate_state 會直接替換值
            clear_values = {
                "reasoning_trace": [],
                "decision_history": [],
                "search_history": [],
                "current_tool_calls": [],
                "error_history": [],
                "unavailable_tools": [],  # 新增
            }
            await chat_graph.aupdate_state(
                config=RunnableConfig(configurable={"thread_id": thread_id}),
                values=clear_values,
                as_node="initialize"  # 作為初始化節點的輸出
            )
            
            # 驗證清空是否成功
            current_state_after = await chat_graph.aget_state(
                config=RunnableConfig(configurable={"thread_id": thread_id})
            )
            state_after = current_state_after.values if current_state_after else {}
            
            after_counts = {
                "reasoning_trace": len(state_after.get("reasoning_trace", [])),
                "decision_history": len(state_after.get("decision_history", [])),
                "search_history": len(state_after.get("search_history", [])),
                "current_tool_calls": len(state_after.get("current_tool_calls", [])),
                "error_history": len(state_after.get("error_history", [])),
                "unavailable_tools": len(state_after.get("unavailable_tools", [])),
            }
            logger.info(f"After clearing - accumulated fields counts: {after_counts}")
            
            # 檢查是否全部清空成功
            all_cleared = all(count == 0 for count in after_counts.values())
            if all_cleared:
                logger.info(f"Successfully cleared all accumulated thinking process fields from state")
            else:
                logger.warning(f"Some fields were not fully cleared: before={before_counts}, after={after_counts}")
                
        except Exception as clear_error:
            logger.warning(f"Failed to clear accumulated fields from state: {clear_error}")
            logger.exception(clear_error)
        
        # 關鍵修復：增強答案提取邏輯，支持多種 fallback
        answer_to_use = None
        if final_answer:
            answer_to_use = final_answer
        elif partial_answer:
            answer_to_use = partial_answer
            logger.info("Using partial_answer as final_answer fallback")
        else:
            # 額外 fallback：如果 agent 選擇了 finish 但沒有答案，使用 reasoning
            current_decision = result.get("current_decision", {})
            if isinstance(current_decision, dict) and current_decision.get("action") == "finish":
                reasoning = current_decision.get("reasoning", "")
                if reasoning and len(reasoning.strip()) > 0:
                    answer_to_use = reasoning.strip()
                    logger.info(f"Using finish action reasoning as answer (length: {len(answer_to_use)} chars)")
        
        # 關鍵修復：優先使用 result 中的 messages（因為 synthesize_answer 現在會將 AI 消息添加到狀態中）
        result_messages = result.get("messages", [])
        
        # 檢查 result_messages 中是否有 AI 消息
        from langchain_core.messages import AIMessage
        has_ai_message = any(
            (hasattr(msg, 'type') and msg.type == 'ai') or isinstance(msg, AIMessage)
            for msg in result_messages
        ) if result_messages else False
        
        if result_messages and has_ai_message:
            # result 中的 messages 已經包含最終的 AI 消息（由 synthesize_answer 或 finish 動作添加）
            messages = result_messages
            logger.info(f"Using messages from result. Result messages count: {len(result_messages)}, has AI message: True")
        elif answer_to_use:
            # Fallback：即使 result_messages 存在，如果沒有 AI 消息且有 answer_to_use，也要創建
            ai_message = AIMessage(content=answer_to_use)
            if result_messages:
                # 如果已有 messages（可能是 Human 消息），追加 AI 消息
                messages = result_messages + [ai_message]
                logger.warning(f"Result messages exist but no AI message found, added AIMessage from answer_to_use. Total messages count: {len(messages)}")
            else:
                # 如果沒有 messages，創建新的列表
                messages = [ai_message]
                logger.warning(f"Result has no messages, manually added AIMessage. Total messages count: {len(messages)}")
        else:
            # 最後的 fallback：生成說明消息
            logger.warning("No answer found in result, generating fallback message")
            fallback_content = "抱歉，我無法生成完整的答案。請嘗試重新表述問題或提供更多上下文。"
            ai_message = AIMessage(content=fallback_content)
            if result_messages:
                messages = result_messages + [ai_message]
            else:
                messages = [ai_message]
            logger.info(f"Fallback AIMessage added to messages. Total messages count: {len(messages)}")

        # Update session timestamp
        await session.save()

        # Convert messages to response format
        response_messages: list[ChatMessage] = []
        import uuid
        logger.info(f"Building response_messages from {len(messages)} messages")
        for idx, msg in enumerate(messages):
            # 確保 id 不是 None
            msg_id = getattr(msg, "id", None)
            if msg_id is None:
                msg_id = str(uuid.uuid4())
            elif not isinstance(msg_id, str):
                msg_id = str(msg_id)
            
            msg_type = msg.type if hasattr(msg, "type") else "unknown"
            thinking = thinking_process if msg_type == "ai" and thinking_process else None
            
            msg_content = msg.content if hasattr(msg, "content") else str(msg)
            logger.debug(f"Message {idx} - type={msg_type}, id={msg_id}, content_length={len(msg_content) if msg_content else 0}, has_thinking={thinking is not None}")
            
            if msg_type == "ai":
                logger.info(f"AI message {idx} - content_length={len(msg_content)}, content_preview={msg_content[:100] if msg_content else 'None'}...")
            
            response_messages.append(
                ChatMessage(
                    id=msg_id,  # 確保不是 None
                    type=msg_type,
                    content=msg_content,
                    timestamp=None,
                    thinking_process=thinking,
                )
            )
        
        # 關鍵修復：記錄最終響應的詳細信息
        ai_messages_in_response = [m for m in response_messages if m.type == "ai"]
        logger.info(f"Response preparation: Total messages={len(response_messages)}, "
                   f"AI messages={len(ai_messages_in_response)}, "
                   f"Human messages={len([m for m in response_messages if m.type == 'human'])}")
        
        if ai_messages_in_response:
            last_ai_msg = ai_messages_in_response[-1]
            logger.info(f"Last AI message in response: type={last_ai_msg.type}, "
                       f"content length={len(last_ai_msg.content)}, "
                       f"content preview={last_ai_msg.content[:100] if len(last_ai_msg.content) > 100 else last_ai_msg.content}, "
                       f"has thinking_process={last_ai_msg.thinking_process is not None}")
        else:
            logger.warning("No AI messages found in response_messages! This may indicate an issue with answer extraction.")
        
        logger.info(f"Returning ExecuteChatResponse with {len(response_messages)} messages")

        # Use actual session_id (may be different if session was auto-created)
        actual_session_id = full_session_id.replace("chat_session:", "") if full_session_id.startswith("chat_session:") else full_session_id
        return ExecuteChatResponse(session_id=actual_session_id, messages=response_messages)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError as e:
        # Model configuration errors - return 400 Bad Request with helpful message
        error_msg = str(e)
        if "not found" in error_msg.lower() or "not configured" in error_msg.lower():
            logger.error(f"Model configuration error: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Model configuration error: {error_msg}. Please configure a default chat model in Settings > Models."
            )
        else:
            logger.error(f"Value error in chat execution: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {error_msg}")
    except Exception as e:
        error_msg = str(e)
        error_str_lower = error_msg.lower()
        
        # Log full exception details for debugging
        logger.error(f"Error executing chat: {error_msg}")
        logger.exception(e)  # Log full stack trace
        
        # 關鍵修復：檢查是否是 socket hang up 或連接中斷錯誤
        is_connection_error = (
            "socket hang up" in error_str_lower or
            "ECONNRESET" in error_msg or
            "connection" in error_str_lower or
            "ConnectionResetError" in str(type(e))
        )
        
        if is_connection_error:
            logger.warning(f"Connection error detected: {error_msg}, attempting to recover state from checkpointer")
            try:
                # 嘗試從 checkpointer 獲取當前狀態
                current_state = await chat_graph.aget_state(
                    config=RunnableConfig(
                        configurable={"thread_id": request.session_id}
                    )
                )
                state_values = current_state.values if current_state and hasattr(current_state, 'values') else {}
                
                if not isinstance(state_values, dict):
                    state_values = {}
                
                # 構建思考過程，即使圖未完成（在清空之前，需要保存完整的 state 數據）
                thinking_process = build_thinking_process_from_state(state_values)
                
                # 如果有部分結果，生成部分響應
                if state_values.get("collected_results") or state_values.get("current_tool_calls"):
                    # 從 state_values 獲取 messages，如果不存在則使用原始 messages
                    recovered_messages = state_values.get("messages", [])
                    if not recovered_messages:
                        # 如果 state 中沒有 messages，嘗試從之前的作用域獲取
                        # 或者重新構建（包含用戶消息）
                        from langchain_core.messages import HumanMessage, AIMessage
                        recovered_messages = [HumanMessage(content=request.message)]
                    
                    # 生成部分答案說明
                    partial_content = "連接中斷，但已收集到部分結果。"
                    if state_values.get("collected_results"):
                        result_count = len(state_values.get("collected_results", []))
                        partial_content += f" 已找到 {result_count} 個相關結果。"
                    if state_values.get("current_tool_calls"):
                        tool_count = len(state_values.get("current_tool_calls", []))
                        partial_content += f" 已執行 {tool_count} 個工具調用。"
                    
                    from langchain_core.messages import AIMessage
                    import uuid
                    ai_message = AIMessage(
                        content=partial_content + " 請重新發送請求以獲取完整答案。",
                        id=str(uuid.uuid4())
                    )
                    recovered_messages.append(ai_message)
                    
                    # 構建響應
                    response_messages: list[ChatMessage] = []
                    import uuid
                    for msg in recovered_messages:
                        msg_id = getattr(msg, "id", None)
                        if msg_id is None:
                            msg_id = str(uuid.uuid4())
                        elif not isinstance(msg_id, str):
                            msg_id = str(msg_id)
                        
                        msg_type = msg.type if hasattr(msg, "type") else "unknown"
                        thinking = thinking_process if msg_type == "ai" and thinking_process else None
                        
                        response_messages.append(
                            ChatMessage(
                                id=msg_id,
                                type=msg_type,
                                content=msg.content if hasattr(msg, "content") else str(msg),
                                timestamp=None,
                                thinking_process=thinking,
                            )
                        )
                    
                    # 關鍵修復：在構建 thinking_process 並返回響應後，清空 state 中的累積字段
                    # 確保下次新問題開始時是乾淨的狀態
                    try:
                        # 記錄清空前的情況
                        before_counts = {
                            "reasoning_trace": len(state_values.get("reasoning_trace", [])),
                            "decision_history": len(state_values.get("decision_history", [])),
                            "search_history": len(state_values.get("search_history", [])),
                            "current_tool_calls": len(state_values.get("current_tool_calls", [])),
                            "error_history": len(state_values.get("error_history", [])),
                            "unavailable_tools": len(state_values.get("unavailable_tools", [])),
                        }
                        logger.info(f"Connection error recovery: Before clearing - accumulated fields counts: {before_counts}")
                        
                        clear_values = {
                            "reasoning_trace": [],
                            "decision_history": [],
                            "search_history": [],
                            "current_tool_calls": [],
                            "error_history": [],
                            "unavailable_tools": [],  # 新增
                        }
                        await chat_graph.aupdate_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id}),
                            values=clear_values,
                            as_node="initialize"  # 作為初始化節點的輸出
                        )
                        
                        # 驗證清空是否成功
                        current_state_after = await chat_graph.aget_state(
                            config=RunnableConfig(configurable={"thread_id": thread_id})
                        )
                        state_after = current_state_after.values if current_state_after else {}
                        
                        after_counts = {
                            "reasoning_trace": len(state_after.get("reasoning_trace", [])),
                            "decision_history": len(state_after.get("decision_history", [])),
                            "search_history": len(state_after.get("search_history", [])),
                            "current_tool_calls": len(state_after.get("current_tool_calls", [])),
                            "error_history": len(state_after.get("error_history", [])),
                            "unavailable_tools": len(state_after.get("unavailable_tools", [])),
                        }
                        logger.info(f"Connection error recovery: After clearing - accumulated fields counts: {after_counts}")
                        
                        all_cleared = all(count == 0 for count in after_counts.values())
                        if all_cleared:
                            logger.info(f"Connection error recovery: Successfully cleared all accumulated thinking process fields from state")
                        else:
                            logger.warning(f"Connection error recovery: Some fields were not fully cleared: before={before_counts}, after={after_counts}")
                            
                    except Exception as clear_error:
                        logger.warning(f"Connection error recovery: Failed to clear accumulated fields: {clear_error}")
                        logger.exception(clear_error)
                    
                    await session.save()
                    # Use actual session_id (may be different if session was auto-created)
                    actual_session_id = full_session_id.replace("chat_session:", "") if full_session_id.startswith("chat_session:") else full_session_id
                    return ExecuteChatResponse(session_id=actual_session_id, messages=response_messages)
                else:
                    logger.warning("Connection error but no partial state found in checkpointer")
            except Exception as recovery_error:
                logger.error(f"Failed to recover state from checkpointer: {recovery_error}")
                logger.exception(recovery_error)
        
        # Check for rate limit errors (429)
        if "429" in error_msg or "rate_limit" in error_str_lower or "tokens per min" in error_str_lower:
            logger.error(f"Rate limit error in chat execution: {error_msg}")
            raise HTTPException(
                status_code=429,
                detail=(
                    "Request too large - the conversation history exceeds token limits. "
                    "The system has automatically truncated older messages. "
                    "If this persists, try starting a new chat session or reducing the context size."
                )
            )
        
        logger.error(f"Error executing chat: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error executing chat: {error_msg}")
        """


@router.post("/chat/context", response_model=BuildContextResponse)
async def build_context(request: BuildContextRequest):
    """Build context for a notebook based on context configuration."""
    try:
        # Verify notebook exists
        notebook = await Notebook.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        context_data: dict[str, list[dict[str, str]]] = {"sources": []}
        total_content = ""

        # Process context configuration if provided
        if request.context_config:
            # Process sources
            for source_id, status in request.context_config.get("sources", {}).items():
                if "not in" in status:
                    continue

                try:
                    # Add table prefix if not present
                    full_source_id = (
                        source_id
                        if source_id.startswith("source:")
                        else f"source:{source_id}"
                    )

                    try:
                        source = await Source.get(full_source_id)
                    except Exception:
                        continue

                    if "insights" in status:
                        source_context = await source.get_context(context_size="short")
                        context_data["sources"].append(source_context)
                        total_content += str(source_context)
                    elif "full content" in status:
                        source_context = await source.get_context(context_size="long")
                        context_data["sources"].append(source_context)
                        total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source {source_id}: {str(e)}")
                    continue

        else:
            # Default behavior - include all sources with short context
            sources = await notebook.get_sources()
            for source in sources:
                try:
                    source_context = await source.get_context(context_size="short")
                    context_data["sources"].append(source_context)
                    total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source {source.id}: {str(e)}")
                    continue

        # Calculate character and token counts
        char_count = len(total_content)
        # Use token count utility if available
        try:
            from open_notebook.utils import token_count

            estimated_tokens = token_count(total_content) if total_content else 0
        except ImportError:
            # Fallback to simple estimation
            estimated_tokens = char_count // 4

        return BuildContextResponse(
            context=context_data, token_count=estimated_tokens, char_count=char_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building context: {str(e)}")
