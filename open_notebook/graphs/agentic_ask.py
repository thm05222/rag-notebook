"""
Agentic RAG graph for dynamic question answering.
Supports iterative search, multi-tool usage, and self-evaluation.
"""

import operator
import time
import uuid
from typing import Annotated, Any, Dict, List, Optional

from ai_prompter import Prompter
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from loguru import logger

from open_notebook.exceptions import (
    CircularReasoningError,
    IterationLimitError,
    TimeoutError as AgentTimeoutError,
    TokenLimitError,
)
from open_notebook.graphs.utils import provision_langchain_model
from open_notebook.services.evaluation_service import evaluation_service
from open_notebook.services.tool_service import tool_registry
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.token_utils import token_count


class AgenticAskState(TypedDict):
    """State for Agentic RAG workflow."""

    question: str
    iteration_count: int
    search_history: List[Dict[str, Any]]  # Search history
    collected_results: List[Dict[str, Any]]  # Accumulated search results
    current_tool_calls: List[Dict[str, Any]]  # Current tool calls
    evaluation_result: Optional[Dict[str, Any]]  # Evaluation result
    partial_answer: str  # Partial answer
    final_answer: Optional[str]  # Final answer
    reasoning_trace: Annotated[List[str], operator.add]  # Reasoning trace
    max_iterations: int  # Maximum iterations (default 10)
    token_count: int  # Token usage
    max_tokens: int  # Token limit (default 50000)
    start_time: float  # Start timestamp
    max_duration: float  # Maximum duration (seconds, default 300)
    decision_history: Annotated[List[str], operator.add]  # Decision history for cycle detection
    error_history: Annotated[List[Dict[str, Any]], operator.add]  # Error history
    request_id: str  # Request ID for state isolation


class Decision(BaseModel):
    """Decision model for agent actions."""

    action: str = Field(
        description="Action to take: 'use_tool', 'evaluate', 'synthesize', or 'finish'"
    )
    tool_name: Optional[str] = Field(None, description="Tool name if action is 'use_tool'")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Tool parameters if action is 'use_tool'"
    )
    reasoning: str = Field(description="Reasoning for this decision")


async def initialize_state(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Initialize AgenticAskState with default values."""
    import os

    question = state.get("question", "")
    # Allow configurable values from request, fallback to env vars or defaults
    max_iterations = state.get("max_iterations") or int(os.getenv("AGENTIC_MAX_ITERATIONS", "10"))
    max_tokens = state.get("max_tokens") or int(os.getenv("AGENTIC_MAX_TOKENS", "50000"))
    max_duration = state.get("max_duration") or float(os.getenv("AGENTIC_MAX_DURATION", "300"))

    return {
        "question": question,
        "iteration_count": 0,
        "search_history": [],
        "collected_results": [],
        "current_tool_calls": [],
        "evaluation_result": None,
        "partial_answer": "",
        "final_answer": None,
        "reasoning_trace": [],
        "max_iterations": max_iterations,
        "token_count": 0,
        "max_tokens": max_tokens,
        "start_time": time.time(),
        "max_duration": max_duration,
        "decision_history": [],
        "error_history": [],
        "request_id": str(uuid.uuid4()),
    }


def check_limits(state: AgenticAskState) -> str:
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

    return "continue"


def detect_circular_reasoning(state: AgenticAskState) -> bool:
    """Detect if we're stuck in circular reasoning."""
    if len(state["decision_history"]) < 3:
        return False

    # Check if last 3 decisions are the same
    recent = state["decision_history"][-3:]
    return len(set(recent)) == 1


async def agent_decision(state: AgenticAskState, config: RunnableConfig) -> Dict[str, Any]:
    """Agent makes decision on next action."""
    # Check limits first
    limit_check = check_limits(state)
    if limit_check != "continue":
        return {
            "reasoning_trace": [f"Stopped due to: {limit_check}"],
            "final_answer": state.get("partial_answer") or "Unable to complete due to limits.",
        }

    # Get available tools
    available_tools = await tool_registry.list_tools()

    # Prepare decision prompt
    prompt_data = {
        "question": state["question"],
        "iteration_count": state["iteration_count"],
        "max_iterations": state["max_iterations"],
        "token_count": state["token_count"],
        "max_tokens": state["max_tokens"],
        "search_history": state["search_history"][-3:],
        "collected_results": state["collected_results"],
        "partial_answer": state.get("partial_answer", ""),
        "reasoning_trace": state["reasoning_trace"][-3:],
        "available_tools": available_tools,
    }

    parser = PydanticOutputParser(pydantic_object=Decision)
    system_prompt = Prompter(
        prompt_template="agentic_ask/decision", parser=parser
    ).render(data=prompt_data)

    # Track tokens
    tokens = token_count(system_prompt)
    new_token_count = state["token_count"] + tokens

    try:
        model = await provision_langchain_model(
            system_prompt,
            config.get("configurable", {}).get("decision_model"),
            "tools",
            max_tokens=2000,
            structured=dict(type="json"),
        )

        response = await model.ainvoke(system_prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        cleaned_content = clean_thinking_content(content)

        # Parse decision
        decision = parser.parse(cleaned_content)

        # Update token count (estimate response tokens)
        new_token_count += 500  # Estimate

        return {
            "decision_history": [decision.action],
            "reasoning_trace": [decision.reasoning],
            "token_count": new_token_count,
            "current_decision": {
                "action": decision.action,
                "tool_name": decision.tool_name,
                "parameters": decision.parameters or {},
                "reasoning": decision.reasoning,
            },
        }
    except Exception as e:
        logger.error(f"Error in agent decision: {e}")
        return {
            "error_history": [
                {"step": "agent_decision", "error": str(e), "iteration": state["iteration_count"]}
            ],
            "reasoning_trace": [f"Decision failed: {str(e)}"],
        }


def route_decision(state: AgenticAskState) -> str:
    """Route based on agent decision."""
    current_decision = state.get("current_decision")
    if not current_decision:
        return "evaluate"

    action = current_decision.get("action", "evaluate")
    return action


async def execute_tool(state: AgenticAskState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the selected tool."""
    current_decision = state.get("current_decision", {})
    tool_name = current_decision.get("tool_name")
    parameters = current_decision.get("parameters", {})

    if not tool_name:
        return {
            "error_history": [
                {
                    "step": "execute_tool",
                    "error": "No tool name specified",
                    "iteration": state["iteration_count"],
                }
            ],
        }

    try:
        result = await tool_registry.execute_tool(tool_name, **parameters)

        # Record search history
        search_entry = {
            "query": parameters.get("query", ""),
            "tool": tool_name,
            "result_count": len(result.get("data", [])) if result.get("success") else 0,
            "success": result.get("success", False),
            "timestamp": time.time(),
        }

        # Collect results if successful
        collected = state["collected_results"].copy()
        if result.get("success") and result.get("data"):
            # Add results to collected results
            for item in result["data"]:
                if item not in collected:
                    collected.append(item)

        return {
            "search_history": [search_entry],
            "collected_results": collected,
            "current_tool_calls": [result],
            "iteration_count": state["iteration_count"] + 1,
        }
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return {
            "error_history": [
                {
                    "step": "execute_tool",
                    "error": str(e),
                    "tool": tool_name,
                    "iteration": state["iteration_count"],
                }
            ],
            "search_history": [
                {
                    "query": parameters.get("query", ""),
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                }
            ],
        }


async def evaluate_results(state: AgenticAskState, config: RunnableConfig) -> Dict[str, Any]:
    """Evaluate current results and decide next step."""
    try:
        eval_result = await evaluation_service.evaluate_results(
            question=state["question"],
            results=state["collected_results"],
            answer=state.get("partial_answer"),
            model_id=config.get("configurable", {}).get("evaluation_model"),
            use_two_stage=config.get("configurable", {}).get("evaluation_use_two_stage", True),
            config=config,
        )

        decision = eval_result.get("decision", "continue")

        return {
            "evaluation_result": eval_result,
            "current_decision": {"action": decision},
            "reasoning_trace": [f"Evaluation: {eval_result.get('reasoning', '')}"],
        }
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        # Fallback to simple decision
        result_count = len(state["collected_results"])
        decision = "synthesize" if result_count >= 3 else "continue"
        return {
            "evaluation_result": {
                "decision": decision,
                "reasoning": "Evaluation failed, using fallback",
            },
            "current_decision": {"action": decision},
        }


async def refine_query(state: AgenticAskState, config: RunnableConfig) -> Dict[str, Any]:
    """Refine search query based on previous attempts."""
    from ai_prompter import Prompter

    prompt_data = {
        "question": state["question"],
        "search_history": state["search_history"][-5:],
        "collected_results": state["collected_results"],
    }

    parser = PydanticOutputParser(pydantic_object=Decision)
    system_prompt = Prompter(
        prompt_template="agentic_ask/refinement", parser=parser
    ).render(data=prompt_data)

    try:
        model = await provision_langchain_model(
            system_prompt,
            config.get("configurable", {}).get("decision_model"),
            "tools",
            max_tokens=1000,
            structured=dict(type="json"),
        )

        response = await model.ainvoke(system_prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
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


async def synthesize_answer(state: AgenticAskState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate final answer from collected results."""
    from ai_prompter import Prompter

    prompt_data = {
        "question": state["question"],
        "collected_results": state["collected_results"],
    }

    system_prompt = Prompter(prompt_template="agentic_ask/synthesis").render(data=prompt_data)

    try:
        model = await provision_langchain_model(
            system_prompt,
            config.get("configurable", {}).get("synthesis_model"),
            "tools",
            max_tokens=4000,
        )

        response = await model.ainvoke(system_prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        answer = clean_thinking_content(content)

        # Evaluate answer quality
        eval_result = await evaluation_service.evaluate_results(
            question=state["question"],
            results=state["collected_results"],
            answer=answer,
            model_id=config.get("configurable", {}).get("evaluation_model"),
            use_two_stage=config.get("configurable", {}).get("evaluation_use_two_stage", True),
            config=config,
        )

        # Check for hallucinations
        hallucination_check = await evaluation_service.detect_hallucination(
            answer, state["collected_results"]
        )

        return {
            "partial_answer": answer,
            "final_answer": answer,
            "evaluation_result": eval_result,
            "reasoning_trace": [
                f"Generated answer with quality score: {eval_result.get('combined_score', 0):.2f}"
            ],
        }
    except Exception as e:
        logger.error(f"Error synthesizing answer: {e}")
        return {
            "error_history": [
                {
                    "step": "synthesize_answer",
                    "error": str(e),
                    "iteration": state["iteration_count"],
                }
            ],
            "final_answer": state.get("partial_answer") or "Error generating answer.",
        }


def should_accept_answer(state: AgenticAskState) -> str:
    """Determine if answer should be accepted."""
    eval_result = state.get("evaluation_result", {})
    combined_score = eval_result.get("combined_score", 0.0)

    if combined_score >= 0.7:
        return "accept"
    elif state["iteration_count"] >= state["max_iterations"] - 1:
        return "accept"  # Accept even if not perfect if we're at limit
    else:
        return "reject"


# Build the graph
agent_state = StateGraph(AgenticAskState)

# Add nodes
agent_state.add_node("initialize", initialize_state)
agent_state.add_node("agent_decision", agent_decision)
agent_state.add_node("execute_tool", execute_tool)
agent_state.add_node("evaluate", evaluate_results)
agent_state.add_node("refine_query", refine_query)
agent_state.add_node("synthesize", synthesize_answer)

# Add edges
agent_state.add_edge(START, "initialize")
agent_state.add_edge("initialize", "agent_decision")

# Conditional routing from agent_decision
agent_state.add_conditional_edges(
    "agent_decision",
    route_decision,
    {
        "use_tool": "execute_tool",
        "evaluate": "evaluate",
        "synthesize": "synthesize",
        "finish": END,
    },
)

agent_state.add_edge("execute_tool", "agent_decision")
agent_state.add_edge("refine_query", "execute_tool")

# Conditional routing from evaluate
agent_state.add_conditional_edges(
    "evaluate",
    lambda state: state.get("evaluation_result", {}).get("decision", "continue"),
    {
        "continue": "agent_decision",
        "refine_search": "refine_query",
        "synthesize": "synthesize",
    },
)

# Conditional routing from synthesize
agent_state.add_conditional_edges(
    "synthesize",
    should_accept_answer,
    {"accept": END, "reject": "agent_decision"},
)

graph = agent_state.compile()

