import json
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from api.models import (
    AgenticAskRequest,
    AgenticAskResponse,
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
)
from open_notebook.domain.models import Model, model_manager
from open_notebook.domain.notebook import text_search, vector_search
from open_notebook.exceptions import DatabaseOperationError, InvalidInputError
from open_notebook.graphs.ask import graph as ask_graph
from open_notebook.graphs.agentic_ask import graph as agentic_ask_graph

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(search_request: SearchRequest):
    """Search the knowledge base using text or vector search."""
    try:
        if search_request.type == "vector":
            # Check if embedding model is available for vector search
            if not await model_manager.get_embedding_model():
                raise HTTPException(
                    status_code=400,
                    detail="Vector search requires an embedding model. Please configure one in the Models section.",
                )

            results = await vector_search(
                keyword=search_request.query,
                results=search_request.limit,
                source=search_request.search_sources,
                minimum_score=search_request.minimum_score,
            )
        else:
            # Text search
            results = await text_search(
                keyword=search_request.query,
                results=search_request.limit,
                source=search_request.search_sources,
            )

        return SearchResponse(
            results=results or [],
            total_count=len(results) if results else 0,
            search_type=search_request.type,
        )

    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseOperationError as e:
        logger.error(f"Database error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def stream_ask_response(
    question: str, strategy_model: Model, answer_model: Model, final_answer_model: Model
) -> AsyncGenerator[str, None]:
    """Stream the ask response as Server-Sent Events."""
    try:
        final_answer = None

        async for chunk in ask_graph.astream(
            input=dict(question=question),  # type: ignore[arg-type]
            config=dict(
                configurable=dict(
                    strategy_model=strategy_model.id,
                    answer_model=answer_model.id,
                    final_answer_model=final_answer_model.id,
                )
            ),
            stream_mode="updates",
        ):
            if "agent" in chunk:
                strategy_data = {
                    "type": "strategy",
                    "reasoning": chunk["agent"]["strategy"].reasoning,
                    "searches": [
                        {"term": search.term, "instructions": search.instructions}
                        for search in chunk["agent"]["strategy"].searches
                    ],
                }
                yield f"data: {json.dumps(strategy_data)}\n\n"

            elif "provide_answer" in chunk:
                for answer in chunk["provide_answer"]["answers"]:
                    answer_data = {"type": "answer", "content": answer}
                    yield f"data: {json.dumps(answer_data)}\n\n"

            elif "write_final_answer" in chunk:
                final_answer = chunk["write_final_answer"]["final_answer"]
                final_data = {"type": "final_answer", "content": final_answer}
                yield f"data: {json.dumps(final_data)}\n\n"

        # Send completion signal
        completion_data = {"type": "complete", "final_answer": final_answer}
        yield f"data: {json.dumps(completion_data)}\n\n"

    except Exception as e:
        logger.error(f"Error in ask streaming: {str(e)}")
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/search/ask")
async def ask_knowledge_base(ask_request: AskRequest):
    """Ask the knowledge base a question using AI models."""
    try:
        # Validate models exist
        strategy_model = await Model.get(ask_request.strategy_model)
        answer_model = await Model.get(ask_request.answer_model)
        final_answer_model = await Model.get(ask_request.final_answer_model)

        if not strategy_model:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy model {ask_request.strategy_model} not found",
            )
        if not answer_model:
            raise HTTPException(
                status_code=400,
                detail=f"Answer model {ask_request.answer_model} not found",
            )
        if not final_answer_model:
            raise HTTPException(
                status_code=400,
                detail=f"Final answer model {ask_request.final_answer_model} not found",
            )

        # Check if embedding model is available
        if not await model_manager.get_embedding_model():
            raise HTTPException(
                status_code=400,
                detail="Ask feature requires an embedding model. Please configure one in the Models section.",
            )

        # For streaming response
        return StreamingResponse(
            stream_ask_response(
                ask_request.question, strategy_model, answer_model, final_answer_model
            ),
            media_type="text/plain",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ask operation failed: {str(e)}")


@router.post("/search/ask/simple", response_model=AskResponse)
async def ask_knowledge_base_simple(ask_request: AskRequest):
    """Ask the knowledge base a question and return a simple response (non-streaming)."""
    try:
        # Validate models exist
        strategy_model = await Model.get(ask_request.strategy_model)
        answer_model = await Model.get(ask_request.answer_model)
        final_answer_model = await Model.get(ask_request.final_answer_model)

        if not strategy_model:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy model {ask_request.strategy_model} not found",
            )
        if not answer_model:
            raise HTTPException(
                status_code=400,
                detail=f"Answer model {ask_request.answer_model} not found",
            )
        if not final_answer_model:
            raise HTTPException(
                status_code=400,
                detail=f"Final answer model {ask_request.final_answer_model} not found",
            )

        # Check if embedding model is available
        if not await model_manager.get_embedding_model():
            raise HTTPException(
                status_code=400,
                detail="Ask feature requires an embedding model. Please configure one in the Models section.",
            )

        # Run the ask graph and get final result
        final_answer = None
        async for chunk in ask_graph.astream(
            input=dict(question=ask_request.question),  # type: ignore[arg-type]
            config=dict(
                configurable=dict(
                    strategy_model=strategy_model.id,
                    answer_model=answer_model.id,
                    final_answer_model=final_answer_model.id,
                )
            ),
            stream_mode="updates",
        ):
            if "write_final_answer" in chunk:
                final_answer = chunk["write_final_answer"]["final_answer"]

        if not final_answer:
            raise HTTPException(status_code=500, detail="No answer generated")

        return AskResponse(answer=final_answer, question=ask_request.question)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask simple endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ask operation failed: {str(e)}")


async def stream_agentic_ask_response(
    question: str,
    decision_model: Optional[Model],
    evaluation_model: Optional[Model],
    synthesis_model: Optional[Model],
    max_iterations: int = 10,
    max_tokens: int = 50000,
    max_duration: float = 300.0,
) -> AsyncGenerator[str, None]:
    """Stream the agentic ask response as Server-Sent Events."""
    try:
        final_answer = None
        execution_time = 0.0

        # Prepare config
        config_dict = {"configurable": {}}
        if decision_model:
            config_dict["configurable"]["decision_model"] = decision_model.id
        if evaluation_model:
            config_dict["configurable"]["evaluation_model"] = evaluation_model.id
        if synthesis_model:
            config_dict["configurable"]["synthesis_model"] = synthesis_model.id

        async for chunk in agentic_ask_graph.astream(
            input={"question": question}, config=config_dict, stream_mode="updates"
        ):
            # Stream decision events
            if "agent_decision" in chunk:
                decision_data = chunk["agent_decision"].get("current_decision", {})
                if decision_data:
                    decision_event = {
                        "type": "decision",
                        "action": decision_data.get("action"),
                        "tool_name": decision_data.get("tool_name"),
                        "reasoning": decision_data.get("reasoning", ""),
                    }
                    yield f"data: {json.dumps(decision_event)}\n\n"

            # Stream tool call events
            if "execute_tool" in chunk:
                tool_calls = chunk["execute_tool"].get("current_tool_calls", [])
                for tool_call in tool_calls:
                    tool_event = {
                        "type": "tool_call",
                        "tool": tool_call.get("tool_name", "unknown"),
                        "status": "success" if tool_call.get("success") else "failed",
                        "result_count": len(tool_call.get("data", [])) if tool_call.get("data") else 0,
                    }
                    yield f"data: {json.dumps(tool_event)}\n\n"

            # Stream evaluation events
            if "evaluate" in chunk:
                eval_result = chunk["evaluate"].get("evaluation_result", {})
                if eval_result:
                    eval_event = {
                        "type": "evaluation",
                        "score": eval_result.get("combined_score", 0.0),
                        "decision": eval_result.get("decision", "continue"),
                        "reasoning": eval_result.get("reasoning", ""),
                    }
                    yield f"data: {json.dumps(eval_event)}\n\n"

            # Stream partial answer
            if "synthesize" in chunk:
                answer = chunk["synthesize"].get("partial_answer") or chunk["synthesize"].get("final_answer")
                if answer:
                    answer_event = {"type": "partial_answer", "content": answer}
                    yield f"data: {json.dumps(answer_event)}\n\n"

            # Check for final answer
            if "synthesize" in chunk and chunk["synthesize"].get("final_answer"):
                final_answer = chunk["synthesize"]["final_answer"]
                final_event = {"type": "final_answer", "content": final_answer}
                yield f"data: {json.dumps(final_event)}\n\n"

        # Send completion signal
        completion_event = {
            "type": "complete",
            "final_answer": final_answer,
            "execution_time": execution_time,
        }
        yield f"data: {json.dumps(completion_event)}\n\n"

    except Exception as e:
        logger.error(f"Error in agentic ask streaming: {str(e)}")
        error_event = {"type": "error", "message": str(e), "recoverable": True}
        yield f"data: {json.dumps(error_event)}\n\n"


@router.post("/search/ask/agentic")
async def ask_knowledge_base_agentic(ask_request: AgenticAskRequest):
    """Ask the knowledge base using Agentic RAG (streaming)."""
    try:
        # Check if embedding model is available
        if not await model_manager.get_embedding_model():
            raise HTTPException(
                status_code=400,
                detail="Agentic ask requires an embedding model. Please configure one in the Models section.",
            )

        # Get models (use defaults if not specified)
        decision_model = None
        evaluation_model = None
        synthesis_model = None

        if ask_request.decision_model:
            decision_model = await Model.get(ask_request.decision_model)
        if ask_request.evaluation_model:
            evaluation_model = await Model.get(ask_request.evaluation_model)
        if ask_request.synthesis_model:
            synthesis_model = await Model.get(ask_request.synthesis_model)

        # For streaming response
        return StreamingResponse(
            stream_agentic_ask_response(
                ask_request.question,
                decision_model,
                evaluation_model,
                synthesis_model,
                ask_request.max_iterations or 10,
                ask_request.max_tokens or 50000,
                ask_request.max_duration or 300.0,
            ),
            media_type="text/event-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agentic ask endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Agentic ask operation failed: {str(e)}"
        )


@router.post("/search/ask/agentic/simple", response_model=AgenticAskResponse)
async def ask_knowledge_base_agentic_simple(ask_request: AgenticAskRequest):
    """Ask the knowledge base using Agentic RAG (non-streaming)."""
    import time

    try:
        # Check if embedding model is available
        if not await model_manager.get_embedding_model():
            raise HTTPException(
                status_code=400,
                detail="Agentic ask requires an embedding model. Please configure one in the Models section.",
            )

        # Get models
        decision_model = None
        evaluation_model = None
        synthesis_model = None

        if ask_request.decision_model:
            decision_model = await Model.get(ask_request.decision_model)
        if ask_request.evaluation_model:
            evaluation_model = await Model.get(ask_request.evaluation_model)
        if ask_request.synthesis_model:
            synthesis_model = await Model.get(ask_request.synthesis_model)

        # Prepare config
        config_dict = {"configurable": {}}
        if decision_model:
            config_dict["configurable"]["decision_model"] = decision_model.id
        if evaluation_model:
            config_dict["configurable"]["evaluation_model"] = evaluation_model.id
        if synthesis_model:
            config_dict["configurable"]["synthesis_model"] = synthesis_model.id

        # Run the agentic ask graph
        start_time = time.time()
        final_state = None

        async for chunk in agentic_ask_graph.astream(
            input={
                "question": ask_request.question,
                "max_iterations": ask_request.max_iterations,
                "max_tokens": ask_request.max_tokens,
                "max_duration": ask_request.max_duration,
            },
            config=config_dict,
            stream_mode="updates"
        ):
            # Collect final state
            for key, value in chunk.items():
                if value:
                    if final_state is None:
                        final_state = {}
                    final_state.update(value)

        execution_time = time.time() - start_time

        if not final_state or not final_state.get("final_answer"):
            raise HTTPException(status_code=500, detail="No answer generated")

        return AgenticAskResponse(
            answer=final_state["final_answer"],
            question=ask_request.question,
            iteration_count=final_state.get("iteration_count", 0),
            token_count=final_state.get("token_count", 0),
            execution_time=execution_time,
            search_count=len(final_state.get("search_history", [])),
            reasoning_trace=final_state.get("reasoning_trace", [])[-10:],  # Last 10 reasoning steps
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agentic ask simple endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Agentic ask operation failed: {str(e)}"
        )
