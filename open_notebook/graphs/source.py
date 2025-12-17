import operator
from typing import Any, Dict, List, Optional

from content_core import extract_content
from content_core.common import ProcessSourceState
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from typing_extensions import Annotated, TypedDict

from open_notebook.domain.content_settings import ContentSettings
from open_notebook.domain.notebook import Asset, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.graphs.transformation import graph as transform_graph


class SourceState(TypedDict):
    content_state: ProcessSourceState
    apply_transformations: List[Transformation]
    source_id: str
    notebook_ids: List[str]
    source: Source
    transformation: Annotated[list, operator.add]
    embed: bool
    build_pageindex: bool


class TransformationState(TypedDict):
    source: Source
    transformation: Transformation


async def content_process(state: SourceState) -> dict:
    content_settings = ContentSettings(
        default_content_processing_engine_doc="auto",
        default_content_processing_engine_url="auto",
        default_embedding_option="ask",
        auto_delete_files="yes",
        youtube_preferred_languages=["en", "pt", "es", "de", "nl", "en-GB", "fr", "hi", "ja"]
    )
    content_state: Dict[str, Any] = state["content_state"]  # type: ignore[assignment]

    # Preserve original file_path and url before processing
    original_file_path = content_state.get("file_path")
    original_url = content_state.get("url")

    content_state["url_engine"] = (
        content_settings.default_content_processing_engine_url or "auto"
    )
    content_state["document_engine"] = (
        content_settings.default_content_processing_engine_doc or "auto"
    )
    content_state["output_format"] = "markdown"

    processed_state = await extract_content(content_state)
    
    # Convert ProcessSourceOutput to dict if needed
    if hasattr(processed_state, 'model_dump'):
        # It's a Pydantic model or similar
        processed_state = processed_state.model_dump()
    elif hasattr(processed_state, '__dict__'):
        # It's an object with __dict__
        processed_state = dict(processed_state.__dict__)
    elif not isinstance(processed_state, dict):
        # Try to convert using dict() constructor
        try:
            processed_state = dict(processed_state)
        except (TypeError, ValueError):
            # If all else fails, convert to dict using vars() or get attributes
            processed_state = vars(processed_state) if hasattr(processed_state, '__dict__') else {}
            logger.warning(f"Had to use vars() to convert processed_state to dict. Type: {type(processed_state)}")
    
    # Ensure file_path and url are preserved in processed state
    # Always preserve file_path and url if they were in the original state
    if original_file_path:
        processed_state["file_path"] = original_file_path
        logger.debug(f"Preserved file_path in processed state: {original_file_path}")
    if original_url:
        processed_state["url"] = original_url
        logger.debug(f"Preserved url in processed state: {original_url}")
    
    return {"content_state": processed_state}


async def save_source(state: SourceState) -> dict:
    content_state = state["content_state"]

    # Get existing source using the provided source_id
    source = await Source.get(state["source_id"])
    if not source:
        raise ValueError(f"Source with ID {state['source_id']} not found")

    # Access content_state as dictionary to safely get values
    # ProcessSourceState is a TypedDict, so we can use both dict and attribute access
    # But using dict access is safer and more explicit
    content_state_dict = content_state if isinstance(content_state, dict) else dict(content_state)
    url = content_state_dict.get("url")
    file_path = content_state_dict.get("file_path")
    content = content_state_dict.get("content")
    title = content_state_dict.get("title")

    # Log content_state for debugging
    logger.info(f"Saving source {source.id}: url={url}, file_path={file_path}, has_content={bool(content)}, content_length={len(content) if content else 0}")
    
    # Log all keys in content_state for debugging
    if not content:
        logger.warning(f"Source {source.id} has no content in processed state. Available keys: {list(content_state_dict.keys())}")
        logger.debug(f"Full content_state (first 500 chars): {str(content_state_dict)[:500]}")

    # Update the source with processed content
    source.asset = Asset(url=url, file_path=file_path)
    source.full_text = content or ""  # Ensure full_text is at least empty string, not None
    
    # Only use content title as fallback if no custom title was provided
    if title and (not source.title or source.title == "Processing..."):
        source.title = title
    
    await source.save()

    # NOTE: Notebook associations are created by the API immediately for UI responsiveness
    # No need to create them here to avoid duplicate edges

    if state["embed"]:
        logger.debug("Embedding content for vector search")
        await source.vectorize()

    # 添加：可選的 PageIndex 索引建立
    if state.get("build_pageindex", False):
        try:
            from open_notebook.services.pageindex_service import pageindex_service
            
            if pageindex_service.is_available() and source.full_text:
                logger.info(f"Building PageIndex for source {source.id}...")
                try:
                    # 同步建立索引（確保狀態更新）
                    await pageindex_service._get_or_create_index_for_source(source.id, source)
                    logger.info(f"Successfully built PageIndex for source {source.id}")
                except Exception as pageindex_error:
                    # 捕獲 PageIndex 建構錯誤，但不影響主流程
                    logger.warning(
                        f"PageIndex build failed for source {source.id} "
                        f"(likely unstructured text or too short). "
                        f"Fallback to Vector Search only. Error: {pageindex_error}"
                    )
                    # Source 的 pageindex_structure 保持為 None，表示不支援 PageIndex
                    # 這已經足夠標記該 Source 為「僅支援向量搜尋」
            elif not pageindex_service.is_available():
                logger.warning(f"PageIndex not available, skipping index building for source {source.id}")
            elif not source.full_text:
                logger.warning(f"Source {source.id} has no content, skipping PageIndex building")
        except Exception as e:
            logger.warning(f"Failed to start PageIndex building for source {source.id}: {e}")
            # 不拋出異常，避免影響主流程

    return {"source": source}


def trigger_transformations(state: SourceState, config: RunnableConfig) -> List[Send]:
    if len(state["apply_transformations"]) == 0:
        return []

    to_apply = state["apply_transformations"]
    logger.debug(f"Applying transformations {to_apply}")

    return [
        Send(
            "transform_content",
            {
                "source": state["source"],
                "transformation": t,
            },
        )
        for t in to_apply
    ]


async def transform_content(state: TransformationState) -> Optional[dict]:
    source = state["source"]
    content = source.full_text
    if not content:
        return None
    transformation: Transformation = state["transformation"]

    logger.debug(f"Applying transformation {transformation.name}")
    result = await transform_graph.ainvoke(
        dict(input_text=content, transformation=transformation)  # type: ignore[arg-type]
    )
    await source.add_insight(transformation.title, result["output"])
    return {
        "transformation": [
            {
                "output": result["output"],
                "transformation_name": transformation.name,
            }
        ]
    }


# Create and compile the workflow
workflow = StateGraph(SourceState)

# Add nodes
workflow.add_node("content_process", content_process)
workflow.add_node("save_source", save_source)
workflow.add_node("transform_content", transform_content)
# Define the graph edges
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")
workflow.add_conditional_edges(
    "save_source", trigger_transformations, ["transform_content"]
)
workflow.add_edge("transform_content", END)

# Compile the graph
source_graph = workflow.compile()
