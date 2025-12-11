
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import SourceInsightResponse
from open_notebook.domain.notebook import SourceInsight

router = APIRouter()


@router.get("/insights/{insight_id}", response_model=SourceInsightResponse)
async def get_insight(insight_id: str):
    """Get a specific insight by ID."""
    try:
        insight = await SourceInsight.get(insight_id)
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        # Get source ID from the insight relationship
        source = await insight.get_source()
        
        return SourceInsightResponse(
            id=insight.id or "",
            source_id=source.id or "",
            insight_type=insight.insight_type,
            content=insight.content,
            created=str(insight.created),
            updated=str(insight.updated),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching insight {insight_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching insight: {str(e)}")


@router.delete("/insights/{insight_id}")
async def delete_insight(insight_id: str):
    """Delete a specific insight."""
    try:
        insight = await SourceInsight.get(insight_id)
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        # Delete from Qdrant first (before deleting from SurrealDB)
        try:
            from open_notebook.services.qdrant_service import qdrant_service
            await qdrant_service.delete_source_insight(insight_id)
        except Exception as e:
            logger.warning(f"Failed to delete insight {insight_id} from Qdrant: {e}")
            # Continue with SurrealDB deletion even if Qdrant deletion fails
        
        # Delete from SurrealDB
        await insight.delete()
        
        return {"message": "Insight deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting insight {insight_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting insight: {str(e)}")