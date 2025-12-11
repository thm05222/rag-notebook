"""
Insights service layer using API.
"""

from typing import List, Optional

from loguru import logger

from api.client import api_client
from open_notebook.domain.notebook import SourceInsight


class InsightsService:
    """Service layer for insights operations using API."""
    
    def __init__(self):
        logger.info("Using API for insights operations")
    
    def get_source_insights(self, source_id: str) -> List[SourceInsight]:
        """Get all insights for a specific source."""
        insights_data = api_client.get_source_insights(source_id)
        # Convert API response to SourceInsight objects
        insights = []
        for insight_data in insights_data:
            insight = SourceInsight(
                insight_type=insight_data["insight_type"],
                content=insight_data["content"],
            )
            insight.id = insight_data["id"]
            insight.created = insight_data["created"]
            insight.updated = insight_data["updated"]
            insights.append(insight)
        return insights
    
    def get_insight(self, insight_id: str) -> SourceInsight:
        """Get a specific insight."""
        insight_response = api_client.get_insight(insight_id)
        insight_data = insight_response if isinstance(insight_response, dict) else insight_response[0]
        insight = SourceInsight(
            insight_type=insight_data["insight_type"],
            content=insight_data["content"],
        )
        insight.id = insight_data["id"]
        insight.created = insight_data["created"]
        insight.updated = insight_data["updated"]
        # Note: source_id from API response is not stored; use await insight.get_source() if needed
        return insight
    
    def delete_insight(self, insight_id: str) -> bool:
        """Delete a specific insight."""
        api_client.delete_insight(insight_id)
        return True
    
    def create_source_insight(self, source_id: str, transformation_id: str, model_id: Optional[str] = None) -> SourceInsight:
        """Create a new insight for a source by running a transformation."""
        insight_response = api_client.create_source_insight(source_id, transformation_id, model_id)
        insight_data = insight_response if isinstance(insight_response, dict) else insight_response[0]
        insight = SourceInsight(
            insight_type=insight_data["insight_type"],
            content=insight_data["content"],
        )
        insight.id = insight_data["id"]
        insight.created = insight_data["created"]
        insight.updated = insight_data["updated"]
        # Note: source_id from API response is not stored; use await insight.get_source() if needed
        return insight


# Global service instance
insights_service = InsightsService()