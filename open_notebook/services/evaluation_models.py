"""
Pydantic models for LLM evaluation results.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class HallucinationDetails(BaseModel):
    """Hallucination 檢測詳細信息"""
    has_risk: bool = Field(description="是否存在幻覺風險")
    risk_score: float = Field(ge=0.0, le=1.0, description="風險分數 (0.0-1.0)")
    unsupported_claims: List[str] = Field(default_factory=list, description="未支持的聲明列表")
    invalid_citations: List[str] = Field(default_factory=list, description="無效引用列表")
    contradictory_info: List[str] = Field(default_factory=list, description="矛盾信息列表")
    over_extrapolation: List[str] = Field(default_factory=list, description="過度推斷列表")
    notes: Optional[str] = Field(None, description="額外說明")


class EvaluationResult(BaseModel):
    """LLM 評估結果模型"""
    score: float = Field(ge=0.0, le=1.0, description="總體質量分數 (0.0-1.0)")
    reasoning: str = Field(description="評估理由和推理過程")
    confidence: float = Field(ge=0.0, le=1.0, description="評估信心 (0.0-1.0)")
    decision: Literal["continue", "refine_search", "synthesize", "reject"] = Field(
        description="決策：continue=繼續搜尋, refine_search=優化搜尋, synthesize=生成答案, reject=拒絕答案"
    )
    hallucination: HallucinationDetails = Field(description="Hallucination 檢測結果")
    
    # 新增評估維度
    completeness_score: float = Field(ge=0.0, le=1.0, description="信息完整性分數")
    relevance_score: float = Field(ge=0.0, le=1.0, description="相關性分數")
    citation_quality_score: float = Field(ge=0.0, le=1.0, description="引用質量分數")
    consistency_score: float = Field(ge=0.0, le=1.0, description="一致性分數（答案內部邏輯）")
    
    # 詳細評估說明
    completeness_notes: Optional[str] = Field(None, description="完整性評估說明")
    relevance_notes: Optional[str] = Field(None, description="相關性評估說明")
    citation_notes: Optional[str] = Field(None, description="引用質量評估說明")
    consistency_notes: Optional[str] = Field(None, description="一致性評估說明")

