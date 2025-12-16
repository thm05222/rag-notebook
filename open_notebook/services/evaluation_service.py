"""
Evaluation service for Agentic RAG.
Provides both LLM-based and rule-based evaluation for search results and answers.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from open_notebook.exceptions import EvaluationError
from open_notebook.graphs.utils import provision_langchain_model
from open_notebook.utils import clean_thinking_content


class EvaluationService:
    """Service for evaluating search results and generated answers."""

    async def evaluate_results(
        self,
        question: str,
        results: List[Dict[str, Any]],
        answer: Optional[str] = None,
        model_id: Optional[str] = None,
        use_two_stage: bool = True,  # 新增參數：是否使用兩階段評估
        config: Optional[Dict[str, Any]] = None,  # 新增參數：配置信息
    ) -> Dict[str, Any]:
        """
        評估搜尋結果和答案質量。
        
        Args:
            use_two_stage: 是否使用兩階段評估方法（默認 True）
            config: 配置信息，用於獲取階段2模型ID等
        """
        # Step 1: 快速規則基礎驗證
        rule_score = self._rule_based_evaluation(results, answer, question)
        
        # Step 2: LLM 評估（兩階段或單階段）
        try:
            if use_two_stage:
                llm_score = await self._llm_evaluation_two_stage(
                    question, results, answer, model_id, config
                )
            else:
                llm_score = await self._llm_evaluation(question, results, answer, model_id)
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}, using fallback")
            llm_score = self._fallback_evaluation(results, answer, question)
        
        # Step 3: 整合規則基礎檢測和 LLM 檢測的 Hallucination 結果
        rule_hallucination = await self.detect_hallucination(answer or "", results)
        llm_hallucination = llm_score.get("hallucination", {})
        
        # 合併 Hallucination 檢測結果（取較高的風險分數）
        combined_hallucination_risk = max(
            rule_hallucination.get("hallucination_risk_score", 0.0),
            llm_hallucination.get("risk_score", 0.0)
        )
        has_hallucination_risk = (
            rule_hallucination.get("has_hallucination_risk", False) or
            llm_hallucination.get("has_risk", False)
        )
        
        # Step 4: 計算綜合分數（考慮新的評估維度）
        # 權重分配：
        # - 規則分數：30%
        # - LLM 總分：40%
        # - 新評估維度：30%（完整性、相關性、引用質量、一致性的平均值）
        
        dimension_scores = [
            llm_score.get("completeness_score", 0.5),
            llm_score.get("relevance_score", 0.5),
            llm_score.get("citation_quality_score", 0.5),
            llm_score.get("consistency_score", 0.5),
        ]
        dimension_avg = sum(dimension_scores) / len(dimension_scores) if dimension_scores else 0.5
        
        combined_score = (
            rule_score * 0.3 +
            llm_score["score"] * 0.4 +
            dimension_avg * 0.3
        )
        
        # 如果存在高 Hallucination 風險，降低綜合分數
        if has_hallucination_risk and combined_hallucination_risk > 0.6:
            combined_score *= 0.7  # 降低 30%
        elif has_hallucination_risk and combined_hallucination_risk > 0.3:
            combined_score *= 0.85  # 降低 15%
        
        # Step 5: 確定決策
        decision = self._determine_decision(
            combined_score,
            llm_score.get("reasoning", ""),
            has_hallucination_risk,
            combined_hallucination_risk
        )
        
        return {
            "rule_score": rule_score,
            "llm_score": llm_score["score"],
            "combined_score": combined_score,
            "decision": decision,
            "reasoning": llm_score.get("reasoning", ""),
            "confidence": llm_score.get("confidence", 0.5),
            
            # Hallucination 檢測結果
            "hallucination": {
                "has_risk": has_hallucination_risk,
                "risk_score": combined_hallucination_risk,
                "rule_based": rule_hallucination,
                "llm_based": llm_hallucination,
            },
            
            # 新評估維度
            "completeness_score": llm_score.get("completeness_score", 0.5),
            "relevance_score": llm_score.get("relevance_score", 0.5),
            "citation_quality_score": llm_score.get("citation_quality_score", 0.5),
            "consistency_score": llm_score.get("consistency_score", 0.5),
            "dimension_avg": dimension_avg,
            
            # 詳細說明
            "completeness_notes": llm_score.get("completeness_notes"),
            "relevance_notes": llm_score.get("relevance_notes"),
            "citation_notes": llm_score.get("citation_notes"),
            "consistency_notes": llm_score.get("consistency_notes"),
        }

    def _rule_based_evaluation(
        self,
        results: List[Dict[str, Any]],
        answer: Optional[str],
        question: str,
    ) -> float:
        """
        Fast rule-based evaluation (no LLM cost).
        Returns score 0.0 to 1.0.
        """
        score = 0.0
        
        # Result count check (max 0.3 points)
        if results and len(results) > 0:
            result_count_score = min(len(results) / 10.0, 1.0) * 0.3
            score += result_count_score
        
        # Answer completeness check (max 0.3 points)
        if answer:
            answer_length = len(answer.strip())
            if answer_length > 100:
                score += 0.2
            if answer_length > 500:
                score += 0.1
        
        # Citation check (max 0.4 points)
        if answer:
            # Simple check: count citations in format [document_id]
            import re
            citations = re.findall(r'\[([^\]]+)\]', answer)
            if citations:
                score += min(len(citations) / 3.0, 1.0) * 0.4
        
        return min(score, 1.0)

    async def _llm_evaluation_two_stage(
        self,
        question: str,
        results: List[Dict[str, Any]],
        answer: Optional[str],
        model_id: Optional[str],
        config: Optional[Dict[str, Any]] = None,
        use_small_model_for_stage2: bool = True,
    ) -> Dict[str, Any]:
        """
        兩階段 LLM 評估方法。
        
        階段1：使用強模型進行自由格式評估（鼓勵自我批評）
        階段2：使用較小模型將評估轉換為結構化輸出
        """
        from ai_prompter import Prompter
        from pydantic import ValidationError
        
        # 階段1：自由格式評估
        stage1_prompt_data = {
            "question": question,
            "collected_results": results,
            "answer": answer or "No answer generated yet",
        }
        
        stage1_prompt = Prompter(
            prompt_template="chat_agentic/evaluation_stage1",
            parser=None,
        )
        stage1_prompt_text = stage1_prompt.render(data=stage1_prompt_data)
        
        # 使用強模型進行第一階段評估
        stage1_model = await provision_langchain_model(
            stage1_prompt_text,
            model_id,
            "evaluation",
            max_tokens=2000,  # 允許更長的推理過程
        )
        
        stage1_response = await stage1_model.ainvoke(stage1_prompt_text)
        stage1_content = stage1_response.content if isinstance(stage1_response.content, str) else str(stage1_response.content)
        evaluation_text = clean_thinking_content(stage1_content)
        
        logger.debug(f"Stage 1 evaluation text length: {len(evaluation_text)}")
        
        # 階段2：結構化輸出
        stage2_prompt_data = {
            "evaluation_text": evaluation_text,
        }
        
        stage2_prompt = Prompter(
            prompt_template="chat_agentic/evaluation_stage2",
            parser=None,
        )
        stage2_prompt_text = stage2_prompt.render(data=stage2_prompt_data)
        
        # 使用較小模型進行格式轉換（節省成本）
        stage2_model_id = None
        if use_small_model_for_stage2 and config:
            stage2_model_id = config.get("configurable", {}).get("evaluation_formatting_model")
        
        stage2_model = await provision_langchain_model(
            stage2_prompt_text,
            stage2_model_id or model_id,
            "evaluation",
            max_tokens=1000,
            structured=dict(type="json"),  # 使用結構化輸出
        )
        
        stage2_response = await stage2_model.ainvoke(stage2_prompt_text)
        stage2_content = stage2_response.content if isinstance(stage2_response.content, str) else str(stage2_response.content)
        cleaned_content = clean_thinking_content(stage2_content)
        
        # 解析為 Pydantic 模型
        try:
            from open_notebook.services.evaluation_models import EvaluationResult
            import json
            import re
            
            # 嘗試提取 JSON
            json_match = re.search(r'\{[^}]+\}', cleaned_content, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                eval_data = json.loads(cleaned_content)
            
            # 驗證並轉換為 Pydantic 模型
            evaluation_result = EvaluationResult(**eval_data)
            
            return {
                "score": evaluation_result.score,
                "reasoning": evaluation_result.reasoning,
                "confidence": evaluation_result.confidence,
                "decision": evaluation_result.decision,
                "hallucination": {
                    "has_risk": evaluation_result.hallucination.has_risk,
                    "risk_score": evaluation_result.hallucination.risk_score,
                    "unsupported_claims": evaluation_result.hallucination.unsupported_claims,
                    "invalid_citations": evaluation_result.hallucination.invalid_citations,
                    "contradictory_info": evaluation_result.hallucination.contradictory_info,
                    "over_extrapolation": evaluation_result.hallucination.over_extrapolation,
                    "notes": evaluation_result.hallucination.notes,
                },
                "completeness_score": evaluation_result.completeness_score,
                "relevance_score": evaluation_result.relevance_score,
                "citation_quality_score": evaluation_result.citation_quality_score,
                "consistency_score": evaluation_result.consistency_score,
                "completeness_notes": evaluation_result.completeness_notes,
                "relevance_notes": evaluation_result.relevance_notes,
                "citation_notes": evaluation_result.citation_notes,
                "consistency_notes": evaluation_result.consistency_notes,
                "stage1_evaluation": evaluation_text,  # 保留原始評估文本
            }
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse structured evaluation, falling back to text parsing: {e}")
            # Fallback 到文本解析
            return self._parse_evaluation_text(cleaned_content)

    async def _llm_evaluation(
        self,
        question: str,
        results: List[Dict[str, Any]],
        answer: Optional[str],
        model_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        LLM-based quality evaluation.
        Returns dict with score, reasoning, and confidence.
        """
        from ai_prompter import Prompter
        
        # Prepare evaluation prompt
        prompt_data = {
            "question": question,
            "result_count": len(results),
            "results_summary": self._summarize_results(results),
            "answer": answer or "No answer generated yet",
        }
        
        system_prompt = Prompter(
            prompt_template="agentic_ask/evaluation",
            parser=None,
        ).render(data=prompt_data)
        
        try:
            model = await provision_langchain_model(
                system_prompt,
                model_id,
                "evaluation",
                max_tokens=1000,
            )
            
            response = await model.ainvoke(system_prompt)
            content = response.content if isinstance(response.content, str) else str(response.content)
            cleaned_content = clean_thinking_content(content)
            
            # Parse evaluation from response
            # Expected format: JSON with score, reasoning, confidence, decision
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', cleaned_content, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                # Fallback: parse from text
                eval_data = self._parse_evaluation_text(cleaned_content)
            
            return {
                "score": float(eval_data.get("score", 0.5)),
                "reasoning": eval_data.get("reasoning", ""),
                "confidence": float(eval_data.get("confidence", 0.5)),
                "decision": eval_data.get("decision", "continue"),
            }
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            raise EvaluationError(f"LLM evaluation failed: {str(e)}") from e

    def _fallback_evaluation(
        self,
        results: List[Dict[str, Any]],
        answer: Optional[str],
        question: str,
    ) -> Dict[str, Any]:
        """Fallback evaluation when LLM evaluation fails."""
        rule_score = self._rule_based_evaluation(results, answer, question)
        
        # Simple decision logic based on rule score
        if rule_score >= 0.7:
            decision = "synthesize"
        elif rule_score >= 0.5:
            decision = "refine_search"
        else:
            decision = "continue"
        
        return {
            "score": rule_score,
            "reasoning": f"Fallback evaluation: rule-based score is {rule_score:.2f}",
            "confidence": 0.6,
            "decision": decision,
        }

    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize search results for evaluation prompt."""
        if not results:
            return "No results found."
        
        summary = f"Found {len(results)} results:\n"
        for i, result in enumerate(results[:5], 1):  # Limit to top 5
            title = result.get("title", "Untitled")
            similarity = result.get("similarity", 0.0)
            # 關鍵修復：支持不同的內容格式
            content = result.get("content", "")
            if not content and result.get("matches"):
                # 支持 matches 數組格式（從 vector_search 返回）
                content = result["matches"][0] if isinstance(result["matches"], list) and result["matches"] else ""
            content_preview = content[:100] if content else "No content"
            summary += f"{i}. {title} (similarity: {similarity:.2f}): {content_preview}...\n"
        
        return summary

    def _parse_evaluation_text(self, text: str) -> Dict[str, Any]:
        """Parse evaluation from text response (fallback)."""
        import re
        
        # Try to extract score
        score_match = re.search(r'score[:\s]+([0-9.]+)', text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5
        
        # Try to extract decision
        decision = "continue"
        if re.search(r'synthesize|generate|final', text, re.IGNORECASE):
            decision = "synthesize"
        elif re.search(r'refine|improve|search', text, re.IGNORECASE):
            decision = "refine_search"
        elif re.search(r'continue|more|additional', text, re.IGNORECASE):
            decision = "continue"
        
        return {
            "score": min(max(score, 0.0), 1.0),
            "reasoning": text[:200],  # First 200 chars
            "confidence": 0.6,
            "decision": decision,
        }

    def _determine_decision(
        self,
        combined_score: float,
        reasoning: str,
        has_hallucination_risk: bool = False,
        hallucination_risk_score: float = 0.0,
    ) -> str:
        """
        根據評估分數和 Hallucination 風險確定下一步行動。
        """
        # 如果存在高 Hallucination 風險，拒絕答案
        if has_hallucination_risk and hallucination_risk_score > 0.6:
            return "reject"
        
        # 標準決策邏輯
        if combined_score >= 0.75:
            return "synthesize"
        elif combined_score >= 0.5:
            # 如果有中等 Hallucination 風險，優化搜尋而不是直接生成
            if has_hallucination_risk and hallucination_risk_score > 0.3:
                return "refine_search"
            return "synthesize"
        elif combined_score >= 0.3:
            return "refine_search"
        else:
            return "continue"

    async def detect_hallucination(
        self,
        answer: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations in the answer.
        Uses rule-based checks (embedding similarity, citation validation).
        """
        import re
        from typing import Set
        
        # 關鍵修復：檢查空答案
        if not answer or len(answer.strip()) == 0:
            logger.warning("detect_hallucination: Answer is empty, skipping hallucination detection")
            return {
                "has_hallucination_risk": False,
                "hallucination_risk_score": 0.0,
                "valid_citations": [],
                "invalid_citations": [],
                "uncited_sentences": [],
                "citation_ratio": 0.0,
                "note": "Answer is empty, cannot detect hallucinations",
            }
        
        # Extract citations from answer
        citations = re.findall(r'\[([^\]]+)\]', answer)
        citation_ids: Set[str] = set(citations)
        
        # Get result IDs
        result_ids: Set[str] = set()
        for result in results:
            result_id = result.get("id", "")
            if result_id:
                result_ids.add(result_id)
            parent_id = result.get("parent_id", "")
            if parent_id:
                result_ids.add(parent_id)
        
        # Validate citations
        valid_citations = citation_ids.intersection(result_ids)
        invalid_citations = citation_ids - result_ids
        
        # Check for uncited claims (simple heuristic: long sentences without citations)
        sentences = re.split(r'[.!?]+', answer)
        uncited_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 50 and not re.search(r'\[[^\]]+\]', sentence):
                uncited_sentences.append(sentence.strip()[:100])
        
        # Calculate hallucination risk score
        total_claims = len([s for s in sentences if len(s.strip()) > 20])
        
        # 關鍵修復：如果沒有有效的斷言（total_claims = 0），不應該判定為高風險
        if total_claims == 0:
            logger.warning("detect_hallucination: No valid claims found in answer, cannot calculate risk")
            return {
                "has_hallucination_risk": False,
                "hallucination_risk_score": 0.0,
                "valid_citations": list(valid_citations),
                "invalid_citations": list(invalid_citations),
                "uncited_sentences": uncited_sentences[:3],
                "citation_ratio": 0.0,
                "note": "No valid claims found, cannot detect hallucinations",
            }
        
        citation_ratio = len(valid_citations) / max(total_claims, 1)
        hallucination_risk = 1.0 - min(citation_ratio, 1.0)
        
        return {
            "has_hallucination_risk": hallucination_risk > 0.3,
            "hallucination_risk_score": hallucination_risk,
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "uncited_sentences": uncited_sentences[:3],  # Top 3
            "citation_ratio": citation_ratio,
        }


# Global evaluation service instance
evaluation_service = EvaluationService()

