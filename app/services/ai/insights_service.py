"""
services/ai/insights_service.py

Generates AI-powered business insights from analytics results.
Combines Gemini's language ability with structured analytics output.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from app.core.exceptions import AIServiceError
from app.core.logging import get_logger
from app.services.ai.gemini_client import GeminiClient
from app.services.ai.prompt_builder import PromptBuilder

logger = get_logger(__name__)


class InsightsService:

    def __init__(self):
        self.gemini = GeminiClient()
        self.prompt_builder = PromptBuilder()

    async def generate_insights(
        self,
        analytics_data: Dict[str, Any],
        dataset_id: str,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """
        Generate structured business insights using Gemini.
        Falls back to rule-based insights if AI call fails.
        """
        prompt = self.prompt_builder.build_insights_prompt(analytics_data, dataset_name)

        try:
            raw_text = await self.gemini.get_insights(prompt)
            parsed = self._parse_insights_response(raw_text)
        except Exception as e:
            logger.warning("ai_insights_fallback", error=str(e), dataset_id=dataset_id)
            parsed = self._rule_based_insights(analytics_data)

        return {
            "dataset_id": dataset_id,
            "executive_summary": parsed.get("executive_summary", "Analysis complete."),
            "insights": parsed.get("insights", []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _parse_insights_response(raw: str) -> Dict[str, Any]:
        """Extract JSON from Gemini response."""
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
        raise AIServiceError("Could not parse insights JSON from Gemini response.")

    @staticmethod
    def _rule_based_insights(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic fallback insights based on analytics data.
        Ensures the endpoint always returns something useful.
        """
        insights = []
        health = analytics_data.get("health_score", 100)

        if health < 70:
            insights.append({
                "category": "risk",
                "title": "Data Quality Issues Detected",
                "description": f"Dataset health score is {health}/100. Review null values and duplicates before relying on this data for decisions.",
                "severity": "warning",
            })

        null_analysis = analytics_data.get("null_analysis", [])
        high_null = [c for c in null_analysis if c.get("null_percent", 0) > 20]
        if high_null:
            cols = ", ".join(c["column"] for c in high_null[:3])
            insights.append({
                "category": "risk",
                "title": "High Missing Values",
                "description": f"Columns with >20% missing data: {cols}. Consider imputation or exclusion.",
                "severity": "warning",
            })

        dups = analytics_data.get("duplicate_count", 0)
        if dups > 0:
            insights.append({
                "category": "anomaly",
                "title": f"{dups} Duplicate Rows Found",
                "description": f"Dataset contains {dups} duplicate rows which may skew aggregations.",
                "severity": "info",
            })

        if not insights:
            insights.append({
                "category": "trend",
                "title": "Dataset Looks Healthy",
                "description": f"Health score {health}/100. No major data quality issues detected.",
                "severity": "info",
            })

        return {
            "executive_summary": f"Dataset has {analytics_data.get('row_count')} rows with a health score of {health}/100.",
            "insights": insights,
        }
