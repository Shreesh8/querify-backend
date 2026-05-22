"""
services/ai/query_service.py

SECURE NATURAL LANGUAGE QUERY SYSTEM

The core security architecture — answer to "how did you prevent hallucinations?":

  ┌─────────────────────────────────────────────────────────┐
  │  WHAT WE DON'T DO (dangerous):                          │
  │    LLM generates Python code → exec(code)               │
  │    This allows arbitrary file I/O, imports, exploits.   │
  │                                                         │
  │  WHAT WE DO (safe):                                     │
  │    LLM generates a structured JSON "operation spec"     │
  │    A whitelisted executor interprets the spec           │
  │    Only pre-approved Pandas operations can run          │
  └─────────────────────────────────────────────────────────┘

Example operation spec the LLM returns:
{
    "operation": "group_aggregate",
    "group_by": "product_category",
    "aggregate_column": "revenue",
    "aggregate_func": "sum",
    "sort_by": "value",
    "sort_order": "desc",
    "limit": 5,
    "chart_type": "bar"
}

The executor only implements ~12 operation types.
Anything outside the spec returns a validation error.
"""

import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd

from app.core.exceptions import AIServiceError, QueryExecutionError, UnsafeQueryError
from app.core.logging import get_logger
from app.services.ai.gemini_client import GeminiClient
from app.services.ai.operation_executor import OperationExecutor
from app.services.ai.prompt_builder import PromptBuilder

logger = get_logger(__name__)


class NLQueryService:

    def __init__(self):
        self.gemini = GeminiClient()
        self.executor = OperationExecutor()
        self.prompt_builder = PromptBuilder()

    async def execute_query(
        self,
        df: pd.DataFrame,
        question: str,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Build context-aware prompt with schema + sample data
        2. Call Gemini → get operation spec (JSON, not code)
        3. Validate the spec against whitelist
        4. Execute via safe executor
        5. Build chart-ready response
        """
        start_time = time.monotonic()

        # Step 1 — build prompt
        prompt = self.prompt_builder.build_query_prompt(df, question)

        # Step 2 — get structured spec from Gemini
        try:
            raw_spec = await self.gemini.get_operation_spec(prompt)
        except Exception as e:
            raise AIServiceError(f"Gemini API failed: {str(e)}")

        # Step 3 — validate (raises UnsafeQueryError on violation)
        validated_spec = self.executor.validate_spec(raw_spec)

        # Step 4 — execute
        try:
            result_df, summary = self.executor.execute(df, validated_spec)
        except Exception as e:
            raise QueryExecutionError(f"Query execution failed: {str(e)}", detail=validated_spec)

        # Step 5 — build response
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        chart_data = self._build_chart(result_df, validated_spec)

        logger.info(
            "query_executed",
            dataset_id=dataset_id,
            operation=validated_spec.get("operation"),
            rows_returned=len(result_df),
            elapsed_ms=elapsed_ms,
        )

        return {
            "answer": summary,
            "result_data": {
                "rows": result_df.to_dict(orient="records"),
                "chart": chart_data,
            },
            "operation_spec": validated_spec,
            "execution_time_ms": elapsed_ms,
        }

    def _build_chart(
        self, df: pd.DataFrame, spec: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if df.empty or len(df.columns) < 1:
            return None
        chart_type = spec.get("chart_type", "bar")
        cols = df.columns.tolist()

        if chart_type == "pie" and len(cols) >= 2:
            return {"type": "pie", "labels": df[cols[0]].tolist(), "values": df[cols[1]].tolist()}
        elif len(cols) >= 2:
            return {
                "type": chart_type,
                "x": df[cols[0]].tolist(),
                "y": df[cols[1]].tolist(),
                "x_label": cols[0],
                "y_label": cols[1],
            }
        return None
