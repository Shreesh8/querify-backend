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
import math

from app.core.exceptions import AIServiceError, QueryExecutionError, UnsafeQueryError
from app.core.logging import get_logger
from app.services.ai.groq_client import GroqClient
from app.services.ai.operation_executor import OperationExecutor
from app.services.ai.prompt_builder import PromptBuilder

logger = get_logger(__name__)


def _sanitize_nans(obj):
    """Recursively replace NaN floats with None so the result is valid JSON
    (Postgres JSONB rejects the literal token 'NaN')."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nans(v) for v in obj]
    return obj


class NLQueryService:

    def __init__(self):
        self.groq = GroqClient()
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
            raw_spec = await self.groq.get_operation_spec(prompt)
        except Exception as e:
            raise AIServiceError(f"Groq API failed: {str(e)}")

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


        # Step 6 — generate smart natural language answer
        try:
            smart_answer = await self._build_smart_answer(question, result_df, summary)
        except Exception:
            smart_answer = summary

        logger.info(
            "query_executed",
            dataset_id=dataset_id,
            operation=validated_spec.get("operation"),
            rows_returned=len(result_df),
            elapsed_ms=elapsed_ms,
        )

        result_data = _sanitize_nans({
            "rows": result_df.to_dict(orient="records"),
            "chart": chart_data,
        })
        return {
            "answer": smart_answer,
            "result_data": result_data,
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
        op = spec.get("operation")

        # top_n / filter_sort return full rows (all original columns).
        # Pick a label column (x) and a numeric value column (y) instead
        # of blindly using the first two columns.
        if op in ("top_n", "filter_sort") and len(cols) > 2:
            y_col = spec.get("sort_column") or spec.get("sort_by")
            if y_col not in cols or not pd.api.types.is_numeric_dtype(df[y_col]):
                numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
                y_col = numeric_cols[0] if numeric_cols else None

            non_numeric_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
            x_col = non_numeric_cols[0] if non_numeric_cols else cols[0]

            if y_col is None or x_col == y_col:
                return None

            if chart_type == "pie":
                return {"type": "pie", "labels": df[x_col].tolist(), "values": df[y_col].tolist()}
            return {
                "type": chart_type,
                "x": df[x_col].tolist(),
                "y": df[y_col].tolist(),
                "x_label": x_col,
                "y_label": y_col,
            }

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

    async def _build_smart_answer(self, question: str, result_df, fallback: str) -> str:
        """Use Groq to generate a natural language answer from query results."""
        rows = result_df.to_dict(orient="records")
        total = len(rows)
        if total == 0:
            return "No results found for your query."

        # Build a compact sample for the prompt
        cols = list(result_df.columns)
        sample = rows[:5]

        # Pick the most "label-like" column for listing examples
        label_col = cols[0]
        examples = [str(r.get(label_col, "")) for r in rows[:5]]
        examples_str = ", ".join(examples)
        more = f" and {total - 5} more" if total > 5 else ""

        prompt = f"""You are a data analyst assistant. Answer this question naturally in 1-2 sentences using the data below.

Question: {question}
Total results: {total}
Columns: {cols}
Sample rows: {sample}

Write a concise, friendly answer. Mention specific values (like top items, counts, names). 
If there are many results, name a few notable ones and say "and X more".
Example: "There are 72 teams in the Group Stage, including Brazil, France, England{more}."
Just the answer, no preamble."""

        text = await self.groq.get_insights(prompt)
        return text.strip() if text else fallback
