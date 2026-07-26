"""
services/ai/prompt_builder.py

Builds prompts that give the LLM exactly the right context
to generate a valid operation spec — no more, no less.

Key principle: never send the full dataset to the LLM.
Send schema + sample rows only. This is faster, cheaper,
and prevents leaking sensitive data to an external API.
"""

import json
from typing import List

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_OPERATIONS_DESCRIPTION = """
ALLOWED OPERATIONS (you MUST use exactly one, with EXACTLY these field names):

1. group_aggregate:
{"operation": "group_aggregate", "group_by": "<column>", "aggregate_column": "<column>", "aggregate_func": "sum|mean|count|max|min|median"}

2. filter_sort:
{"operation": "filter_sort", "filter_column": "<column>", "filter_op": "eq|ne|gt|gte|lt|lte|contains", "filter_value": <value>, "sort_by": "<column>", "sort_order": "asc|desc"}

3. top_n:
{"operation": "top_n", "sort_column": "<column>", "n": <int>, "sort_order": "asc|desc"}

4. time_series:
{"operation": "time_series", "date_column": "<column>", "value_column": "<column>", "frequency": "D|W|M|Q|Y"}

5. value_counts:
{"operation": "value_counts", "column": "<column>"}

6. describe:
{"operation": "describe", "column": "<column>"}

7. correlation:
{"operation": "correlation", "col_x": "<column>", "col_y": "<column>"}

8. pivot:
{"operation": "pivot", "index_column": "<column>", "columns_column": "<column>", "values_column": "<column>", "aggregate_func": "sum|mean|count|max|min|median"}

You MUST use the EXACT field names shown above for the chosen operation. Do not omit any required field. Do not invent alternate field names.
"""


class PromptBuilder:

    def build_query_prompt(self, df: pd.DataFrame, question: str) -> str:
        schema = self._build_schema(df)
        sample = self._build_sample(df)

        return f"""You are an analytics assistant. Given a dataset schema and a user question, 
return ONLY a valid JSON operation spec. No explanation. No markdown. Just JSON.

{ALLOWED_OPERATIONS_DESCRIPTION}

DATASET SCHEMA:
{json.dumps(schema, indent=2)}

SAMPLE DATA (first 3 rows):
{sample}

USER QUESTION: {question}

Return a single JSON object with an "operation" field (one of the allowed operations above)
plus the required fields for that operation.
Add a "chart_type" field: one of bar, line, pie, scatter, heatmap, area.

JSON ONLY:"""

    def build_insights_prompt(
        self,
        analytics_data: dict,
        dataset_name: str,
    ) -> str:
        return f"""You are a senior business analyst. Analyze this dataset analytics and generate 
business insights.

DATASET: {dataset_name}
ANALYTICS SUMMARY:
- Rows: {analytics_data.get('row_count')}
- Columns: {analytics_data.get('column_count')}
- Health Score: {analytics_data.get('health_score')}/100
- Duplicates: {analytics_data.get('duplicate_count')}
- Null Analysis: {json.dumps(analytics_data.get('null_analysis', [])[:5], indent=2)}
- Summary Stats: {json.dumps(analytics_data.get('summary_stats', [])[:5], indent=2)}
- Top Categories: {json.dumps(analytics_data.get('top_categories', {}), indent=2)}

Generate a JSON response with this exact structure:
{{
  "executive_summary": "2-3 sentence business overview",
  "insights": [
    {{
      "category": "trend|anomaly|risk|opportunity",
      "title": "Short title",
      "description": "Detailed insight with specific numbers",
      "severity": "info|warning|critical"
    }}
  ]
}}

Generate 4-6 insights. Use specific numbers from the data. Be concise and business-focused.
JSON ONLY:"""

    @staticmethod
    def _build_schema(df: pd.DataFrame) -> List[dict]:
        schema = []
        for col in df.columns:
            series = df[col]
            schema.append({
                "column": col,
                "dtype": str(series.dtype),
                "is_numeric": pd.api.types.is_numeric_dtype(series),
                "sample_values": series.dropna().head(3).tolist(),
                "null_count": int(series.isna().sum()),
            })
        return schema

    @staticmethod
    def _build_sample(df: pd.DataFrame, n: int = 3) -> str:
        try:
            return df.head(n).to_string(index=False, max_cols=10)
        except Exception:
            return "(sample unavailable)"
