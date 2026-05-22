"""
services/analytics/engine.py

Core analytics engine — pure Pandas/NumPy computation.
No FastAPI, no DB, no AI here. Just data transformation.

This is the most testable layer: every method takes a DataFrame
and returns a dict. Unit tests don't need a running server.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.core.exceptions import AnalyticsError, InsufficientDataError
from app.core.logging import get_logger

logger = get_logger(__name__)

MIN_ROWS_FOR_CORRELATION = 5


class AnalyticsEngine:

    # ── Public API ────────────────────────────────────────────

    def full_analysis(self, df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
        """
        Run complete analytics suite and return a single structured dict.
        Called by the analytics route handler.
        """
        if len(df) == 0:
            raise InsufficientDataError("Dataset has no rows.")

        logger.info("analytics_start", dataset_id=dataset_id, rows=len(df), cols=len(df.columns))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "duplicate_count": int(df.duplicated().sum()),
            "health_score": self.compute_health_score(df),
            "summary_stats": self._summary_stats(df, numeric_cols),
            "null_analysis": self._null_analysis(df),
            "outliers": self._outlier_detection(df, numeric_cols),
            "correlation_matrix": self._correlation_matrix(df, numeric_cols),
            "top_categories": self._top_categories(df, categorical_cols),
            "chart_data": self._auto_chart_data(df, numeric_cols, categorical_cols),
        }

    def compute_health_score(self, df: pd.DataFrame) -> float:
        """
        Score 0–100 based on:
        - null rate          (40 pts)
        - duplicate rate     (30 pts)
        - column type mix    (30 pts)

        Recruiters ask "how did you score data quality?" — this is the answer.
        """
        if len(df) == 0:
            return 0.0

        total_cells = df.shape[0] * df.shape[1]
        null_rate = df.isna().sum().sum() / total_cells if total_cells > 0 else 0
        dup_rate = df.duplicated().sum() / len(df)

        has_numeric = any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        has_categorical = any(
            pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
            for c in df.columns
        )
        type_score = 30 if (has_numeric and has_categorical) else 15

        null_score = max(0, 40 * (1 - null_rate))
        dup_score = max(0, 30 * (1 - dup_rate))

        return round(null_score + dup_score + type_score, 1)

    # ── Private methods ───────────────────────────────────────

    def _summary_stats(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict]:
        results = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            results.append({
                "column": col,
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "q25": round(float(series.quantile(0.25)), 4),
                "q75": round(float(series.quantile(0.75)), 4),
            })
        return results

    def _null_analysis(self, df: pd.DataFrame) -> List[Dict]:
        results = []
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            results.append({
                "column": col,
                "null_count": null_count,
                "null_percent": round(null_count / len(df) * 100, 2),
            })
        return sorted(results, key=lambda x: x["null_count"], reverse=True)

    def _outlier_detection(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict]:
        """IQR-based outlier detection — robust to non-normal distributions."""
        results = []
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            if len(outliers) > 0:
                results.append({
                    "column": col,
                    "outlier_count": len(outliers),
                    "outlier_values": [round(v, 4) for v in outliers.head(10).tolist()],
                })
        return results

    def _correlation_matrix(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        if len(numeric_cols) < 2 or len(df) < MIN_ROWS_FOR_CORRELATION:
            return None
        corr = df[numeric_cols].corr().round(4)
        return {
            col: {k: v for k, v in row.items() if not np.isnan(v)}
            for col, row in corr.to_dict().items()
        }

    def _top_categories(
        self, df: pd.DataFrame, categorical_cols: List[str], top_n: int = 10
    ) -> Dict[str, List[Dict]]:
        result: Dict[str, List[Dict]] = {}
        for col in categorical_cols[:5]:   # cap at 5 cols to avoid huge payloads
            counts = df[col].value_counts().head(top_n)
            result[col] = [
                {"value": str(k), "count": int(v)}
                for k, v in counts.items()
            ]
        return result

    def _auto_chart_data(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Generate Plotly-ready chart specs for the most useful charts
        given this dataset's shape.
        """
        charts: Dict[str, Any] = {}

        # Bar chart — first categorical vs first numeric
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            grouped = df.groupby(cat_col)[num_col].mean().head(15).reset_index()
            charts["bar"] = {
                "type": "bar",
                "x": grouped[cat_col].tolist(),
                "y": grouped[num_col].round(2).tolist(),
                "x_label": cat_col,
                "y_label": f"avg {num_col}",
            }

        # Histogram — first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            series = df[col].dropna()
            hist, edges = np.histogram(series, bins=20)
            charts["histogram"] = {
                "type": "histogram",
                "bins": [round(float(e), 4) for e in edges.tolist()],
                "counts": hist.tolist(),
                "column": col,
            }

        # Pie chart — top 8 categories of first categorical
        if categorical_cols:
            col = categorical_cols[0]
            counts = df[col].value_counts().head(8)
            charts["pie"] = {
                "type": "pie",
                "labels": counts.index.tolist(),
                "values": counts.tolist(),
                "column": col,
            }

        # Scatter — first two numeric columns
        if len(numeric_cols) >= 2:
            sample = df[[numeric_cols[0], numeric_cols[1]]].dropna().head(500)
            charts["scatter"] = {
                "type": "scatter",
                "x": sample[numeric_cols[0]].tolist(),
                "y": sample[numeric_cols[1]].tolist(),
                "x_label": numeric_cols[0],
                "y_label": numeric_cols[1],
            }

        return charts
