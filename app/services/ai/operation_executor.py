"""
services/ai/operation_executor.py

The operation whitelist — the security boundary of the query system.

Every operation type is a named method with explicit Pandas calls.
Nothing is dynamic. Nothing is eval'd. Nothing reads files.

Supported operations:
  - group_aggregate    : GROUP BY + SUM/MEAN/COUNT/MAX/MIN
  - filter_sort        : WHERE column op value ORDER BY
  - top_n              : top N rows by a column
  - time_series        : resample by time period
  - value_counts       : frequency of a categorical column
  - describe           : summary stats for a numeric column
  - correlation        : correlation between two numeric columns
  - pivot              : simple pivot table
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from app.core.exceptions import UnsafeQueryError

# ── Whitelist definitions ─────────────────────────────────────

ALLOWED_OPERATIONS = {
    "group_aggregate",
    "filter_sort",
    "top_n",
    "time_series",
    "value_counts",
    "describe",
    "correlation",
    "pivot",
}

ALLOWED_AGG_FUNCS = {"sum", "mean", "count", "max", "min", "median"}
ALLOWED_SORT_ORDERS = {"asc", "desc"}
ALLOWED_FILTER_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}
ALLOWED_CHART_TYPES = {"bar", "line", "pie", "scatter", "heatmap", "area"}
ALLOWED_FREQ = {"D", "W", "M", "Q", "Y"}


class OperationExecutor:

    def validate_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the LLM-generated spec against the whitelist.
        Raises UnsafeQueryError with a clear message on any violation.
        """
        if not isinstance(spec, dict):
            raise UnsafeQueryError("Operation spec must be a JSON object.")

        op = spec.get("operation")
        if op not in ALLOWED_OPERATIONS:
            raise UnsafeQueryError(
                f"Operation '{op}' is not allowed.",
                detail={"allowed": list(ALLOWED_OPERATIONS)},
            )

        # Validate per-operation fields
        if op == "group_aggregate":
            self._require_fields(spec, ["group_by", "aggregate_column", "aggregate_func"])
            if spec["aggregate_func"] not in ALLOWED_AGG_FUNCS:
                raise UnsafeQueryError(f"Aggregate function '{spec['aggregate_func']}' not allowed.")

        elif op == "filter_sort":
            self._require_fields(spec, ["filter_column", "filter_op", "filter_value"])
            if spec["filter_op"] not in ALLOWED_FILTER_OPS:
                raise UnsafeQueryError(f"Filter op '{spec['filter_op']}' not allowed.")

        elif op == "top_n":
            self._require_fields(spec, ["sort_column"])
            spec.setdefault("n", 10)
            spec.setdefault("sort_order", "desc")

        elif op == "time_series":
            self._require_fields(spec, ["date_column", "value_column"])
            freq = spec.get("frequency", "M")
            if freq not in ALLOWED_FREQ:
                raise UnsafeQueryError(f"Frequency '{freq}' not allowed.")

        elif op == "correlation":
            self._require_fields(spec, ["col_x", "col_y"])

        # Validate chart type if present
        chart_type = spec.get("chart_type")
        if chart_type and chart_type not in ALLOWED_CHART_TYPES:
            spec["chart_type"] = "bar"   # safe fallback

        return spec

    def execute(
        self, df: pd.DataFrame, spec: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str]:
        """
        Execute a validated spec against a dataframe.
        Returns (result_df, human_readable_summary).
        """
        op = spec["operation"]

        if op == "group_aggregate":
            return self._group_aggregate(df, spec)
        elif op == "filter_sort":
            return self._filter_sort(df, spec)
        elif op == "top_n":
            return self._top_n(df, spec)
        elif op == "time_series":
            return self._time_series(df, spec)
        elif op == "value_counts":
            return self._value_counts(df, spec)
        elif op == "describe":
            return self._describe(df, spec)
        elif op == "correlation":
            return self._correlation(df, spec)
        elif op == "pivot":
            return self._pivot(df, spec)
        else:
            raise UnsafeQueryError(f"Unhandled operation: {op}")

    # ── Operation implementations ─────────────────────────────

    def _group_aggregate(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        group_col = spec["group_by"]
        agg_col = spec["aggregate_column"]
        func = spec["aggregate_func"]
        limit = spec.get("limit", 15)
        sort_order = spec.get("sort_order", "desc")

        result = (
            df.groupby(group_col)[agg_col]
            .agg(func)
            .reset_index()
            .rename(columns={agg_col: f"{func}_{agg_col}"})
            .sort_values(f"{func}_{agg_col}", ascending=(sort_order == "asc"))
            .head(limit)
        )
        summary = (
            f"{func.capitalize()} of '{agg_col}' grouped by '{group_col}'. "
            f"Top result: {result.iloc[0][group_col]} = {result.iloc[0][f'{func}_{agg_col}']:.2f}"
            if len(result) > 0 else "No results."
        )
        return result, summary

    def _filter_sort(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        col = spec["filter_column"]
        op = spec["filter_op"]
        val = spec["filter_value"]
        sort_col = spec.get("sort_column")
        sort_order = spec.get("sort_order", "desc")
        limit = spec.get("limit", 100)

        op_map = {
            "eq": lambda s, v: s == v,
            "ne": lambda s, v: s != v,
            "gt": lambda s, v: s > v,
            "gte": lambda s, v: s >= v,
            "lt": lambda s, v: s < v,
            "lte": lambda s, v: s <= v,
            "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
        }
        mask = op_map[op](df[col], val)
        result = df[mask]
        if sort_col and sort_col in result.columns:
            result = result.sort_values(sort_col, ascending=(sort_order == "asc"))
        result = result.head(limit)
        summary = f"Filtered '{col}' where {op} '{val}'. Found {len(result)} rows."
        return result.reset_index(drop=True), summary

    def _top_n(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        sort_col = spec["sort_column"]
        n = min(spec.get("n", 10), 100)   # cap at 100 for safety
        sort_order = spec.get("sort_order", "desc")
        select_cols = spec.get("select_columns", None)

        result = df.sort_values(sort_col, ascending=(sort_order == "asc")).head(n)
        if select_cols:
            valid_cols = [c for c in select_cols if c in result.columns]
            if valid_cols:
                result = result[valid_cols]

        summary = f"Top {n} rows by '{sort_col}' ({sort_order})."
        return result.reset_index(drop=True), summary

    def _time_series(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        date_col = spec["date_column"]
        val_col = spec["value_column"]
        freq = spec.get("frequency", "M")
        agg_func = spec.get("aggregate_func", "sum")

        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
        df_copy = df_copy.dropna(subset=[date_col])
        df_copy = df_copy.set_index(date_col)

        result = (
            df_copy[val_col]
            .resample(freq)
            .agg(agg_func)
            .reset_index()
        )
        result.columns = ["date", val_col]
        result["date"] = result["date"].dt.strftime("%Y-%m-%d")

        summary = f"Time series of '{val_col}' aggregated by {freq} frequency."
        return result, summary

    def _value_counts(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        col = spec.get("column", df.select_dtypes(include="object").columns[0])
        limit = spec.get("limit", 15)
        counts = df[col].value_counts().head(limit).reset_index()
        counts.columns = [col, "count"]
        summary = f"Value counts for '{col}'. Most common: '{counts.iloc[0][col]}' ({counts.iloc[0]['count']} times)."
        return counts, summary

    def _describe(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        col = spec.get("column")
        if col and col in df.columns:
            result = df[[col]].describe().reset_index()
        else:
            result = df.describe().reset_index()
        summary = f"Summary statistics for '{col or 'all numeric columns'}'."
        return result, summary

    def _correlation(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        col_x = spec["col_x"]
        col_y = spec["col_y"]
        corr_val = df[col_x].corr(df[col_y])
        result = pd.DataFrame({"column_x": [col_x], "column_y": [col_y], "correlation": [round(corr_val, 4)]})
        direction = "positive" if corr_val > 0 else "negative"
        strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.4 else "weak"
        summary = f"{strength.capitalize()} {direction} correlation ({corr_val:.3f}) between '{col_x}' and '{col_y}'."
        return result, summary

    def _pivot(self, df: pd.DataFrame, spec: Dict) -> Tuple[pd.DataFrame, str]:
        index = spec.get("index_column")
        columns = spec.get("pivot_column")
        values = spec.get("value_column")
        agg_func = spec.get("aggregate_func", "sum")

        result = df.pivot_table(
            index=index, columns=columns, values=values, aggfunc=agg_func
        ).reset_index()
        result.columns = [str(c) for c in result.columns]
        summary = f"Pivot table: '{values}' by '{index}' × '{columns}'."
        return result.head(50), summary

    @staticmethod
    def _require_fields(spec: Dict, fields: List[str]) -> None:
        missing = [f for f in fields if f not in spec]
        if missing:
            raise UnsafeQueryError(f"Operation spec missing required fields: {missing}")
