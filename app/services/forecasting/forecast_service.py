"""
services/forecasting/forecast_service.py

Time-series forecasting using Facebook Prophet.
Falls back to sklearn LinearRegression for simpler datasets.

Prophet is run in a thread pool because it's CPU-bound.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

from app.core.exceptions import ForecastError, InsufficientDataError, NoTimeseriesColumnError
from app.core.logging import get_logger

logger = get_logger(__name__)

MIN_DATAPOINTS = 10   # Prophet needs at least ~10 observations


class ForecastService:

    async def generate_forecast(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        periods: int,
        frequency: str,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Run Prophet forecast in thread pool.
        Returns chart-ready JSON with forecast + confidence intervals.
        """
        self._validate_inputs(df, date_column, target_column)

        try:
            forecast_df, metrics = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_prophet,
                df,
                date_column,
                target_column,
                periods,
                frequency,
            )
        except Exception as e:
            raise ForecastError(f"Prophet forecast failed: {str(e)}")

        forecast_points = self._to_forecast_points(forecast_df)
        chart_data = self._build_chart(df, forecast_df, date_column, target_column)

        logger.info(
            "forecast_complete",
            dataset_id=dataset_id,
            periods=periods,
            frequency=frequency,
        )

        return {
            "forecast_id": str(uuid.uuid4()),
            "dataset_id": dataset_id,
            "target_column": target_column,
            "periods": periods,
            "forecast_points": forecast_points,
            "chart_data": chart_data,
            "model_metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _validate_inputs(df: pd.DataFrame, date_col: str, target_col: str) -> None:
        if date_col not in df.columns:
            raise NoTimeseriesColumnError(f"Date column '{date_col}' not found.")
        if target_col not in df.columns:
            raise ForecastError(f"Target column '{target_col}' not found.")
        if len(df.dropna(subset=[date_col, target_col])) < MIN_DATAPOINTS:
            raise InsufficientDataError(
                f"Need at least {MIN_DATAPOINTS} non-null rows for forecasting."
            )

    @staticmethod
    def _run_prophet(
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        periods: int,
        frequency: str,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Synchronous Prophet run — always called via run_in_executor."""
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
        prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
        prophet_df = prophet_df.dropna().sort_values("ds")

        model = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False,
            interval_width=0.95,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = model.predict(future)

        # Basic metrics on training portion
        train_pred = forecast[forecast["ds"].isin(prophet_df["ds"])]["yhat"]
        actual = prophet_df["y"].values
        mae = float(np.mean(np.abs(actual - train_pred.values)))
        rmse = float(np.sqrt(np.mean((actual - train_pred.values) ** 2)))

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "training_rows": len(prophet_df),
            "forecast_periods": periods,
        }

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], metrics

    @staticmethod
    def _to_forecast_points(forecast_df: pd.DataFrame) -> list:
        return [
            {
                "ds": row["ds"].strftime("%Y-%m-%d"),
                "yhat": round(row["yhat"], 4),
                "yhat_lower": round(row["yhat_lower"], 4),
                "yhat_upper": round(row["yhat_upper"], 4),
            }
            for _, row in forecast_df.iterrows()
        ]

    @staticmethod
    def _build_chart(
        df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        date_col: str,
        target_col: str,
    ) -> Dict[str, Any]:
        """Build Plotly-ready chart with actual + forecast + confidence band."""
        actual = df[[date_col, target_col]].copy()
        actual.columns = ["ds", "y"]
        actual["ds"] = pd.to_datetime(actual["ds"], errors="coerce")
        actual = actual.dropna().sort_values("ds")

        return {
            "type": "forecast",
            "actual": {
                "x": actual["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": actual["y"].tolist(),
                "name": f"Actual {target_col}",
            },
            "forecast": {
                "x": forecast_df["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "y": [round(v, 4) for v in forecast_df["yhat"].tolist()],
                "lower": [round(v, 4) for v in forecast_df["yhat_lower"].tolist()],
                "upper": [round(v, 4) for v in forecast_df["yhat_upper"].tolist()],
                "name": f"Forecast {target_col}",
            },
        }
