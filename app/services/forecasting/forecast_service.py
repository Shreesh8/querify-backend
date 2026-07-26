import asyncio
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from prophet import Prophet
from app.core.exceptions import ForecastError, InsufficientDataError, NoTimeseriesColumnError
from app.core.logging import get_logger
logger = get_logger(__name__)
MIN_DATAPOINTS = 10

class ForecastService:
    async def generate_forecast(self, df, date_column, target_column, periods, frequency, dataset_id):
        if date_column not in df.columns:
            raise NoTimeseriesColumnError(f"Date column '{date_column}' not found.")
        if target_column not in df.columns:
            raise ForecastError(f"Target column '{target_column}' not found.")
        try:
            forecast_df, metrics = await asyncio.get_event_loop().run_in_executor(
                None, self._run_prophet, df, date_column, target_column, periods, frequency)
        except Exception as e:
            raise ForecastError(f"Prophet forecast failed: {str(e)}")
        return {
            "forecast_id": str(uuid.uuid4()),
            "dataset_id": dataset_id,
            "target_column": target_column,
            "periods": periods,
            "forecast_points": [{"ds": row["ds"].strftime("%Y-%m-%d"), "yhat": round(row["yhat"], 4), "yhat_lower": round(row["yhat_lower"], 4), "yhat_upper": round(row["yhat_upper"], 4)} for _, row in forecast_df.iterrows()],
            "chart_data": self._build_chart(df, forecast_df, date_column, target_column),
            "model_metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _run_prophet(df, date_col, target_col, periods, frequency):
        p = df[[date_col, target_col]].copy()
        p.columns = ["ds", "y"]
        p["ds"] = pd.to_datetime(p["ds"], errors="coerce")
        p["y"] = pd.to_numeric(p["y"], errors="coerce")
        p = p.dropna(subset=["ds", "y"])
        p = p.groupby("ds")["y"].mean().reset_index().sort_values("ds").reset_index(drop=True)
        if len(p) < MIN_DATAPOINTS:
            raise InsufficientDataError(f"Need at least {MIN_DATAPOINTS} rows.")
        model = Prophet(yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality=False, interval_width=0.95)
        model.fit(p)
        future = model.make_future_dataframe(periods=periods, freq=frequency)
        forecast = model.predict(future)
        t = forecast[forecast["ds"].isin(p["ds"])]["yhat"].values
        a = p["y"].values
        n = min(len(a), len(t))
        mae = float(np.mean(np.abs(a[:n] - t[:n])))
        rmse = float(np.sqrt(np.mean((a[:n] - t[:n]) ** 2)))
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], {"mae": round(mae, 4), "rmse": round(rmse, 4), "training_rows": len(p), "forecast_periods": periods}

    @staticmethod
    def _build_chart(df, forecast_df, date_col, target_col):
        a = df[[date_col, target_col]].copy()
        a.columns = ["ds", "y"]
        a["ds"] = pd.to_datetime(a["ds"], errors="coerce")
        a["y"] = pd.to_numeric(a["y"], errors="coerce")
        a = a.dropna().sort_values("ds")
        return {"type": "forecast", "actual": {"x": a["ds"].dt.strftime("%Y-%m-%d").tolist(), "y": a["y"].tolist()}, "forecast": {"x": forecast_df["ds"].dt.strftime("%Y-%m-%d").tolist(), "y": [round(v, 4) for v in forecast_df["yhat"].tolist()], "lower": [round(v, 4) for v in forecast_df["yhat_lower"].tolist()], "upper": [round(v, 4) for v in forecast_df["yhat_upper"].tolist()]}}
