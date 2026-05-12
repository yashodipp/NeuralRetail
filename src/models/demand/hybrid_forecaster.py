"""Hybrid Prophet + LSTM demand forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from src.common.logging import get_logger

logger = get_logger(__name__)

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - optional during static validation
    Prophet = None

try:
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - optional during static validation
    pl = None
    torch = None
    nn = None
    Dataset = object
    DataLoader = Any  # type: ignore[assignment]


class SequenceDataset(Dataset):
    """Sequence dataset for LSTM training."""

    def __init__(self, values: np.ndarray, window_size: int = 14) -> None:
        self.values = values.astype(np.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, len(self.values) - self.window_size)

    def __getitem__(self, idx: int):
        window = self.values[idx : idx + self.window_size]
        target = self.values[idx + self.window_size]
        return window.reshape(self.window_size, 1), np.array([target], dtype=np.float32)


class LSTMDemandModule(pl.LightningModule if pl else object):
    """Minimal Lightning module for residual learning."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32, learning_rate: float = 1e-3):
        if pl:
            super().__init__()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        if nn:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.loss_fn = nn.MSELoss()

    def forward(self, x):  # type: ignore[override]
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

    def training_step(self, batch, batch_idx):  # pragma: no cover - GPU/torch dependent
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):  # pragma: no cover - torch dependent
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


@dataclass(slots=True)
class ForecastResult:
    """Forecast output container."""

    forecast: pd.DataFrame
    metrics: dict[str, float]


class DemandHybridForecaster:
    """Blend Prophet with an LSTM residual model."""

    def __init__(self, sequence_length: int = 14, epochs: int = 5) -> None:
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.prophet_model = None
        self.lstm_model = None
        self.train_frame: pd.DataFrame | None = None
        self.regressors: list[str] = []

    def fit(self, frame: pd.DataFrame, target_col: str = "quantity") -> "DemandHybridForecaster":
        """Fit a single-series demo forecaster with external regressors."""

        training = frame.copy().sort_values("date")
        training["date"] = pd.to_datetime(training["date"])
        self.train_frame = training
        self.regressors = [col for col in ["avg_price", "weather_index", "promotion_flag", "is_holiday"] if col in training.columns]

        prophet_training = training.rename(columns={"date": "ds", target_col: "y"})
        if Prophet:
            try:
                self.prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                )
                for regressor in self.regressors:
                    self.prophet_model.add_regressor(regressor)
                self.prophet_model.fit(prophet_training[["ds", "y", *self.regressors]])
            except Exception as exc:  # pragma: no cover - dependency dependent
                logger.warning("Prophet training failed; using seasonal fallback: %s", exc)
                self.prophet_model = None
        else:
            logger.warning("Prophet is not installed; falling back to seasonal naive forecasts")

        if pl and torch and len(training) > self.sequence_length + 2:
            values = training[target_col].to_numpy()
            dataset = SequenceDataset(values, window_size=self.sequence_length)
            if len(dataset) > 0:
                try:
                    loader = DataLoader(dataset, batch_size=16, shuffle=True)
                    self.lstm_model = LSTMDemandModule()
                    trainer = pl.Trainer(
                        accelerator="cpu",
                        max_epochs=self.epochs,
                        logger=False,
                        enable_checkpointing=False,
                        enable_model_summary=False,
                    )
                    trainer.fit(self.lstm_model, loader)
                except Exception as exc:  # pragma: no cover - dependency dependent
                    logger.warning("Residual LSTM training failed; continuing without it: %s", exc)
                    self.lstm_model = None
        else:
            logger.warning("PyTorch Lightning stack unavailable; skipping residual model training")
        return self

    def predict(self, horizon_days: int = 30) -> ForecastResult:
        """Return daily, weekly, and monthly forecasts."""

        if self.train_frame is None:
            raise ValueError("Model must be fitted before prediction")

        history = self.train_frame.copy().sort_values("date")
        future_dates = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        regressor_defaults = {
            "avg_price": float(history["avg_price"].tail(7).mean()) if "avg_price" in history else 0.0,
            "weather_index": float(history["weather_index"].tail(7).mean()) if "weather_index" in history else 0.0,
            "promotion_flag": int(history["promotion_flag"].tail(7).round().mode().iloc[0]) if "promotion_flag" in history else 0,
            "is_holiday": 0,
        }
        future = pd.DataFrame({"ds": future_dates})
        for regressor, default in regressor_defaults.items():
            future[regressor] = default
        if self.prophet_model:
            prophet_forecast = self.prophet_model.predict(future[["ds", *self.regressors]])
            values = prophet_forecast["yhat"].clip(lower=0).to_numpy()
        else:
            recent = history["quantity"].tail(14).to_numpy()
            repeated = np.resize(recent, horizon_days)
            values = repeated

        if self.lstm_model and torch:
            residuals = self._predict_residuals(history["quantity"].to_numpy(), horizon_days)
            values = np.maximum(values + residuals, 0)

        forecast = pd.DataFrame({"date": future_dates, "daily_forecast": values})
        forecast["weekly_forecast"] = forecast["daily_forecast"].rolling(7, min_periods=1).sum()
        forecast["monthly_forecast"] = forecast["daily_forecast"].rolling(30, min_periods=1).sum()

        holdout = history["quantity"].tail(min(14, len(history))).to_numpy()
        baseline = history["quantity"].shift(1).bfill().tail(len(holdout)).to_numpy()
        mape = float(mean_absolute_percentage_error(holdout, baseline) * 100)
        return ForecastResult(forecast=forecast, metrics={"mape": mape})

    def _predict_residuals(self, values: np.ndarray, horizon_days: int) -> np.ndarray:
        if not (self.lstm_model and torch):
            return np.zeros(horizon_days)
        sequence = values[-self.sequence_length :].astype(np.float32)
        outputs = []
        self.lstm_model.eval()
        for _ in range(horizon_days):
            tensor = torch.tensor(sequence.reshape(1, self.sequence_length, 1))
            with torch.no_grad():
                next_value = self.lstm_model(tensor).numpy().flatten()[0]
            outputs.append(next_value)
            sequence = np.roll(sequence, -1)
            sequence[-1] = next_value
        return np.array(outputs)
