"""Feature engineering for forecasting, churn, segmentation, and pricing."""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
import polars as pl

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class FeatureArtifacts:
    """Container for engineered feature outputs."""

    time_series_features: pd.DataFrame
    customer_features: pd.DataFrame


class FeatureEngineeringService:
    """Build reusable retail features with pandas and Polars."""

    def __init__(self, country_code: str = "IN") -> None:
        self.country_holidays = holidays.country_holidays(country_code)

    def load_transactions(self, path: str) -> pd.DataFrame:
        """Load transactions efficiently with Polars and convert to pandas."""

        frame = pl.read_csv(path)
        return frame.to_pandas()

    def build(self, transactions: pd.DataFrame) -> FeatureArtifacts:
        """Return all modeling feature tables."""

        transactions = transactions.copy()
        transactions["date"] = pd.to_datetime(transactions["date"])
        transactions.sort_values(["sku", "date"], inplace=True)

        time_series_features = self._build_time_series_features(transactions)
        customer_features = self._build_customer_features(transactions)
        return FeatureArtifacts(
            time_series_features=time_series_features,
            customer_features=customer_features,
        )

    def _build_time_series_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        daily = (
            transactions.groupby(["date", "sku", "region"], as_index=False)
            .agg(
                quantity=("quantity", "sum"),
                revenue=("revenue", "sum"),
                avg_price=("unit_price", "mean"),
                promotion_flag=("promotion_flag", "max"),
                weather_index=("weather_index", "mean"),
            )
            .sort_values(["sku", "region", "date"])
        )
        for window in (7, 14, 30):
            daily[f"rolling_qty_{window}"] = (
                daily.groupby(["sku", "region"])["quantity"]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
            daily[f"rolling_revenue_{window}"] = (
                daily.groupby(["sku", "region"])["revenue"]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
        for lag in (1, 7, 14):
            daily[f"lag_qty_{lag}"] = daily.groupby(["sku", "region"])["quantity"].shift(lag)
            daily[f"lag_revenue_{lag}"] = daily.groupby(["sku", "region"])["revenue"].shift(lag)

        daily["day_of_week"] = daily["date"].dt.dayofweek
        daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
        daily["month"] = daily["date"].dt.month
        daily["quarter"] = daily["date"].dt.quarter
        daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
        daily["is_holiday"] = daily["date"].dt.date.astype("object").isin(self.country_holidays).astype(int)
        daily.fillna(0, inplace=True)
        logger.info("Built %s time-series feature rows", len(daily))
        return daily

    def _build_customer_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        snapshot_date = transactions["date"].max() + pd.Timedelta(days=1)
        customer = transactions.groupby("customer_id").agg(
            last_purchase=("date", "max"),
            frequency=("date", "count"),
            monetary=("revenue", "sum"),
            avg_order_value=("revenue", "mean"),
            avg_discount=("discount_pct", "mean"),
            churn_label=("churned", "max"),
            region=("region", "last"),
            preferred_channel=("channel", "last"),
        )
        customer["recency"] = (snapshot_date - customer["last_purchase"]).dt.days
        recency_score = self._quartile_score(customer["recency"], ascending=False)
        frequency_score = self._quartile_score(customer["frequency"], ascending=True)
        monetary_score = self._quartile_score(customer["monetary"], ascending=True)
        customer["rfm_score"] = recency_score + frequency_score + monetary_score
        customer["tenure_days"] = (
            transactions.groupby("customer_id")["date"].min().rsub(snapshot_date).dt.days
        )
        customer["purchase_velocity"] = customer["frequency"] / customer["tenure_days"].clip(lower=1)
        customer["monetary_per_visit"] = customer["monetary"] / customer["frequency"].clip(lower=1)

        encoded = pd.get_dummies(customer[["region", "preferred_channel"]], drop_first=False)
        customer = pd.concat([customer.drop(columns=["last_purchase", "region", "preferred_channel"]), encoded], axis=1)
        customer.replace([np.inf, -np.inf], 0, inplace=True)
        customer.fillna(0, inplace=True)
        customer.reset_index(inplace=True)
        logger.info("Built %s customer feature rows", len(customer))
        return customer

    @staticmethod
    def _quartile_score(series: pd.Series, ascending: bool) -> pd.Series:
        """Convert ranked values into a 1-4 quartile score."""

        percentiles = series.rank(method="first", ascending=ascending, pct=True)
        return pd.cut(
            percentiles,
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=[1, 2, 3, 4],
            include_lowest=True,
        ).astype(int)
