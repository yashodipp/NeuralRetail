"""Inventory optimization with EOQ, safety stock, and ABC-XYZ classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ReorderRecommendation:
    """Inventory reorder action for a single SKU."""

    sku: str
    reorder_point: float
    order_quantity: float
    safety_stock: float
    abc_class: str
    xyz_class: str
    recommendation: str


class InventoryOptimizer:
    """Optimize inventory policy for retail SKUs."""

    def economic_order_quantity(self, annual_demand: float, ordering_cost: float, holding_cost: float) -> float:
        return float(np.sqrt((2 * annual_demand * ordering_cost) / max(holding_cost, 1e-6)))

    def safety_stock(self, lead_time_days: float, demand_std: float, service_level_z: float = 1.65) -> float:
        return float(service_level_z * demand_std * np.sqrt(max(lead_time_days, 1e-6)))

    def abc_xyz_classification(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Classify SKUs by value contribution and demand volatility."""

        summary = frame.groupby("sku", as_index=False).agg(
            annual_revenue=("revenue", "sum"),
            avg_demand=("quantity", "mean"),
            demand_std=("quantity", "std"),
        )
        summary["demand_std"] = summary["demand_std"].fillna(0.0)
        summary = summary.sort_values("annual_revenue", ascending=False)
        summary["cum_share"] = summary["annual_revenue"].cumsum() / summary["annual_revenue"].sum()
        summary["abc_class"] = pd.cut(
            summary["cum_share"],
            bins=[0, 0.8, 0.95, 1.0],
            labels=["A", "B", "C"],
            include_lowest=True,
        ).astype(str)
        summary["cv"] = summary["demand_std"] / summary["avg_demand"].replace(0, np.nan)
        summary["cv"] = summary["cv"].fillna(0.0)
        summary["xyz_class"] = pd.cut(
            summary["cv"],
            bins=[-np.inf, 0.5, 1.0, np.inf],
            labels=["X", "Y", "Z"],
        ).astype(str)
        return summary

    def recommend(self, transactions: pd.DataFrame, lead_time_days: int = 7, ordering_cost: float = 150.0, holding_cost: float = 8.0) -> pd.DataFrame:
        """Generate reorder recommendations for each SKU."""

        classified = self.abc_xyz_classification(transactions)
        demand_summary = transactions.groupby("sku", as_index=False).agg(
            annual_demand=("quantity", "sum"),
            avg_daily_demand=("quantity", "mean"),
            operational_demand_std=("quantity", "std"),
            current_inventory=("inventory_on_hand", "last"),
        )
        demand_summary["operational_demand_std"] = demand_summary["operational_demand_std"].fillna(0.0)
        merged = classified.merge(demand_summary, on="sku", how="left")

        merged["eoq"] = merged.apply(
            lambda row: self.economic_order_quantity(row["annual_demand"], ordering_cost, holding_cost),
            axis=1,
        )
        merged["safety_stock"] = merged.apply(
            lambda row: self.safety_stock(lead_time_days, row["operational_demand_std"]),
            axis=1,
        )
        merged["reorder_point"] = merged["avg_daily_demand"] * lead_time_days + merged["safety_stock"]
        merged["recommendation"] = np.where(
            merged["current_inventory"] <= merged["reorder_point"],
            "Reorder now",
            "Inventory healthy",
        )
        merged["recommended_order_qty"] = np.where(
            merged["recommendation"] == "Reorder now",
            merged["eoq"],
            0.0,
        )
        return merged[
            [
                "sku",
                "abc_class",
                "xyz_class",
                "reorder_point",
                "recommended_order_qty",
                "safety_stock",
                "current_inventory",
                "recommendation",
            ]
        ]
