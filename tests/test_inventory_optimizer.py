"""Inventory optimization tests."""

from __future__ import annotations

import pandas as pd

from src.models.inventory.optimizer import InventoryOptimizer


def test_inventory_optimizer_returns_recommendation_frame():
    frame = pd.DataFrame(
        [
            {"sku": "SKU-1", "revenue": 100.0, "quantity": 10, "inventory_on_hand": 5},
            {"sku": "SKU-1", "revenue": 130.0, "quantity": 12, "inventory_on_hand": 4},
            {"sku": "SKU-2", "revenue": 70.0, "quantity": 7, "inventory_on_hand": 15},
        ]
    )
    result = InventoryOptimizer().recommend(frame)

    assert {"sku", "reorder_point", "recommended_order_qty", "recommendation"}.issubset(result.columns)
    assert len(result) == 2


def test_inventory_optimizer_handles_internal_demand_std_merge_columns():
    frame = pd.DataFrame(
        [
            {"sku": "SKU-1", "revenue": 100.0, "quantity": 10, "inventory_on_hand": 5},
            {"sku": "SKU-1", "revenue": 110.0, "quantity": 11, "inventory_on_hand": 4},
            {"sku": "SKU-2", "revenue": 90.0, "quantity": 8, "inventory_on_hand": 20},
            {"sku": "SKU-2", "revenue": 120.0, "quantity": 12, "inventory_on_hand": 18},
        ]
    )

    result = InventoryOptimizer().recommend(frame)

    assert "safety_stock" in result.columns
    assert result["safety_stock"].notna().all()
