"""Generate reproducible sample retail datasets for NeuralRetail."""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Store", "Web", "App"]
SKUS = [
    ("SKU-1001", "Premium Coffee Beans"),
    ("SKU-1002", "Organic Tea Pack"),
    ("SKU-1003", "Protein Bar Box"),
    ("SKU-1004", "Vitamin Bundle"),
]


def daily_multiplier(day_index: int) -> float:
    """Create trend + seasonality for synthetic demand."""

    weekly = 1 + 0.2 * math.sin(day_index * (2 * math.pi / 7))
    monthly = 1 + 0.12 * math.sin(day_index * (2 * math.pi / 30))
    trend = 1 + day_index * 0.002
    return weekly * monthly * trend


def build_transactions(rows_per_day: int = 18, days: int = 120) -> list[dict]:
    """Create a realistic transaction table."""

    start = date(2025, 1, 1)
    rows: list[dict] = []
    for day_idx in range(days):
        current_date = start + timedelta(days=day_idx)
        holiday_flag = 1 if current_date.day in {1, 15, 26} else 0
        weather_index = round(0.6 + 0.4 * math.sin(day_idx / 10), 2)
        for row_idx in range(rows_per_day):
            customer_num = (day_idx * rows_per_day + row_idx) % 80 + 1
            customer_id = f"CUST-{customer_num:03d}"
            sku, _ = SKUS[(day_idx + row_idx) % len(SKUS)]
            region = REGIONS[(row_idx + day_idx) % len(REGIONS)]
            channel = CHANNELS[(row_idx + customer_num) % len(CHANNELS)]
            promotion_flag = 1 if (day_idx + row_idx) % 9 == 0 else 0
            base_price = 18 + (row_idx % 4) * 4 + (0.8 if promotion_flag else 2.0)
            unit_price = round(base_price + random.uniform(-1.25, 1.25), 2)
            demand_signal = daily_multiplier(day_idx) * (1.18 if promotion_flag else 1.0)
            quantity = max(1, int(round(3 + (row_idx % 5) + demand_signal * random.uniform(1.5, 4.0))))
            discount_pct = round(0.12 if promotion_flag else random.choice([0.0, 0.03, 0.05]), 2)
            effective_price = round(unit_price * (1 - discount_pct), 2)
            revenue = round(quantity * effective_price, 2)
            churned = 1 if (customer_num % 11 == 0 and row_idx % 3 == 0) else 0
            inventory_on_hand = max(20, 140 - (day_idx % 25) * 3 + row_idx * 2)
            rows.append(
                {
                    "date": current_date.isoformat(),
                    "customer_id": customer_id,
                    "sku": sku,
                    "region": region,
                    "channel": channel,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount_pct": discount_pct,
                    "promotion_flag": promotion_flag,
                    "inventory_on_hand": inventory_on_hand,
                    "weather_index": weather_index,
                    "holiday_flag": holiday_flag,
                    "revenue": revenue,
                    "churned": churned,
                }
            )
    return rows


def build_customers() -> list[dict]:
    """Create a basic customer dimension."""

    customers = []
    for idx in range(1, 81):
        customers.append(
            {
                "customer_id": f"CUST-{idx:03d}",
                "customer_name_hash": f"hash-{idx:03d}",
                "region": REGIONS[idx % len(REGIONS)],
                "preferred_channel": CHANNELS[idx % len(CHANNELS)],
            }
        )
    return customers


def build_catalog() -> list[dict]:
    """Create a simple SKU catalog."""

    return [{"sku": sku, "description": description} for sku, description in SKUS]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    transactions = build_transactions()
    customers = build_customers()
    catalog = build_catalog()

    write_csv(RAW_DIR / "transactions_sample.csv", transactions)
    write_csv(RAW_DIR / "customers_sample.csv", customers)
    write_csv(RAW_DIR / "sku_catalog.csv", catalog)
    print("Sample datasets generated in", RAW_DIR)


if __name__ == "__main__":
    main()
