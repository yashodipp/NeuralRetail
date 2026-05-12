"""Great Expectations-driven retail data validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

try:
    import great_expectations as gx
except ImportError:  # pragma: no cover - optional during static validation
    gx = None


class RetailDataValidator:
    """Validate core transaction fields before writing to Delta Lake."""

    required_columns = {
        "date",
        "customer_id",
        "sku",
        "region",
        "quantity",
        "unit_price",
        "revenue",
    }

    def validate(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Validate a pandas frame and return a compact result payload."""

        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if gx:
            dataset = gx.from_pandas(frame)
            results = [
                dataset.expect_column_values_to_not_be_null("customer_id"),
                dataset.expect_column_values_to_not_be_null("sku"),
                dataset.expect_column_values_to_be_between("quantity", min_value=0),
                dataset.expect_column_values_to_be_between("unit_price", min_value=0),
                dataset.expect_column_values_to_be_between("revenue", min_value=0),
            ]
            success = all(result.success for result in results)
            payload = {
                "success": success,
                "results": [result.to_json_dict() for result in results],
            }
            logger.info("Great Expectations validation success=%s", success)
            return payload

        logger.warning("great_expectations is not installed; using fallback validation")
        success = (
            frame["customer_id"].notna().all()
            and frame["sku"].notna().all()
            and (frame["quantity"] >= 0).all()
            and (frame["unit_price"] >= 0).all()
            and (frame["revenue"] >= 0).all()
        )
        return {"success": success, "results": [{"fallback": True}]}
