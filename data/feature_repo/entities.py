"""Feast entities for NeuralRetail."""

from feast import Entity

customer = Entity(name="customer_id", join_keys=["customer_id"])
sku = Entity(name="sku", join_keys=["sku"])
