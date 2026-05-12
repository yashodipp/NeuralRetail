"""Run the local NeuralRetail training pipeline."""

from __future__ import annotations

from pprint import pprint

from src.mlops.training_pipeline import run_training_pipeline


def main() -> None:
    result = run_training_pipeline()
    pprint(result)


if __name__ == "__main__":
    main()
