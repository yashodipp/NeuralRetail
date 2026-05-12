"""Airflow DAG orchestrating NeuralRetail MLOps."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.common.config import get_settings
from src.mlops.drift import calculate_psi
from src.mlops.retrain import evaluate_retrain_policy
from src.mlops.training_pipeline import run_training_pipeline


def ingest_task(**context):  # noqa: ANN003
    """Kick off training pipeline ingestion phase."""

    result = run_training_pipeline()
    context["ti"].xcom_push(key="training_result", value=result)


def validate_task(**context):  # noqa: ANN003
    """Run drift checks after training."""

    result = context["ti"].xcom_pull(key="training_result", task_ids="train_models")
    psi = calculate_psi([1, 2, 3, 4], [1.05, 2.1, 3.2, 4.1])
    decision = evaluate_retrain_policy(psi=psi, mape=9.8)
    context["ti"].xcom_push(key="validation_result", value={"psi": psi, "decision": asdict(decision), "result": result})


def deploy_task(**context):  # noqa: ANN003
    """Placeholder deployment task."""

    validation_result = context["ti"].xcom_pull(key="validation_result", task_ids="validate_models")
    return {"status": "deployed", "validation": validation_result}


settings = get_settings()

default_args = {
    "owner": "neuralretail",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="neuralretail_mlops_pipeline",
    default_args=default_args,
    description="NeuralRetail ingestion, training, validation, and deployment DAG",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["neuralretail", "mlops"],
) as dag:
    ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_task)
    train = PythonOperator(task_id="train_models", python_callable=run_training_pipeline)
    validate = PythonOperator(task_id="validate_models", python_callable=validate_task)
    deploy = PythonOperator(task_id="deploy_models", python_callable=deploy_task)

    ingest >> train >> validate >> deploy
