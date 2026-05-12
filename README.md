# NeuralRetail - AI Sales Intelligence & Predictive Analytics Platform

NeuralRetail is a production-style retail AI platform that combines demand forecasting, churn prediction, customer segmentation, price optimization, inventory planning, dashboards, and MLOps automation in one modular Python codebase.

## Included capabilities

- Spark + Delta Lake ingestion for CSV, Parquet, API, and Kafka sources
- Great Expectations validation and OpenLineage event emission
- Feature engineering for RFM, rolling windows, lags, holidays, and external regressors
- Feast feature store definitions backed by Redis and offline parquet files
- Demand forecasting with a Prophet + PyTorch Lightning hybrid forecaster
- Churn prediction with stacked XGBoost + LightGBM and SHAP summaries
- Customer segmentation with K-Means, DBSCAN, and Gaussian Mixture selection
- Price elasticity estimation with DoWhy and EconML fallbacks plus what-if simulation
- Inventory optimization with EOQ, safety stock, and ABC-XYZ classification
- FastAPI service with JWT auth, API keys, Pydantic validation, and Redis caching
- Streamlit dashboard with five pages, Plotly charts, and PDF/Excel export
- MLflow tracking, Airflow DAG automation, Evidently drift reporting, Prometheus, and Grafana

## Project structure

```text
neuralretail/
  data/
  notebooks/
  src/
    common/
    ingestion/
    features/
    models/
    api/
    dashboard/
    mlops/
  tests/
  scripts/
  infra/
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md
```

## Quick start

1. Copy `.env.example` to `.env`.
2. Generate sample data:

```bash
python scripts/generate_sample_data.py
```

3. Start the core stack:

```bash
docker compose up --build api dashboard postgres redis mlflow minio prometheus grafana
```

4. Optional MLOps services:

```bash
docker compose --profile mlops up airflow-init airflow-webserver airflow-scheduler
```

## Local development

Install dependencies and run the services directly:

```bash
pyenv exec python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lite.txt
python scripts/generate_sample_data.py
uvicorn src.api.main:app --reload
```

In another terminal:

```bash
cd "/Users/yashodip/Documents/New project/neuralretail"
source .venv/bin/activate
streamlit run src/dashboard/Home.py
```

## Full local development

Install the full platform dependencies and run the full training stack directly:

```bash
pyenv exec python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/train_demo_models.py
uvicorn src.api.main:app --reload
```

## Authentication

Request a JWT:

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

Call a protected endpoint:

```bash
curl -X POST http://localhost:8000/predict/demand \
  -H "Authorization: Bearer <token>" \
  -H "x-api-key: neuralretail-local-key" \
  -H "Content-Type: application/json" \
  -d '{"sku":"SKU-1001","region":"North","horizon_days":30}'
```

## Dashboard pages

1. Executive Overview
2. Demand Forecasting
3. Customer Intelligence
4. Inventory Optimization
5. MLOps Monitoring

## Training and automation

- `scripts/train_demo_models.py` runs the end-to-end training flow
- `src/mlops/training_pipeline.py` orchestrates ingestion, feature generation, training, and logging
- `src/mlops/airflow/dags/neuralretail_pipeline.py` defines the Airflow DAG

## Monitoring and drift

- Prometheus scrapes `/metrics` from the API
- Grafana provisions Prometheus automatically
- `src/mlops/drift.py` computes PSI and can generate an Evidently HTML report
- `src/mlops/retrain.py` triggers retraining when PSI exceeds `0.2` or MAPE exceeds `15`

## Notes

- PII hashing helpers live in `src/common/security.py`
- Feature data can be materialized to Feast-compatible parquet files
- The bundled dataset is synthetic and intended for local testing and demos
