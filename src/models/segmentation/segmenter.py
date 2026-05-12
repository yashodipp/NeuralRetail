"""Customer segmentation with model selection by silhouette score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class SegmentationResult:
    """Selected segmentation model output."""

    algorithm: str
    score: float
    segments: pd.DataFrame
    profile: pd.DataFrame


class CustomerSegmentationService:
    """Try multiple clustering algorithms and retain the best performer."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit_predict(self, frame: pd.DataFrame) -> SegmentationResult:
        """Score KMeans, DBSCAN, and Gaussian Mixture candidates."""

        features = frame.drop(columns=["customer_id", "churn_label"], errors="ignore")
        scaled = self.scaler.fit_transform(features)
        candidates: list[tuple[str, np.ndarray, float]] = []

        for clusters in range(6, 11):
            kmeans = KMeans(n_clusters=clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(scaled)
            score = silhouette_score(scaled, labels)
            candidates.append((f"kmeans_{clusters}", labels, float(score)))

            gmm = GaussianMixture(n_components=clusters, random_state=42)
            labels = gmm.fit_predict(scaled)
            score = silhouette_score(scaled, labels)
            candidates.append((f"gmm_{clusters}", labels, float(score)))

        for eps in (0.5, 0.75, 1.0):
            labels = DBSCAN(eps=eps, min_samples=8).fit_predict(scaled)
            distinct_labels = set(labels) - {-1}
            if len(distinct_labels) >= 2:
                score = silhouette_score(scaled[labels != -1], labels[labels != -1])
                candidates.append((f"dbscan_{eps}", labels, float(score)))

        algorithm, labels, best_score = max(candidates, key=lambda item: item[2])
        segments = frame.copy()
        segments["segment_id"] = labels
        profile = (
            segments.groupby("segment_id", as_index=False)
            .agg(
                customers=("customer_id", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                avg_churn=("churn_label", "mean"),
            )
            .sort_values("customers", ascending=False)
        )
        logger.info("Selected segmentation algorithm=%s silhouette=%.4f", algorithm, best_score)
        return SegmentationResult(algorithm=algorithm, score=best_score, segments=segments, profile=profile)
