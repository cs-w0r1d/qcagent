# -*- coding: utf-8 -*-
"""
Patch Selection Tool-1: Cluster-based Patch Selection
=====================================================

Implements the initial-round patch selection strategy described in the pipeline:
1. Load CONCH patch embeddings for a WSI
2. K-Means clustering into K clusters (default K=50)
3. Select the centroid-closest patch from each cluster
4. Return selected patch indices and coordinates for Patho-R1 analysis

This provides a diverse, representative sampling of the entire WSI,
ensuring that different tissue regions are covered in the initial report.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None
    logger.warning("sklearn not available, will use simple KMeans fallback")


class PatchClusterSelector:
    """
    Patch Selection Tool-1: Cluster-based representative patch selection.

    Given CONCH patch embeddings for a slide, clusters them into K groups
    and selects the most representative patch (closest to centroid) from each.
    """

    def __init__(self, n_clusters: int = 50, random_state: int = 42):
        """
        Args:
            n_clusters: Number of clusters (default: 50 as per pipeline design)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

    def select_patches(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> List[Dict]:
        """
        Cluster patches and select one representative per cluster.

        Args:
            features: (N, D) CONCH patch embeddings
            coords: (N, 2) patch coordinates (x, y)
            n_clusters: Override default cluster count

        Returns:
            List of dicts, one per cluster:
            [{"patch_idx": int, "x": int, "y": int, "cluster": int}, ...]
        """
        n_patches = features.shape[0]
        k = n_clusters or self.n_clusters

        # Adjust K if fewer patches than clusters
        if n_patches <= k:
            logger.warning(
                f"Only {n_patches} patches, selecting all (requested K={k})"
            )
            return [
                {
                    "patch_idx": i,
                    "x": int(coords[i, 0]),
                    "y": int(coords[i, 1]),
                    "cluster": i,
                }
                for i in range(n_patches)
            ]

        # L2 normalize features for better clustering
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        features_normed = features / norms

        # K-Means clustering
        logger.info(f"Clustering {n_patches} patches into {k} clusters...")

        if MiniBatchKMeans is not None:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=self.random_state,
                batch_size=min(1024, n_patches),
                n_init=3,
                max_iter=100,
            )
        else:
            # Fallback: simple numpy-based KMeans
            return self._simple_kmeans_select(features_normed, coords, k)

        labels = kmeans.fit_predict(features_normed)
        centroids = kmeans.cluster_centers_

        # Select the patch closest to each centroid
        selected = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if not mask.any():
                continue

            cluster_indices = np.where(mask)[0]
            cluster_features = features_normed[cluster_indices]

            # Distance to centroid (cosine distance via dot product since normalized)
            centroid = centroids[cluster_id]
            similarities = cluster_features @ centroid
            best_local_idx = np.argmax(similarities)
            best_global_idx = int(cluster_indices[best_local_idx])

            selected.append({
                "patch_idx": best_global_idx,
                "x": int(coords[best_global_idx, 0]),
                "y": int(coords[best_global_idx, 1]),
                "cluster": cluster_id,
                "score": float(similarities[best_local_idx]),
            })

        logger.info(f"Selected {len(selected)} representative patches from {k} clusters")
        return selected

    def _simple_kmeans_select(
        self,
        features_normed: np.ndarray,
        coords: np.ndarray,
        k: int,
    ) -> List[Dict]:
        """Fallback: uniform sampling when sklearn not available."""
        n = features_normed.shape[0]
        indices = np.linspace(0, n - 1, k, dtype=int)
        return [
            {
                "patch_idx": int(idx),
                "x": int(coords[idx, 0]),
                "y": int(coords[idx, 1]),
                "cluster": i,
            }
            for i, idx in enumerate(indices)
        ]
