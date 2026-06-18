# -*- coding: utf-8 -*-
"""
CONCH Retriever: Text encoding + FAISS retrieval
=================================================

Features:
- Load WSI CONCH patch embeddings (H5 format)
- Encode query text using CONCH text encoder
- Retrieve most relevant patches via FAISS cosine similarity
- Return patch coordinates and similarity scores
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import h5py
import torch

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("faiss not installed, will use numpy brute-force search (pip install faiss-cpu)")

# Add CONCH to path (configurable via env var)
CONCH_PATH = os.getenv("CONCH_REPO_PATH", "")
if CONCH_PATH and CONCH_PATH not in sys.path:
    sys.path.insert(0, CONCH_PATH)


class CONCHRetriever:
    """
    CONCH-based Patch Retriever.

    Workflow:
    1. Load CONCH patch embeddings for a slide
    2. Encode query text into CONCH embedding
    3. FAISS (or numpy) cosine similarity search
    4. Return top-k most relevant patch indices/coordinates/scores
    """

    def __init__(
        self,
        features_dir: str = "./data/features_conch_v1",
        coords_dir: str = "./data/patches",
        device: str = "cuda:4",
    ):
        self.features_dir = Path(features_dir)
        self.coords_dir = Path(coords_dir)
        self.device = device

        self._text_encoder = None
        self._slide_cache: Dict[str, Dict] = {}

    def _get_text_encoder(self):
        """Lazy-load CONCH text encoder."""
        if self._text_encoder is None:
            if not CONCH_PATH:
                raise RuntimeError(
                    "CONCH_REPO_PATH is not set. "
                    "Please export CONCH_REPO_PATH=/path/to/CONCH"
                )
            from conch_text_encoder import CONCHTextEncoder
            self._text_encoder = CONCHTextEncoder(device=self.device)
        return self._text_encoder

    def _load_slide_data(self, slide_id: str) -> Dict:
        """Load CONCH features and coordinates for a slide.

        Supports multiple file naming conventions:
        - TCGA:  features/{slide_id}.h5, coords/{slide_id}_patches.h5
        - 301:   features/{base_id}.pt,   coords/{slide_id}.h5
          (where base_id = slide_id stripped of '-HE' suffix)
        """
        if slide_id in self._slide_cache:
            return self._slide_cache[slide_id]

        # --- Resolve feature file (try multiple patterns) ---
        feat_path = None
        feat_format = None
        base_id = slide_id.replace("-HE", "")  # B0697179-HE -> B0697179

        # Priority: exact match first, then stripped suffix
        candidates = [
            (self.features_dir / f"{slide_id}.h5", "h5"),
            (self.features_dir / f"{slide_id}.pt", "pt"),
            (self.features_dir / f"{base_id}.h5", "h5"),
            (self.features_dir / f"{base_id}.pt", "pt"),
        ]
        for path, fmt in candidates:
            if path.exists():
                feat_path = path
                feat_format = fmt
                break

        if feat_path is None:
            raise FileNotFoundError(
                f"CONCH feature file not found for slide_id={slide_id}. "
                f"Searched: {[str(c[0]) for c in candidates]}"
            )

        logger.info(f"Loading CONCH features: {feat_path.name} (format={feat_format})")

        # --- Load features ---
        if feat_format == "h5":
            with h5py.File(feat_path, "r") as f:
                features = f["features"][:].astype(np.float32)
                coords_from_feat = f["coords"][:]
        else:  # pt format (301 dataset)
            pt_data = torch.load(feat_path, map_location="cpu")
            if isinstance(pt_data, dict) and "emb_by_idx" in pt_data:
                features = pt_data["emb_by_idx"].numpy().astype(np.float32)
            elif isinstance(pt_data, torch.Tensor):
                features = pt_data.numpy().astype(np.float32)
            elif isinstance(pt_data, dict) and "features" in pt_data:
                t = pt_data["features"]
                features = t.numpy().astype(np.float32) if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported .pt format, keys={list(pt_data.keys()) if isinstance(pt_data, dict) else type(pt_data)}")
            coords_from_feat = None  # will rely on coord file

        # --- Resolve coordinate file (try multiple patterns) ---
        patch_size = 512
        coord_path = None
        coord_candidates = [
            self.coords_dir / f"{slide_id}_patches.h5",   # TCGA style
            self.coords_dir / f"{slide_id}.h5",            # 301 style
            self.coords_dir / f"{base_id}_patches.h5",
            self.coords_dir / f"{base_id}.h5",
        ]
        for path in coord_candidates:
            if path.exists():
                coord_path = path
                break

        if coord_path is not None:
            with h5py.File(coord_path, "r") as f:
                coords = f["coords"][:]
                for attr_name in ("patch_size_level0", "patch_size"):
                    if attr_name in f["coords"].attrs:
                        patch_size = int(f["coords"].attrs[attr_name])
                        break
        elif coords_from_feat is not None:
            coords = coords_from_feat
        else:
            logger.warning(f"No coordinate file found for {slide_id}, using dummy coords")
            coords = np.zeros((features.shape[0], 2), dtype=np.int64)

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        features_normed = features / norms

        # FAISS index
        dim = features_normed.shape[1]
        if faiss is not None:
            index = faiss.IndexFlatIP(dim)
            index.add(features_normed)
        else:
            index = None

        data = {
            "features_normed": features_normed,
            "coords": coords,
            "patch_size": patch_size,
            "n_patches": features.shape[0],
            "dim": dim,
            "index": index,
        }

        self._slide_cache[slide_id] = data
        logger.info(
            f"Loaded: {data['n_patches']} patches, dim={dim}, "
            f"patch_size={patch_size}"
        )
        return data

    def search(
        self,
        slide_id: str,
        queries: List[str],
        topk: int = 3,
        exclude_indices: Optional[Set[int]] = None,
    ) -> List[List[Dict]]:
        """
        Search for most relevant patches using text queries.

        Args:
            slide_id: WSI slide ID
            queries: List of query texts
            topk: Number of patches per query
            exclude_indices: Patch indices to exclude (avoid re-retrieval)

        Returns:
            List of hit lists per query:
            [
                [{"patch_idx": int, "x": int, "y": int, "score": float}, ...],
                ...
            ]
        """
        slide_data = self._load_slide_data(slide_id)
        encoder = self._get_text_encoder()

        logger.info(f"Encoding {len(queries)} query texts...")
        query_embs = encoder.encode_text(queries, normalize=True)

        actual_topk = topk + (len(exclude_indices) if exclude_indices else 0)
        actual_topk = min(actual_topk, slide_data["n_patches"])

        results = []
        if slide_data["index"] is not None:
            scores, indices = slide_data["index"].search(
                query_embs.astype(np.float32), actual_topk
            )
            for i in range(len(queries)):
                hits = []
                for j in range(actual_topk):
                    idx = int(indices[i, j])
                    if exclude_indices and idx in exclude_indices:
                        continue
                    if len(hits) >= topk:
                        break
                    x = int(slide_data["coords"][idx, 0])
                    y = int(slide_data["coords"][idx, 1])
                    hits.append({
                        "patch_idx": idx,
                        "x": x, "y": y,
                        "score": float(scores[i, j]),
                    })
                results.append(hits)
        else:
            # Numpy brute-force
            sims = query_embs @ slide_data["features_normed"].T
            for i in range(len(queries)):
                sorted_idx = np.argsort(sims[i])[::-1]
                hits = []
                for idx in sorted_idx:
                    idx = int(idx)
                    if exclude_indices and idx in exclude_indices:
                        continue
                    if len(hits) >= topk:
                        break
                    x = int(slide_data["coords"][idx, 0])
                    y = int(slide_data["coords"][idx, 1])
                    hits.append({
                        "patch_idx": idx,
                        "x": x, "y": y,
                        "score": float(sims[i, idx]),
                    })
                results.append(hits)

        return results

    def get_patch_size(self, slide_id: str) -> int:
        data = self._load_slide_data(slide_id)
        return data["patch_size"]

    def release_encoder(self):
        """Release text encoder GPU memory."""
        if self._text_encoder is not None:
            import torch
            del self._text_encoder
            self._text_encoder = None
            torch.cuda.empty_cache()
            logger.info("CONCH text encoder released")
