# -*- coding: utf-8 -*-
"""
Patch Cutter: Cut patches from SVS files at specified coordinates.
"""

import logging
from pathlib import Path
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import openslide
except ImportError:
    openslide = None
    logger.warning("openslide not installed (pip install openslide-python)")


class PatchCutter:
    """Cut patches from SVS files by coordinates."""

    def __init__(
        self,
        svs_dir: str = "./data/TCGA-STAD",
        output_dir: str = "./qc_retrieved_patches",
    ):
        self.svs_dir = Path(svs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cut_patches(
        self,
        slide_id: str,
        patch_coords: List[Dict],
        patch_size: int = 512,
        dest_dir: str = None,
    ) -> List[str]:
        """
        Cut patches from an SVS file.

        Args:
            slide_id: WSI slide ID
            patch_coords: [{"patch_idx": int, "x": int, "y": int, ...}, ...]
            patch_size: Patch size at level 0
            dest_dir: Optional override for output directory (instead of self.output_dir/slide_id)

        Returns:
            List of saved patch image paths
        """
        if openslide is None:
            raise RuntimeError("openslide is not installed")

        # Search for slide file with multiple extensions
        svs_path = None
        for ext in (".svs", ".tiff", ".tif", ".ndpi", ".mrxs"):
            candidate = self.svs_dir / f"{slide_id}{ext}"
            if candidate.exists():
                svs_path = candidate
                break

        # Also try stripping -HE suffix for 301 dataset compatibility
        if svs_path is None:
            base_id = slide_id.replace("-HE", "")
            for ext in (".svs", ".tiff", ".tif", ".ndpi", ".mrxs"):
                candidate = self.svs_dir / f"{base_id}{ext}"
                if candidate.exists():
                    svs_path = candidate
                    break

        if svs_path is None:
            raise FileNotFoundError(
                f"Slide file not found for {slide_id} in {self.svs_dir} "
                f"(tried .svs/.tiff/.tif/.ndpi/.mrxs)"
            )

        slide = openslide.OpenSlide(str(svs_path))
        saved_paths = []

        if dest_dir is not None:
            qc_dir = Path(dest_dir)
        else:
            qc_dir = self.output_dir / slide_id
        qc_dir.mkdir(parents=True, exist_ok=True)

        for coord in patch_coords:
            x, y = coord["x"], coord["y"]
            idx = coord["patch_idx"]

            try:
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                patch = patch.convert("RGB")

                filename = f"qc_patch_{idx}_x{x}_y{y}.png"
                save_path = qc_dir / filename
                patch.save(save_path)
                saved_paths.append(str(save_path))
            except Exception as e:
                logger.warning(f"Failed to cut patch {idx}: {e}")
                continue

        slide.close()
        logger.info(f"Cut {len(saved_paths)} patches → {qc_dir}")
        return saved_paths
