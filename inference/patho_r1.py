# -*- coding: utf-8 -*-
"""
Patho-R1 Reporter: Generate patch-level pathology reports using Patho-R1.
"""

import sys
import os
import logging
import traceback
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Configurable via env var
PATHO_R1_REPO_PATH = os.getenv("PATHO_R1_REPO_PATH", "")


class PathoR1Reporter:
    """Generate patch-level pathology reports using Patho-R1."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:5",
        cancer_type: str = "gastric adenocarcinoma/STAD",
    ):
        self.model_path = model_path or os.getenv("PATHO_R1_MODEL_PATH", "")
        self.device = device
        self.cancer_type = cancer_type
        self._generator = None

    def _get_generator(self):
        """Lazy-load Patho-R1 model."""
        if self._generator is None:
            if not PATHO_R1_REPO_PATH:
                raise RuntimeError(
                    "PATHO_R1_REPO_PATH is not set. "
                    "Please export PATHO_R1_REPO_PATH=/path/to/Patho-R1"
                )
            if not self.model_path:
                raise RuntimeError(
                    "Patho-R1 model path is not set. "
                    "Please pass model_path or export PATHO_R1_MODEL_PATH"
                )
            if PATHO_R1_REPO_PATH not in sys.path:
                sys.path.insert(0, PATHO_R1_REPO_PATH)
            from generate_pathology_report import PathologyReportGenerator
            self._generator = PathologyReportGenerator(
                self.model_path, self.device
            )
        return self._generator

    def generate_report(
        self,
        patch_paths: List[str],
        query_context: str = "",
    ) -> Dict[str, str]:
        """
        Generate pathology report for a batch of patches.

        Args:
            patch_paths: Patch image paths (max 5)
            query_context: Additional context prompt

        Returns:
            {"thinking": str, "answer": str, "raw_output": str}
        """
        import torch

        generator = self._get_generator()
        result = generator.generate_report(
            image_path=patch_paths,
            prompt_type="detailed",
            max_tokens=2048,
            cancer_type=self.cancer_type,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def generate_batch_reports(
        self,
        patch_paths: List[str],
        query_texts: Optional[List[str]] = None,
        batch_size: int = 5,
    ) -> List[Dict]:
        """
        Generate reports in batches.

        Args:
            patch_paths: All patch paths
            query_texts: Corresponding query texts (optional)
            batch_size: Batch size

        Returns:
            [{"batch_idx": int, "paths": [...], "query": str, "report": str}, ...]
        """
        reports = []

        for batch_start in range(0, len(patch_paths), batch_size):
            batch_paths = patch_paths[batch_start:batch_start + batch_size]
            batch_idx = batch_start // batch_size + 1

            logger.info(f"Batch {batch_idx}: {len(batch_paths)} patches")

            try:
                result = self.generate_report(batch_paths)
                answer = result.get("answer") or result.get("raw_output", "")

                reports.append({
                    "batch_idx": batch_idx,
                    "paths": batch_paths,
                    "report": answer,
                })
            except Exception as e:
                logger.error(f"Batch {batch_idx} generation failed: {e}")
                traceback.print_exc()
                reports.append({
                    "batch_idx": batch_idx,
                    "paths": batch_paths,
                    "report": f"[Generation failed: {e}]",
                })

        return reports

    def release(self):
        """Release GPU memory."""
        import torch
        if self._generator is not None:
            del self._generator
            self._generator = None
            torch.cuda.empty_cache()
            logger.info("Patho-R1 model released")
