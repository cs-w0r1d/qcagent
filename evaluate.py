# -*- coding: utf-8 -*-
"""
Pathology Report Evaluation Metrics
====================================

Evaluate generated pathology reports against ground-truth TCGA reports using:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-1, ROUGE-2, ROUGE-L
- METEOR
- BERTScore (optional, requires GPU)
- Exact-match field extraction accuracy

Ground truth: tcga_reports_cleaned.csv → `slide_reports` column.
Generated reports: qc_results/{slide_id}/final_report.txt

Usage:
    # Evaluate a single slide
    python evaluate.py --slide_id TCGA-3M-AB47-01Z-00-DX1.xxx --results_dir ./qc_results

    # Evaluate all slides in results directory
    python evaluate.py --results_dir ./qc_results --gt_csv ./data/tcga_reports_cleaned.csv

    # Evaluate with BERTScore (needs GPU)
    python evaluate.py --results_dir ./qc_results --bert_score --bert_gpu 6
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Text Preprocessing
# ============================================================
def normalize_text(text: str) -> str:
    """Normalize text for evaluation: lowercase, strip, normalize whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove common formatting artifacts
    text = re.sub(r"[=\-]{3,}", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for pathology text."""
    text = normalize_text(text)
    # Split on whitespace and keep punctuation as separate tokens
    tokens = re.findall(r"[a-zA-Z0-9]+(?:\.[0-9]+)?|[^\s\w]", text)
    return tokens


# ============================================================
# BLEU Score Implementation
# ============================================================
def compute_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """Compute n-gram frequency counts."""
    ngrams = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams[ngram] += 1
    return dict(ngrams)


def compute_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-N.

    Args:
        reference: Ground truth text
        hypothesis: Generated text
        max_n: Maximum n-gram order (default: 4)
        smoothing: Use smoothing for short texts (add-1 smoothing)

    Returns:
        {"bleu1": float, "bleu2": float, ..., "bleu_avg": float}
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return {f"bleu{i}": 0.0 for i in range(1, max_n + 1)}

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        hyp_ngrams = compute_ngrams(hyp_tokens, n)

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            if smoothing:
                precisions.append(1.0 / (len(hyp_tokens) + 1))
            else:
                precisions.append(0.0)
        else:
            if smoothing and clipped == 0:
                precisions.append(1.0 / (total + 1))
            else:
                precisions.append(clipped / total)

    results = {}
    for n in range(1, max_n + 1):
        # Cumulative BLEU (geometric mean of precisions 1..n)
        log_prec = sum(np.log(max(p, 1e-10)) for p in precisions[:n]) / n
        results[f"bleu{n}"] = bp * np.exp(log_prec)

    results["brevity_penalty"] = bp
    return results


# ============================================================
# ROUGE Score Implementation
# ============================================================
def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L (F1 scores).

    Args:
        reference: Ground truth text
        hypothesis: Generated text

    Returns:
        {"rouge1": float, "rouge2": float, "rougeL": float}
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # ROUGE-1 (unigram F1)
    ref_1grams = set(ref_tokens)
    hyp_1grams = set(hyp_tokens)
    overlap_1 = ref_1grams & hyp_1grams
    p1 = len(overlap_1) / len(hyp_1grams) if hyp_1grams else 0
    r1 = len(overlap_1) / len(ref_1grams) if ref_1grams else 0
    rouge1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0

    # ROUGE-2 (bigram F1)
    ref_2grams = set(compute_ngrams(ref_tokens, 2).keys())
    hyp_2grams = set(compute_ngrams(hyp_tokens, 2).keys())
    overlap_2 = ref_2grams & hyp_2grams
    p2 = len(overlap_2) / len(hyp_2grams) if hyp_2grams else 0
    r2 = len(overlap_2) / len(ref_2grams) if ref_2grams else 0
    rouge2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0

    # ROUGE-L (LCS-based F1)
    lcs_len = _lcs_length(ref_tokens, hyp_tokens)
    pL = lcs_len / len(hyp_tokens) if hyp_tokens else 0
    rL = lcs_len / len(ref_tokens) if ref_tokens else 0
    rougeL = 2 * pL * rL / (pL + rL) if (pL + rL) > 0 else 0

    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length."""
    m, n = len(x), len(y)
    # Optimize memory: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ============================================================
# METEOR Score (Simplified)
# ============================================================
def compute_meteor(reference: str, hypothesis: str) -> float:
    """
    Compute simplified METEOR score.
    Based on unigram precision, recall, and chunk penalty.

    Args:
        reference: Ground truth text
        hypothesis: Generated text

    Returns:
        METEOR score (float)
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # Unigram matches
    ref_set = defaultdict(int)
    for t in ref_tokens:
        ref_set[t] += 1

    hyp_set = defaultdict(int)
    for t in hyp_tokens:
        hyp_set[t] += 1

    matches = 0
    for token, count in hyp_set.items():
        matches += min(count, ref_set.get(token, 0))

    precision = matches / len(hyp_tokens) if hyp_tokens else 0
    recall = matches / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0

    # Harmonic mean with recall weight (alpha = 0.9 as in original METEOR)
    alpha = 0.9
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    # Chunk penalty (simplified: count contiguous matched chunks)
    chunks = _count_chunks(ref_tokens, hyp_tokens)
    if matches > 0:
        frag = chunks / matches
    else:
        frag = 0

    penalty = 0.5 * (frag ** 3)
    score = f_mean * (1 - penalty)

    return max(0.0, score)


def _count_chunks(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """Count the number of contiguous matched chunks."""
    ref_matched = [False] * len(ref_tokens)
    hyp_matched = [False] * len(hyp_tokens)

    # Greedy matching
    for i, ht in enumerate(hyp_tokens):
        for j, rt in enumerate(ref_tokens):
            if ht == rt and not ref_matched[j]:
                hyp_matched[i] = True
                ref_matched[j] = True
                break

    # Count chunks in hypothesis
    chunks = 0
    in_chunk = False
    for m in hyp_matched:
        if m and not in_chunk:
            chunks += 1
            in_chunk = True
        elif not m:
            in_chunk = False

    return max(chunks, 1)


# ============================================================
# BERTScore (Optional)
# ============================================================
class BERTScoreEvaluator:
    """
    Cached BERTScore evaluator — loads the model once and reuses it.
    """

    def __init__(self, device: str = "cuda:6", model_type: str = "microsoft/deberta-xlarge-mnli"):
        self.device = device
        self.model_type = model_type
        self._scorer = None
        self._load_attempted = False  # avoid repeated import warnings

    @property
    def available(self) -> bool:
        """Whether BERTScore model is loaded and ready."""
        if not self._load_attempted:
            self._ensure_loaded()
        return self._scorer is not None

    def _ensure_loaded(self):
        """Lazy-load the BERTScore model (only tries once)."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from bert_score import BERTScorer
            logger.info(f"Loading BERTScore model ({self.model_type}) on {self.device} ...")
            self._scorer = BERTScorer(
                model_type=self.model_type,
                device=self.device,
                lang="en",
                rescale_with_baseline=True,
            )
            logger.info("BERTScore model loaded.")
        except ImportError:
            logger.warning(
                "bert-score not installed. Install with: pip install bert-score"
            )
            self._scorer = None
        except Exception as e:
            logger.warning(f"Failed to load BERTScore model: {e}")
            self._scorer = None

    def score(
        self,
        references: List[str],
        hypotheses: List[str],
    ) -> Optional[Dict[str, List[float]]]:
        """
        Compute BERTScore for a batch of (reference, hypothesis) pairs.

        Returns:
            {"precision": [...], "recall": [...], "f1": [...]}  or None if unavailable.
        """
        self._ensure_loaded()
        if self._scorer is None:
            return None

        P, R, F1 = self._scorer.score(hypotheses, references)
        return {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
        }


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    device: str = "cuda:6",
) -> Dict[str, List[float]]:
    """
    Convenience wrapper — creates a one-shot BERTScoreEvaluator.

    For repeated calls prefer instantiating BERTScoreEvaluator directly.
    """
    evaluator = BERTScoreEvaluator(device=device)
    return evaluator.score(references, hypotheses)


# ============================================================
# Pathology-specific Field Extraction Metrics
# ============================================================
PATHOLOGY_FIELDS = {
    "tumor_type": [
        # English
        r"adenocarcinoma", r"squamous\s*cell\s*carcinoma",
        r"signet\s*ring", r"poorly\s*differentiated",
        r"moderately\s*differentiated", r"well\s*differentiated",
        # Chinese
        r"腺癌", r"鳞[状]*[细胞]*癌", r"印戒细胞癌", r"粘液[腺]*癌",
        r"低分化", r"中分化", r"高分化", r"中-?低分化", r"中-?高分化",
        r"未分化[癌]?", r"管状腺癌", r"乳头状[腺]*癌",
    ],
    "tumor_size": [
        # English
        r"\d+(?:\.\d+)?\s*(?:x|×)\s*\d+(?:\.\d+)?(?:\s*(?:x|×)\s*\d+(?:\.\d+)?)?\s*cm",
        r"\d+(?:\.\d+)?\s*cm",
        r"\d+(?:\.\d+)?\s*mm",
        # Chinese
        r"\d+(?:\.\d+)?\s*(?:x|×|X)\s*\d+(?:\.\d+)?(?:\s*(?:x|×|X)\s*\d+(?:\.\d+)?)?\s*(?:cm|厘米|mm|毫米)",
        r"大小[约为:：]?\s*\d+",
        r"肿[物瘤][^。，]*\d+(?:\.\d+)?\s*(?:cm|mm|厘米|毫米)",
    ],
    "lymph_nodes": [
        # English
        r"\d+\s*/\s*\d+",
        r"\d+\s*of\s*\d+",
        r"lymph\s*node.*(?:positive|negative|metast)",
        # Chinese
        r"淋巴结.*\d+\s*/\s*\d+",
        r"淋巴结[^。]*(?:转移|阳性|阴性|未见)",
        r"送检淋巴结",
        r"\d+[枚个]淋巴结",
    ],
    "margins": [
        # English
        r"margin.*(?:positive|negative|free|involved|clear)",
        r"(?:proximal|distal)\s*margin",
        # Chinese
        r"切缘[^。]*(?:阳性|阴性|未见|阴|净|干净|受累|累及|未累及)",
        r"[上下两]切缘",
        r"断端",
    ],
    "invasion": [
        # English
        r"(?:lymphovascular|perineural|vascular|neural)\s*invasion",
        r"(?:lvi|pni)\s*(?:present|absent|identified|not\s*identified)",
        # Chinese
        r"[脉管淋巴管血管][^。]*[侵犯浸润]",
        r"神经[侵犯浸润]",
        r"脉管[内][^。]*癌栓",
        r"浸润[深度至达]",
    ],
    "staging": [
        # English
        r"p?T\d[a-d]?\s*(?:p?N\d[a-c]?)?\s*(?:p?M[01x]?)?",
        r"stage\s*(?:I{1,3}[AB]?|IV[AB]?)",
        # Chinese
        r"p?T\d[a-d]?\s*N\d",
        r"[分期][^。]*(?:I{1,3}[AB]?|IV[AB]?|Ⅰ|Ⅱ|Ⅲ|Ⅳ)",
    ],
    "ihc": [
        # English
        r"(?:HER[-\s]?2|Ki[-\s]?67|PD[-\s]?L1|p53|CDX2|CK)",
        r"immunohistochem",
        # Chinese
        r"(?:HER[-\s]?2|Ki[-\s]?67|PD[-\s]?L1|p53|CDX2|CK|MLH1|MSH2|MSH6|PMS2)",
        r"免疫[组化]",
    ],
}


def compute_field_coverage(
    reference: str,
    hypothesis: str,
) -> Dict[str, Dict[str, bool]]:
    """
    Check if key pathology fields present in reference are also present in generated report.

    Returns:
        {
            "field_name": {
                "in_reference": bool,
                "in_hypothesis": bool,
                "match": bool
            },
            ...
        }
    """
    results = {}
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()

    for field, patterns in PATHOLOGY_FIELDS.items():
        in_ref = any(re.search(p, ref_lower) for p in patterns)
        in_hyp = any(re.search(p, hyp_lower) for p in patterns)

        results[field] = {
            "in_reference": in_ref,
            "in_hypothesis": in_hyp,
            "match": (in_ref and in_hyp) or (not in_ref),
        }

    return results


# ============================================================
# Main Evaluation Pipeline
# ============================================================
class ReportEvaluator:
    """
    Evaluate generated pathology reports against ground truth.
    """

    def __init__(
        self,
        gt_csv_path: str,
        results_dir: str = "./qc_results",
        use_bertscore: bool = False,
        bert_gpu: int = 6,
    ):
        self.results_dir = Path(results_dir)
        self.use_bertscore = use_bertscore
        self.bert_device = f"cuda:{bert_gpu}"
        self._bert_scorer: Optional[BERTScoreEvaluator] = None

        # Load ground truth
        logger.info(f"Loading ground truth from: {gt_csv_path}")
        self.gt_df = pd.read_csv(gt_csv_path)
        self.gt_lookup = dict(
            zip(self.gt_df["slide_id"], self.gt_df["slide_reports"])
        )
        logger.info(f"Loaded {len(self.gt_lookup)} ground truth reports")

    def get_ground_truth(self, slide_id: str) -> Optional[str]:
        """Look up ground truth report by slide ID."""
        return self.gt_lookup.get(slide_id, None)

    def load_generated_report(self, slide_id: str) -> Optional[str]:
        """Load the generated final report for a slide."""
        report_path = self.results_dir / slide_id / "final_report.txt"
        if report_path.exists():
            return report_path.read_text(encoding="utf-8").strip()

        # Also check draft_report
        draft_path = self.results_dir / slide_id / "draft_report.txt"
        if draft_path.exists():
            return draft_path.read_text(encoding="utf-8").strip()

        return None

    def _get_bert_scorer(self) -> BERTScoreEvaluator:
        """Get or create a cached BERTScoreEvaluator."""
        if self._bert_scorer is None:
            self._bert_scorer = BERTScoreEvaluator(device=self.bert_device)
        return self._bert_scorer

    def evaluate_single(
        self,
        slide_id: str,
        reference: Optional[str] = None,
        hypothesis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single generated report.

        Args:
            slide_id: Slide ID
            reference: Ground truth text (auto-loaded if None)
            hypothesis: Generated text (auto-loaded if None)

        Returns:
            Dict with all metric scores
        """
        if reference is None:
            reference = self.get_ground_truth(slide_id)
        if hypothesis is None:
            hypothesis = self.load_generated_report(slide_id)

        if not reference:
            return {"slide_id": slide_id, "error": "No ground truth found"}
        if not hypothesis:
            return {"slide_id": slide_id, "error": "No generated report found"}

        # Compute metrics
        bleu_scores = compute_bleu(reference, hypothesis)
        rouge_scores = compute_rouge(reference, hypothesis)
        meteor_score = compute_meteor(reference, hypothesis)
        field_coverage = compute_field_coverage(reference, hypothesis)

        # Field coverage summary
        total_fields = sum(1 for v in field_coverage.values() if v["in_reference"])
        matched_fields = sum(
            1 for v in field_coverage.values()
            if v["in_reference"] and v["in_hypothesis"]
        )
        field_recall = matched_fields / total_fields if total_fields > 0 else 1.0

        result = {
            "slide_id": slide_id,
            "ref_length": len(tokenize(reference)),
            "hyp_length": len(tokenize(hypothesis)),
            **bleu_scores,
            **rouge_scores,
            "meteor": meteor_score,
            "field_recall": field_recall,
            "field_details": field_coverage,
        }

        # BERTScore (single-slide)
        if self.use_bertscore:
            scorer = self._get_bert_scorer()
            bs = scorer.score([reference], [hypothesis])
            if bs is not None:
                result["bertscore_precision"] = bs["precision"][0]
                result["bertscore_recall"] = bs["recall"][0]
                result["bertscore_f1"] = bs["f1"][0]

        return result

    def evaluate_batch(
        self,
        slide_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Evaluate a batch of slides.

        Args:
            slide_ids: List of slide IDs (auto-discover if None)

        Returns:
            (individual_results, aggregate_metrics)
        """
        if slide_ids is None:
            # Auto-discover from results directory
            slide_ids = [
                d.name for d in self.results_dir.iterdir()
                if d.is_dir() and (d / "final_report.txt").exists()
            ]

        if not slide_ids:
            logger.warning("No slides found for evaluation")
            return [], {}

        logger.info(f"Evaluating {len(slide_ids)} slides...")

        # Collect individual results
        all_results = []
        references = []
        hypotheses = []

        for slide_id in slide_ids:
            result = self.evaluate_single(slide_id)
            all_results.append(result)

            if "error" not in result:
                ref = self.get_ground_truth(slide_id)
                hyp = self.load_generated_report(slide_id)
                if ref and hyp:
                    references.append(ref)
                    hypotheses.append(hyp)

        # BERTScore (batch computation is more efficient)
        if self.use_bertscore and references:
            scorer = self._get_bert_scorer()
            if not scorer.available:
                logger.warning("BERTScore unavailable, skipping.")
            else:
                # Check if already computed with real values in evaluate_single
                already_done = all(
                    r.get("bertscore_f1", 0) != 0
                    for r in all_results if "error" not in r
                )
                if not already_done:
                    logger.info("Computing BERTScore (batch, %d pairs)...", len(references))
                    bert_results = scorer.score(references, hypotheses)
                    if bert_results is not None:
                        bert_idx = 0
                        for result in all_results:
                            if "error" not in result:
                                result["bertscore_precision"] = bert_results["precision"][bert_idx]
                                result["bertscore_recall"] = bert_results["recall"][bert_idx]
                                result["bertscore_f1"] = bert_results["f1"][bert_idx]
                                bert_idx += 1
                else:
                    logger.info("BERTScore already computed per-slide, skipping batch.")

        # Compute aggregate metrics
        valid_results = [r for r in all_results if "error" not in r]
        if not valid_results:
            return all_results, {}

        metric_keys = [
            "bleu1", "bleu2", "bleu3", "bleu4",
            "rouge1", "rouge2", "rougeL",
            "meteor", "field_recall",
        ]
        if self.use_bertscore:
            metric_keys.extend(["bertscore_precision", "bertscore_recall", "bertscore_f1"])

        aggregate = {}
        for key in metric_keys:
            values = [r[key] for r in valid_results if key in r]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_std"] = float(np.std(values))
                aggregate[f"{key}_min"] = float(np.min(values))
                aggregate[f"{key}_max"] = float(np.max(values))

        aggregate["n_evaluated"] = len(valid_results)
        aggregate["n_errors"] = len(all_results) - len(valid_results)

        return all_results, aggregate

    def print_results(
        self,
        individual: List[Dict],
        aggregate: Dict[str, float],
    ):
        """Print evaluation results in a nice table format."""
        print("\n" + "=" * 80)
        print("  Pathology Report Evaluation Results")
        print("=" * 80)

        if not aggregate:
            print("  No valid results to display.")
            return

        print(f"\n  Evaluated: {aggregate.get('n_evaluated', 0)} slides")
        print(f"  Errors:    {aggregate.get('n_errors', 0)} slides\n")

        # Aggregate table
        print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        metric_display = [
            ("BLEU-1", "bleu1"),
            ("BLEU-2", "bleu2"),
            ("BLEU-3", "bleu3"),
            ("BLEU-4", "bleu4"),
            ("ROUGE-1", "rouge1"),
            ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"),
            ("METEOR", "meteor"),
            ("Field Recall", "field_recall"),
        ]

        if self.use_bertscore:
            metric_display.extend([
                ("BERTScore-P", "bertscore_precision"),
                ("BERTScore-R", "bertscore_recall"),
                ("BERTScore-F1", "bertscore_f1"),
            ])

        for display_name, key in metric_display:
            mean = aggregate.get(f"{key}_mean", 0)
            std = aggregate.get(f"{key}_std", 0)
            mn = aggregate.get(f"{key}_min", 0)
            mx = aggregate.get(f"{key}_max", 0)
            print(f"  {display_name:<25} {mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")

        print(f"\n{'='*80}")

        # Per-slide details (top 5 and bottom 5 by BLEU-4)
        valid = [r for r in individual if "error" not in r]
        if valid:
            valid.sort(key=lambda x: x.get("bleu4", 0), reverse=True)

            print(f"\n  Top 5 (by BLEU-4):")
            for r in valid[:5]:
                sid = r["slide_id"][:40]
                print(
                    f"    {sid:<42} "
                    f"B4={r['bleu4']:.4f} R1={r['rouge1']:.4f} "
                    f"M={r['meteor']:.4f} FR={r['field_recall']:.4f}"
                )

            if len(valid) > 5:
                print(f"\n  Bottom 5 (by BLEU-4):")
                for r in valid[-5:]:
                    sid = r["slide_id"][:40]
                    print(
                        f"    {sid:<42} "
                        f"B4={r['bleu4']:.4f} R1={r['rouge1']:.4f} "
                        f"M={r['meteor']:.4f} FR={r['field_recall']:.4f}"
                    )

        # Show errors
        errors = [r for r in individual if "error" in r]
        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for r in errors[:10]:
                print(f"    {r['slide_id']}: {r['error']}")

    def save_results(
        self,
        individual: List[Dict],
        aggregate: Dict[str, float],
        output_path: str = "./eval_results.json",
    ):
        """Save evaluation results to JSON."""
        # Remove field_details for cleaner output
        clean_individual = []
        for r in individual:
            r_clean = {k: v for k, v in r.items() if k != "field_details"}
            clean_individual.append(r_clean)

        output = {
            "aggregate": aggregate,
            "individual": clean_individual,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated pathology reports against TCGA ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--gt_csv", type=str, default="./data/tcga_reports_cleaned.csv",
        help="Path to ground truth CSV (data/tcga_reports_cleaned.csv)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="./qc_results",
        help="Directory containing QC results (one subfolder per slide)",
    )
    parser.add_argument(
        "--slide_id", type=str, default=None,
        help="Evaluate a single slide (optional)",
    )
    parser.add_argument(
        "--output", type=str, default="./eval_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--bert_score", action="store_true",
        help="Compute BERTScore (requires bert-score package and GPU)",
    )
    parser.add_argument(
        "--bert_gpu", type=int, default=6,
        help="GPU for BERTScore model",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    evaluator = ReportEvaluator(
        gt_csv_path=args.gt_csv,
        results_dir=args.results_dir,
        use_bertscore=args.bert_score,
        bert_gpu=args.bert_gpu,
    )

    if args.slide_id:
        # Single evaluation
        result = evaluator.evaluate_single(args.slide_id)
        if "error" in result:
            print(f"❌ {result['error']}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  Slide: {args.slide_id}")
        print(f"{'='*60}")
        display_keys = [
            "bleu1", "bleu2", "bleu3", "bleu4",
            "rouge1", "rouge2", "rougeL", "meteor", "field_recall",
        ]
        if args.bert_score:
            display_keys.extend(["bertscore_precision", "bertscore_recall", "bertscore_f1"])
        for key in display_keys:
            if key in result:
                print(f"  {key:<20}: {result[key]:.4f}")

        # Field details
        print(f"\n  Field Coverage:")
        for field, info in result.get("field_details", {}).items():
            if info["in_reference"]:
                status = "✓" if info["in_hypothesis"] else "✗"
                print(f"    {status} {field}")

        print(f"{'='*60}")

    else:
        # Batch evaluation
        individual, aggregate = evaluator.evaluate_batch()
        evaluator.print_results(individual, aggregate)
        evaluator.save_results(individual, aggregate, args.output)


if __name__ == "__main__":
    main()
