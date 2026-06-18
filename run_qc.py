#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI Pathology Report QC Pipeline — CLI Entry Point
====================================================

Usage:

  # Single slide with Qwen API (recommended)
  python run_qc.py \\
      --slide_id TCGA-3M-AB47-01Z-00-DX1.D7D70922-A91F-4798-8D6E-DFD84D145A59 \\
      --backend qwen \\
      --conch_gpu 4 --patho_gpu 5

  # With Dify Agent Workflow
  python run_qc.py \\
      --slide_id TCGA-3M-AB47-01Z-00-DX1.D7D70922-A91F-4798-8D6E-DFD84D145A59 \\
      --backend dify

  # Batch processing
  python run_qc.py --batch --slide_list slides.txt --backend qwen
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import QCPipeline


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level), format=fmt, handlers=handlers
    )


def run_single(args):
    """Run QC for a single slide. Returns the result dict."""
    pipeline = QCPipeline(
        slide_id=args.slide_id,
        cancer_type=args.cancer_type,
        max_rounds=args.max_rounds,
        topk_patches=args.topk,
        conch_gpu=args.conch_gpu,
        patho_gpu=args.patho_gpu,
        agent_backend=args.backend,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        dify_url=args.dify_url,
        dify_key=args.dify_key,
        qc_rules=args.qc_rules,
        language=args.language,
        dataset_name=args.dataset_name,
        anatomical_site=args.anatomical_site,
        wsi_report_dir=args.wsi_report_dir,
        patch_report_dir=args.patch_report_dir,
        features_dir=args.features_dir,
        coords_dir=args.coords_dir,
        svs_dir=args.svs_dir,
        output_dir=args.output_dir,
        no_retrieval=args.no_retrieval,
        no_patho_r1=args.no_patho_r1,
        n_clusters=args.n_clusters,
    )
    return pipeline.run()


def run_batch(args) -> int:
    """Batch process multiple slides with shared model instances."""
    slide_list_path = Path(args.slide_list)
    if not slide_list_path.exists():
        print(f"❌ slide_list file not found: {slide_list_path}")
        return 1

    slides = [
        line.strip()
        for line in slide_list_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # Skip already completed slides
    output_dir = Path(args.output_dir)
    todo = []
    skipped = 0
    for sid in slides:
        report_path = output_dir / sid / "final_report.txt"
        if report_path.exists() and report_path.stat().st_size > 0:
            skipped += 1
        else:
            todo.append(sid)

    print(f"📋 Batch processing: {len(slides)} total, {skipped} already done, {len(todo)} to run")
    print("=" * 70)

    if not todo:
        print("✅ All slides already processed!")
        return 0

    # Pre-create shared components (load models ONCE)
    from qc_agent import QCEngine
    from retrieval import CONCHRetriever
    from inference import PathoR1Reporter

    print("🔧 Loading shared models (one-time)...")
    shared_qc_engine = QCEngine(
        backend=args.backend,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        dify_url=args.dify_url,
        dify_key=args.dify_key,
        qc_rules=args.qc_rules,
        language=args.language,
    )
    # Always init retriever (needed for Step 0 clustering even in --no_retrieval mode)
    shared_retriever = CONCHRetriever(
        features_dir=args.features_dir,
        coords_dir=args.coords_dir,
        device=f"cuda:{args.conch_gpu}",
    )
    shared_reporter = None
    if not args.no_patho_r1:
        shared_reporter = PathoR1Reporter(
            device=f"cuda:{args.patho_gpu}",
            cancer_type=args.cancer_type,
        )
    print("✅ Models ready.\n")

    results = {"passed": 0, "needs_review": 0, "error": 0}

    for i, slide_id in enumerate(todo, 1):
        print(f"\n\n{'#'*70}")
        print(f"# [{i}/{len(todo)}] (overall {skipped + i}/{len(slides)}) {slide_id}")
        print(f"{'#'*70}")

        try:
            pipeline = QCPipeline(
                slide_id=slide_id,
                cancer_type=args.cancer_type,
                max_rounds=args.max_rounds,
                topk_patches=args.topk,
                conch_gpu=args.conch_gpu,
                patho_gpu=args.patho_gpu,
                agent_backend=args.backend,
                qwen_api_key=args.qwen_api_key,
                qwen_model=args.qwen_model,
                qwen_base_url=args.qwen_base_url,
                dify_url=args.dify_url,
                dify_key=args.dify_key,
                qc_rules=args.qc_rules,
                language=args.language,
                dataset_name=args.dataset_name,
                anatomical_site=args.anatomical_site,
                wsi_report_dir=args.wsi_report_dir,
                patch_report_dir=args.patch_report_dir,
                features_dir=args.features_dir,
                coords_dir=args.coords_dir,
                svs_dir=args.svs_dir,
                output_dir=args.output_dir,
                shared_retriever=shared_retriever,
                shared_reporter=shared_reporter,
                shared_qc_engine=shared_qc_engine,
                no_retrieval=args.no_retrieval,
                no_patho_r1=args.no_patho_r1,
                n_clusters=args.n_clusters,
            )
            result = pipeline.run()
            if result["qc_passed"]:
                results["passed"] += 1
            else:
                results["needs_review"] += 1
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback; traceback.print_exc()
            results["error"] += 1

    print(f"\n\n{'='*70}")
    print(f"Batch processing complete!")
    print(f"  Total:        {len(slides)}")
    print(f"  Skipped:      {skipped}")
    print(f"  Passed:       {results['passed']}")
    print(f"  Needs Review: {results['needs_review']}")
    print(f"  Errors:       {results['error']}")
    print(f"{'='*70}")

    return 0 if results["error"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="WSI Pathology Report QC Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--slide_id", type=str, help="Single WSI Slide ID")
    parser.add_argument("--slide_list", type=str, help="Batch mode: slide ID list file")

    # QC Agent config
    parser.add_argument(
        "--backend", type=str, default="qwen",
        choices=["qwen", "dify"],
        help="QC Agent backend: qwen (direct LLM) / dify (Dify workflow)",
    )
    parser.add_argument(
        "--qwen_api_key", type=str,
        default=os.getenv("QWEN_API_KEY", ""),
        help="Qwen API Key (or set QWEN_API_KEY env var)",
    )
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument(
        "--qwen_base_url", type=str, default="",
        help="Qwen API base URL (default: use QwenClient default)",
    )
    parser.add_argument(
        "--dify_url", type=str,
        default=os.getenv("DIFY_BASE_URL", "http://localhost"),
    )
    parser.add_argument(
        "--dify_key", type=str,
        default=os.getenv("DIFY_API_KEY", ""),
    )
    parser.add_argument("--qc_rules", type=str, default="")

    # GPU config
    parser.add_argument("--conch_gpu", type=int, default=4, help="CONCH encoder GPU")
    parser.add_argument("--patho_gpu", type=int, default=5, help="Patho-R1 GPU")

    # Pipeline config
    parser.add_argument("--cancer_type", type=str, default="gastric adenocarcinoma/STAD")
    parser.add_argument(
        "--language", type=str, default="en",
        choices=["en", "zh"],
        help="Report output language: en (English) / zh (Chinese)",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="TCGA-STAD",
        help="Dataset name for context (e.g., TCGA-STAD, 301-Hospital)",
    )
    parser.add_argument(
        "--anatomical_site", type=str, default="stomach (gastric)",
        help="Anatomical site for context",
    )
    parser.add_argument("--max_rounds", type=int, default=3, help="Max QC iterations")
    parser.add_argument("--topk", type=int, default=3, help="TopK patches per query")

    # Ablation flags
    parser.add_argument("--no_retrieval", action="store_true",
                        help="Ablation: disable CONCH retrieval (LLM-only QC)")
    parser.add_argument("--no_patho_r1", action="store_true",
                        help="Ablation: disable Patho-R1 supplement reports")
    parser.add_argument("--n_clusters", type=int, default=50,
                        help="Patch Selection Tool-1: number of clusters (default: 50)")

    # Path config
    parser.add_argument("--wsi_report_dir", type=str, default="./data/reports_batch_TCGA")
    parser.add_argument("--patch_report_dir", type=str, default="./data/tcga_stad_reports")
    parser.add_argument("--features_dir", type=str, default="./data/features_conch_v1")
    parser.add_argument("--coords_dir", type=str, default="./data/patches")
    parser.add_argument("--svs_dir", type=str, default="./data/TCGA-STAD")
    parser.add_argument("--output_dir", type=str, default="./qc_results")

    # Logging
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--log_file", type=str, default=None)

    args = parser.parse_args()
    setup_logging(args.log_level, args.log_file)

    if args.batch:
        if not args.slide_list:
            parser.error("Batch mode requires --slide_list")
        return run_batch(args)
    else:
        if not args.slide_id:
            parser.error("Single mode requires --slide_id")
        run_single(args)
        return 0


if __name__ == "__main__":
    sys.exit(main())
