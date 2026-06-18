#!/usr/bin/env python3
"""
Run Patho-R1 inference on selected patches and generate a visualization figure.

Usage:
    conda activate patho-r1
    python run_patho_r1_inference.py \
    --patch_dir /path/to/selected_patches/TCGA-XX-XXXX \
        --output_dir ./patho_r1_results \
    --gpu 0 \
    --model_path /path/to/Patho-R1/models/Patho-R1-7B
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

# Add Patho-R1 repo to path
PATHO_R1_REPO = os.getenv("PATHO_R1_REPO_PATH", "")
if PATHO_R1_REPO and PATHO_R1_REPO not in sys.path:
    sys.path.insert(0, PATHO_R1_REPO)


def run_inference(
    patch_dir: str,
    output_dir: str,
    gpu: int,
    cancer_type: str,
    model_path: str,
    max_patches_per_batch: int = 5,
    max_tokens: int = 2048,
):
    """Run Patho-R1 on all patches, saving per-cluster reports."""
    import torch
    from generate_pathology_report import PathologyReportGenerator

    patch_dir = Path(patch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort patches by cluster
    patches = sorted(patch_dir.glob("*.png"))
    if not patches:
        patches = sorted(patch_dir.glob("*.jpg"))
    print(f"Found {len(patches)} patches in {patch_dir.name}")

    # Group by cluster
    cluster_map = {}
    for p in patches:
        # e.g. cluster_00_patch_9494_x48640_y37888.png
        parts = p.stem.split("_")
        cluster_id = int(parts[1])
        cluster_map.setdefault(cluster_id, []).append(str(p))

    print(f"Found {len(cluster_map)} clusters")

    # Init model
    device = f"cuda:{gpu}"
    print(f"Loading Patho-R1 on {device}...")
    generator = PathologyReportGenerator(
        model_path=model_path,
        device=device,
    )

    # Run inference per cluster (each cluster has 1 representative patch)
    all_results = []
    for cluster_id in sorted(cluster_map.keys()):
        cluster_patches = cluster_map[cluster_id]
        print(f"\n[Cluster {cluster_id:02d}] {len(cluster_patches)} patch(es)")

        try:
            result = generator.generate_report(
                image_path=cluster_patches,
                prompt_type="detailed",
                max_tokens=max_tokens,
                cancer_type=cancer_type,
            )
            answer = result.get("answer") or result.get("raw_output", "")
            thinking = result.get("thinking", "")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            answer = f"[Generation failed: {e}]"
            thinking = ""

        entry = {
            "cluster_id": cluster_id,
            "patch_paths": cluster_patches,
            "patch_name": Path(cluster_patches[0]).name,
            "answer": answer,
            "thinking": thinking,
        }
        all_results.append(entry)

        # Preview
        preview = answer[:120].replace("\n", " ")
        if len(answer) > 120:
            preview += "..."
        print(f"  → {preview}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results JSON
    results_path = output_dir / "patho_r1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "slide_id": patch_dir.name,
            "cancer_type": cancer_type,
            "timestamp": datetime.now().isoformat(),
            "n_clusters": len(cluster_map),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved: {results_path}")

    # Save combined report text
    report_path = output_dir / "combined_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Patho-R1 Patch-Level Reports\n")
        f.write(f"{'='*80}\n")
        f.write(f"Slide: {patch_dir.name}\n")
        f.write(f"Cancer type: {cancer_type}\n")
        f.write(f"Total clusters: {len(cluster_map)}\n\n")
        for r in all_results:
            f.write(f"--- Cluster {r['cluster_id']:02d}: {r['patch_name']} ---\n")
            f.write(f"{r['answer']}\n\n")
    print(f"✓ Combined report: {report_path}")

    return all_results


def generate_figure(patch_dir: str, output_dir: str, results: list,
                    ncols: int = 5, max_text_lines: int = 8):
    """Generate a grid figure: each cell = patch image + truncated report text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    output_dir = Path(output_dir)
    n = len(results)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 5.5))
    fig.suptitle(
        f"Patho-R1 Patch Reports — {Path(patch_dir).name}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    if nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue

        r = results[idx]
        img_path = r["patch_paths"][0]
        img = Image.open(img_path).convert("RGB")

        ax.imshow(np.array(img))
        ax.set_xticks([])
        ax.set_yticks([])

        # Title = cluster ID
        ax.set_title(f"Cluster {r['cluster_id']:02d}", fontsize=10, fontweight="bold")

        # Wrap and truncate the report text for display below the image
        answer = r["answer"].replace("\n", " ").strip()
        wrapped = textwrap.wrap(answer, width=45)
        if len(wrapped) > max_text_lines:
            wrapped = wrapped[:max_text_lines]
            wrapped[-1] += " ..."
        display_text = "\n".join(wrapped)

        ax.text(
            0.5, -0.02, display_text,
            transform=ax.transAxes, fontsize=5.5,
            verticalalignment="top", horizontalalignment="center",
            wrap=False, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=4.0)

    fig_path = output_dir / "patho_r1_patch_reports_grid.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✓ Figure saved: {fig_path}")

    # Also generate a compact summary figure (top 10)
    n_show = min(10, n)
    fig2, axes2 = plt.subplots(2, 5, figsize=(24, 12))
    fig2.suptitle(
        f"Patho-R1 Top-10 Cluster Reports — {Path(patch_dir).name}",
        fontsize=16, fontweight="bold",
    )
    axes2_flat = axes2.flatten()

    for idx in range(10):
        ax = axes2_flat[idx]
        if idx >= n_show:
            ax.axis("off")
            continue
        r = results[idx]
        img = Image.open(r["patch_paths"][0]).convert("RGB")
        ax.imshow(np.array(img))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Cluster {r['cluster_id']:02d}", fontsize=12, fontweight="bold")

        answer = r["answer"].replace("\n", " ").strip()
        wrapped = textwrap.wrap(answer, width=50)
        if len(wrapped) > 6:
            wrapped = wrapped[:6]
            wrapped[-1] += " ..."
        display_text = "\n".join(wrapped)
        ax.text(
            0.5, -0.02, display_text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=4.5)
    fig2_path = output_dir / "patho_r1_top10_summary.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig2)
    print(f"✓ Summary figure saved: {fig2_path}")


def main():
    parser = argparse.ArgumentParser(description="Patho-R1 inference on selected patches")
    parser.add_argument("--patch_dir", type=str, required=True, help="Patch image directory")
    parser.add_argument("--output_dir", type=str, default="./patho_r1_results", help="Output dir")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--cancer_type", type=str, default="gastric adenocarcinoma/STAD")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("PATHO_R1_MODEL_PATH", ""),
        help="Patho-R1 model path. Or set PATHO_R1_MODEL_PATH.",
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference, only regenerate figure from existing results JSON")
    args = parser.parse_args()

    if not PATHO_R1_REPO:
        print("❌ PATHO_R1_REPO_PATH is not set.")
        print("   Please export PATHO_R1_REPO_PATH=/path/to/Patho-R1")
        return
    if not args.model_path:
        print("❌ model_path is empty.")
        print("   Please pass --model_path or export PATHO_R1_MODEL_PATH")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_inference:
        results_path = output_dir / "patho_r1_results.json"
        if not results_path.exists():
            print(f"❌ No results JSON at {results_path}")
            return
        with open(results_path, "r") as f:
            data = json.load(f)
        results = data["results"]
        print(f"Loaded {len(results)} results from {results_path}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # After setting CUDA_VISIBLE_DEVICES, device becomes cuda:0
        results = run_inference(
            args.patch_dir, args.output_dir, gpu=0,
            cancer_type=args.cancer_type,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
        )

    generate_figure(args.patch_dir, args.output_dir, results)
    print("\n✅ All done!")


if __name__ == "__main__":
    main()
