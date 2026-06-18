# -*- coding: utf-8 -*-
"""
QC Pipeline: Main workflow orchestrator
========================================

Complete workflow:
1. Load WSI report (Prism) + Patch report (Patho-R1)
2. Submit to QCEngine (Qwen LLM / Dify) for QC assessment
3. LLM returns structured QC JSON (missing fields / conflicts / draft_report)
4. Generate CONCH retrieval queries → FAISS retrieve most relevant patches
5. Cut patches from SVS → Patho-R1 generates supplement reports
6. Feed supplement evidence back to LLM → iterate
7. Output final QC report

GPU allocation:
- CONCH text encoder: ~500MB (e.g., GPU 4)
- Patho-R1 (7B): ~16GB (e.g., GPU 5)
"""

import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from qc_agent import QCEngine, QCOutputParser
from retrieval import CONCHRetriever, PatchClusterSelector
from inference import PatchCutter, PathoR1Reporter

logger = logging.getLogger(__name__)


# ===========================================================================
# Report Loader
# ===========================================================================
class ReportLoader:
    """Load WSI reports and Patch reports."""

    def __init__(
        self,
        wsi_report_dir: str = "./data/reports_batch_TCGA",
        patch_report_dir: str = "./data/tcga_stad_reports",
    ):
        self.wsi_report_dir = Path(wsi_report_dir)
        self.patch_report_dir = Path(patch_report_dir)

    def load(self, slide_id: str) -> List[Dict[str, str]]:
        """
        Load and format reports for QCEngine.

        Returns:
            [{"id": "WSI_REPORT", "text": "..."}, {"id": "PATCH_REPORT", "text": "..."}]
        """
        reports = []

        wsi_path = self.wsi_report_dir / f"{slide_id}.txt"
        if wsi_path.exists():
            wsi_text = wsi_path.read_text(encoding="utf-8").strip()
            reports.append({
                "id": "WSI_REPORT",
                "source": "Prism (WSI-level overview)",
                "text": wsi_text,
            })
            logger.info(f"Loaded WSI report: {wsi_path.name} ({len(wsi_text)} chars)")
        else:
            logger.warning(f"WSI report not found: {wsi_path}")

        patch_path = self.patch_report_dir / f"{slide_id}_report.txt"
        if patch_path.exists():
            patch_text = patch_path.read_text(encoding="utf-8").strip()
            reports.append({
                "id": "PATCH_REPORT",
                "source": "Patho-R1 (Patch-level detailed analysis)",
                "text": patch_text,
            })
            logger.info(f"Loaded Patch report: {patch_path.name} ({len(patch_text)} chars)")
        else:
            logger.warning(f"Patch report not found: {patch_path}")

        if not reports:
            raise FileNotFoundError(f"No reports found for {slide_id}")

        return reports


# ===========================================================================
# QC Pipeline
# ===========================================================================
class QCPipeline:
    """
    Pathology Report QC Pipeline.

    Orchestrates: Report Loading → LLM QC → CONCH Retrieval → SVS Cutting → Patho-R1 → Iteration
    """

    def __init__(
        self,
        # Basic config
        slide_id: str,
        cancer_type: str = "gastric adenocarcinoma/STAD",
        max_rounds: int = 3,
        topk_patches: int = 3,
        # GPU config
        conch_gpu: int = 4,
        patho_gpu: int = 5,
        # QC Agent config
        agent_backend: str = "qwen",
        qwen_api_key: str = "",
        qwen_model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        qwen_base_url: str = "",
        dify_url: str = "",
        dify_key: str = "",
        qc_rules: str = "",
        # Language & dataset config
        language: str = "en",
        dataset_name: str = "TCGA-STAD",
        anatomical_site: str = "stomach (gastric)",
        # Path config
        wsi_report_dir: str = "./data/reports_batch_TCGA",
        patch_report_dir: str = "./data/tcga_stad_reports",
        features_dir: str = "./data/features_conch_v1",
        coords_dir: str = "./data/patches",
        svs_dir: str = "./data/TCGA-STAD",
        output_dir: str = "./qc_results",
        # Shared components (for batch mode — reuse across slides)
        shared_retriever: Optional[CONCHRetriever] = None,
        shared_reporter: Optional[PathoR1Reporter] = None,
        shared_qc_engine: Optional[QCEngine] = None,
        # Ablation flags
        no_retrieval: bool = False,
        no_patho_r1: bool = False,
        # Patch Selection Tool-1 config
        n_clusters: int = 50,
    ):
        self.slide_id = slide_id
        self.cancer_type = cancer_type
        self.max_rounds = max_rounds
        self.topk_patches = topk_patches
        self.conch_gpu = conch_gpu
        self.patho_gpu = patho_gpu
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.dataset_name = dataset_name
        self.anatomical_site = anatomical_site

        # Ablation flags
        self.no_retrieval = no_retrieval
        self.no_patho_r1 = no_patho_r1

        # Patch Selection Tool-1 config
        self.n_clusters = n_clusters

        # Initialize components (reuse shared if provided)
        self.report_loader = ReportLoader(wsi_report_dir, patch_report_dir)

        self.qc_engine = shared_qc_engine or QCEngine(
            backend=agent_backend,
            qwen_api_key=qwen_api_key,
            qwen_model=qwen_model,
            qwen_base_url=qwen_base_url,
            dify_url=dify_url,
            dify_key=dify_key,
            qc_rules=qc_rules,
            language=language,
        )

        # Always initialize retriever & patch_cutter (needed for Step 0 clustering).
        # --no_retrieval only skips CONCH retrieval during QC iterations.
        self.retriever = shared_retriever or CONCHRetriever(
            features_dir=features_dir,
            coords_dir=coords_dir,
            device=f"cuda:{conch_gpu}",
        )

        self.patch_cutter = PatchCutter(svs_dir=svs_dir)

        self.reporter = None
        if not self.no_patho_r1:
            self.reporter = shared_reporter or PathoR1Reporter(
                device=f"cuda:{patho_gpu}",
                cancer_type=cancer_type,
            )

        # Runtime state
        self.history: List[Dict] = []
        self.retrieved_indices: Set[int] = set()

        # Per-round data (for visualization / demo)
        self._cluster_data: Dict = {}       # Step 0 cluster result
        self._round_data: List[Dict] = []  # QC round-level data

    def _generate_cluster_patch_report(self) -> Optional[str]:
        """
        Patch Selection Tool-1: Cluster-based initial patch selection.
        
        Clusters WSI patches into K groups, selects representative patches,
        and runs Patho-R1 to generate initial patch-level reports.
        
        Returns:
            Concatenated patch report string, or None if failed.
        """
        if self.n_clusters <= 0:
            logger.info("n_clusters=0, skipping cluster-based patch report")
            return None

        if self.no_patho_r1 or self.reporter is None:
            logger.info("Patho-R1 disabled, skipping cluster-based patch report")
            return None

        if self.retriever is None:
            logger.info("Retriever disabled, skipping cluster-based patch report")
            return None

        try:
            # Load slide data from CONCH retriever (reuse cached data)
            slide_data = self.retriever._load_slide_data(self.slide_id)
            features = slide_data["features_normed"]
            coords = slide_data["coords"]
            n_patches = slide_data["n_patches"]
            patch_size = slide_data["patch_size"]

            print(f"  📊 Slide has {n_patches} patches, clustering into {self.n_clusters}...")

            # Cluster and select representative patches
            selector = PatchClusterSelector(
                n_clusters=self.n_clusters,
                random_state=42,
            )
            selected = selector.select_patches(features, coords)
            print(f"  ✓ Selected {len(selected)} representative patches")

            # Store cluster selection coords
            self._cluster_data["selected_coords"] = selected

            # Track selected indices
            for s in selected:
                self.retrieved_indices.add(s["patch_idx"])

            # Cut patches from SVS
            if self.patch_cutter is None:
                logger.warning("PatchCutter not available, skipping")
                return None

            # Save cluster patches to per-slide output dir
            cluster_patch_dir = self.output_dir / self.slide_id / "round_0_cluster_patches"
            cluster_patch_dir.mkdir(parents=True, exist_ok=True)

            print(f"  ✂️ Cutting {len(selected)} patches from SVS...")
            cut_paths = self.patch_cutter.cut_patches(
                self.slide_id, selected, patch_size,
                dest_dir=str(cluster_patch_dir),
            )

            # Save cluster coords JSON
            cluster_coords_path = self.output_dir / self.slide_id / "round_0_cluster_coords.json"
            cluster_coords_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cluster_coords_path, "w", encoding="utf-8") as _f:
                json.dump({"selected": selected, "patch_size": patch_size, "cut_paths": cut_paths}, _f, ensure_ascii=False, indent=2, default=str)
            self._cluster_data["cut_paths"] = cut_paths

            if not cut_paths:
                print("  ⚠️ Failed to cut patches")
                return None

            print(f"  ✓ Cut {len(cut_paths)} patches")

            # Run Patho-R1 on selected patches
            print(f"  🔬 Running Patho-R1 on {len(cut_paths)} cluster-representative patches...")
            batch_reports = self.reporter.generate_batch_reports(
                cut_paths, batch_size=5
            )

            # Collect valid reports
            report_parts = []
            for br in batch_reports:
                report_text = br["report"]
                if report_text and "[Generation failed" not in report_text:
                    report_parts.append(report_text)

            if not report_parts:
                print("  ⚠️ No valid patch reports generated")
                return None

            # Consolidate into N_GROUPS summary reports to reduce token cost
            n_groups = 5
            group_size = max(1, len(report_parts) // n_groups)
            consolidated = []
            for i in range(0, len(report_parts), group_size):
                group = report_parts[i : i + group_size]
                group_text = f"[Patch Group {len(consolidated)+1}] " + " ".join(group)
                consolidated.append(group_text)

            combined = "\n\n".join(consolidated)
            print(f"  ✓ Generated {len(report_parts)} patch reports → consolidated into {len(consolidated)} groups")

            # Save cluster patch report text
            cluster_report_path = self.output_dir / self.slide_id / "round_0_cluster_patch_report.txt"
            cluster_report_path.parent.mkdir(parents=True, exist_ok=True)
            cluster_report_path.write_text(combined, encoding="utf-8")
            self._cluster_data["report"] = combined
            self._cluster_data["batch_reports"] = [
                {"paths": br["paths"], "report": br["report"]} for br in batch_reports
            ]
            print(f"  ✓ Saved cluster patch data to round_0_cluster_*/")

            return combined

        except Exception as e:
            logger.error(f"Cluster patch report generation failed: {e}")
            import traceback; traceback.print_exc()
            return None

    def run(self) -> Dict[str, Any]:
        """Run the complete QC workflow."""
        self._print_header()

        # Step 0: Patch Selection Tool-1 — Cluster-based initial patch analysis
        print("\n🔬 Step 0: Patch Selection Tool-1 (Cluster-based)")
        print("-" * 50)
        cluster_patch_report = self._generate_cluster_patch_report()

        # Step 1: Load initial reports
        print("\n📋 Step 1: Load initial reports")
        print("-" * 50)
        reports = self.report_loader.load(self.slide_id)

        # Add cluster-based patch report if available
        if cluster_patch_report:
            reports.append({
                "id": "PATCH_REPORT",
                "source": "Patho-R1 (Cluster-based patch analysis, Tool-1)",
                "text": cluster_patch_report,
            })
            print(f"  ✓ Loaded {len(reports)} reports (including cluster patch analysis)")
        else:
            print(f"  ✓ Loaded {len(reports)} reports")

        current_draft = None
        supplement_evidence = ""
        final_report = ""
        qc_passed = False

        # Step 2: Initial synthesis — integrate PRISM + cluster patches into one initial report
        if self.max_rounds >= 1:
            print("\n📝 Step 2: Initial Report Synthesis (Round 0 — pre-QC)")
            print("-" * 50)
            try:
                init_assessment = self.qc_engine.assess(
                    reports=reports,
                    current_draft=None,
                    supplement_evidence=None,
                    round_num=0,
                    cancer_type=self.cancer_type,
                    dataset_name=self.dataset_name,
                    anatomical_site=self.anatomical_site,
                )
                initial_draft = init_assessment.get("draft_report", "")
                if initial_draft:
                    init_report_path = self.output_dir / self.slide_id / "round_0_initial_report.txt"
                    init_report_path.parent.mkdir(parents=True, exist_ok=True)
                    init_report_path.write_text(initial_draft, encoding="utf-8")
                    current_draft = initial_draft
                    preview = initial_draft[:300].replace("\n", " ")
                    if len(initial_draft) > 300:
                        preview += "..."
                    print(f"  ✓ Initial synthesis complete ({len(initial_draft)} chars)")
                    print(f"  ✓ Saved → round_0_initial_report.txt")
                    print(f"  📝 Initial Draft: {preview}")
                    # Record in round_data as round 0
                    self._round_data.append({
                        "round": 0,
                        "qc_passed": False,
                        "draft_report": initial_draft,
                        "retrieval_queries": [],
                        "retrieved_coords": [],
                        "cut_paths": [],
                        "patch_reports": [],
                    })
                else:
                    print("  ⚠️ Initial synthesis returned empty draft, proceeding without pre-QC draft")
            except Exception as e:
                logger.error(f"Initial synthesis failed: {e}")
                import traceback as _tb; _tb.print_exc()
                print(f"  ❌ Initial synthesis failed: {e}")

        # Step 3-N: Iterative QC
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n{'='*60}")
            print(f"🔄 QC Round {round_num}/{self.max_rounds}")
            print(f"{'='*60}")

            # LLM QC assessment
            print(f"\n  📊 {round_num}.1 LLM QC Assessment...")
            try:
                assessment = self.qc_engine.assess(
                    reports=reports,
                    current_draft=current_draft,
                    supplement_evidence=supplement_evidence if supplement_evidence else None,
                    round_num=round_num,
                    cancer_type=self.cancer_type,
                    dataset_name=self.dataset_name,
                    anatomical_site=self.anatomical_site,
                )
            except Exception as e:
                logger.error(f"QC assessment failed: {e}")
                traceback.print_exc()
                print(f"  ❌ QC assessment failed: {e}")
                break

            qc_result = assessment["qc_result"]
            qc_passed = assessment["qc_passed"]
            current_draft = assessment["draft_report"]
            final_report = assessment["final_report"] or current_draft
            classified = assessment["classified_fields"]

            print(f"\n  📋 QC Summary:")
            for line in assessment["summary"].split("\n"):
                print(f"    {line}")

            if assessment.get("thinking"):
                thinking_preview = assessment["thinking"][:200]
                if len(assessment["thinking"]) > 200:
                    thinking_preview += "..."
                print(f"\n  💭 LLM Thinking: {thinking_preview}")

            if current_draft:
                preview = current_draft[:300].replace("\n", " ")
                if len(current_draft) > 300:
                    preview += "..."
                print(f"\n  📝 Draft Preview: {preview}")

            # Save per-round draft report immediately
            round_draft_path = self.output_dir / self.slide_id / f"round_{round_num}_draft_report.txt"
            round_draft_path.parent.mkdir(parents=True, exist_ok=True)
            round_draft_path.write_text(current_draft or "", encoding="utf-8")

            # Initialize round data entry (append now, update in-place)
            _round_entry: Dict = {
                "round": round_num,
                "qc_passed": qc_passed,
                "draft_report": current_draft or "",
                "retrieval_queries": classified.get("retrieval_queries", []),
                "retrieved_coords": [],
                "cut_paths": [],
                "patch_reports": [],
            }
            self._round_data.append(_round_entry)  # append by reference, update in-place below

            # Save history
            self.history.append({
                "round": round_num,
                "qc_passed": qc_passed,
                "n_non_compliant": len(qc_result.get("non_compliant", [])),
                "n_need_more_info": len(qc_result.get("need_more_info", [])),
                "n_pass": len(qc_result.get("pass", [])),
                "n_image_fields": len(classified["image_related"]),
                "n_admin_fields": len(classified["admin_required"]),
                "n_queries": len(classified["retrieval_queries"]),
                "summary": assessment["summary"],
                "draft_report_len": len(current_draft or ""),
            })

            if qc_passed:
                print(f"\n  ✅ QC PASSED! Report quality meets requirements.")
                break

            if not QCOutputParser.needs_iteration(qc_result):
                print(f"\n  ⚠️ Remaining missing items are administrative info, cannot supplement from WSI.")
                if classified["admin_required"]:
                    print(f"  Needs upstream: {classified['admin_required']}")
                break

            # --- Ablation: skip retrieval entirely ---
            if self.no_retrieval:
                print(f"\n  ⛔ Retrieval disabled (ablation: --no_retrieval), ending iteration.")
                break

            # Generate retrieval queries
            queries = self.qc_engine.generate_retrieval_queries(
                qc_result, cancer_type=self.cancer_type
            )

            if not queries:
                print(f"\n  ⚠️ No retrieval queries available, ending iteration.")
                break

            print(f"\n  🔍 {round_num}.2 CONCH Retrieval ({len(queries)} queries)...")
            for i, q in enumerate(queries, 1):
                print(f"    {i}. {q}")

            # CONCH retrieval
            retrieval_results = self.retriever.search(
                self.slide_id,
                queries,
                topk=self.topk_patches,
                exclude_indices=self.retrieved_indices,
            )

            all_coords = []
            for i, (query, hits) in enumerate(zip(queries, retrieval_results)):
                print(f"\n    Query {i+1}: {query}")
                for hit in hits:
                    print(
                        f"      → patch_{hit['patch_idx']} "
                        f"({hit['x']}, {hit['y']}) "
                        f"score={hit['score']:.4f}"
                    )
                    if hit["patch_idx"] not in self.retrieved_indices:
                        all_coords.append(hit)
                        self.retrieved_indices.add(hit["patch_idx"])

            if not all_coords:
                print(f"\n  ⚠️ All retrieved patches already processed, ending iteration.")
                break

            # Track retrieved coords
            _round_entry["retrieved_coords"] = all_coords

            # Save round coords JSON
            round_coords_path = self.output_dir / self.slide_id / f"round_{round_num}_coords.json"
            with open(round_coords_path, "w", encoding="utf-8") as _f:
                json.dump({"queries": queries, "coords": all_coords}, _f, ensure_ascii=False, indent=2, default=str)

            # Cut patches — save to per-round sub-directory
            print(f"\n  ✂️ {round_num}.3 Cutting {len(all_coords)} patches...")
            patch_size = self.retriever.get_patch_size(self.slide_id)
            round_patch_dir = self.output_dir / self.slide_id / f"round_{round_num}_patches"
            round_patch_dir.mkdir(parents=True, exist_ok=True)
            cut_paths = self.patch_cutter.cut_patches(
                self.slide_id, all_coords, patch_size,
                dest_dir=str(round_patch_dir),
            )

            if not cut_paths:
                print(f"  ⚠️ Failed to cut any patches")
                break

            _round_entry["cut_paths"] = cut_paths
            print(f"  ✓ Cut {len(cut_paths)} patches → {round_patch_dir}")

            # --- Ablation: skip Patho-R1 ---
            if self.no_patho_r1 or self.reporter is None:
                print(f"\n  ⛔ Patho-R1 disabled (ablation: --no_patho_r1), ending iteration.")
                break

            # Patho-R1 supplement reports
            print(f"\n  🔬 {round_num}.4 Patho-R1 generating supplement reports...")
            batch_reports = self.reporter.generate_batch_reports(
                cut_paths, batch_size=5
            )

            round_evidence = f"\n--- QC Round {round_num} Supplement Evidence ---\n"
            valid_reports = 0
            round_patch_report_lines = []
            for i, br in enumerate(batch_reports):
                report_text = br["report"]
                if report_text and "[Generation failed" not in report_text:
                    valid_reports += 1
                    query_idx = min(i, len(queries) - 1)
                    round_evidence += (
                        f"\n[Query]: {queries[query_idx]}\n"
                        f"[Supplement Diagnosis]:\n{report_text}\n"
                    )
                    round_patch_report_lines.append(
                        f"[Query {query_idx+1}]: {queries[query_idx]}\n{report_text}"
                    )

            # Save round patch reports (update in-place)
            _round_entry["patch_reports"] = [{"paths": br["paths"], "report": br["report"]} for br in batch_reports]
            round_patch_report_text = "\n\n".join(round_patch_report_lines)
            round_pr_path = self.output_dir / self.slide_id / f"round_{round_num}_patch_reports.txt"
            round_pr_path.write_text(round_patch_report_text, encoding="utf-8")

            if valid_reports > 0:
                supplement_evidence += round_evidence
                print(f"\n  ✓ Generated {valid_reports} valid supplement reports")
            else:
                print(f"\n  ⚠️ No valid supplement reports generated, ending iteration.")
                break

        # Fallback: if no QC rounds ran (max_rounds=0), use cluster patch report as final
        if not final_report and cluster_patch_report:
            final_report = cluster_patch_report
            current_draft = cluster_patch_report
            print(f"\n  ℹ️  max_rounds=0: using cluster patch report as final report.")

        # Step 4: Save results
        print(f"\n📝 Step 4: Save final results")
        print("-" * 50)

        result = {
            "slide_id": self.slide_id,
            "cancer_type": self.cancer_type,
            "qc_passed": qc_passed,
            "total_rounds": len(self.history),
            "agent_backend": self.qc_engine.backend,
            "final_report": final_report,
            "draft_report": current_draft or "",
            "history": self.history,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_results(result, supplement_evidence)
        self._print_footer(result)

        return result

    def _print_header(self):
        print("=" * 70)
        print("  Pathology Report QC Pipeline (LLM-Powered)")
        print("=" * 70)
        print(f"  Slide ID:     {self.slide_id}")
        print(f"  Cancer Type:  {self.cancer_type}")
        print(f"  QC Backend:   {self.qc_engine.backend}")
        print(f"  CONCH GPU:    {self.conch_gpu}")
        print(f"  Patho GPU:    {self.patho_gpu}")
        print(f"  Max Rounds:   {self.max_rounds}")
        print(f"  TopK:         {self.topk_patches}")
        print(f"  Language:     {self.language}")
        print(f"  Dataset:      {self.dataset_name}")
        if self.no_retrieval:
            print(f"  ⛔ Ablation:   CONCH retrieval DISABLED")
        if self.no_patho_r1:
            print(f"  ⛔ Ablation:   Patho-R1 DISABLED")
        print("=" * 70)

    def _print_footer(self, result: Dict):
        print(f"\n{'='*70}")
        print(f"  QC Complete!")
        print(f"  Slide:    {result['slide_id']}")
        print(f"  Status:   {'✅ PASSED' if result['qc_passed'] else '⚠️ NEEDS REVIEW'}")
        print(f"  Rounds:   {result['total_rounds']}")
        print(f"  Backend:  {result['agent_backend']}")
        print(f"  Output:   {self.output_dir / self.slide_id}")
        print(f"{'='*70}")

    def _save_results(self, result: Dict, supplement: str):
        """Save all results."""
        save_dir = self.output_dir / self.slide_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Final report
        report_path = save_dir / "final_report.txt"
        report_path.write_text(result.get("final_report", ""), encoding="utf-8")

        # Draft report
        draft_path = save_dir / "draft_report.txt"
        draft_path.write_text(result.get("draft_report", ""), encoding="utf-8")

        # Supplement evidence
        if supplement.strip():
            supp_path = save_dir / "supplement_evidence.txt"
            supp_path.write_text(supplement, encoding="utf-8")

        # QC history (JSON)
        history_path = save_dir / "qc_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        # Human-readable summary
        summary_path = save_dir / "qc_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Pathology Report QC Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Slide ID:   {result['slide_id']}\n")
            f.write(f"Cancer:     {result['cancer_type']}\n")
            f.write(f"QC Status:  {'PASSED' if result['qc_passed'] else 'NEEDS REVIEW'}\n")
            f.write(f"Backend:    {result['agent_backend']}\n")
            f.write(f"Rounds:     {result['total_rounds']}\n")
            f.write(f"Timestamp:  {result['timestamp']}\n")
            f.write(f"{'='*60}\n\n")

            for h in result["history"]:
                f.write(f"Round {h['round']}:\n")
                f.write(f"  {h['summary']}\n")
                f.write(f"  (Image-related missing: {h['n_image_fields']}, ")
                f.write(f"Admin missing: {h['n_admin_fields']}, ")
                f.write(f"Retrieval queries: {h['n_queries']})\n\n")

            f.write(f"\n{'='*60}\n")
            f.write("Final Report:\n")
            f.write(f"{'='*60}\n")
            f.write(result.get("final_report", "(none)"))

        print(f"  ✓ Results saved:")
        print(f"    - {report_path}")
        print(f"    - {draft_path}")
        print(f"    - {history_path}")
        print(f"    - {summary_path}")

        # Save per-round visualization data (for paper figures)
        viz_data = {
            "slide_id": self.slide_id,
            "cluster_data": {
                "selected_num": len(self._cluster_data.get("selected_coords", [])),
                "coords": self._cluster_data.get("selected_coords", []),
                "cut_paths": self._cluster_data.get("cut_paths", []),
                "report_preview": (self._cluster_data.get("report", "") or "")[:500],
                "per_patch_reports": [
                    {"paths": br["paths"], "report_preview": (br["report"] or "")[:300]}
                    for br in self._cluster_data.get("batch_reports", [])
                ],
            },
            "rounds": [
                {
                    "round": rd["round"],
                    "qc_passed": rd["qc_passed"],
                    "queries": rd.get("retrieval_queries", []),
                    "retrieved_coords": rd.get("retrieved_coords", []),
                    "cut_paths": rd.get("cut_paths", []),
                    "draft_report_preview": (rd.get("draft_report", "") or "")[:500],
                    "patch_report_previews": [
                        {"paths": pr["paths"], "report_preview": (pr.get("report", "") or "")[:300]}
                        for pr in rd.get("patch_reports", [])
                    ],
                }
                for rd in self._round_data
            ],
        }
        viz_path = save_dir / "round_data_summary.json"
        with open(viz_path, "w", encoding="utf-8") as f:
            json.dump(viz_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"    - {viz_path}")
        print(f"    - round_0_cluster_patches/ (cluster patch images)")
        print(f"    - round_0_initial_report.txt (synthesized from PRISM + cluster patches)")
        for rd in self._round_data:
            r = rd["round"]
            if r == 0:
                continue  # round 0 is synthesis only, no patches dir
            print(f"    - round_{r}_patches/ ({len(rd.get('cut_paths', []))} patches), round_{r}_draft_report.txt, round_{r}_coords.json")

