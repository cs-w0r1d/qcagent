# -*- coding: utf-8 -*-
"""
QC Engine: Orchestrate LLM-based QC workflow
=============================================

Supports two backends:
- qwen: Direct Qwen API calls (default, recommended)
- dify:  Dify Workflow API calls
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .llm_client import QwenClient, DifyClient
from .prompts import QCPromptBuilder
from .output_parser import QCOutputParser, extract_qc_json

logger = logging.getLogger(__name__)


class QCEngine:
    """
    Pathology Report QC Engine.

    Uses LLM for QC assessment:
    - Identify non-compliant reports
    - Find missing information
    - Generate integrated draft_report
    - Classify missing fields (image-derivable vs admin info)
    """

    def __init__(
        self,
        backend: str = "qwen",
        # Qwen config
        qwen_api_key: str = "",
        qwen_model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        qwen_base_url: str = "",
        # Dify config
        dify_url: str = "",
        dify_key: str = "",
        # QC config
        qc_rules: str = "",
        temperature: float = 0.7,
        language: str = "en",
    ):
        self.backend = backend
        self.qc_rules = qc_rules
        self.language = language
        self.prompt_builder = QCPromptBuilder(qc_rules=qc_rules, language=language)
        self.parser = QCOutputParser()

        if backend == "qwen":
            kwargs = dict(
                api_key=qwen_api_key,
                model=qwen_model,
                temperature=temperature,
            )
            if qwen_base_url:
                kwargs["base_url"] = qwen_base_url
            self.llm = QwenClient(**kwargs)
            logger.info(f"QCEngine using Qwen backend: model={qwen_model}")
        elif backend == "dify":
            self.dify = DifyClient(
                base_url=dify_url,
                api_key=dify_key,
            )
            logger.info(f"QCEngine using Dify backend: {dify_url}")
        else:
            raise ValueError(f"Unsupported backend: {backend}, options: qwen, dify")

    def assess(
        self,
        reports: List[Dict[str, str]],
        current_draft: Optional[str] = None,
        supplement_evidence: Optional[str] = None,
        round_num: int = 1,
        cancer_type: str = "gastric adenocarcinoma/STAD",
        dataset_name: str = "TCGA-STAD",
        anatomical_site: str = "stomach (gastric)",
    ) -> Dict[str, Any]:
        """
        Perform one round of QC assessment.

        Args:
            reports: [{"id": "R1", "text": "..."}, ...]
            current_draft: Previous round's draft_report
            supplement_evidence: Supplemental patch-level evidence
            round_num: Current round number
            cancer_type: Known cancer type from dataset context
            dataset_name: Dataset identifier (e.g., "TCGA-STAD", "301-Hospital")
            anatomical_site: Anatomical site description

        Returns:
            {
                "qc_result": dict,
                "qc_passed": bool,
                "draft_report": str,
                "final_report": str,
                "classified_fields": dict,
                "summary": str,
                "thinking": str,
            }
        """
        logger.info(f"QC Round {round_num}: {len(reports)} reports")

        if self.backend == "qwen":
            qc_result, thinking = self._assess_qwen(
                reports, current_draft, supplement_evidence,
                cancer_type, dataset_name, anatomical_site,
            )
        else:
            qc_result, thinking = self._assess_dify(
                reports, current_draft, supplement_evidence, cancer_type
            )

        classified = self.parser.classify_missing_fields(qc_result)
        qc_passed = self.parser.is_qc_passed(qc_result)
        summary = self.parser.format_summary(qc_result, round_num)

        return {
            "qc_result": qc_result,
            "qc_passed": qc_passed,
            "draft_report": qc_result.get("draft_report", ""),
            "final_report": qc_result.get("final_report", ""),
            "classified_fields": classified,
            "summary": summary,
            "thinking": thinking,
        }

    def _assess_qwen(self, reports, current_draft, supplement_evidence,
                      cancer_type="gastric adenocarcinoma/STAD",
                      dataset_name="TCGA-STAD",
                      anatomical_site="stomach (gastric)"):
        """Assess using Qwen API."""
        system_prompt = self.prompt_builder.get_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(
            reports=reports,
            current_draft=current_draft,
            supplement_evidence=supplement_evidence,
            cancer_type=cancer_type,
            dataset_name=dataset_name,
            anatomical_site=anatomical_site,
        )

        response = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        content = response["content"]
        thinking = response.get("thinking", "")
        qc_result = extract_qc_json(content)

        usage = response.get("usage", {})
        if usage:
            logger.info(
                f"Token usage: prompt={usage.get('prompt_tokens', '?')}, "
                f"completion={usage.get('completion_tokens', '?')}, "
                f"total={usage.get('total_tokens', '?')}"
            )

        return qc_result, thinking

    def _assess_dify(self, reports, current_draft, supplement_evidence, cancer_type="gastric adenocarcinoma/STAD"):
        """Assess using Dify Workflow."""
        all_reports = list(reports)
        if current_draft:
            all_reports.append({
                "id": "CURRENT_DRAFT",
                "text": f"[Previous round integrated report]\n{current_draft}",
            })
        if supplement_evidence:
            all_reports.append({
                "id": "SUPPLEMENT_EVIDENCE",
                "text": f"[WSI patch supplement evidence]\n{supplement_evidence}",
            })

        reports_str = json.dumps(all_reports, ensure_ascii=False)

        outputs = self.dify.run(
            reports=reports_str,
            qc_rules=self.qc_rules,
        )

        qc_json_str = outputs.get("qc_json", "{}")
        if isinstance(qc_json_str, str):
            qc_result = extract_qc_json(qc_json_str)
        elif isinstance(qc_json_str, dict):
            qc_result = qc_json_str
        else:
            qc_result = extract_qc_json(str(qc_json_str))

        return qc_result, ""

    def generate_retrieval_queries(
        self,
        qc_result: Dict[str, Any],
        cancer_type: str = "gastric adenocarcinoma",
    ) -> List[str]:
        """
        Generate CONCH retrieval queries based on QC result.

        Prioritizes LLM-returned questions_to_request,
        supplements with auto-generated queries.
        """
        classified = self.parser.classify_missing_fields(qc_result)
        queries = list(classified["retrieval_queries"])

        if not queries and classified["image_related"]:
            for field in classified["image_related"]:
                queries.append(
                    f"Histopathological evidence for {field} in {cancer_type}"
                )

        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        return queries
