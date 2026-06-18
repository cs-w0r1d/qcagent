# -*- coding: utf-8 -*-
"""
QC Agent Output Parser
======================

Robustly extract structured QC JSON from LLM output.
Includes multi-strategy parsing with fallbacks.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def strip_think_tags(s: str) -> str:
    """Remove <think>...</think> tags."""
    if not s:
        return ""
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()


def strip_markdown_code_block(s: str) -> str:
    """Remove ```json ... ``` wrapper."""
    if not s:
        return ""
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def find_json_candidates(s: str) -> List[str]:
    """
    Find all balanced {...} fragments in a string.
    Returns candidate JSON strings (prioritized from last to first).
    """
    candidates = []
    stack = []
    in_str = False
    esc = False
    start_idx = None

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if not stack:
                    start_idx = i
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        candidates.append(s[start_idx:i + 1])
                        start_idx = None
    return candidates


def try_load_json(txt: str) -> Optional[dict]:
    """Try to parse text as dict, supports double-parsing (escaped JSON string)."""
    if not txt:
        return None
    txt = txt.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            return obj2 if isinstance(obj2, dict) else None
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# QC JSON required keys
QC_REQUIRED_KEYS = {"non_compliant", "need_more_info", "pass", "summary",
                     "draft_report", "final_report"}
QC_INDICATOR_KEYS = {"non_compliant", "need_more_info", "pass", "summary"}


def extract_qc_json(llm_output: str) -> Dict[str, Any]:
    """
    Extract QC JSON from LLM output.

    Strategies:
    1. Strip <think> tags and markdown code blocks
    2. Try direct parsing of full text
    3. Find all {...} candidates
    4. Prioritize candidate containing most QC key fields
    5. Fallback to empty QC structure

    Args:
        llm_output: Raw LLM output text

    Returns:
        Parsed QC dict
    """
    s = strip_think_tags(llm_output)
    s = strip_markdown_code_block(s)

    # Strategy 1: Direct parse
    direct = try_load_json(s)
    if direct and any(k in direct for k in QC_INDICATOR_KEYS):
        return _normalize_qc_dict(direct)

    # Strategy 2: Find all candidate JSONs
    candidates = find_json_candidates(s)

    best_match = None
    best_score = -1

    for cand in reversed(candidates):
        obj = try_load_json(cand)
        if not isinstance(obj, dict):
            continue
        score = sum(1 for k in QC_INDICATOR_KEYS if k in obj)
        if score > best_score:
            best_score = score
            best_match = obj

    if best_match is not None:
        return _normalize_qc_dict(best_match)

    # Strategy 3: Fallback — try any parsed dict
    for cand in candidates:
        obj = try_load_json(cand)
        if isinstance(obj, dict):
            return _normalize_qc_dict(obj)

    logger.warning("Cannot extract QC JSON from LLM output, using empty result")
    logger.debug(f"Raw output first 500 chars: {llm_output[:500]}")
    return _empty_qc_dict()


def _normalize_qc_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize QC dict, ensure all required fields exist with correct types."""
    result = {
        "non_compliant": _ensure_list(d.get("non_compliant")),
        "need_more_info": _ensure_list(d.get("need_more_info")),
        "pass": _ensure_list(d.get("pass")),
        "summary": _ensure_summary(d.get("summary")),
        "draft_report": str(d.get("draft_report", "")),
        "final_report": str(d.get("final_report", "")),
    }

    for item in result["need_more_info"]:
        if isinstance(item, dict):
            item["missing_fields"] = _ensure_list(item.get("missing_fields"))
            item["questions_to_request"] = _ensure_list(
                item.get("questions_to_request")
            )

    for item in result["non_compliant"]:
        if isinstance(item, dict):
            item["reasons"] = _ensure_list(item.get("reasons"))
            item["missing_fields"] = _ensure_list(item.get("missing_fields"))

    return result


def _ensure_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]


def _ensure_summary(x: Any) -> Dict[str, int]:
    default = {"total": 0, "non_compliant": 0, "need_more_info": 0, "pass": 0}
    if not isinstance(x, dict):
        return default
    return {
        "total": int(x.get("total", 0)),
        "non_compliant": int(x.get("non_compliant", 0)),
        "need_more_info": int(x.get("need_more_info", 0)),
        "pass": int(x.get("pass", 0)),
    }


def _empty_qc_dict() -> Dict[str, Any]:
    return {
        "non_compliant": [],
        "need_more_info": [],
        "pass": [],
        "summary": {"total": 0, "non_compliant": 0, "need_more_info": 0, "pass": 0},
        "draft_report": "",
        "final_report": "",
    }


class QCOutputParser:
    """QC Output Parser."""

    # Administrative fields: cannot be inferred from WSI patches
    ADMIN_FIELD_KEYWORDS = [
        "patient", "identifier", "mrn", "name",
        "gender", "sex", "age",
        "referring", "department",
        "report date", "date",
        "signature", "pathologist",
        "姓名", "性别", "年龄", "病人", "患者",
        "科室", "送检", "日期", "签名",
        "标本编号", "specimen number",
    ]

    @staticmethod
    def parse(llm_output: str) -> Dict[str, Any]:
        """Parse LLM output into structured QC dict."""
        return extract_qc_json(llm_output)

    @classmethod
    def is_admin_field(cls, field_name: str) -> bool:
        """Check if a field is administrative (non-image-derivable)."""
        f = (field_name or "").lower()
        return any(k in f for k in cls.ADMIN_FIELD_KEYWORDS)

    @classmethod
    def classify_missing_fields(
        cls, qc_result: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Classify missing fields into image-derivable vs admin-required.

        Returns:
            {
                "image_related": ["field1", "field2", ...],
                "admin_required": ["field3", "field4", ...],
                "retrieval_queries": ["query1", "query2", ...],
            }
        """
        image_related = []
        admin_required = []
        retrieval_queries = []

        for item in qc_result.get("need_more_info", []):
            if not isinstance(item, dict):
                continue
            for f in item.get("missing_fields", []):
                if cls.is_admin_field(f):
                    admin_required.append(f)
                else:
                    image_related.append(f)

            for q in item.get("questions_to_request", []):
                if not cls.is_admin_field(q):
                    retrieval_queries.append(q)

        for item in qc_result.get("non_compliant", []):
            if not isinstance(item, dict):
                continue
            for r in item.get("reasons", []):
                if r and not cls.is_admin_field(r):
                    retrieval_queries.append(r)

        # Deduplicate
        image_related = list(dict.fromkeys(image_related))
        admin_required = list(dict.fromkeys(admin_required))
        retrieval_queries = list(dict.fromkeys(retrieval_queries))

        # Auto-generate queries if needed
        if not retrieval_queries and image_related:
            for f in image_related:
                retrieval_queries.append(
                    f"Histopathological evidence for {f}"
                )

        return {
            "image_related": image_related,
            "admin_required": admin_required,
            "retrieval_queries": retrieval_queries,
        }

    @classmethod
    def needs_iteration(cls, qc_result: Dict[str, Any]) -> bool:
        """Check if further iteration is needed."""
        classified = cls.classify_missing_fields(qc_result)
        has_image_fields = len(classified["image_related"]) > 0
        has_queries = len(classified["retrieval_queries"]) > 0
        has_conflicts = len(qc_result.get("non_compliant", [])) > 0
        return has_image_fields or has_queries or has_conflicts

    @classmethod
    def is_qc_passed(cls, qc_result: Dict[str, Any]) -> bool:
        """Check if QC passed."""
        return (
            len(qc_result.get("non_compliant", [])) == 0
            and len(qc_result.get("need_more_info", [])) == 0
        )

    @staticmethod
    def format_summary(qc_result: Dict[str, Any], round_num: int = 1) -> str:
        """Generate human-readable QC summary."""
        summary = qc_result.get("summary", {})
        n_non = summary.get("non_compliant", len(qc_result.get("non_compliant", [])))
        n_need = summary.get("need_more_info", len(qc_result.get("need_more_info", [])))
        n_pass = summary.get("pass", len(qc_result.get("pass", [])))

        passed = (n_non == 0 and n_need == 0)
        status = "✓ PASSED" if passed else "✗ NEEDS SUPPLEMENT"

        lines = [
            f"QC Round {round_num} Result: {status}",
            f"  Non-compliant: {n_non}, Need more info: {n_need}, Pass: {n_pass}",
        ]

        for item in qc_result.get("non_compliant", []):
            if isinstance(item, dict):
                rid = item.get("id", "?")
                reasons = item.get("reasons", [])
                lines.append(f"  [NON_COMPLIANT {rid}]")
                for r in reasons:
                    lines.append(f"    - {r}")

        for item in qc_result.get("need_more_info", []):
            if isinstance(item, dict):
                rid = item.get("id", "?")
                missing = item.get("missing_fields", [])
                questions = item.get("questions_to_request", [])
                lines.append(f"  [NEED_MORE_INFO {rid}]")
                for m in missing:
                    lines.append(f"    Missing: {m}")
                for q in questions:
                    lines.append(f"    Suggest: {q}")

        return "\n".join(lines)
