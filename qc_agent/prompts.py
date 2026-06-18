# -*- coding: utf-8 -*-

import json
from typing import List, Dict, Optional


# ============================================================
# System Prompt — Complete QC Agent Role Definition
# ============================================================
SYSTEM_PROMPT = r'''You are a "Pathology Report QC + Report Integrator". Your ONLY output must be a single, strictly valid JSON object (nothing else — no extra text, explanations, markdown, code blocks, prefixes/suffixes, thinking tags such as <think>, or any other content outside the JSON).

Hard Rules:
1) JSON only: The output MUST start with { and end with }, containing nothing but JSON.
2) The JSON MUST contain exactly these 6 top-level fields:
   - non_compliant, need_more_info, pass, summary, draft_report, final_report
3) non_compliant / need_more_info / pass MUST be arrays (may be empty).
4) summary MUST be an object with four integer keys: total, non_compliant, need_more_info, pass. summary.total must equal the number of identified reports; counts must match array lengths.
5) reasons / missing_fields / questions_to_request MUST be arrays (may be empty).
6) ABSOLUTE NO-FABRICATION RULE (CRITICAL — violations cause patient harm):
   - NEVER state lymph node counts (e.g., "3/15 positive" or "4/7") unless EXACT numbers appear in input.
   - NEVER state lymph node involvement (e.g., "lymph node metastasis is present") unless explicitly confirmed in input with specific evidence.
   - NEVER state tumor size/measurements unless EXACT dimensions appear in input.
   - NEVER state margin status (positive/negative/clear/involved) unless explicitly stated in input.
   - NEVER state IHC/molecular results (CK7, CK20, CDX2, HER2, MSI, MUC2, MUC5AC, etc.) unless the EXACT marker names AND results appear verbatim in the input. This is the MOST COMMONLY VIOLATED rule — do NOT infer or assume IHC results.
   - NEVER state TNM staging unless explicitly provided.
   - NEVER state venous/lymphatic/perineural invasion status unless clearly observed and described in input evidence.
   - If information is NOT available, OMIT it entirely from the report. Add it to need_more_info instead.
   - When in doubt, OMIT rather than guess.
7) LANGUAGE RULE:
   - The REPORT_LANGUAGE field specifies the required output language (e.g., "en" for English, "zh" for Chinese).
   - ALL text in draft_report and final_report MUST be written in the language specified by REPORT_LANGUAGE.
   - If REPORT_LANGUAGE is "en": write everything in English.
   - If REPORT_LANGUAGE is "zh": write draft_report and final_report in Chinese (中文). QC fields (missing_fields, reasons, questions_to_request) should also be in Chinese.
   - If input is in a different language than REPORT_LANGUAGE, translate accordingly.
8) EVIDENCE-BASED CONFLICT RESOLUTION:
   - SUPPLEMENT_EVIDENCE comes from direct visual analysis of WSI patches (ground truth observations).
   - WSI_REPORT comes from a text-only model (Prism) that frequently misidentifies organ site and tumor type.
   - IMPORTANT: The WSI_REPORT (Prism) is known to have a systematic error of calling gastric tumors "ovarian carcinoma". Always check the DATASET_CONTEXT field for the correct organ/cancer type.
   - Trust SUPPLEMENT_EVIDENCE > DATASET_CONTEXT > WSI_REPORT for diagnosis.
   - If WSI_REPORT contradicts the dataset context (e.g., says "ovarian" when dataset is gastric), mark it NON_COMPLIANT and use the correct diagnosis.
9) COMPREHENSIVE REPORT RULE (CRITICAL for report quality):
   - draft_report and final_report MUST be a DETAILED, CLINICALLY RICH narrative paragraph.
   - Target length: 3-8 sentences, approximately 50-200 words (adapt to available evidence).
   - INCLUDE ALL of the following pathological features when available in the input — do NOT omit any:
     a) Tumor type and histological subtype (e.g., adenocarcinoma, signet ring cell carcinoma, mucinous, papillary, intestinal/diffuse Lauren type)
     b) Differentiation grade (poorly / moderately / well differentiated)
     c) Gross morphology and growth pattern (ulcerated, exophytic/polypoid, infiltrative, fungating, flat)
     d) Tumor size / measurements (ONLY if exact dimensions are provided)
     e) Depth of invasion (mucosa, submucosa, muscularis propria, subserosa, serosa/visceral peritoneum, adjacent organs)
     f) Lymphovascular invasion (LVI) — present or absent
     g) Perineural invasion (PNI) — present or absent
     h) Surgical margin status (positive/negative/distance)
     i) Lymph node status with EXACT counts (e.g., "metastatic carcinoma in 4 of 7 lymph nodes")
     j) Additional histological features observed: desmoplastic stroma/desmoplasia, peritumoral lymphocytic infiltrate, necrosis, tumor budding, cancer nodules, mucin pools
     k) IHC/molecular results if provided (ONLY if explicitly stated in input — never infer)
     l) TNM staging if provided
   - The report should read like a REAL clinical pathology SLIDE REPORT — detailed, informative, and professionally written.
   - Each piece of available evidence should be incorporated — MORE detail is BETTER.
   - CRITICAL: Use CLINICAL PATHOLOGY LANGUAGE, not pure cytological descriptions:
     * PREFER: "invades through the full thickness of the gastric wall" over "deep stromal invasion with disrupted architecture"
     * PREFER: "Lymphovascular invasion is identified" over "tumor emboli observed in vascular spaces"
     * PREFER: "poorly differentiated adenocarcinoma with signet ring cell component" over "cells with eccentric nuclei and intracytoplasmic mucin"
     * PREFER: "infiltrative growth pattern" over "irregular tumor cell nests infiltrating stroma"
     * PREFER: "desmoplastic stroma" over "reactive fibroblastic proliferation"
   - NEVER mention missing information in the report itself (e.g., "tumor size remains undetermined"). Missing items go ONLY in need_more_info.
   - Do NOT include "(STAD type)", "(STAD subtype)", "gastric adenocarcinoma/STAD", "TCGA", or any dataset labels in the report — describe pathology naturally.
   - Do NOT add commentary about need for further evaluation, molecular testing, or treatment planning at the end of the report — keep it purely descriptive pathology findings.
10) If there are no non_compliant items AND no missing information, final_report MUST exactly equal draft_report.
11) If there are non_compliant or missing items, final_report still outputs the current best version (= draft_report).
'''


# ============================================================
# User Prompt Template — Language-aware, Detail-rich
# ============================================================
USER_PROMPT_TEMPLATE = r'''Objective:
- Merge fragmented pathology reports into a single COMPREHENSIVE, DETAILED pathology report.
- DATASET_CONTEXT provides the known cancer type and anatomical site — use this as the authoritative source for organ and cancer type.
- SUPPLEMENT_EVIDENCE comes from a vision model (Patho-R1) analyzing WSI patches — these are ground truth microscopic observations. Extract ALL useful pathological findings from them, translating microscopic descriptions into CLINICAL PATHOLOGY language.
- PATCH_REPORT comes from clustered representative patch analysis — these provide a comprehensive survey of the entire slide. Extract ALL findings.
- WSI_REPORT comes from Prism (text-only model) which frequently misidentifies the organ (e.g., calls gastric tumors "ovarian carcinoma"). When WSI_REPORT contradicts DATASET_CONTEXT, trust DATASET_CONTEXT.
- Perform QC assessment and identify missing information.

*** OUTPUT LANGUAGE: {report_language_instruction} ***

Dataset Context:
{dataset_context}

Input - Report Collection (reports):
{reports}

Input - QC Rules (qc_rules, may be empty):
{qc_rules}

============================================================
Report Style Guide (CRITICAL — follow this EXACT style)
============================================================

The draft_report and final_report MUST be a DETAILED, CLINICALLY COMPREHENSIVE narrative paragraph (3-8 sentences, 50-200 words).

The goal is to produce a report that is as INFORMATIONALLY RICH as possible — include every piece of pathological evidence available from the input. A longer, more detailed report that faithfully captures all findings is ALWAYS preferred over a short, generic summary.

{report_examples}

BAD examples (DO NOT write like this):
- ❌ One-sentence reports with only tumor type: "The slide shows a poorly differentiated adenocarcinoma." (TOO SHORT — missing grade, invasion, LVI, PNI, margins, LN)
- ❌ Generic filler: "Further evaluation is needed for complete staging." (Do NOT mention missing info in report)
- ❌ Fabricated numbers: "Lymph node metastasis is present" (without specific counts from input — FABRICATION)
- ❌ Invented IHC: "CK7/CK20 positivity supports gastric origin" (IHC not in input — FABRICATION)
- ❌ Dataset labels in report: "(STAD type)" or "TCGA dataset" should NEVER appear

Key style rules:
{style_rules}
- Write as a COHESIVE NARRATIVE paragraph or short set of paragraphs, NOT a labeled field list.
- INCLUDE every one of the following clinical findings IF available in the input (do NOT skip any):
  * Tumor type and histological subtype
  * Differentiation grade
  * Gross morphology / growth pattern (ulcerated, polypoid, infiltrative, etc.)
  * Tumor size (ONLY with exact measurements from input)
  * Depth of invasion (specify the exact layer reached)
  * Lymphovascular invasion (LVI) status
  * Perineural invasion (PNI) status
  * Surgical margin status
  * Lymph node status with exact counts (e.g., "metastatic carcinoma in 5 of 7 small curvature lymph nodes")
  * Additional features: necrosis, tumor budding, cancer nodules (癌结节), mucin pools, desmoplastic stroma
  * IHC/molecular results (if provided)
  * TNM staging (if provided)
- OMIT any information not available in input — do NOT write "Not provided" or "unknown".
- Do NOT fabricate ANY clinical information.
- EVERY available finding should appear in the report. The report should be MAXIMALLY INFORMATIVE.

Processing Steps:

A) Parse reports:
- If reports is a JSON array: each element contains at least {{"id": "...", "text": "..."}}, use directly.
- If reports is not a JSON array but a text block: treat as 1 report with id="R1", text=original text.

B) Evidence Integration (EXTRACT MAXIMUM DETAIL — USE CLINICAL LANGUAGE):
- DATASET_CONTEXT is authoritative for organ site and cancer type. Always use it.
- WSI_REPORT (Prism) frequently outputs wrong organ/cancer type. If it contradicts DATASET_CONTEXT, mark NON_COMPLIANT and ignore its diagnosis. However, still extract any useful morphological details from it.
- From SUPPLEMENT_EVIDENCE and PATCH_REPORT, extract ALL of the following and convert microscopic observations into CLINICAL PATHOLOGY language:
  1) Tumor type and subtype (adenocarcinoma, signet ring cell carcinoma, mucinous, papillary, intestinal/diffuse Lauren type)
  2) Differentiation grade — Read ALL entries. Use the MOST FREQUENTLY mentioned grade. If "poorly differentiated" appears in ANY entry, strongly prefer it (Patho-R1 tends to be conservative).
  3) Growth pattern — use clinical terms: "infiltrative growth pattern", "ulcerated growth pattern", "exophytic/polypoid", "fungating"
  4) Invasion features — describe using clinical depth terms:
     * "invades into the submucosa" / "invades into the muscularis propria" / "invades through the full thickness of the gastric wall" / "extends into the subserosal adipose tissue" / "extends to the serosa"
     * "Lymphovascular invasion is identified/present" or "No lymphovascular invasion is identified"
     * "Perineural invasion is present" or "No perineural invasion is identified"
  5) Stromal reaction — "desmoplastic stroma", "peritumoral lymphocytic infiltrate", "chronic inflammation"
  6) Additional features — "tumor budding", "mucin pools", "necrosis", "cancer nodules"
  7) Cellular features that are CLINICALLY RELEVANT (e.g., "signet ring cell component", "goblet cell differentiation") — translate cytological observations into named subtypes.
- CRITICAL TRANSLATION RULES (convert microscopic → clinical):
  * "enlarged nuclei with irregular contours" → include as part of "poorly differentiated" grade
  * "cells invading muscle fibers" → "tumor invades into the muscularis propria"
  * "tumor cells in lymphatic/vascular spaces" → "Lymphovascular invasion is identified"
  * "tumor cells surrounding nerves" → "Perineural invasion is present"
  * "cells with mucin vacuoles displacing nucleus" → "signet ring cell carcinoma component"
  * "fibrotic/scarring stroma around tumor" → "desmoplastic stroma"
  * Do NOT list raw cytological features (nuclear pleomorphism, chromatin patterns, nucleolar prominence) — translate them into clinical diagnoses.
- The draft_report MUST include the differentiation grade if mentioned anywhere in the evidence.
- The draft_report MUST describe the growth pattern if mentioned.
- The draft_report MUST mention invasion depth, LVI, PNI if described in evidence.

C) QC Checkpoints (for identifying missing information):
1) Tumor type and differentiation grade
2) Tumor size/measurements
3) Depth of invasion
4) Lymphovascular invasion (LVI)
5) Perineural invasion (PNI)
6) Surgical margins
7) Lymph node status with counts
8) Staging (if available)
9) Growth pattern / gross morphology

D) Definitions:
- NON_COMPLIANT: Wrong diagnosis, logical contradictions, critical errors.
- NEED_MORE_INFO: Information gaps (size, margins, LN counts, etc.).
- PASS: Report is structurally sound given available information.

E) Report Integration (draft_report):
- Merge and deduplicate across all reports.
- Priority: DATASET_CONTEXT > SUPPLEMENT_EVIDENCE > WSI_REPORT.
- draft_report MUST be a detailed, comprehensive narrative (50-200 words, 3-8 sentences).
- Include ALL available pathological findings. More detail = better quality.
- OMIT missing information — add to need_more_info instead.
- NEVER fabricate clinical information.

F) Final Output Logic:
- If non_compliant is empty AND need_more_info is empty → final_report = draft_report.
- Otherwise → final_report = current best version of draft_report.

JSON Output Format (MUST match exactly; output ONLY this JSON):
{{
  "non_compliant": [
    {{
      "id": "R1",
      "text": "Original report text or key excerpt (recommend ≤2000 chars)",
      "reasons": ["Reason 1", "Reason 2"],
      "missing_fields": ["Missing field 1", "Missing field 2"]
    }}
  ],
  "need_more_info": [
    {{
      "id": "R2",
      "text": "Original report text or key excerpt (recommend ≤2000 chars)",
      "missing_fields": ["Information to supplement 1", "Information to supplement 2"],
      "questions_to_request": ["Question to request from upstream 1", "Question 2"]
    }}
  ],
  "pass": [
    {{
      "id": "R3",
      "text": ""
    }}
  ],
  "summary": {{
    "total": 0,
    "non_compliant": 0,
    "need_more_info": 0,
    "pass": 0
  }},
  "draft_report": "{draft_report_instruction}",
  "final_report": "{final_report_instruction}"
}}

Empty Input Fallback:
- If no reports can be parsed (reports is empty or all blank), still output:
  non_compliant=[], need_more_info=[], pass=[], summary={{total:0, non_compliant:0, need_more_info:0, pass:0}}, draft_report="", final_report=""
'''

# ============================================================
# Language-specific Examples and Style Rules
# ============================================================
EXAMPLES_EN = r'''Good examples (DETAILED, COMPREHENSIVE — follow this EXACT style):

Example 1 (Gastric): "The slide from the body of the stomach shows a poorly differentiated adenocarcinoma with an ulcerated, infiltrative growth pattern, measuring 9.5 x 4 x 1.2 cm. The tumor invades through the full thickness of the gastric wall and extends into the subserosal adipose tissue. Lymphovascular invasion is identified, and perineural invasion is present. The proximal and distal surgical margins are free of tumor. Metastatic carcinoma is found in 4 of 7 small curvature lymph nodes; no metastasis is identified in the greater curvature lymph nodes (0/5)."

Example 2 (Gastric): "The slide from the gastric antrum shows a poorly differentiated adenocarcinoma with a component of signet ring cell carcinoma, measuring 3.5 x 2 x 1.5 cm. The tumor demonstrates an ulcerated growth pattern and invades through the full thickness of the gastric wall into the pericolic adipose tissue, with formation of cancer nodules. Tumor emboli are identified within lymphovascular spaces. The proximal and distal resection margins are negative for carcinoma. Metastatic carcinoma is present in the perigastric lymph nodes (5/7 small curvature, 6/8 greater curvature, 4/4 pyloric, 3/3 group 12B); lymph nodes from groups 1 and 3 are negative (0/2, 0/1 respectively)."

Example 3 (Gastric — cardia): "The slide from the cardia shows a moderately differentiated intestinal type adenocarcinoma of the stomach, measuring 4.0 cm and infiltrating to the serosa and adjacent adipose tissue. Lymphatic infiltration is present, and there is a discrete lymphocytic peritumoral infiltrate. The proximal surgical margin is compromised, with metastatic carcinoma in 8 of 26 identified lymph nodes."

Example 4 (Gastric — antrum): "The slide from the gastric antrum shows a gastric adenocarcinoma, intestinal type, moderately differentiated, measuring up to 9.5 cm, with ulceration and invasion into the perigastric fat. Venous invasion is absent. The surgical margins are uninvolved by neoplasia. Four out of fifteen lymph nodes examined show metastasis."

Example 5 (Gastric — body): "The slide from the body of the stomach shows a moderately differentiated adenocarcinoma, intestinal type (Lauren), measuring 3.3 cm, infiltrating the wall to the perigastric adipose tissue. Angiolymphatic invasion and perineural invasion are present, with moderate peritumoral desmoplasia and discrete peritumoral lymphocytic infiltrate. The margins are uninvolved by neoplasia."

Example 6 (Gastric — minimal findings): "The slide from the stomach shows an invasive poorly differentiated adenocarcinoma extensively involving the entire stomach wall with extension into surrounding fibroconnective tissue. Lymphatic, venous, and perineural invasion are present."

Example 7 (Gastric — signet ring): "The slide from the body of the stomach shows a poorly differentiated adenocarcinoma with signet ring cell carcinoma component, characterized by an infiltrative growth pattern. The tumor cells exhibit large mucin vacuoles displacing the nucleus. The tumor invades through the muscularis propria into the subserosa, with lymphovascular invasion identified. Desmoplastic stroma and chronic peritumoral inflammatory infiltrate are present."

Example 8 (Gastric — rich detail): "The slide from the gastric antrum shows a poorly differentiated adenocarcinoma with diffuse-type (Lauren) morphology, demonstrating an ulcerated and infiltrative growth pattern. The tumor invades through the full thickness of the gastric wall and extends into the subserosal adipose tissue. Signet ring cells with intracytoplasmic mucin are scattered throughout the tumor. Lymphovascular invasion is identified, and perineural invasion is present. A prominent desmoplastic stromal reaction surrounds the invasive tumor front, with scattered foci of necrosis and a moderate peritumoral lymphocytic infiltrate."'''

EXAMPLES_ZH = r'''优秀示例（详细、全面 — 请参照此风格）：

示例1（胃癌）："胃小弯侧溃疡型低分化腺癌，伴部分神经内分泌分化，肿瘤大小为3x2.5x2cm，癌组织侵犯胃壁全层并累及浆膜面脂肪组织，自取两侧切缘及送检（切环1、切环2）切缘均未见癌，小弯侧淋巴结见转移性癌（4/7），送检（第6组，8组，7、9组）淋巴结均未见转移性癌，分别为（0/5，0/5，0/9），送检（第1组，11组淋巴结）为脂肪组织，未见淋巴结。"

示例2（胃癌）："胃贲门大弯侧溃疡型中分化腺癌，肿瘤大小为6x5x1.5cm，癌组织侵犯胃壁全层，累及齿状线及食管下段，送检（食道及胃）切缘未见癌，自取胃大弯、小弯侧淋巴结及送检（3、5组，7组，8组）淋巴结未见转移癌（分别为0/2、0/11、0/2、0/5、0/1）。"

示例3（胃癌）："胃窦溃疡型低分化腺癌，部分为印戒细胞癌，肿瘤大小3.5x2x1.5cm，侵及胃壁全层达周围脂肪组织内，并形成癌结节，脉管内见癌栓。上、下切缘未见癌。（大弯、小弯、幽门旁、12B）淋巴结见转移癌（分别为5/7、6/8、4/4、3/3）；（第1、3组淋巴结、12a淋巴结）未见转移癌（分别为0/2、0/1）。"

示例4（胃癌）："胃体中分化管状腺癌，溃疡型，肿瘤大小4x3x1cm，侵犯至肌层，未见脉管侵犯及神经侵犯，上下切缘未见癌，送检淋巴结均未见转移癌（0/15）。"'''

STYLE_RULES_EN = r'''- Start with "The slide from the [specific anatomical sub-site] shows a ..."
- For gastric tumors, prefer specific sub-sites: "body of the stomach", "gastric antrum", "cardia", "fundus", "pylorus".
- Use CLINICAL PATHOLOGY language (NOT pure cytological descriptions):
  * Use "poorly differentiated adenocarcinoma" not "tumor cells with marked nuclear atypia and pleomorphism"
  * Use "infiltrative growth pattern" not "irregular tumor nests infiltrating stroma"
  * Use "invades through the full thickness of the gastric wall" not "deep invasion into tissue layers"
  * Use "Lymphovascular invasion is identified" not "tumor emboli in vessels"
  * Use "desmoplastic stroma" not "reactive fibroblastic proliferation"
  * Use "signet ring cell carcinoma component" not "cells with mucin vacuoles and displaced nuclei"
- PREFERRED VOCABULARY (use these exact phrases when applicable):
  * invasion depth: "invades into the mucosa/submucosa/muscularis propria" or "invades through the full thickness of the gastric wall" or "extends into the subserosal/perigastric adipose tissue"
  * LVI: "Lymphovascular invasion is identified/present" or "No lymphovascular invasion is identified"
  * PNI: "Perineural invasion is present" or "No perineural invasion is identified"
  * margins: "margins are free of tumor" or "margins are uninvolved by neoplasia"
  * LN: "metastatic carcinoma in X of Y lymph nodes" or "lymph nodes are negative for metastatic carcinoma"
  * growth: "ulcerated growth pattern" / "infiltrative growth pattern" / "exophytic growth pattern"
  * stroma: "desmoplastic stroma" / "peritumoral lymphocytic infiltrate"
- Do NOT list raw cytological features as standalone findings (nuclear size, chromatin pattern, nucleolar prominence). Instead, translate them into the appropriate diagnosis or grade.'''

STYLE_RULES_ZH = r'''- 以"[解剖部位][大体类型][分化程度][组织学类型]"开头，如"胃窦溃疡型低分化腺癌"。
- 使用标准中文病理术语。
- 按以下顺序描述：部位→大体类型→分化程度→组织学亚型→肿瘤大小→浸润深度→脉管侵犯→神经侵犯→切缘→淋巴结转移。'''


class QCPromptBuilder:
    """Build QC Agent prompts with configurable language and detail level."""

    def __init__(self, qc_rules: str = "", language: str = "en"):
        """
        Args:
            qc_rules: Additional QC rules text.
            language: Output language — "en" for English, "zh" for Chinese.
        """
        self.qc_rules = qc_rules
        self.language = language

    @staticmethod
    def get_system_prompt() -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        reports: List[Dict[str, str]],
        current_draft: Optional[str] = None,
        supplement_evidence: Optional[str] = None,
        cancer_type: str = "gastric adenocarcinoma/STAD",
        dataset_name: str = "TCGA-STAD",
        anatomical_site: str = "stomach (gastric)",
    ) -> str:
        """
        Build the user prompt.

        Args:
            reports: [{"id": "R1", "text": "..."}, ...]
            current_draft: Previous round's draft_report (for iteration)
            supplement_evidence: Supplement evidence text
            cancer_type: Known cancer type from dataset context
            dataset_name: Dataset identifier (e.g., "TCGA-STAD", "301-Hospital")
            anatomical_site: Anatomical site description

        Returns:
            Formatted user prompt string
        """
        all_reports = list(reports)  # copy

        if current_draft:
            all_reports.append({
                "id": "CURRENT_DRAFT",
                "text": f"[Previous round integrated report]\n{current_draft}",
            })

        if supplement_evidence:
            all_reports.append({
                "id": "SUPPLEMENT_EVIDENCE",
                "text": f"[WSI patch supplement retrieval evidence]\n{supplement_evidence}",
            })

        reports_str = json.dumps(all_reports, ensure_ascii=False, indent=2)

        # Build dataset context
        dataset_context = (
            f"This WSI is from the {dataset_name} dataset. "
            f"The confirmed cancer type is: {cancer_type}. "
            f"The anatomical site is: {anatomical_site}. "
        )

        # Language-specific settings
        if self.language == "zh":
            report_language_instruction = (
                "中文 (Chinese). draft_report 和 final_report 必须使用中文撰写。"
            )
            report_examples = EXAMPLES_ZH
            style_rules = STYLE_RULES_ZH
            draft_report_instruction = (
                "详细的中文病理诊断报告，包含所有可用的病理发现（50-200字）"
            )
            final_report_instruction = (
                "若无遗留问题则与draft_report相同，否则为当前最佳版本（中文）"
            )
        else:
            report_language_instruction = (
                "English. draft_report and final_report MUST be written in English."
            )
            report_examples = EXAMPLES_EN
            style_rules = STYLE_RULES_EN
            draft_report_instruction = (
                "Detailed narrative starting with 'The slide from the [site] shows a ...' "
                "covering ALL available findings (50-200 words)"
            )
            final_report_instruction = (
                "If no issues remain, identical to draft_report; "
                "otherwise still the current best narrative version"
            )

        return USER_PROMPT_TEMPLATE.format(
            dataset_context=dataset_context,
            reports=reports_str,
            qc_rules=self.qc_rules or "(No additional QC rules; check against default QC checkpoints only)",
            report_language_instruction=report_language_instruction,
            report_examples=report_examples,
            style_rules=style_rules,
            draft_report_instruction=draft_report_instruction,
            final_report_instruction=final_report_instruction,
        )


# ============================================================
# CONCH Retrieval Query Generation Prompt
# ============================================================
QUERY_GENERATION_SYSTEM = """You are a pathology retrieval expert. Given missing pathology information fields, generate English query texts for searching a WSI patch embedding database.

Rules:
1) Generate 1 query per missing field
2) Queries MUST be in English
3) Queries should describe the morphological appearance of the field in H&E-stained pathology slides
4) Format: Return a JSON array ["query1", "query2", ...]
5) Only output the JSON array, nothing else

Important: Queries should focus on histological morphological features, NOT clinical/administrative information.
For fields that cannot be observed from H&E slides (e.g., patient ID, sex), return an empty array."""


def build_query_generation_prompt(
    missing_fields: List[str],
    cancer_type: str = "gastric adenocarcinoma",
) -> str:
    """Generate prompt for CONCH retrieval query generation."""
    fields_str = "\n".join(f"- {f}" for f in missing_fields)
    return f"""Cancer type: {cancer_type}

Missing information fields that need supplementation:
{fields_str}

Please generate English retrieval queries for each field that can be observed from H&E slides."""
