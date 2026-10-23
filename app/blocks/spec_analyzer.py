"""Spec Analyzer Block - Extract grade requirements, material specs, compliance flags from PDFs"""

import os
import re
from typing import Any, Dict, List, Tuple
from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)


class SpecAnalyzerBlock(UniversalBlock):
    auto_validate = False
    name = "spec_analyzer"
    version = "1.0.0"
    description = "Extract grade requirements, material specs, and compliance flags from specification PDFs"
    layer = 3
    tags = ["domain", "construction", "specs", "pdf", "compliance", "materials"]
    # Require the ocr block so scanned/image-only spec PDFs (no text layer)
    # still extract real content instead of silently returning empty lists.
    # When the text layer comes back below the threshold, _extract_text falls
    # back to OCR via the wired dep — mirroring document_engine.
    requires = ["ocr"]

    default_config = {
        "max_pages": 100,
        "compliance_standards": ["astm", "aci", "aisc", "iso", "bs", "en", "saso"],
        # If PyMuPDF's text layer yields fewer than this many chars across
        # the whole sampled range, treat the PDF as image-only and OCR it.
        "ocr_fallback_min_chars": 200,
    }

    ui_schema = {
        "input": {
            "type": "file",
            "accept": [".pdf"],
            "placeholder": "Upload project specification PDF...",
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "grade_requirements", "type": "list", "label": "Grade Requirements"},
                {"name": "material_specs", "type": "list", "label": "Material Specs"},
                {"name": "compliance_flags", "type": "list", "label": "Compliance Flags"},
                {"name": "sections_found", "type": "number", "label": "Sections"},
            ],
        },
        "quick_actions": [
            {"icon": "", "label": "Extract Specs", "prompt": "Extract all material specifications and grade requirements"},
            {"icon": "", "label": "Check Compliance", "prompt": "Check specification for ASTM/ACI/SASO compliance requirements"},
            {"icon": "", "label": "Find Standards", "prompt": "List all referenced standards and codes"},
        ],
    }

    _GRADE_PATTERNS: List[Tuple[str, str]] = [
        (r"\b(?:grade|class|type)\s*[:\-]?\s*([A-Z0-9\-]+)", "grade"),
        (r"\bfc['\"]?\s*=\s*([\d,]+)\s*(?:psi|MPa|kPa)", "concrete_strength"),
        (r"\bfy\s*=\s*([\d,]+)\s*(?:psi|MPa)", "rebar_yield"),
        # ASTM: allow trailing year suffix like A615-22 and slash-grade variants.
        (r"\bASTM\s+([A-Z]\d+(?:[/-][A-Z]?\d*)*)", "astm_standard"),
        (r"\bACI\s+(\d+\w*)", "aci_standard"),
        (r"\bAISC\s+(\d+\w*)", "aisc_standard"),
        (r"\bISO\s+(\d+(?:[-:]\d+)?)", "iso_standard"),
        # BS EN must come before BS alone so "BS EN 1992" matches the EN variant.
        (r"\bBS\s+EN\s+(\d+(?:[-:]\d+)?)", "bs_en_standard"),
        (r"\bBS\s+(\d+(?:[-:]\d+)?)", "bs_standard"),
        (r"\bEN\s+(\d+(?:[-:]\d+)?)", "en_standard"),
        (r"\bSASO\s+(\d+(?:[-:]\d+)?)", "saso_standard"),
        # AS (Australian Standard), e.g. "AS 3600", "AS 1170.4", "AS 1170-2002".
        (r"\bAS\s+(\d+(?:\.\d+)?(?:-\d+)?)", "as_standard"),
        # IBC year code, e.g. "IBC 2021".
        (r"\bIBC\s+(\d{4})", "ibc_standard"),
        (r"\b(\d+)\s*MPa\s*(?:concrete|compressive)", "concrete_strength_mpa"),
        (r"\b(\d+)\s*N/mm[²2]", "strength_nmm2"),
    ]

    _MATERIAL_PATTERNS: List[Tuple[str, str]] = [
        (r"(?:reinforcing|reinforcement)\s+(?:steel|bars?)\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "rebar"),
        (r"concrete\s+(?:shall\s+be\s+)?(?:of\s+)?(?:class|grade|strength)?\s*([^\.\n]{5,80})", "concrete"),
        (r"structural\s+steel\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "structural_steel"),
        (r"(?:cement|portland\s+cement)\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "cement"),
        (r"aggregate\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "aggregate"),
        (r"waterproof(?:ing)?\s+(?:membrane|material|system)?\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "waterproofing"),
        (r"insulation\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "insulation"),
        (r"paint\s+(?:system|coat|finish)?\s+(?:shall\s+be\s+)?([^\.\n]{5,80})", "paint"),
    ]

    _COMPLIANCE_KEYWORDS: List[Tuple[str, str]] = [
        ("shall conform to", "conformance"),
        ("shall comply with", "compliance"),
        ("shall be in accordance with", "accordance"),
        ("minimum requirement", "minimum_requirement"),
        ("not less than", "minimum_value"),
        ("not exceed", "maximum_value"),
        ("must be approved", "approval_required"),
        ("submittal required", "submittal"),
        ("shop drawing", "shop_drawing"),
        ("test certificate", "test_certificate"),
        ("material approval", "material_approval"),
        ("mock-up", "mockup_required"),
    ]

    _SECTION_PATTERNS = [
        r"SECTION\s+\d+\s*[-–]\s*([A-Z][A-Z\s,/]{3,60})",
        r"^\s*(\d{2}\s+\d{2}\s+\d{2})\s+([A-Z][A-Z\s,/]{3,50})",
        r"(?:PART|ARTICLE)\s+\d+[:\s]+([A-Z][A-Z\s,/]{3,60})",
    ]

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        file_path = data.get("file_path") or params.get("file_path")

        # Accept raw text directly (InputAdapter may wrap it as {"text": "..."})
        raw_text = data.get("text") or data.get("input") or (str(input_data) if isinstance(input_data, str) else "")

        if file_path:
            if not os.path.exists(file_path):
                return {"status": "error", "error": f"File not found: {file_path}"}
            try:
                text, page_count = self._extract_text(file_path, params)
            except ImportError:
                return {"status": "error", "error": "pymupdf not installed. Run: pip install pymupdf"}
            except Exception as e:
                return {"status": "error", "error": f"PDF extraction failed: {e}"}

            # OCR fallback for image-only / scanned specs (no text layer).
            # Without this, _extract_grades / _extract_materials all return
            # empty lists with status: success — the LLM gets "no specs
            # found" for a perfectly valid scanned PDF.
            min_chars = int(self.config.get("ocr_fallback_min_chars", 200))
            if len(text.strip()) < min_chars:
                ocr_block = self.get_dep("ocr")
                if ocr_block is not None:
                    try:
                        ocr_result = await ocr_block.process({"file_path": file_path})
                        ocr_text = (ocr_result.get("text") or "").strip()
                        if len(ocr_text) > len(text.strip()):
                            text = ocr_text
                            # Page count from OCR if available (PDF → N images).
                            page_count = ocr_result.get("pages") or page_count
                    except Exception:
                        # OCR is best-effort — if it fails, fall through with
                        # whatever the text layer gave us (may still be empty
                        # but at least we don't crash the spec analysis).
                        logger.debug(
                            "swallowed %s in process() — continuing",
                            "Exception", exc_info=True,
                        )
        elif raw_text:
            text = raw_text
            page_count = 0
        else:
            return {"status": "error", "error": "Provide file_path (PDF) or raw spec text as input"}

        grade_requirements = self._extract_grades(text)
        material_specs = self._extract_materials(text)
        compliance_flags = self._extract_compliance(text)
        sections = self._extract_sections(text)

        # Resolve each extracted grade against the central grade table.
        # Adds {"resolved": {...kind/system/fck_mpa or fy_mpa.../strength info...}}
        # to entries where the label matches a known concrete/rebar/steel grade.
        from app.core.construction_constants import lookup_grade, lookup_standard
        for g in grade_requirements:
            if g.get("type") in ("grade", "concrete_grade", "rebar_grade",
                                  "steel_grade"):
                info = lookup_grade(g.get("value", ""))
                if info:
                    g["resolved"] = info

        # Emit standards as {type, value, resolved} dicts so callers see the
        # actual matched standard code (e.g. "A615") AND its domain meaning
        # ("Deformed carbon-steel rebar"). Deduplicate on (type, value).
        _seen_std = set()
        referenced_standards: List[Dict] = []
        for g in grade_requirements:
            if "standard" not in g["type"]:
                continue
            key = (g["type"], g["value"])
            if key in _seen_std:
                continue
            _seen_std.add(key)
            # Resolve against STANDARDS_PURPOSE. The matched value is the
            # code (e.g. "A615"); the type label tells us the system
            # (astm_standard, aci_standard, etc). Try the most likely
            # prefixed form first; lookup_standard handles fallback.
            type_to_prefix = {
                "astm_standard": "ASTM",
                "aci_standard": "ACI",
                "bs_standard": "BS",
                "bs_en_standard": "BS EN",
                "as_standard": "AS",
                "ibc_standard": "IBC",
            }
            prefix = type_to_prefix.get(g["type"], "")
            full_ref = f"{prefix} {g['value']}".strip() if prefix else g["value"]
            resolved = lookup_standard(full_ref)
            entry = {"type": g["type"], "value": g["value"]}
            if resolved:
                entry["resolved"] = resolved
            referenced_standards.append(entry)

        return {
            "status": "success",
            "grade_requirements": grade_requirements,
            "material_specs": material_specs,
            "compliance_flags": compliance_flags,
            "sections_found": len(sections),
            "sections": sections[:20],
            "page_count": page_count,
            "standards_referenced": referenced_standards,
        }

    def _extract_text(self, file_path: str, params: Dict) -> Tuple[str, int]:
        import fitz
        from app.core.file_crypto import open_plaintext
        max_pages = int(params.get("max_pages", self.config.get("max_pages", 100)))
        # open_plaintext transparently decrypts when DATA_ENCRYPTION_KEY is set
        # (uploads go through file_crypto.write_document); plaintext files
        # short-circuit to their original path.
        with open_plaintext(file_path) as plain_path:
            doc = fitz.open(plain_path)
            pages = min(len(doc), max_pages)
            text = "\n".join(doc[i].get_text() for i in range(pages))
            doc.close()
        return text, pages

    # Post-filter stopwords for the loose grade/class/type pattern. The pattern is
    # compiled IGNORECASE so the capture group can match lowercase tokens like
    # "of" in "type of concrete" — drop these. We also drop common ALL-CAPS
    # English words ("CONCRETE", "INSTALL", ...) that the loose pattern picks
    # up from headings and instructional text.
    _GRADE_STOPWORDS = {
        # short prepositions/articles
        "of", "as", "be", "is", "to", "in", "on", "or", "at", "by",
        # ALL-CAPS common words that leak through the loose grade regex
        "CONCRETE", "STEEL", "INSTALL", "APPROVAL", "REQUIREMENT",
        "PROVIDE", "FURNISH", "PERFORM", "VERIFY",
    }

    def _extract_grades(self, text: str) -> List[Dict]:
        found: Dict[str, Dict] = {}
        for pattern, ptype in self._GRADE_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(1).strip()
                # Drop spurious matches for the loose grade/class/type pattern.
                if ptype == "grade":
                    if len(val) < 2:
                        continue
                    if val.islower():
                        continue
                    # Compare against stopword set in both raw and uppercased
                    # forms (set already contains both short lowercase tokens
                    # and ALL-CAPS English common words).
                    if val in self._GRADE_STOPWORDS:
                        continue
                    if val.lower() in self._GRADE_STOPWORDS:
                        continue
                    if val.upper() in self._GRADE_STOPWORDS:
                        continue
                    # Real grade strings (C30, B500B, M25, Grade-60) always
                    # contain at least one digit or a hyphen. ALL-CAPS English
                    # words ("CONCRETE", "FURNISH") do not — drop them.
                    if not any(c.isdigit() for c in val) and "-" not in val:
                        continue
                key = f"{ptype}:{val}"
                if key not in found:
                    ctx_start = max(0, m.start() - 80)
                    ctx_end = min(len(text), m.end() + 80)
                    found[key] = {
                        "type": ptype,
                        "value": val,
                        "context": text[ctx_start:ctx_end].replace("\n", " ").strip(),
                    }
        return list(found.values())

    def _extract_materials(self, text: str) -> List[Dict]:
        found: Dict[str, Dict] = {}
        for pattern, mtype in self._MATERIAL_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(1).strip()
                if len(val) < 5:
                    continue
                key = f"{mtype}:{val[:40]}"
                if key not in found:
                    found[key] = {
                        "material_type": mtype,
                        "specification": val[:200],
                        "full_match": m.group(0)[:200],
                    }
        return list(found.values())

    def _extract_compliance(self, text: str) -> List[Dict]:
        flags: List[Dict] = []
        lower = text.lower()
        for keyword, flag_type in self._COMPLIANCE_KEYWORDS:
            idx = 0
            count = 0
            while count < 3:
                pos = lower.find(keyword, idx)
                if pos == -1:
                    break
                ctx_start = max(0, pos - 40)
                ctx_end = min(len(text), pos + len(keyword) + 120)
                flags.append(
                    {
                        "flag_type": flag_type,
                        "keyword": keyword,
                        "context": text[ctx_start:ctx_end].replace("\n", " ").strip(),
                    }
                )
                count += 1
                idx = pos + len(keyword)
        return flags

    def _extract_sections(self, text: str) -> List[str]:
        sections: List[str] = []
        seen: set = set()
        for pattern in self._SECTION_PATTERNS:
            for m in re.finditer(pattern, text, re.MULTILINE):
                if m.lastindex and m.lastindex >= 2:
                    title = f"{m.group(1)} {m.group(2)}".strip()
                else:
                    title = m.group(1).strip()
                if title not in seen and len(title) > 3:
                    seen.add(title)
                    sections.append(title)
        return sections
