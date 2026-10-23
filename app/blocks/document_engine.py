"""Document Engine Block — Platform wrapper for Parse → Reason → Map pipeline.

Exposes the document_engine block to the Cerebrum platform via:
  POST /execute { "block": "document_engine", "input": { "pdf_path": "..." } }

Integrates with platform blocks:
  - pdf  : PDF text extraction (PyMuPDF)
  - ocr  : Image / scanned PDF OCR fallback
"""

import os
from typing import Any, Dict, Optional
from app.core.universal_base import UniversalBlock


class DocumentEngineBlock(UniversalBlock):
    """Technical document reasoning engine.

    Ingests PDF / DOCX / XLSX, runs 8 semantic reasoning pipelines,
    and outputs structured YAML/JSON consumable by schedule_engine,
    cost_engine, and risk_engine downstream blocks.
    """

    auto_validate = False
    name = "document_engine"
    version = "1.0.0"
    description = "Parse → Reason → Map pipeline for technical document intelligence"
    layer = 3
    tags = ["domain", "construction", "documents", "reasoning", "scheduling"]
    requires = ["pdf", "ocr"]

    default_config = {
        "extract_tables": True,
        "extract_glossary": True,
        "output_format": "yaml",
        "use_platform_pdf": True,
        "use_platform_ocr": True,
    }

    ui_schema = {
        "input": {
            "type": "files",
            "accept": [".pdf", ".docx", ".xlsx"],
            "placeholder": "Upload BOD, RFP, or spec documents...",
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "glossary", "type": "list", "label": "Glossary Terms"},
                {"name": "requirements", "type": "list", "label": "Requirements"},
                {"name": "constraints", "type": "list", "label": "Constraints"},
                {"name": "schedule_targets", "type": "list", "label": "Schedule Targets"},
                {"name": "equipment_specs", "type": "list", "label": "Equipment Specs"},
                {"name": "risks", "type": "list", "label": "Risks"},
                {"name": "downstream", "type": "object", "label": "Downstream Feed"},
            ],
        },
        "quick_actions": [
            {"icon": "📄", "label": "Analyze BOD", "prompt": "Extract glossary, constraints, and equipment lead times from Basis of Design"},
            {"icon": "📋", "label": "Analyze RFP", "prompt": "Extract requirements, schedule targets, and risks from RFP"},
            {"icon": "📊", "label": "Schedule Feed", "prompt": "Generate procurement activities and milestones for schedule_engine"},
            {"icon": "⚠️", "label": "Risk Register", "prompt": "Extract all risks and output risk_engine feed"},
        ],
    }

    async def _parse_with_platform_pdf(self, file_path: str) -> Optional[str]:
        """Use platform PDF block for text extraction (universal connector)."""
        pdf_block = self.get_dep("pdf")
        if pdf_block is None:
            return None
        result = await pdf_block.process({"file_path": file_path})
        if result.get("status") == "success":
            return result.get("text", "")
        return None

    async def _parse_with_platform_ocr(self, file_path: str) -> Optional[str]:
        """Use platform OCR block for image/scanned PDF fallback (universal connector).

        Side effect: captures the OCR markup/redline verdict (Roadmap V2 · Epic 5)
        on ``self._last_ocr_markup`` so `process()` can surface it — annotated
        regions are flagged for the user, not presented as clean extracted data.
        """
        ocr_block = self.get_dep("ocr")
        if ocr_block is None:
            return None
        result = await ocr_block.process({"file_path": file_path})
        if isinstance(result, dict) and result.get("markup"):
            self._last_ocr_markup = result.get("markup")
        if result.get("status") == "success":
            return result.get("text", "")
        return None

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Main entry point — run the 3-layer pipeline."""
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        # Reset the per-call OCR markup verdict (Roadmap V2 · Epic 5). Populated
        # by `_parse_with_platform_ocr` when a scanned input carries redlines.
        self._last_ocr_markup = None

        # InputAdapter may wrap bare file path as {"text": "/path/to/file.pdf"} — detect by extension
        raw_path = data.get("text") or data.get("input") or (input_data if isinstance(input_data, str) else "")
        raw_ext = os.path.splitext(raw_path)[1].lower() if raw_path else ""

        file_paths = {
            "pdf": data.get("pdf_path") or data.get("pdf") or params.get("pdf_path") or (raw_path if raw_ext == ".pdf" else None),
            "docx": data.get("docx_path") or data.get("docx") or params.get("docx_path") or (raw_path if raw_ext in (".docx", ".doc") else None),
            "xlsx": data.get("xlsx_path") or data.get("xlsx") or params.get("xlsx_path") or (raw_path if raw_ext in (".xlsx", ".xls") else None),
        }

        # A raw string that is NOT a recognised document path is an inline brief:
        # reason over it directly via a synthetic text document instead of
        # erroring. This is the no-file path used by scope_extractor and the
        # generate-from-brief flows. An explicit ``brief`` key also feeds here.
        brief_text = data.get("brief") or params.get("brief") or ""
        if not brief_text and raw_path and not any(file_paths.values()):
            known_ext = raw_ext in (".pdf", ".docx", ".doc", ".xlsx", ".xls")
            if not known_ext and not os.path.exists(raw_path):
                brief_text = raw_path

        if not any(file_paths.values()) and not (brief_text and brief_text.strip()):
            return {"status": "error", "error": "No input provided — pass a document (pdf/docx/xlsx) via pdf_path/docx_path/xlsx_path, or inline text via 'text'/'brief'."}

        # Decrypt-to-temp if the stored files are encrypted at rest. Both the
        # platform pdf/ocr blocks and the fallback parsers (fitz / python-docx /
        # openpyxl) read by raw path. open_plaintext is a no-op for plaintext /
        # legacy files (see app/core/file_crypto.py).
        from contextlib import ExitStack
        from app.core.file_crypto import open_plaintext
        with ExitStack() as _crypto_stack:
            file_paths = {
                key: (_crypto_stack.enter_context(open_plaintext(p)) if p else None)
                for key, p in file_paths.items()
            }
            return await self._run_pipeline(file_paths, brief_text=brief_text)

    async def _run_pipeline(self, file_paths: Dict, brief_text: str = "") -> Dict:
        """Run the 3-layer pipeline on already-decrypted plaintext paths.

        When ``brief_text`` is provided (and no files), the brief is wrapped in a
        synthetic text document so the reasoner/mapper run over it exactly like a
        parsed PDF — this is the inline-brief scope-extraction path.
        """
        try:
            from app.document_engine.main import parse_all
            from app.document_engine.reasoner import DocumentReasoner
            from app.document_engine.mapper import DocumentMapper
            from app.document_engine.parsers.pdf_parser import PDFParser, PDFDocument
            from app.document_engine.parsers.docx_parser import DOCXParser
            from app.document_engine.parsers.xlsx_parser import XLSXParser
            import yaml

            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "document_engine",
                "config.yaml",
            )

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    full_config = yaml.safe_load(f)
                config = full_config.get("document_engine", full_config)
            else:
                config = {}

            # ------------------------------------------------------------------
            # Layer 1: Parse — use platform blocks when available, fallback to own parsers
            # ------------------------------------------------------------------
            documents = []

            # PDF → platform pdf block → OCR fallback when text layer is empty
            # → final fallback to own PDFParser. The OCR fallback is what
            # makes CAD drawings (vector / rasterised, zero text layer)
            # actually readable — without it `pdf_text` comes back as "" and
            # the LLM ends up guessing from the filename.
            #
            # Threshold: 200 chars is the "is this a real text PDF" tripwire.
            # A 3-page drawing with empty pages produces a few whitespace
            # chars at most; a real text PDF reliably clears this on page 1.
            if file_paths.get("pdf"):
                pdf_text = None
                if self.config.get("use_platform_pdf", True):
                    pdf_text = await self._parse_with_platform_pdf(file_paths["pdf"])

                # OCR fallback for empty-text-layer PDFs (drawings, scans).
                # _parse_with_platform_ocr renders each page → image → reads
                # text via the OCR block (Tesseract or Vision API). Only
                # invoke when the existing text really is empty so we don't
                # pay OCR latency on every text PDF.
                if (
                    self.config.get("use_platform_ocr", True)
                    and self.get_dep("ocr") is not None
                    and (pdf_text is None or len(pdf_text.strip()) < 200)
                ):
                    ocr_text = await self._parse_with_platform_ocr(file_paths["pdf"])
                    if ocr_text and len(ocr_text.strip()) > len((pdf_text or "").strip()):
                        pdf_text = ocr_text

                if pdf_text is not None:
                    doc = PDFDocument(source=file_paths["pdf"], text=pdf_text)
                    documents.append(doc)
                else:
                    parser = PDFParser(config)
                    documents.append(parser.parse(file_paths["pdf"]))

            # DOCX → own parser (no platform block available)
            if file_paths.get("docx"):
                parser = DOCXParser(config)
                documents.append(parser.parse(file_paths["docx"]))

            # XLSX → own parser (no platform block available)
            if file_paths.get("xlsx"):
                parser = XLSXParser(config)
                documents.append(parser.parse(file_paths["xlsx"]))

            # Inline brief → synthetic text document (no file). PDFDocument is a
            # plain (source, text) carrier; the reasoner reads `.text` the same
            # way it reads a parsed PDF, so scope/requirements/constraints/etc.
            # are extracted from the brief itself — not fabricated.
            if brief_text and brief_text.strip():
                documents.append(PDFDocument(source="inline-brief", text=brief_text))

            # ------------------------------------------------------------------
            # Layer 2: Reason
            # ------------------------------------------------------------------
            reasoner = DocumentReasoner(config)
            reasoned = reasoner.reason(documents)

            # ------------------------------------------------------------------
            # Layer 3: Map
            # ------------------------------------------------------------------
            mapper = DocumentMapper(config)
            structured = mapper.map_to_structured(reasoned)

            result = structured.to_dict()
            result["status"] = "success"
            result["documents_parsed"] = len(documents)

            # Surface the RAW extracted text alongside the mapper's structured
            # output. The reasoner+mapper layers throw away the source content
            # in favour of construction-ontology defaults (equipment lead
            # times, WBS placeholders) — which is fine for QTO/schedule
            # documents but a disaster for cost spreadsheets. Without this,
            # a BOQ-style xlsx upload reaches the LLM as 4 kB of generic
            # equipment defaults with zero rows from the actual file, and the
            # LLM correctly says "No Cost Data Available" because that is
            # literally what it was given.
            raw_chunks: list = []
            for doc in documents:
                src = getattr(doc, "source", "") or ""
                # PDF / DOCX: `.text` attribute
                txt = getattr(doc, "text", None)
                if txt:
                    raw_chunks.append(f"--- {os.path.basename(src)} ---\n{txt}")
                    continue
                # XLSX: `.sheets` attribute → dict of sheet_name → list[list[str]]
                sheets = getattr(doc, "sheets", None)
                if isinstance(sheets, dict):
                    for sheet_name, rows in sheets.items():
                        if not rows:
                            continue
                        # Render as TSV-ish text so the LLM sees columns aligned.
                        lines = ["\t".join(str(c) for c in row) for row in rows]
                        raw_chunks.append(
                            f"--- {os.path.basename(src)} :: {sheet_name} ---\n"
                            + "\n".join(lines)
                        )
            if raw_chunks:
                # Cap to ~50 kB so an enormous workbook can't blow up the
                # response. The frontend further truncates to 8 kB before
                # putting it in the chat prompt.
                joined = "\n\n".join(raw_chunks)
                result["raw_text"] = joined[:50000]
                result["raw_text_truncated"] = len(joined) > 50000

            # Surface the OCR markup / redline verdict (Roadmap V2 · Epic 5).
            # If a scanned input carried coloured annotations, flag them rather
            # than presenting the extracted text as clean data.
            if self._last_ocr_markup is not None:
                result["markup"] = self._last_ocr_markup
                result["has_markup"] = bool(self._last_ocr_markup.get("has_markup"))

            result["platform_blocks_used"] = []
            if self.config.get("use_platform_pdf") and self.get_dep("pdf"):
                result["platform_blocks_used"].append("pdf")
            if self.config.get("use_platform_ocr") and self.get_dep("ocr"):
                result["platform_blocks_used"].append("ocr")
            return result

        except Exception as e:
            return {"status": "error", "error": f"Document engine pipeline failed: {str(e)}"}
