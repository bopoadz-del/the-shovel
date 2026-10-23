"""PDF Block v2 - Extract text from PDF files using TypedBlock

This is the v2 implementation using the TypedBlock system with schema validation.
"""

import os
from typing import Any, Dict

from app.core.typed_block import TypedBlock
from app.core.schema_registry import TextContent
import logging

logger = logging.getLogger(__name__)


class PDFBlockV2(TypedBlock):
    """Extract text from PDF files - v2 with schema support"""
    
    name = "pdf_v2"
    version = "2.0"
    description = "Extract text from PDF files with typed output"
    layer = 3
    tags = ["domain", "documents", "pdf", "v2"]
    # OCR is wired so image-only / scanned PDFs (CAD drawings, photocopied
    # specs) still produce real text instead of silently returning "". Same
    # pattern as document_engine — fall back when the PyMuPDF text layer is
    # below `ocr_fallback_min_chars`.
    requires = ["ocr"]

    default_config = {
        "extract_tables": True,
        "max_pages": 100,
        "text_limit": 20000,
        "ocr_fallback_min_chars": 200,
    }
    
    # Input: file path or dict with file info
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "path": {"type": "string"},
            "url": {"type": "string"}
        },
        "anyOf": [
            {"required": ["file_path"]},
            {"required": ["path"]},
            {"required": ["url"]}
        ]
    }
    
    # Output: TextContent schema
    output_schema = TextContent
    
    # Type declarations for orchestrator
    accepted_input_types = ["PDFContent", "FileContent"]
    produced_output_types = ["TextContent"]
    
    ui_schema = {
        "input": {
            "type": "file",
            "accept": [".pdf"],
            "placeholder": "Upload PDF...",
            "multiline": False
        },
        "output": {
            "type": "json",
            "fields": [
                {"name": "text", "type": "text", "label": "Text"},
                {"name": "pages", "type": "number", "label": "Pages"},
                {"name": "source", "type": "string", "label": "Source"}
            ]
        }
    }
    
    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Extract text from PDF and return TextContent format"""
        params = params or {}
        
        # Download from URL if needed
        pdf_path = self._get_pdf_path(input_data)
        if not pdf_path:
            return self._error_response("No PDF provided")

        if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
            import httpx, tempfile
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    resp = await client.get(pdf_path, timeout=30)
                    resp.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(resp.content)
                        pdf_path = f.name
            except Exception as e:
                return self._error_response(f"Download failed: {str(e)}")

        if not os.path.exists(pdf_path):
            return self._error_response(f"File not found: {pdf_path}")
        
        # Extract using PyMuPDF
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            text = ""

            max_pages = params.get("max_pages", self.config.get("max_pages", 100))
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text += page.get_text()

            pages = len(doc)
            doc.close()

            # OCR fallback for image-only / scanned PDFs (zero text layer).
            # When the text layer is below threshold, render → OCR → use that
            # text. Mirrors document_engine. Without this, drawings come out
            # as empty `text` and any downstream block (LLM, spec_analyzer,
            # boq_processor) gets nothing.
            source = "pdf"
            min_chars = int(self.config.get("ocr_fallback_min_chars", 200))
            if len(text.strip()) < min_chars:
                ocr_block = self.get_dep("ocr")
                if ocr_block is not None:
                    try:
                        ocr_result = await ocr_block.process({"file_path": pdf_path})
                        ocr_text = (ocr_result.get("text") or "").strip()
                        if len(ocr_text) > len(text.strip()):
                            text = ocr_text
                            source = "pdf+ocr"
                    except Exception:
                        # OCR is best-effort — if it fails, return whatever
                        # PyMuPDF gave us (may be empty).
                        logger.debug(
                            "swallowed %s in process() — continuing",
                            "Exception", exc_info=True,
                        )

            # Return TextContent format
            return {
                "text": text[:self.config.get("text_limit", 20000)],
                "source": source,
                "pages": pages,
                "metadata": {
                    "filename": os.path.basename(pdf_path),
                    "total_pages": pages,
                    "extracted_at": self._timestamp()
                }
            }
            
        except ImportError:
            return self._error_response("PyMuPDF not installed")
        except Exception as e:
            return self._error_response(f"PDF extraction failed: {str(e)}")
    
    def _get_pdf_path(self, input_data: Any) -> str:
        """Extract PDF path from input"""
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict):
            return input_data.get("file_path") or input_data.get("path") or input_data.get("url")
        return None
    
    def _error_response(self, message: str) -> Dict:
        """Return standardized error response"""
        return {
            "text": "",
            "source": "pdf",
            "pages": 0,
            "metadata": {
                "error": message,
                "extracted_at": self._timestamp()
            }
        }
    
    def _timestamp(self) -> str:
        """Get current ISO timestamp"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
