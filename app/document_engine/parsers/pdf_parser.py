"""PDF Parser — Layer 1 syntactic extraction for document engine."""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PDFDocument:
    source: str
    text: str = ""
    pages: List[str] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    figures: List[Dict[str, Any]] = field(default_factory=list)


class PDFParser:
    """Extract text, tables, glossary entries and figure captions from PDF."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def parse(self, file_path: str) -> PDFDocument:
        path = Path(file_path)
        doc = PDFDocument(source=str(path))

        # Try pymupdf first (already in root requirements), then pdfplumber, then PyPDF2
        parsed = False
        if not parsed:
            try:
                doc = self._parse_with_pymupdf(file_path, doc)
                parsed = True
            except Exception:
                logger.warning(
                    "pymupdf could not parse %s — falling back to pdfplumber",
                    file_path, exc_info=True,
                )

        if not parsed:
            try:
                doc = self._parse_with_pdfplumber(file_path, doc)
                parsed = True
            except Exception:
                logger.warning(
                    "pdfplumber could not parse %s — falling back to PyPDF2",
                    file_path, exc_info=True,
                )

        if not parsed:
            try:
                doc = self._parse_with_pypdf2(file_path, doc)
                parsed = True
            except Exception:
                logger.warning(
                    "PyPDF2 could not parse %s either — falling back to raw text, "
                    "so this document will have no page structure and no tables",
                    file_path, exc_info=True,
                )

        if not parsed:
            # Ultimate fallback — treat as text
            doc.text = path.read_text(errors="ignore")
            doc.pages = [doc.text]

        doc.glossary = self._extract_glossary(doc.text)
        doc.figures = self._extract_figures(doc.text)
        return doc

    def _parse_with_pymupdf(self, file_path: str, doc: PDFDocument) -> PDFDocument:
        import fitz  # pymupdf
        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                doc.pages.append(text)
                doc.text += text + "\n"
                doc.tables.extend(self._tables_from_page(page, page_num))
        return doc

    def _tables_from_page(self, page, page_num: int) -> List[List[List[str]]]:
        """Extract tables from one page in the shape `PDFDocument.tables`
        declares: a list of tables, each a list of rows, each a list of cell
        strings — identical to what `pdfplumber.extract_tables()` returns, so
        both parser branches produce the same thing.

        This replaces `_blocks_to_tables`, which returned `[]` unconditionally
        with a comment saying real extraction was "handled by pdfplumber".
        It was not: `_parse_with_pymupdf` is tried FIRST in `parse()` and
        succeeds for essentially every PDF, so the pdfplumber branch never ran
        and `doc.tables` was always empty — for specifications, BOQs and
        drawing schedules alike.
        """
        finder = getattr(page, "find_tables", None)
        if finder is None:
            logger.warning(
                "PyMuPDF on this host has no Page.find_tables() (need >= 1.23); "
                "no tables extracted from page %s",
                page_num + 1,
            )
            return []
        try:
            found = finder()
        except Exception:
            logger.warning(
                "table detection failed on page %s — that page contributes no tables",
                page_num + 1, exc_info=True,
            )
            return []

        tables: List[List[List[str]]] = []
        for table in getattr(found, "tables", None) or []:
            try:
                rows = [
                    ["" if cell is None else str(cell).replace("\n", " ").strip()
                     for cell in row]
                    for row in (table.extract() or [])
                ]
            except Exception:
                logger.warning(
                    "could not read a detected table on page %s — skipping it",
                    page_num + 1, exc_info=True,
                )
                continue
            rows = [r for r in rows if any(c for c in r)]
            if rows:
                tables.append(rows)
        return tables

    def _parse_with_pdfplumber(self, file_path: str, doc: PDFDocument) -> PDFDocument:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                doc.pages.append(text)
                doc.text += text + "\n"
                tables = page.extract_tables()
                if tables:
                    doc.tables.extend(tables)
        return doc

    def _parse_with_pypdf2(self, file_path: str, doc: PDFDocument) -> PDFDocument:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text() or ""
            doc.pages.append(text)
            doc.text += text + "\n"
        return doc


    def _extract_glossary(self, text: str) -> Dict[str, str]:
        glossary = {}
        patterns = self.config.get("patterns", {}).get("glossary", [])
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = (
                    match.group("term").strip()
                    if "term" in match.groupdict()
                    else match.group(1).strip()
                )
                definition = (
                    match.group("definition").strip()
                    if "definition" in match.groupdict()
                    else match.group(2).strip()
                )
                if 1 < len(term) < 40:
                    glossary[term] = definition
        return glossary

    def _extract_figures(self, text: str) -> List[Dict[str, Any]]:
        figures = []
        fig_pattern = r"Figure\s+(\d+[.-]?\d*)\s*[—:\-\.]\s*(.+?)(?:\n|$)"
        for match in re.finditer(fig_pattern, text, re.IGNORECASE):
            figures.append({
                "id": match.group(1),
                "caption": match.group(2).strip(),
            })
        return figures
