#!/usr/bin/env python3
"""
Cerebrum-Blocks / Document Engine / Main Orchestrator

Usage:
    python main.py --pdf BOD.pdf --docx RFP.docx --xlsx Appendix.xlsx --output analysis.yaml

Triggers the 3-layer pipeline:
    Parse → Reason → Map → Export
"""
import argparse
import sys
from pathlib import Path
import yaml

from app.document_engine.parsers.pdf_parser import PDFParser
from app.document_engine.parsers.docx_parser import DOCXParser
from app.document_engine.parsers.xlsx_parser import XLSXParser
from app.document_engine.reasoner import DocumentReasoner
from app.document_engine.mapper import DocumentMapper

_HERE = Path(__file__).parent


def parse_all(file_paths: dict, config: dict):
    """Layer 1: Parse all provided files into typed document objects."""
    documents = []

    if file_paths.get("pdf"):
        parser = PDFParser(config)
        documents.append(parser.parse(file_paths["pdf"]))

    if file_paths.get("docx"):
        parser = DOCXParser(config)
        documents.append(parser.parse(file_paths["docx"]))

    if file_paths.get("xlsx"):
        parser = XLSXParser(config)
        documents.append(parser.parse(file_paths["xlsx"]))

    return documents


def main():
    parser = argparse.ArgumentParser(description="Document Reasoning Engine")
    parser.add_argument("--pdf", help="Path to PDF document")
    parser.add_argument("--docx", help="Path to DOCX document")
    parser.add_argument("--xlsx", help="Path to XLSX document")
    parser.add_argument("--config", default=str(_HERE / "config.yaml"))
    parser.add_argument("--output", default="analysis.yaml")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not any([args.pdf, args.docx, args.xlsx]):
        parser.error("At least one input file (--pdf, --docx, --xlsx) is required")

    # Load config
    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)
    config = full_config.get("document_engine", full_config)

    if args.verbose:
        print(f"📄 Document Engine v{config.get('version', '1.0.0')}")
        print(f"   PDF={args.pdf}, DOCX={args.docx}, XLSX={args.xlsx}")

    # Layer 1: Parse
    if args.verbose:
        print("\n[1/3] Parsing documents...")
    file_paths = {"pdf": args.pdf, "docx": args.docx, "xlsx": args.xlsx}
    documents = parse_all(file_paths, config)

    # Layer 2: Reason
    if args.verbose:
        print("[2/3] Reasoning over extracted content...")
    reasoner = DocumentReasoner(config)
    reasoned = reasoner.reason(documents)
    reasoned._document_count = len(documents)

    # Layer 3: Map
    if args.verbose:
        print("[3/3] Mapping to structured output...")
    mapper = DocumentMapper(config)
    structured = mapper.map_to_structured(reasoned)

    # Export
    output_path = Path(args.output)
    if args.format == "yaml":
        output_path.write_text(structured.to_yaml(), encoding="utf-8")
    else:
        output_path.write_text(structured.to_json(), encoding="utf-8")

    if args.verbose:
        print(f"\n✅ Analysis exported: {args.output}")
        print(f"   Glossary terms: {len(structured.glossary)}")
        print(f"   Requirements: {len(structured.requirements)}")
        print(f"   Constraints: {len(structured.constraints)}")
        print(f"   Schedule targets: {len(structured.schedule_targets)}")
        print(f"   Equipment specs: {len(structured.equipment_specs)}")
        print(f"   Risks: {len(structured.risks)}")
        print(f"   WBS mappings: {sum(len(v) for v in structured.wbs_mapping.values())}")
        print(f"   Downstream activities: {len(structured.downstream.get('schedule_engine', {}).get('procurement_activities', []))}")
    else:
        print(f"✅ {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
