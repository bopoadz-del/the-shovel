"""BIM Block - IFC/DXF/PDF file processor.

DWG is not read (owner ruling): any action given a DWG answers with
``app.core.cad_formats.DWG_NOT_SUPPORTED``.
"""
from app.core.cad_formats import DWG_NOT_SUPPORTED, dwg_not_supported, is_dwg
from app.core.universal_base import UniversalBlock
from typing import Dict, Any, List, Optional
import os
import json
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BIMBlock(UniversalBlock):
    """
    BIM File Processor - parses IFC, OCRs PDFs
    """
    
    auto_validate = False
    name = "bim"
    version = "1.1.0"
    # Original `requires = ["config", "storage", "vector", "pdf", "ocr"]`
    # referenced four blocks (`config`, `storage`, `vector`, `database`) that
    # don't exist in BLOCK_REGISTRY — the platform moved to a direct-file-path
    # architecture (document_engine, bim_extractor) and storage was never
    # implemented. We keep `pdf` + `ocr` so `process_pdf` can fall back to
    # them, and we delegate IFC parsing to `bim_extractor` (which actually
    # works today). Storage-backed actions (`index_folder`, `compare_versions`)
    # surface a clear error instead of crashing on a missing dep.
    requires = ["pdf", "ocr", "bim_extractor", "document_engine"]
    layer = 6  # Domain layer
    tags = ["construction", "bim", "cad", "domain"]
    default_config = {
        "ifc_enabled": True,
        "auto_index": True
    }
    
    SUPPORTED_FORMATS = {
        ".ifc": "bim_model",
        ".ifczip": "bim_model_compressed",
        ".dxf": "cad_exchange",
        ".pdf": "drawing_pdf",
        ".rvt": "revit_model",
        ".nwd": "navisworks",
        ".xlsx": "schedule",
        ".csv": "data_export"
    }
    
    def __init__(self, hal_block, config: Dict[str, Any]):
        super().__init__(hal_block, config)
        # NB: `self.storage_block` / `self.pdf_block` / etc. used to be set
        # to None here, but the platform's `wire()` populates
        # `self._dependencies[<name>]` — not those attributes. So even after
        # the dependency container wired the deps in, every action below
        # checked `self.pdf_block` (still None) and short-circuited with
        # "block not connected". The whole block was unreachable.
        #
        # The lazy properties below resolve `self.pdf_block` etc. from
        # `self.get_dep(...)` at access time, so wired deps flow through
        # transparently without touching the 200+ usage sites.
        self.projects = {}
        self._ifc_cache = {}

    @property
    def storage_block(self): return self.get_dep("storage")

    @property
    def pdf_block(self): return self.get_dep("pdf")

    @property
    def ocr_block(self): return self.get_dep("ocr")

    @property
    def database_block(self): return self.get_dep("database")
        
    async def _legacy_initialize(self):
        print(f"BIM File Processor ready")
        print(f"   Supports: {list(self.SUPPORTED_FORMATS.keys())}")
        
        # Check ifcopenshell availability
        try:
            import ifcopenshell
            print(f"   IFC parsing: ENABLED (ifcopenshell {ifcopenshell.version})")
        except ImportError:
            print("   IFC parsing: LIMITED (pip install ifcopenshell for full support)")
        
        return True
    
    # Set of action names this block dispatches. Anything outside this set
    # is rejected with a status:error + a list of valid actions, so
    # callers (and agents) see a complete contract instead of a bare error.
    _KNOWN_ACTIONS = frozenset({
        "index_folder", "parse_ifc",
        "process_pdf", "get_elements", "spatial_query", "compare_versions",
    })

    #: Input keys that name a file an action reads.
    _FILE_KEYS = ("file_path", "path", "old_path", "new_path")

    async def process(self, input_data: Dict, params: Dict = None) -> Dict:
        action = (params or {}).get("action") or (input_data.get("action") if isinstance(input_data, dict) else None)

        # A DWG is refused by format, whatever the action.
        for source in (input_data, params):
            if isinstance(source, dict) and any(
                is_dwg(source.get(k)) for k in self._FILE_KEYS
            ):
                return dwg_not_supported()

        if action == "index_folder":
            return await self._index_folder(input_data)
        elif action == "parse_ifc":
            return await self._parse_ifc_real(input_data)
        elif action == "process_pdf":
            return await self._process_pdf_real(input_data)
        elif action == "get_elements":
            return await self._get_ifc_elements(input_data)
        elif action == "spatial_query":
            return await self._spatial_query(input_data)
        elif action == "compare_versions":
            return await self._compare_versions_real(input_data)

        return {
            "status": "error",
            "error": (
                f"Unknown action: {action!r}. "
                f"Valid actions: {sorted(self._KNOWN_ACTIONS)}."
            ),
            "valid_actions": sorted(self._KNOWN_ACTIONS),
        }
    
    async def _index_folder(self, data: Dict) -> Dict:
        """Walk `folder_path` on disk and parse every supported BIM file's headers.

        History, because this wiring has been wrong twice: it originally listed
        via `self.storage_block.execute(list)` and embedded via
        `self.vector_block.execute(add)`, and neither block existed. It was then
        repointed at the `zvec` block — which computed an embedding and returned
        it WITHOUT STORING ANYTHING, while this method discarded the return
        value. So the "indexing" step indexed nothing; it only cost an embedding.

        That call is now gone. Folder listing uses `os.scandir`. Files uploaded
        through the normal path are indexed for retrieval by doc_index /
        app.core.rag, which is the only retrieval system the platform has.
        """
        project_id = data.get("project_id")
        folder_path = data.get("folder_path")
        if not folder_path or not os.path.isdir(folder_path):
            return {
                "status": "error",
                "error": f"folder_path required and must be a directory (got {folder_path!r})",
            }

        bim_files: List[Dict] = []
        not_supported: List[Dict] = []
        for entry in os.scandir(folder_path):
            if not entry.is_file():
                continue
            if is_dwg(entry.name):
                not_supported.append({"name": entry.name, "error": DWG_NOT_SUPPORTED})
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                continue
            stat = entry.stat()
            file_info = {
                "name": entry.name,
                "type": self.SUPPORTED_FORMATS[ext],
                "path": entry.path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
            if ext == ".ifc":
                meta = await self._parse_ifc_headers(file_info)
            elif ext == ".pdf":
                meta = {"type": "drawing_pdf", "needs_ocr": True}
            else:
                meta = {"type": file_info["type"]}
            file_info.update(meta)
            bim_files.append(file_info)

        self.projects[project_id] = {
            "folder": folder_path,
            "files": bim_files,
            "indexed_at": datetime.now().isoformat(),
            "total_files": len(bim_files),
        }
        return {
            "status": "success",
            "project_id": project_id,
            "indexed": len(bim_files),
            "by_type": self._count_by_type(bim_files),
            "not_supported": not_supported,
        }
    
    async def _parse_ifc_headers(self, file_info: Dict) -> Dict:
        """Parse IFC file headers + element counts directly from disk.

        Rewired: ifcopenshell.open() reads files by path, so the old
        storage→tempfile dance was pure overhead even when it worked. With
        the storage block now correctly identified as nonexistent, this
        method just reads file_info["path"] directly.
        """
        path = file_info.get("path") or file_info.get("file_path")
        if not path or not os.path.exists(path):
            return {
                "status": "error",
                "error": f"file_path required and must exist (got {path!r})",
            }
        try:
            import ifcopenshell
            from app.core.file_crypto import open_plaintext

            # open_plaintext transparently decrypts when DATA_ENCRYPTION_KEY
            # is set on the server (uploads encrypt at rest); plaintext files
            # short-circuit to their original path.
            with open_plaintext(path) as plain_path:
                ifc_file = ifcopenshell.open(plain_path)

            project = ifc_file.by_type("IfcProject")[0] if ifc_file.by_type("IfcProject") else None
            site = ifc_file.by_type("IfcSite")[0] if ifc_file.by_type("IfcSite") else None
            building = ifc_file.by_type("IfcBuilding")[0] if ifc_file.by_type("IfcBuilding") else None

            element_counts = {}
            for ifc_class in ["IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcColumn", "IfcBeam"]:
                count = len(ifc_file.by_type(ifc_class))
                if count > 0:
                    element_counts[ifc_class.replace("Ifc", "")] = count

            # Cache the parsed summary (a plain dict — not the live
            # ifcopenshell.file). LRU-bounded so a long-running session that
            # touches many IFCs doesn't blow memory.
            cache_entry = {
                "schema": ifc_file.schema,
                "project_name": project.Name if project else None,
                "site": site.Name if site else None,
                "building": building.Name if building else None,
                "element_counts": dict(element_counts),
            }
            _IFC_CACHE_CAP = 32
            if len(self._ifc_cache) >= _IFC_CACHE_CAP:
                self._ifc_cache.pop(next(iter(self._ifc_cache)))
            self._ifc_cache[path] = cache_entry

            return {
                "status": "success",
                "description": f"IFC Model: {project.Name if project else 'Unknown'}",
                "schema": ifc_file.schema,
                "project_name": project.Name if project else None,
                "site_name": site.Name if site else None,
                "building_name": building.Name if building else None,
                "element_counts": element_counts,
                "total_elements": sum(element_counts.values()),
                "extracted": True,
            }
        except ImportError:
            return {
                "status": "error",
                "error": "ifcopenshell not installed (pip install ifcopenshell).",
                "description": "IFC Model (ifcopenshell not installed)",
                "schema": "IFC2X3 (assumed)",
                "extracted": False,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"IFC parse error: {str(e)}",
                "description": f"IFC Model (parse error: {str(e)})",
                "extracted": False,
            }
    
    async def _parse_ifc_real(self, data: Dict) -> Dict:
        """Full IFC parsing with element extraction. Delegates to the
        bim_extractor block (which actually works today). The previous
        implementation fetched the file via a `storage` block that doesn't
        exist in the current platform — so this action was unreachable."""
        import os
        file_path = data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"file_path required and must exist (got {file_path!r})",
            }
        element_types = data.get("element_types", ["IfcWall", "IfcDoor", "IfcWindow"])

        extractor = self.get_dep("bim_extractor")
        if extractor is None:
            return {
                "status": "error",
                "error": "bim_extractor block not available — domain kit may not be loaded.",
            }
        result = await extractor.process(
            {"file_path": file_path},
            {"element_types": element_types},
        )
        if result.get("status") == "error":
            return {
                "status": "error",
                "error": result.get("error", "bim_extractor failed"),
            }
        # bim_extractor's `building_elements` is the canonical list; older
        # callers expect `elements`. bim_extractor does NOT honour the
        # `element_types` filter, so apply it here — otherwise the caller's
        # filter is silently dropped and they get the full element list.
        all_elements = result.get("building_elements", []) or result.get("elements", [])
        if element_types:
            wanted = {t.lower() for t in element_types}
            elements = [
                e for e in all_elements
                if str(e.get("ifc_type", "")).lower() in wanted
            ]
            filter_applied = sorted(element_types)
        else:
            elements = all_elements
            filter_applied = None
        return {
            "status": "success",
            "file": file_path,
            "elements": elements,
            "count": len(elements),
            "total_elements_in_file": len(all_elements),
            "element_types_filter": filter_applied,
            "schema": result.get("ifc_schema")
                or result.get("schema")
                or result.get("project_info", {}).get("schema"),
        }
    
    async def _process_pdf_real(self, data: Dict) -> Dict:
        """Process a PDF drawing — text-layer extraction, OCR-fallback for
        image-only drawings, then heuristic drawing-info extraction.

        Rewired to read `file_path` directly from disk via document_engine
        (which already wires `pdf` + `ocr`). The original implementation
        called `self.storage_block.execute({...retrieve...})` against a
        `storage` block that doesn't exist in this codebase — so this action
        crashed for everyone.
        """
        import os
        file_path = data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"file_path required and must exist (got {file_path!r})",
            }

        # Delegate to document_engine, which does PDF → text-layer → OCR
        # fallback. Use it via the wired dep so we share the platform's
        # singleton (and its already-wired pdf+ocr).
        engine = self.get_dep("document_engine")
        if engine is None:
            return {
                "status": "error",
                "error": "document_engine block not available — extraction stack not wired.",
            }
        eng_result = await engine.process({}, {"pdf_path": file_path})
        if eng_result.get("status") == "error":
            return {
                "status": "error",
                "error": eng_result.get("error", "document_engine failed"),
            }

        text = (eng_result.get("raw_text") or "").strip()
        drawing_info = self._extract_drawing_info(text)
        return {
            "status": "success",
            "file": file_path,
            "text": text[:5000],
            "pages": (eng_result.get("documents_parsed") or 0),
            "drawing_info": drawing_info,
            "extracted_via": eng_result.get("platform_blocks_used", []),
            "processed": True,
        }
    
    def _extract_drawing_info(self, text: str) -> Dict:
        """Extract structured info from drawing text"""
        import re
        
        # Look for drawing numbers (A-101, etc.)
        drawing_nums = re.findall(r'[A-Z]-\d{3,4}', text)
        
        # Look for dates
        dates = re.findall(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', text)
        
        # Look for scales
        scales = re.findall(r'\d+\s*:\s*\d+', text)
        
        # Look for sheet sizes
        sheet_sizes = re.findall(r'\b(A0|A1|A2|A3|A4)\b', text)
        
        return {
            "drawing_numbers": list(set(drawing_nums))[:10],
            "dates": list(set(dates))[:5],
            "scales": list(set(scales))[:5],
            "sheet_sizes": list(set(sheet_sizes))
        }
    
    async def _get_ifc_elements(self, data: Dict) -> Dict:
        """Query IFC elements with filters"""
        file_path = data.get("file_path")
        element_type = data.get("element_type", "IfcWall")
        properties = data.get("properties", [])
        
        return await self._parse_ifc_real({
            "file_path": file_path,
            "element_types": [element_type]
        })
    
    async def _spatial_query(self, data: Dict) -> Dict:
        """Find IFC elements whose AABB intersects a query bounding box.

        Uses ifcopenshell.geom to compute world-coord AABBs per element, then
        keeps those whose box overlaps the query box. Returns elements grouped
        by ifc_type with counts and IDs. Requires the `geom` extension; the
        method returns an honest error when it's not installed.
        """
        file_path = data.get("file_path")
        if not file_path:
            return {"status": "error", "error": "file_path is required"}
        bbox = data.get("bbox") or data.get("bounding_box")
        if not bbox or len(bbox) != 6:
            return {
                "status": "error",
                "error": "Provide bbox = [xmin, ymin, zmin, xmax, ymax, zmax] (world coords, metres).",
            }
        # Default to the major IFC types if none specified.
        element_types = data.get("element_types") or [
            "IfcWall", "IfcSlab", "IfcBeam", "IfcColumn",
            "IfcDoor", "IfcWindow", "IfcPipeSegment",
            "IfcDuctSegment", "IfcCableSegment",
        ]

        try:
            import ifcopenshell
            import ifcopenshell.geom
        except Exception:
            return {
                "status": "error",
                "error": "ifcopenshell.geom not available. Install ifcopenshell[all] for spatial queries.",
            }

        try:
            from app.core.file_crypto import open_plaintext
            with open_plaintext(file_path) as plain_path:
                model = ifcopenshell.open(plain_path)
        except Exception as e:
            return {"status": "error", "error": f"IFC open failed: {e}"}

        settings = ifcopenshell.geom.settings()
        try:
            settings.set(settings.USE_WORLD_COORDS, True)
        except Exception:
            logger.debug(
                "swallowed %s in _spatial_query() — continuing",
                "Exception", exc_info=True,
            )

        qmin = (float(bbox[0]), float(bbox[1]), float(bbox[2]))
        qmax = (float(bbox[3]), float(bbox[4]), float(bbox[5]))

        hits: Dict[str, List[str]] = {}
        considered = 0
        for ifc_type in element_types:
            try:
                items = model.by_type(ifc_type)
            except Exception:
                continue
            for el in items:
                considered += 1
                try:
                    shape = ifcopenshell.geom.create_shape(settings, el)
                    verts = shape.geometry.verts
                except Exception:
                    continue
                if not verts:
                    continue
                xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
                emin = (min(xs), min(ys), min(zs))
                emax = (max(xs), max(ys), max(zs))
                if (emin[0] <= qmax[0] and emax[0] >= qmin[0]
                        and emin[1] <= qmax[1] and emax[1] >= qmin[1]
                        and emin[2] <= qmax[2] and emax[2] >= qmin[2]):
                    hits.setdefault(ifc_type, []).append(getattr(el, "GlobalId", str(el.id())))

        total = sum(len(v) for v in hits.values())
        return {
            "status": "success",
            "bbox": bbox,
            "elements_considered": considered,
            "match_count": total,
            "matches_by_type": {k: {"count": len(v), "ids": v[:50]} for k, v in hits.items()},
        }
    
    async def _compare_versions_real(self, data: Dict) -> Dict:
        """Compare two versions of same file. Both paths must parse — if
        either side fails the action surfaces the failure instead of
        silently computing a diff against zero (the prior behaviour:
        ``old_count = old_data.get("count", 0)`` swallowed every parse
        error and reported it as ``old_count = 0`` plus a misleading
        change_percent value)."""
        old_path = data.get("old_version")
        new_path = data.get("new_version")
        if not old_path or not new_path:
            return {
                "status": "error",
                "error": "Both 'old_version' and 'new_version' file_paths required.",
            }

        old_data = await self._parse_ifc_real({"file_path": old_path, "element_types": ["IfcWall"]})
        if old_data.get("status") == "error":
            return {
                "status": "error",
                "error": f"old_version parse failed: {old_data.get('error')}",
                "old_version": old_path,
                "new_version": new_path,
            }
        new_data = await self._parse_ifc_real({"file_path": new_path, "element_types": ["IfcWall"]})
        if new_data.get("status") == "error":
            return {
                "status": "error",
                "error": f"new_version parse failed: {new_data.get('error')}",
                "old_version": old_path,
                "new_version": new_path,
            }

        old_count = old_data.get("count", 0)
        new_count = new_data.get("count", 0)
        return {
            "status": "success",
            "old_version": old_path,
            "new_version": new_path,
            "old_count": old_count,
            "new_count": new_count,
            "difference": new_count - old_count,
            "change_percent": ((new_count - old_count) / old_count * 100) if old_count > 0 else 0,
        }
    
    def _count_by_type(self, files: List[Dict]) -> Dict:
        counts = {}
        for f in files:
            t = f["type"]
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    def health(self) -> Dict:
        h = {"name": self.name, "version": self.version}
        h["projects_indexed"] = len(self.projects)
        h["files_tracked"] = sum(p.get("total_files", 0) for p in self.projects.values())
        h["ifc_cache_size"] = len(self._ifc_cache)
        
        try:
            import ifcopenshell
            h["ifcopenshell"] = True
            h["ifcopenshell_version"] = ifcopenshell.version
        except ImportError:
            h["ifcopenshell"] = False
        
        return h
