"""BIM Extractor Block - Extract quantities, elements, and clash report from IFC BIM models"""

import logging
import os
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from app.core.universal_base import UniversalBlock

logger = logging.getLogger(__name__)

_logger = logging.getLogger(__name__)


# IFC type → construction category mapping
IFC_CATEGORY_MAP: Dict[str, str] = {
    "IfcWall": "walls",
    "IfcWallStandardCase": "walls",
    "IfcSlab": "slabs",
    "IfcBeam": "beams",
    "IfcColumn": "columns",
    "IfcStair": "stairs",
    "IfcRoof": "roofs",
    "IfcDoor": "doors",
    "IfcWindow": "windows",
    "IfcCurtainWall": "curtain_walls",
    "IfcRailing": "railings",
    "IfcPile": "piles",
    "IfcFooting": "footings",
    "IfcMember": "members",
    "IfcPlate": "plates",
    "IfcPipeSegment": "pipes",
    "IfcDuctSegment": "ducts",
    "IfcCableSegment": "cables",
    "IfcFlowTerminal": "terminals",
    "IfcSpaceHeater": "hvac",
    "IfcLightFixture": "lighting",
    "IfcSpace": "spaces",
    "IfcZone": "zones",
    "IfcBuildingStorey": "storeys",
}


_CATEGORY_ITEM_CAP = 200
_ELEMENT_CAP = 500
_SPACE_CAP = 50

_CLASH_DISCLAIMER = (
    "basic (bounding-box — not geometric intersection. Verify critical "
    "clashes in Navisworks Clash Detective or Solibri)"
)


def _container_storey_name(el, ifc_util) -> Optional[str]:
    """Name of the IfcBuildingStorey that contains ``el`` in the model.

    Read from the model's own spatial containment (``get_container``
    walks up through enclosing spaces). ``None`` when the element is not
    contained in a storey or the lookup fails; an unnamed storey is "".
    """
    try:
        container = ifc_util.get_container(el, ifc_class="IfcBuildingStorey")
    except Exception:
        logger.debug("get_container failed for %s", el, exc_info=True)
        return None
    if container is None:
        return None
    return getattr(container, "Name", None) or ""


def _types_outside_map(model) -> Dict[str, int]:
    """Count physical elements whose IFC class no IFC_CATEGORY_MAP entry
    (or a supertype of it) covers, keyed by class name.

    Coverage is read from the schema's own inheritance (``is_a``), so a
    subtype of a mapped class is covered and only genuinely unmapped
    classes are reported.
    """
    try:
        physical = model.by_type("IfcElement")
    except Exception:
        logger.debug("by_type(IfcElement) failed", exc_info=True)
        return {}
    counts: Dict[str, int] = {}
    for el in physical:
        if any(el.is_a(mapped) for mapped in IFC_CATEGORY_MAP):
            continue
        cls = el.is_a()
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))


_QTO_GROUP_BY_DEFAULT: Tuple[str, ...] = ("storey", "ifc_type")


def _qto_number(value: Any) -> Optional[float]:
    """A quantity figure: an int or float that is not a bool and is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _qto_field(element: Dict, field: str) -> Any:
    """Read ``field`` from the element itself, else from its flattened
    ``properties`` (``"Pset.Prop"`` keys)."""
    if field in element:
        return element[field]
    props = element.get("properties")
    if isinstance(props, dict):
        return props.get(field)
    return None


def _qto_default_columns(elements: List[Dict]) -> List[Dict[str, Any]]:
    """One column per numeric top-level key the elements themselves carry,
    each read from that key only, in first-seen order."""
    seen: List[str] = []
    for el in elements:
        for key, value in el.items():
            if key not in seen and _qto_number(value) is not None:
                seen.append(key)
    return [{"column": key, "fields": [key]} for key in seen]


def _qto_columns_from_table(table: Any) -> List[Dict[str, Any]]:
    """Validate a caller's column table: a list of
    ``{"column": name, "fields": [field, ...], "unit": optional}``.

    ``fields`` is read in order and the first numeric one an element
    carries is that element's figure. Raises ``ValueError`` naming the
    first bad row; a table is never half-applied.
    """
    if not isinstance(table, list) or not table:
        raise ValueError("qto_columns must be a non-empty list of column rows")
    columns: List[Dict[str, Any]] = []
    names: set = set()
    for i, row in enumerate(table):
        if not isinstance(row, dict):
            raise ValueError(f"qto_columns row {i} is not an object")
        name = row.get("column")
        fields = row.get("fields")
        if isinstance(fields, str):
            fields = [fields]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"qto_columns row {i} has no column name")
        if name in names:
            raise ValueError(f"qto_columns row {i} repeats column {name!r}")
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(f, str) and f for f in fields)
        ):
            raise ValueError(f"qto_columns row {i} ({name}) has no fields")
        names.add(name)
        col: Dict[str, Any] = {"column": name, "fields": list(fields)}
        if row.get("unit") is not None:
            col["unit"] = str(row["unit"])
        columns.append(col)
    return columns


def group_quantities(
    elements: List[Dict],
    columns: Optional[Any] = None,
    group_by: Optional[Any] = None,
) -> Dict[str, Any]:
    """Quantity take-off grouped by element keys (storey x IFC class by
    default), with the column mapping taken from ``columns``.

    Per group and column it reports the total of the figures found, how
    many elements gave a figure (``counted``), how many gave none
    (``missing``) and which field each figure came from (``from``), so a
    total mixing net and gross figures, or covering part of a group, says
    so. A column with no figure in a group totals ``None``, never 0. An
    element without a group key value (for example not contained in any
    storey) is grouped under ``None``; the value is never guessed.
    """
    keys = list(group_by) if group_by else list(_QTO_GROUP_BY_DEFAULT)
    if not all(isinstance(k, str) and k for k in keys):
        raise ValueError("qto_group_by must be a list of element keys")
    cols = (
        _qto_columns_from_table(columns)
        if columns is not None
        else _qto_default_columns(elements)
    )
    groups: Dict[Tuple, Dict[str, Any]] = {}
    sums: Dict[Tuple, Dict[str, List[float]]] = {}
    for el in elements:
        gkey = tuple(_qto_field(el, k) for k in keys)
        group = groups.get(gkey)
        if group is None:
            group = {k: v for k, v in zip(keys, gkey)}
            group["count"] = 0
            group["columns"] = {
                c["column"]: {"counted": 0, "missing": 0, "from": {}}
                for c in cols
            }
            groups[gkey] = group
            sums[gkey] = {c["column"]: [] for c in cols}
        group["count"] += 1
        for c in cols:
            cell = group["columns"][c["column"]]
            for field in c["fields"]:
                figure = _qto_number(_qto_field(el, field))
                if figure is not None:
                    sums[gkey][c["column"]].append(figure)
                    cell["counted"] += 1
                    cell["from"][field] = cell["from"].get(field, 0) + 1
                    break
            else:
                cell["missing"] += 1
    rows: List[Dict[str, Any]] = []
    for gkey, group in groups.items():
        for c in cols:
            figures = sums[gkey][c["column"]]
            cell = group["columns"][c["column"]]
            cell["total"] = math.fsum(figures) if figures else None
            if "unit" in c:
                cell["unit"] = c["unit"]
        rows.append(group)
    rows.sort(key=lambda r: tuple(
        (r[k] is None, str(r[k]) if r[k] is not None else "") for k in keys
    ))
    return {
        "group_by": keys,
        "columns": [c["column"] for c in cols],
        "elements_grouped": len(elements),
        "groups": rows,
    }


class BIMExtractorBlock(UniversalBlock):
    auto_validate = False
    name = "bim_extractor"
    version = "1.2.0"
    description = "Extract building elements, quantities, and clash report from IFC BIM models"
    layer = 3
    tags = ["domain", "construction", "bim", "ifc", "quantities", "clash"]
    requires = []

    default_config = {
        "extract_properties": True,
        "run_clash_detection": False,
        "clash_timeout_s": 8.0,
        "clash_tolerance_mm": 10.0,
        "max_elements": 10000,
    }

    # Proprietary formats this block does NOT parse natively. Each is mapped to
    # the conversion / SDK path the operator needs to take, so the chat
    # response is actionable instead of a vague "unsupported".
    _PROPRIETARY_FORMATS = {
        ".nwd": (
            "Navisworks Document (.nwd) is Autodesk-proprietary. Convert to IFC "
            "(in Navisworks: File > Export > IFC) or use the Autodesk Platform "
            "Services Model Derivative API to convert. Then upload the .ifc."
        ),
        ".rvt": (
            "Revit (.rvt) is Autodesk-proprietary. Export from Revit as IFC "
            "(File > Export > IFC, IFC 4 or IFC 2x3) and upload the .ifc file."
        ),
        ".nwc": (
            "Navisworks Cache (.nwc) is Autodesk-proprietary. Open in Navisworks "
            "and export as IFC."
        ),
    }

    ui_schema = {
        "input": {
            "type": "file",
            "accept": [".ifc"],
            "placeholder": "Upload IFC BIM model (.rvt/.nwd: export to IFC first)...",
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "building_elements", "type": "list", "label": "Elements"},
                {"name": "quantities", "type": "json", "label": "Quantities"},
                {"name": "clash_report", "type": "json", "label": "Clashes"},
                {"name": "element_count", "type": "number", "label": "Total Elements"},
            ],
        },
        "quick_actions": [
            {"icon": "️", "label": "Extract All", "prompt": "Extract all building elements and quantities"},
            {"icon": "", "label": "Clash Detection", "prompt": "Run clash detection on this BIM model"},
            {"icon": "", "label": "Quantities", "prompt": "Extract material quantities for cost estimation"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        file_path = data.get("file_path") or params.get("file_path") or data.get("text") or data.get("input") or (input_data if isinstance(input_data, str) else "")
        if not file_path:
            return {"status": "error", "error": "No file_path provided — requires an IFC file"}
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}

        ext = os.path.splitext(str(file_path).lower())[1]
        # Helpful, actionable error for known proprietary formats — instead of
        # returning a vague "unsupported" string when the operator uploads an
        # .rvt or .nwd. Maps each to the conversion path needed to get to IFC.
        if ext in self._PROPRIETARY_FORMATS:
            return {
                "status": "error",
                "error": self._PROPRIETARY_FORMATS[ext],
                "format_extension": ext,
                "required_format": ".ifc",
            }
        if ext != ".ifc":
            return {
                "status": "error",
                "error": f"File must be an .ifc IFC model (got '{ext}')",
                "format_extension": ext,
                "required_format": ".ifc",
            }

        try:
            import ifcopenshell
            import ifcopenshell.util.element as ifc_util
        except ImportError:
            return {
                "status": "error",
                "error": "ifcopenshell not installed. Run: pip install ifcopenshell",
            }

        try:
            # Decrypt-on-read for encrypted-at-rest uploads; no-op for plaintext.
            from app.core.file_crypto import open_plaintext
            with open_plaintext(file_path) as plain_path:
                model = ifcopenshell.open(plain_path)
        except Exception as e:
            return {"status": "error", "error": f"IFC open error: {e}"}

        max_el = int(params.get("max_elements", self.config.get("max_elements", 10000)))
        extract_props = params.get("extract_properties", self.config.get("extract_properties", True))
        run_clash = params.get(
            "run_clash_detection",
            self.config.get("run_clash_detection", False),
        )

        (
            building_elements,
            quantities,
            duplicate_subtypes_skipped,
            quantities_truncated,
            categories_skipped,
            psets_failed,
        ) = self._extract_elements(model, ifc_util, max_el, extract_props)
        types_not_extracted = _types_outside_map(model)
        project_info = self._extract_project_info(model)
        storeys = self._extract_storeys(model)
        spaces = self._extract_spaces(model)
        # Take-off grouped by storey x IFC class over every extracted
        # element (before the payload cap). The column mapping is the
        # caller's table (``qto_columns``); without one, each quantity key
        # the elements carry is its own column. A bad table is reported,
        # never half-applied, and does not lose the extraction.
        try:
            qto_groups = group_quantities(
                building_elements,
                params.get("qto_columns"),
                params.get("qto_group_by"),
            )
        except ValueError as exc:
            qto_groups = {"error": str(exc)}

        clash_report = {}
        if run_clash:
            clash_report = self._basic_clash_report(
                model, building_elements, params
            )

        # Cap response payload so a 50k-element model can't blow up the chat
        # context. Each cap fires independently; the top-level ``truncated``
        # flag is a single signal callers can read without inspecting every
        # sub-object.
        building_elements_capped = building_elements[:_ELEMENT_CAP]
        spaces_capped = spaces[:_SPACE_CAP]
        building_elements_truncated = len(building_elements) > _ELEMENT_CAP
        spaces_truncated = len(spaces) > _SPACE_CAP
        clash_truncated = bool(clash_report.get("pair_cap_reached"))

        any_truncated = (
            bool(quantities_truncated)
            or building_elements_truncated
            or spaces_truncated
            or clash_truncated
        )

        return {
            "status": "success",
            "ifc_schema": model.schema,
            "element_count": len(building_elements),
            "element_count_returned": len(building_elements_capped),
            "qto_groups": qto_groups,
            "quantities": quantities,
            "quantities_truncated": list(quantities_truncated),
            "building_elements": building_elements_capped,
            "building_elements_truncated": building_elements_truncated,
            "spaces_truncated": spaces_truncated,
            "clash_report": clash_report,
            "project_info": project_info,
            "storeys": storeys,
            "spaces": spaces_capped,
            "duplicate_subtypes_skipped": duplicate_subtypes_skipped,
            # Categories whose model.by_type() raised — quantities for these
            # IFC types are 0 in this result but the underlying model may
            # contain them.
            "categories_skipped": categories_skipped,
            # Elements for which get_psets() raised — properties and pset-
            # derived quantities (NetVolume, NetArea, etc.) are missing.
            "psets_failed": psets_failed,
            # Physical elements of an IFC class no category above covers:
            # {ifc_class: count}. They are in the model but not in
            # ``quantities`` or ``building_elements``.
            "types_not_extracted": types_not_extracted,
            "truncated": any_truncated,
            "truncation_caps": {
                "category_item_cap": _CATEGORY_ITEM_CAP,
                "building_elements_cap": _ELEMENT_CAP,
                "spaces_cap": _SPACE_CAP,
            },
        }

    def _extract_elements(
        self, model, ifc_util, max_el: int, extract_props: bool
    ) -> Tuple[List[Dict], Dict, int, List[str], List[str], List[Dict[str, str]]]:
        """Returns (elements, quantities, duplicate_subtypes_skipped,
        quantities_truncated, categories_skipped, psets_failed).

        ``categories_skipped`` lists IFC type strings for which
        ``model.by_type()`` raised — those categories contributed zero rows
        despite possibly existing in the file.

        ``psets_failed`` lists ``{"global_id", "ifc_type", "error"}`` dicts
        for elements where ``ifc_util.get_psets()`` raised. The element is
        still returned (without the ``properties`` block) but the caller now
        knows quantities derived from psets are incomplete.
        """
        elements: List[Dict] = []
        quantities: Dict[str, Any] = {
            cat: {"count": 0, "items": []}
            for cat in set(IFC_CATEGORY_MAP.values())
        }
        # `ifcopenshell.model.by_type(t)` returns subtypes by default, so
        # IFC_CATEGORY_MAP entries like IfcWall + IfcWallStandardCase would
        # otherwise double-count every IfcWallStandardCase. Track GlobalId
        # (preferred) or step-id fallback to dedupe.
        seen_guids: set = set()
        duplicate_subtypes_skipped = 0
        categories_skipped: List[str] = []
        psets_failed: List[Dict[str, str]] = []

        for ifc_type, category in IFC_CATEGORY_MAP.items():
            try:
                items = model.by_type(ifc_type)
            except Exception as exc:
                _logger.warning(
                    "bim_extractor: model.by_type(%s) failed — dropping entire "
                    "category from results: %s",
                    ifc_type, exc,
                )
                categories_skipped.append(ifc_type)
                continue
            for el in items:
                if len(elements) >= max_el:
                    break
                key = getattr(el, "GlobalId", None) or el.id()
                if key in seen_guids:
                    duplicate_subtypes_skipped += 1
                    continue
                seen_guids.add(key)
                el_data = self._element_to_dict(
                    el, category, ifc_util, extract_props, psets_failed,
                )
                elements.append(el_data)
                quantities[category]["count"] += 1
                if len(quantities[category]["items"]) < _CATEGORY_ITEM_CAP:
                    quantities[category]["items"].append(el_data)

        quantities = {k: v for k, v in quantities.items() if v["count"] > 0}
        # Per-category items array is capped at _CATEGORY_ITEM_CAP. Surface
        # which categories actually hit the cap so the caller knows what was
        # dropped instead of inferring it.
        quantities_truncated = [
            cat for cat, v in quantities.items()
            if v["count"] > len(v["items"])
        ]
        for cat in quantities_truncated:
            quantities[cat]["truncated"] = True
            quantities[cat]["items_returned"] = len(quantities[cat]["items"])
        return (
            elements,
            quantities,
            duplicate_subtypes_skipped,
            quantities_truncated,
            categories_skipped,
            psets_failed,
        )

    def _element_to_dict(
        self,
        el,
        category: str,
        ifc_util,
        extract_props: bool,
        psets_failed: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        el_dict: Dict = {
            "id": el.GlobalId if hasattr(el, "GlobalId") else str(el.id()),
            "ifc_type": el.is_a(),
            "category": category,
            "name": getattr(el, "Name", "") or "",
            "description": getattr(el, "Description", "") or "",
            "object_type": getattr(el, "ObjectType", "") or "",
        }

        # Extract psets
        if extract_props:
            try:
                psets = ifc_util.get_psets(el)
                # Flatten to key/value — keep only scalar values
                flat_props: Dict[str, Any] = {}
                for pset_name, pset_vals in psets.items():
                    if isinstance(pset_vals, dict):
                        for prop_name, prop_val in pset_vals.items():
                            if isinstance(prop_val, (str, int, float, bool)):
                                flat_props[f"{pset_name}.{prop_name}"] = prop_val
                if flat_props:
                    el_dict["properties"] = flat_props

                # Pull common quantities
                for pset_name, pset_vals in psets.items():
                    if isinstance(pset_vals, dict):
                        for qname in ("NetVolume", "GrossVolume", "NetSideArea", "GrossSideArea",
                                      "NetFootprintArea", "GrossFootprintArea", "Length", "Width",
                                      "Height", "Depth", "Thickness", "NetArea", "GrossArea"):
                            if qname in pset_vals and isinstance(pset_vals[qname], (int, float)):
                                el_dict[qname.lower()] = pset_vals[qname]
            except Exception as exc:
                # Silent before; the element still gets returned (id/name/etc.)
                # but properties + pset-derived quantities are missing. Log
                # and accumulate so the caller knows quantities are partial.
                gid = el_dict.get("id", "?")
                ifc_type = el_dict.get("ifc_type", "?")
                _logger.warning(
                    "bim_extractor: get_psets failed for %s (id=%s): %s",
                    ifc_type, gid, exc,
                )
                if psets_failed is not None:
                    psets_failed.append({
                        "global_id": str(gid),
                        "ifc_type": str(ifc_type),
                        "error": str(exc),
                    })

        # Spatial containment: the storey the model itself places the element
        # on (direct or through an enclosing space). Absent when the model
        # does not contain the element in a storey; never guessed.
        storey = _container_storey_name(el, ifc_util)
        if storey is not None:
            el_dict["storey"] = storey

        # Material
        try:
            mats = ifc_util.get_materials(el)
            if mats:
                el_dict["materials"] = [
                    m.Name if hasattr(m, "Name") else str(m) for m in mats[:5]
                ]
        except Exception:
            logger.debug(
                "swallowed %s in _element_to_dict() — continuing",
                "Exception", exc_info=True,
            )

        return el_dict

    def _extract_project_info(self, model) -> Dict:
        try:
            projects = model.by_type("IfcProject")
            if projects:
                p = projects[0]
                return {
                    "name": getattr(p, "Name", "") or "",
                    "description": getattr(p, "Description", "") or "",
                    "global_id": p.GlobalId if hasattr(p, "GlobalId") else "",
                    "schema": model.schema,
                }
        except Exception:
            logger.debug(
                "swallowed %s in _extract_project_info() — continuing",
                "Exception", exc_info=True,
            )
        return {"schema": model.schema}

    def _extract_storeys(self, model) -> List[Dict]:
        try:
            storeys = model.by_type("IfcBuildingStorey")
            return [
                {
                    "name": getattr(s, "Name", "") or "",
                    "elevation": getattr(s, "Elevation", None),
                    "description": getattr(s, "Description", "") or "",
                }
                for s in storeys
            ]
        except Exception:
            return []

    def _extract_spaces(self, model) -> List[Dict]:
        try:
            spaces = model.by_type("IfcSpace")
            return [
                {
                    "name": getattr(s, "Name", "") or "",
                    "long_name": getattr(s, "LongName", "") or "",
                    "description": getattr(s, "Description", "") or "",
                }
                for s in spaces
            ]
        except Exception:
            return []

    def _clash_opt(self, params: Dict, key: str, default):
        """Resolve a clash knob: per-call params, then block config, then default.

        ``run_clash_detection`` already followed this precedence but the knobs
        that govern the RESULT did not -- they read self.config only. Asking for
        a 25mm tolerance through params silently ran at 10mm AND reported
        ``tolerance_mm: 10.0``, so the output looked authoritative while having
        ignored the request.
        """
        if params and key in params:
            return params[key]
        return self.config.get(key, default)

    def _basic_clash_report(
        self, model, elements: List[Dict], params: Dict | None = None
    ) -> Dict:
        """Clash detection: AABB intersection via ifcopenshell.geom when available.

        Builds an axis-aligned bounding box per element in world coordinates,
        then pairwise-tests them with a small tolerance. Pairs that overlap
        across DIFFERENT categories (e.g. pipe-vs-wall, beam-vs-duct) are
        flagged. Same-category overlaps are skipped because adjacent walls,
        stacked slabs, and side-by-side columns are routine and not clashes.

        Falls back to the legacy name-duplicate heuristic if `ifcopenshell.geom`
        isn't importable (some installs ship without the geometry extension).
        """
        try:
            import ifcopenshell.geom  # noqa: F401  (probe only)
        except Exception:
            return self._name_duplicate_clash_fallback(elements)
        return self._geometric_clash_report(model, elements, params)

    def _geometric_clash_report(
        self, model, elements: List[Dict], params: Dict | None = None
    ) -> Dict:
        """Real AABB clash pass. Returns method='aabb_intersection'."""
        import ifcopenshell.geom

        settings = ifcopenshell.geom.settings()
        try:
            settings.set(settings.USE_WORLD_COORDS, True)
        except Exception:
            # Older ifcopenshell builds do not expose this attribute. Geometry
            # is then read in LOCAL coordinates, so clash positions are
            # relative rather than absolute — worth a trace, not a warning.
            logger.debug(
                "ifcopenshell build does not expose USE_WORLD_COORDS; "
                "clash geometry stays in local coordinates", exc_info=True,
            )
        try:
            settings.set(settings.WELD_VERTICES, True)
        except Exception:
            logger.debug(
                "swallowed %s in _geometric_clash_report() — continuing",
                "Exception", exc_info=True,
            )

        params = params or {}
        tol_mm = float(self._clash_opt(params, "clash_tolerance_mm", 10.0))
        tol_m = tol_mm / 1000.0
        pair_cap = int(self._clash_opt(params, "clash_pair_cap", 200))
        # Bound the geometry pass — generating shapes is the expensive step.
        elem_cap = int(self._clash_opt(params, "clash_elem_cap", 2000))

        boxes: List[Tuple[Dict, Tuple[float, float, float, float, float, float]]] = []
        skipped_no_geom = 0
        deadline = time.monotonic() + float(
            self._clash_opt(params, "clash_timeout_s", 8.0)
        )
        timed_out = False
        for el in elements[:elem_cap]:
            if time.monotonic() > deadline:
                timed_out = True
                break
            ifc_el = self._lookup_ifc_element(model, el)
            if ifc_el is None:
                skipped_no_geom += 1
                continue
            try:
                shape = ifcopenshell.geom.create_shape(settings, ifc_el)
                verts = shape.geometry.verts  # flat [x0,y0,z0,x1,y1,z1,...]
            except Exception:
                # Element has no representable geometry (IfcSpace, IfcZone, etc.)
                skipped_no_geom += 1
                continue
            if not verts:
                skipped_no_geom += 1
                continue
            xs = verts[0::3]
            ys = verts[1::3]
            zs = verts[2::3]
            aabb = (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
            boxes.append((el, aabb))

        clashes: List[Dict] = []
        clash_count = 0
        for i in range(len(boxes)):
            if len(clashes) >= pair_cap:
                break
            ela, ba = boxes[i]
            for j in range(i + 1, len(boxes)):
                if len(clashes) >= pair_cap:
                    break
                elb, bb = boxes[j]
                # Per-axis interpenetration DEPTH, so tol_m means what it says:
                # the minimum overlap that counts as a clash.
                #
                # This used to compare each box's min against the other's MAX
                # (ba[0] + tol_m <= bb[3]), which is not the overlap depth -- it
                # is the distance from one box's near face to the other's FAR
                # face, so it grew with the other element's length. A 2.5m pipe
                # poking 200mm through a wall survived a 500mm tolerance while a
                # 300mm stub with the identical 200mm interpenetration did not.
                # Tolerance was therefore untunable in practice: raising it
                # dropped clashes in an order that tracked element length rather
                # than how badly things actually intersected.
                ox = min(ba[3], bb[3]) - max(ba[0], bb[0])
                oy = min(ba[4], bb[4]) - max(ba[1], bb[1])
                oz = min(ba[5], bb[5]) - max(ba[2], bb[2])
                if ox > tol_m and oy > tol_m and oz > tol_m:
                    # Same-category collocation is usually expected (walls
                    # touching at corners, slabs stacked). Skip it; cross-
                    # discipline overlaps are the real clashes.
                    if ela["category"] == elb["category"]:
                        continue
                    clash_count += 1
                    clashes.append({
                        "type": "aabb_overlap",
                        "element_a": ela["id"],
                        "element_b": elb["id"],
                        "category_a": ela["category"],
                        "category_b": elb["category"],
                        "ifc_type_a": ela.get("ifc_type"),
                        "ifc_type_b": elb.get("ifc_type"),
                        "name_a": ela.get("name") or "",
                        "name_b": elb.get("name") or "",
                        # Smallest per-axis interpenetration, in mm: how deep
                        # the two solids actually run into each other. This is
                        # the number to compare against tolerance_mm when
                        # judging whether a hit is worth chasing.
                        "overlap_mm": round(min(ox, oy, oz) * 1000.0, 1),
                        "severity": "warning",
                    })

        return {
            "clash_count": clash_count,
            "clashes": clashes,
            "detection_method": "aabb_intersection",
            "detection_method_disclaimer": _CLASH_DISCLAIMER,
            "tolerance_mm": tol_mm,
            "elements_analyzed": len(boxes),
            "elements_without_geometry": skipped_no_geom,
            "pair_cap_reached": len(clashes) >= pair_cap,
            "timed_out": timed_out,
            "note": (
                "AABB overlap is a coarse-pass: it catches every real clash "
                "but may flag false positives for adjacent elements that touch "
                "but don't intersect. Precise OBB/mesh intersection still "
                "needs a dedicated tool (Navisworks, Solibri)."
            ),
        }

    def _lookup_ifc_element(self, model, el_dict: Dict):
        """Look up the live ifcopenshell entity from the element dict.

        The dict's `id` field is the IFC GlobalId string (an IfcGuid). Falls
        back to the entity step-id if GlobalId lookup fails (legacy data).
        """
        key = el_dict.get("id")
        if not key:
            return None
        try:
            return model.by_guid(key)
        except Exception:
            logger.debug(
                "swallowed %s in _lookup_ifc_element() — continuing",
                "Exception", exc_info=True,
            )
        try:
            return model.by_id(int(key)) if str(key).isdigit() else None
        except Exception:
            return None

    def _name_duplicate_clash_fallback(self, elements: List[Dict]) -> Dict:
        """Legacy fallback: flag elements that share a name+type string.

        Used only when `ifcopenshell.geom` is unavailable. Real clashes
        between differently-named elements are invisible to this method —
        the result is named accordingly so callers don't mistake it for
        geometric clash detection.
        """
        clash_count = 0
        clashes: List[Dict] = []
        seen_positions: Dict[str, List[str]] = {}

        for el in elements:
            key = f"{el.get('object_type', '')}:{el.get('name', '')}"
            if key and key.strip(":"):
                if key in seen_positions:
                    clash_count += 1
                    if len(clashes) < 20:
                        clashes.append({
                            "type": "name_duplicate",
                            "element_a": seen_positions[key][0],
                            "element_b": el["id"],
                            "category": el["category"],
                            "description": f"Duplicate {el['ifc_type']} name: {el.get('name')}",
                            "severity": "warning",
                        })
                else:
                    seen_positions[key] = [el["id"]]

        return {
            "clash_count": clash_count,
            "clashes": clashes,
            "detection_method": "name_duplicate_fallback",
            "detection_method_disclaimer": _CLASH_DISCLAIMER,
            "note": (
                "ifcopenshell.geom not available — using name-duplicate heuristic. "
                "This catches mis-labelled duplicates but NOT real geometric clashes. "
                "Install ifcopenshell[all] (or a build that includes the geometry "
                "extension) for real AABB intersection."
            ),
        }
