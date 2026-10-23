"""Schedule Excel Writer block — roadmap-named shim over app.lib.pm_excel.

The L2 Schedule Engine roadmap expected ``app/blocks/schedule_excel_writer.py``.
The platform's Excel rendering lives in ``app.lib.pm_excel`` (CPM workbook and
cost-loaded L2 schedule workbook). This block is a thin wrapper that exposes
the same capability under the roadmap name.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

from app.core.universal_base import UniversalBlock


class ScheduleExcelWriterBlock(UniversalBlock):
    auto_validate = False
    name = "schedule_excel_writer"
    version = "1.0.0"
    description = "Write CPM or cost-loaded L2 schedule workbooks to Excel"
    layer = 3
    tags = ["domain", "construction", "schedule", "excel", "export"]
    requires = []

    default_config = {
        "output_dir": None,  # None -> tempfile
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        """Write a schedule workbook.

        Modes:
          * ``cpm_output``: dict matching ``CPMOutput`` -> writes a CPM/Gantt sheet
            via ``write_schedule_excel``.
          * ``activities``: list of cost-loaded activity dicts -> writes a full
            cost-loaded L2 schedule via ``generate_cost_loaded_schedule``.
        """
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        cpm_output = data.get("cpm_output") or params.get("cpm_output")
        activities = data.get("activities") or params.get("activities") or []
        meta = data.get("meta") or params.get("meta") or {}
        histogram = data.get("histogram") or params.get("histogram")

        output_dir = self.config.get("output_dir") or tempfile.gettempdir()
        os.makedirs(output_dir, exist_ok=True)

        try:
            if cpm_output:
                from app.lib.pm_excel import write_schedule_excel
                from app.schemas.cpm import CPMOutput

                output = CPMOutput(**cpm_output) if isinstance(cpm_output, dict) else cpm_output
                path = os.path.join(output_dir, "schedule.xlsx")
                hist = None
                if histogram:
                    from app.schemas.cpm import ResourceHistogram

                    hist = ResourceHistogram(**histogram) if isinstance(histogram, dict) else histogram
                written = write_schedule_excel(output, path, histogram=hist)
                return {
                    "status": "success",
                    "file_path": written,
                    "format": "cpm",
                }

            if activities:
                from app.lib.pm_excel import generate_cost_loaded_schedule

                project = meta.get("project", "Project")
                currency = meta.get("currency", "SAR")
                schedule_meta = {"project": project, "currency": currency}
                if meta.get("start_date"):
                    schedule_meta["start_date"] = meta["start_date"]
                if meta.get("cost_basis"):
                    schedule_meta["cost_basis"] = meta["cost_basis"]
                if meta.get("target_milestones"):
                    schedule_meta["target_milestones"] = meta["target_milestones"]

                wb = generate_cost_loaded_schedule(schedule_meta, activities)
                path = os.path.join(output_dir, "l2_schedule.xlsx")
                wb.save(path)
                return {
                    "status": "success",
                    "file_path": path,
                    "format": "l2_cost_loaded",
                    "sheets": wb.sheetnames,
                }

            return {"status": "error", "error": "Provide cpm_output or activities"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": f"Excel write failed: {exc}"}
