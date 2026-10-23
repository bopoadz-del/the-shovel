# Interior Finishes Take-off and Estimate — the Process Contract

How interior quantities and resource costs are derived on this platform.
This document states the PROCESS. It deliberately contains **no heights, no
productivity figures, no rates**: those are project variables that change
per project and even per floor, and they must come from the project's own
documents, the project facts store, or the operator — never from a global
default.

## The process

1. **Measure** — the drawing take-off gives, per floor: net room floor area,
   summed room perimeter, and room count (`drawing_qto`).
2. **Derive interior quantities** — `interior_finishes_takeoff` computes:
   - floor screed / floor tiling / ceiling finish areas = net floor area;
   - skirting length = perimeter − (doors × door width);
   - wall finish area = perimeter × floor-to-ceiling height − door area −
     window deductions;
   - blockwork element area = wall face area × shared-wall fraction
     (internal walls are shared between two rooms; the fraction is a
     declared modelling parameter).
   The calculator REFUSES when the height is missing — it names the
   variable to supply instead of assuming one.
3. **Split resources** — `resource_line_cost` computes, per line:
   labour = quantity ÷ daily output × trade day rate; plant = declared
   fraction of labour; material = quantity × material rate. Productivity
   and day rate are required inputs; a missing material rate is reported as
   NO SOURCED RATE, never as zero and never invented.
4. **Source the variables, per project** — in this order:
   - the project's own documents (sections and door schedules for heights
     and openings; priced bills for rates);
   - the project facts / memory store (values the operator confirmed for
     THIS project — heights per floor, agreed norms, market rates);
   - the curated knowledge base's published references (composite rates,
     trade day rates, material prices) — cited by source;
   - the operator, asked directly, when none of the above carries the value.
5. **Declare** — every deliverable restates the variables used and their
   source. An estimate whose basis section cannot name where each variable
   came from is not finished.

## Why no defaults

A default height or norm silently wrong for one floor poisons every
quantity derived from it, and the error is invisible because nothing asked
for the value. A refusal that names the missing variable costs one
question; a wrong hardwired fact costs a re-measure. The calculators
therefore fail loudly on missing required variables, and agents ask.
