# BOQ Standard Units of Measurement — CESMM4 (Civil/Infra) + POMI (Buildings)

Authoritative unit-of-measurement reference so the platform can determine the
CORRECT unit for any BOQ line item from its work type — independent of how badly
a scanned PDF or OCR mangles the printed unit. When an extracted unit contradicts
the item's work type, trust this reference and flag the extracted unit as
suspect.

Sources: CESMM4 (Civil Engineering Standard Method of Measurement, 4th ed., ICE)
for civil/infrastructure works; POMI (Principles of Measurement International,
RICS 1979) for building works.

Unit abbreviations: `m3` cubic metre (volume) · `m2` square metre (area) ·
`m` / `LM` linear (running) metre · `nr`/`no` number (enumerated) ·
`t` tonne / `kg` kilogram (weight) · `ha` hectare · `sum`/`item` lump sum.

---

## Part 1 — CESMM4 work classes (civil / infrastructure BOQs)

KSA civil-infrastructure bills of this class are CESMM4 (items carry refs like `D999.x`).

| Class | Work | Primary unit(s) |
|---|---|---|
| A | General items (method-related & time-related charges) | sum, nr |
| B | Ground investigation (trial holes, boreholes, sampling) | nr, m, m3 |
| C | Geotechnical & other specialist processes (grouting, diaphragm walls) | m2, m3, nr, m |
| D | Demolition & site clearance | nr, m2, m3, sum |
| **E** | **Earthworks** — excavation, filling, backfill, disposal | **m3** (surface prep m2; areas ha) |
| **F** | **In-situ concrete** | **m3** |
| **G** | **Concrete ancillaries** | **formwork m2 · reinforcement bar t · fabric/mesh m2 · joints m · inserts nr** |
| H | Precast / prestressed concrete units | nr, m, t |
| **I** | **Pipework — pipes** (laying, by bore & depth band) | **m (LM)** |
| **J** | **Pipework — fittings & valves** | **nr** |
| **K** | **Pipework — manholes & pipework ancillaries** | **nr** (some m) |
| **L** | **Pipework — supports & protection; ancillaries to laying & excavation** (beds, surrounds, extra excavation) | **m3, m2, m** |
| **M** | **Structural metalwork** | **t** (erection nr) |
| N | Miscellaneous metalwork | t, kg, m, nr |
| O | Timber | m3, m2, m, nr |
| P | Piles (bored, driven, cast in place) | nr, m |
| Q | Piling ancillaries | nr, m, sum |
| **R** | **Roads & pavings** — sub-base/roadbase m3 or m2, surfacing/asphalt m2, kerbs/edgings m | **m2, m, m3, nr** |
| S | Rail track | m, nr, t |
| T | Tunnels | m, m3, m2, nr |
| **U** | **Brickwork, blockwork & masonry** | **m2** (bands/copings m; isolated nr) |
| **V** | **Painting** | **m2** (narrow widths m) |
| **W** | **Waterproofing / tanking / membranes** | **m2** |
| X | Miscellaneous work (fences, gates, drainage to structures) | m, m2, nr, sum |
| Y | Sewer & water main renovation & ancillary works | m, nr, sum |
| Z | Simple building works incidental to civils | m2, m3, m, nr |

---

## Part 2 — POMI sections (building BOQs)

Building bills (tower, residential and fit-out bills) follow
POMI / building-SMM conventions.

| Section | Trade | Primary unit(s) |
|---|---|---|
| A | General requirements / preliminaries | sum, item |
| B | Site work — excavation, disposal, filling, hardcore, piling | excavation/fill **m3** · site clearance m2/ha · hardcore m3 or m2 · piling m/nr |
| C | Concrete work | in-situ concrete **m3** · formwork **m2** · bar reinforcement **t** · mesh m2 |
| D | Masonry (br/block/stone) | **m2** (thick/mass m3; bands & copings m) |
| E | Metalwork | structural **t** · misc nr/m/kg |
| F | Woodwork / carpentry & joinery | timber runs m · boarding m2 · doors & units nr |
| G | Thermal & moisture protection | waterproofing/tanking/insulation **m2** · DPC m or m2 |
| H | Doors & windows | **nr** (enumerated) · glazing m2 |
| J | Finishes — plaster/render, paint, tiling, flooring, ceilings | **m2** (skirtings, borders m) |
| K | Accessories / specialties (toilet accessories, mirrors, signage) | **nr** |
| L | Equipment | nr, sum |
| M | Furnishings (loose furniture, blinds) | **nr** |
| N | Special construction | sum, nr |
| P | Conveying systems (lifts, escalators) | **nr** |
| Q | Mechanical engineering (HVAC, plumbing) | plant **nr** · ducts/pipes **m** · systems sum |
| R | Electrical engineering | fittings/points **nr** · cable/conduit/containment **m** · systems sum |

---

## Part 3 — OCR-robust keyword → unit lookup (use this to validate extracted units)

If a line-item description contains these terms, the unit SHOULD be:

- **m3 (volume):** excavation, excavate, cut, dig, earthwork, backfill, back
  filling, filling, fill, disposal, cart away, muck away, hardcore (by volume),
  concrete, RCC, blinding, mass concrete, grade C/M, screed (structural),
  sub-base / roadbase (by volume), pipe bedding / surround / haunch / protection,
  imported fill, selected fill
- **m2 (area):** formwork, shuttering, falsework, fabric/mesh reinforcement,
  blockwork, brickwork, masonry, blockwall, cladding, plaster, render, skim,
  paint, painting, decoration, tiling, screed (finish), flooring, floor finish,
  ceiling, gypsum board, partition, waterproofing, tanking, membrane, DPM,
  insulation, surfacing, asphalt, bituminous, wearing course, interlock paving,
  paving, geotextile, site clearance, topsoil strip
- **m / LM (linear):** pipe, pipework, pipe laying, gravity sewer, foul/waste
  water pipe, rising main, culvert (linear), duct/ducting (linear), cable
  (linear), conduit, containment, trunking, kerb, edging, channel, coping band,
  skirting, handrail, balustrade, DPC (linear), string course, expansion joint
- **nr / no (number):** manhole, chamber, inspection chamber, catchpit, gully,
  headwall, valve, hydrant, fitting, bend, junction, tee, connection, door,
  window, ironmongery, sanitary fitting, WC, basin, sink, mirror, accessory,
  light fixture, socket, switch, DB, panel, transformer, pump, chiller, AHU,
  manhole cover, tree, plant (single), pile (also m), lift, precast unit
- **t / tonne (weight):** structural steel, steelwork, reinforcement (bar),
  rebar, high-yield steel, plate, RSJ, purlin, tonnage
- **sum / item:** preliminaries, mobilisation, insurance, provisional sum, testing
  & commissioning, dayworks allowance, contingency, method-related charge

### Notes on ambiguous / dual-unit cases
- **Pipe bedding / surround:** measured in **m3** by default (Class L). If the
  section is fixed and stated (e.g. 150 mm bed depth × trench width), it may be
  given per **linear metre (LM)** of pipe run instead — either is valid if the
  basis is stated. Do NOT sum bedding LM into pipe-laying LM (double count).
- **Pipe laying (Class I):** LM by nominal bore AND depth band — the trench
  excavation is priced INTO the per-metre rate (that is why it is depth-banded),
  so there is usually no separate m3 excavation line for the pipe itself.
- **Different diameters are separate items** — never sum pipe LM across bores
  into one "total length"; keep per-diameter.
- **Reinforcement** is weight (t); its supporting concrete is volume (m3);
  its formwork is area (m2) — three different units for one pour.
