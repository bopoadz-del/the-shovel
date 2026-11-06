# AISC 360 — structural steel design, stability, and quality evidence

Original guidance for **ANSI/AISC 360-22**, checked **2026-11-06**. The directly
reviewed specification copy is a public steel-industry mirror; AISC's official
scope and **January 2025/September 2026 errata** were also checked. Distinguish
mandatory specification provisions from commentary/user guidance. This edition
is not automatically the one referenced by the adopted building code; the 2021
IBC, for example, references 360-16. [Sources 1–4]

## Design basis and responsibilities

**A1–A3** establish scope, references, and material requirements. **A4–A5**
address design documents/specifications and approvals. A steel grade or mill
certificate establishes neither complete member capacity nor approval of the
connection/erection design. Record the adopted standard, material, geometry,
loads, connection responsibilities, and required evidence. [Source 2]

**B2–B3.2** distinguish loads/combinations and design methods. LRFD checks
factored demand against reduced nominal resistance; ASD checks its compatible
required strength against nominal resistance divided by the appropriate safety
factor. Do not call ASD demand simply an arbitrary unfactored load or interchange
phi and omega in the same check. [Source 2]

**C1–C3** require a coherent stability analysis and compatible strength evaluation.
Member buckling, frame effects, bracing, imperfections, stiffness assumptions,
and connection behaviour cannot be replaced by a gross-section stress check.
[Source 2]

## Verified strength example — tension is not one limit state

Under **D2(a), Equation D2-1**, gross-section yielding uses **Pn = Fy Ag**,
with **phi = 0.90** or **omega = 1.67** for the respective design method.
Under **D2(b), Equation D2-2**, net-section rupture uses **Pn = Fu Ae**,
with **phi = 0.75** or **omega = 2.00**. Check both as applicable; **D3**
addresses effective net area. Bolt holes and load transfer can therefore matter
even when gross-section yielding looks adequate. These are verified factors,
not universal factors for all steel limit states. [Source 2]

## Coordinated review map

| 360-22 reference | Own-word review focus |
|---|---|
| **B4; Chapters D–H** | Establish section properties and local slenderness, then examine tension, compression, flexure, shear, and combined actions. |
| **Chapter I** | Use composite-member provisions only with the relevant steel/concrete interaction and construction assumptions. |
| **Chapters J–K** | Coordinate welded/bolted connections, connected parts, and applicable HSS/box connection limit states with member forces. |
| **Chapter L** | Check service behaviour rather than treating strength adequacy as complete performance. |
| **Chapter M; B8** | Coordinate fabrication/erection and dimensional tolerances with the design assumptions. |
| **N1–N8** | Separate fabricator/erector quality control, independent assurance, qualified inspection/NDT, nonconformance, and coating evidence. Quality is Chapter N, not an invented Appendix Q. |

The building code's special-inspection obligations and any applicable seismic
standards remain separate interfaces. Ordinary 360 member checks alone do not
establish seismic system detailing or connection prequalification. [Sources 1–2]

## Errata and remaining decisions

AISC's **23 September 2026** list is additional to earlier corrections. It includes
reference, symbol, unit, and numerical corrections; retain the relevant errata
with the chosen printing rather than relying on an old uncorrected PDF.
[Sources 3–4]

The actual adopted edition, applicable seismic package, material grade, bracing,
connection details, and QA/inspection conditions remain design/authority decisions:
**confirm against the official code**. No section capacity or bolt/weld value is
invented for an unspecified configuration.

## Sources

1. [AISC — specification scope and ASD/LRFD status](https://www.aisc.org/aisc/publications/current-standards/aisc-360/).
2. [Public steel-industry mirror — ANSI/AISC 360-22, August 2022](https://welovesteelconstruction.ssi-steel.com/wp-content/uploads/2024/08/Specification-AISC-360-22_Specification-for-Structural-Steel-Buildings.pdf).
3. [AISC — official revision/errata index](https://www.aisc.org/aisc/publications/revisions-and-errata/).
4. [AISC — September 2026 additional errata](https://www.aisc.org/media/kgjb33g5/errata_360-22_09232026.pdf).

Related: [ASCE 7](us_asce_7.md), [ASTM](us_astm_construction_standards.md), and
[IBC inspection interfaces](us_ibc_irc.md).
