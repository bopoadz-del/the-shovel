# ACI 318-19 — SI examples and US structural-concrete review

The existing SI tables below are calculated examples for their stated assumptions,
not universal design defaults. ACI 318 is a structural-concrete code; the separate
ACI 209R/curing examples are not ACI 318 acceptance requirements. Saudi SBC 304
adaptations and US building-code adoption both require the applicable edition and
local provisions. The added US review section cites public sources checked on
2026-11-06; it does not change the existing numerical tables.

## Modulus of elasticity and modulus of rupture (normal-weight concrete)

Ec = 4700·√f'c (ACI 318-19 Eq. 19.2.2.1b, SI); fr = 0.62·√f'c (Eq. 19.2.3.1).

| f'c (MPa, cylinder) | Ec (MPa) | fr (MPa) |
|---|---|---|
| 20 | 21,019 | 2.77 |
| 25 | 23,500 | 3.10 |
| 28 | 24,870 | 3.28 |
| 30 | 25,743 | 3.40 |
| 32 | 26,587 | 3.51 |
| 35 | 27,806 | 3.67 |
| 40 | 29,725 | 3.92 |
| 45 | 31,529 | 4.16 |
| 50 | 33,234 | 4.38 |
| 60 | 36,406 | 4.80 |

## Minimum thickness of non-prestressed one-way solid slabs (ACI 318-19 Table 7.3.1.1)

For slabs not supporting or attached to partitions likely to be damaged by large 
deflections. fy = 420 MPa (for other fy multiply by 0.4 + fy/700).

| Span (m) | Simply supported (mm) | One end continuous (mm) | Both ends continuous (mm) | Cantilever (mm) |
|---|---|---|---|---|
| 3 | 150 | 125 | 107 | 300 |
| 3.6 | 180 | 150 | 129 | 360 |
| 4 | 200 | 167 | 143 | 400 |
| 4.8 | 240 | 200 | 171 | 480 |
| 5 | 250 | 208 | 179 | 500 |
| 6 | 300 | 250 | 214 | 600 |
| 7.2 | 360 | 300 | 257 | 720 |
| 8 | 400 | 333 | 286 | 800 |

## Tension lap splice length, Class B (ACI 318-19 §25.4.2 / §25.5.2)

fy = 420 MPa, confinement term (cb+Ktr)/db = 1.5, uncoated bottom bars. The bar-size factor psi_s is taken as 1.0 for all sizes, so values for bars of 19 mm and smaller are conservative (ACI permits 0.8).

| Bar Ø (mm) | f'c 25 MPa (mm) | f'c 30 MPa (mm) | f'c 35 MPa (mm) | f'c 40 MPa (mm) |
|---|---|---|---|---|
| 10 | 662 | 604 | 559 | 523 |
| 12 | 794 | 725 | 671 | 628 |
| 16 | 1059 | 967 | 895 | 837 |
| 20 | 1324 | 1208 | 1119 | 1046 |
| 25 | 1654 | 1510 | 1398 | 1308 |
| 32 | 2118 | 1933 | 1790 | 1674 |

## Concrete shear strength of beams, φVc (ACI 318-19 §22.5, φ = 0.75)

Simplified Vc = 0.17·λ·√f'c·bw·d (members with at least minimum shear reinforcement); longitudinal steel ratio ρl = 1%.

| b × d (mm) | f'c 25 MPa (kN) | f'c 30 MPa (kN) | f'c 35 MPa (kN) | f'c 40 MPa (kN) |
|---|---|---|---|---|
| 250 × 450 | 71.7 | 78.6 | 84.9 | 90.7 |
| 300 × 500 | 95.6 | 104.8 | 113.2 | 121.0 |
| 300 × 600 | 114.8 | 125.7 | 135.8 | 145.2 |
| 400 × 700 | 178.5 | 195.5 | 211.2 | 225.8 |
| 500 × 900 | 286.9 | 314.3 | 339.4 | 362.9 |

## Drying shrinkage and strength gain (ACI 209R)

Hyperbolic shrinkage, ultimate 780 microstrain, time constant 35 days.

| Age (days) | Shrinkage (microstrain) |
|---|---|
| 7 | 130 |
| 14 | 223 |
| 28 | 347 |
| 56 | 480 |
| 90 | 562 |
| 180 | 653 |
| 365 | 712 |

| Target fraction of 28-day strength | Curing time (days) |
|---|---|
| 50% | 3.5 |
| 60% | 4.9 |
| 70% | 6.9 |
| 75% | 8.3 |
| 90% | 15.3 |


## US application — design, detailing, durability, and acceptance

This independently authored addition uses **ACI 318-19** as the selected reference
edition. The public 2021 IBC Chapter 35 sample references that edition; this is not
a claim that 318-19 is the latest publication or the adopted edition everywhere.
The ACI public preview establishes the following review locations, not every
exception or numerical table. [Sources 1–2]

| Reference | Own-word review focus |
|---|---|
| **Chapters 4–6** | Establish system performance, loads/combinations, analysis, and the full load path before member checks. |
| **Chapters 7–13** | Apply the relevant slab, beam, column, wall, diaphragm, and foundation provisions rather than extending a beam shortcut to every member. |
| **17.6–17.10; 26.7** | Coordinate anchor tension/shear failure modes, interaction, edge/spacing conditions, seismic use, and installation evidence. |
| **Chapter 18** | Determine the seismic system and detailing obligations; adequate gravity strength alone does not establish ductile performance. |
| **19.3; 20.5** | Assign exposure and coordinate mixture durability with reinforcement protection. Strength, water/cementitious ratio, cover, and detailing are not interchangeable measures. |
| **21.2; Chapter 22** | Use the applicable strength-reduction rule and failure mechanism. One-way and two-way shear are different checks. |
| **24.2–24.4; 25.2–25.7** | Check service behaviour, spacing, development, splices, and confinement; an illustrative lap table does not cover every bar position/coating/confinement case. |
| **26.4–26.6; 26.11–26.13** | Connect design documents to materials, execution, temporary support, inspection, and hardened-concrete acceptance. |

### Publicly sourced durability example

ACI's chloride-exposure FAQ confirms that **Exposure C2** involves moisture plus
an external chloride source; it does not offer a relaxed mixture just because a
particular moist soil contains less salt than seawater. For that class, the
reported maximum **w/cm is 0.40**, with minimum specified strength **5,000 psi**
(on the cited US-customary FAQ basis; do not silently substitute a rounded unit
conversion for the governing edition's SI designation). Other exposure classes and
concurrent exposures require their own assessment. [Sources 3–4]

### Publicly sourced strength acceptance — not a universal early-age percentage

ACI's FAQ identifies **19.2.1.3**: the default specified-strength age is **28 days**
unless the construction documents specify otherwise. Early results can inform
progress but do not create a universal seven-day percentage acceptance rule.
[Source 3]

For **26.12.3.1**, both conditions apply: the mean of each consecutive set of
**three strength tests** reaches at least the specified strength; an individual
test is no more than **3.5 MPa (500 psi)** below it when the specified strength is
at most **35 MPa (5,000 psi)**, or no more than **10%** below it at higher strength.
A strength test is not one cylinder: the cited FAQ identifies the required
averaging of at least **two 150 × 300 mm** or **three 100 × 200 mm** cylinders
under **26.5.3.2/26.12.1.1**. Sampling, curing, test age, and method still matter.
A low result calls for the applicable evaluation process, not automatic acceptance
from an unrelated cube or field-cured specimen. [Source 3]

### Remaining applicability and scope

The adopted edition/local amendments, actual exposure, seismic system, material
properties, anchor configuration, and field acceptance decisions remain
unspecified: **confirm against the official code**. The public sources resolve
the examples above. This addition does not independently recertify the older
illustrative tables or make ACI 209R model constants mandatory code values.

## Sources for the added US scope

1. [ACI — official 318-19 preview and contents](https://www.concrete.org/Portals/0/Files/PDF/Previews/318-19_preview.pdf).
2. [ICC — 2021 IBC referenced standards](https://shop.iccsafe.org/media/wysiwyg/material/3000S21-Sample.pdf).
3. [ACI — strength-age, test definition, and 26.12.3.1 acceptance FAQ](https://www.concrete.org/frequentlyaskedquestions.aspx?faqid=13).
4. [ACI — chloride exposure and C2 limits](https://www.concrete.org/frequentlyaskedquestions/faqid/904.aspx).

Related: [ASCE 7](us_asce_7.md), [ASTM practices/tests](us_astm_construction_standards.md),
and [IBC inspection coordination](us_ibc_irc.md).
