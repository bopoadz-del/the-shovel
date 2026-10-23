# ACI 318-19 design values (SI / metric) — as used in KSA practice

Metric (SI) design values computed by the platform's construction calculator from 
ACI 318-19 and ACI 209R/308. ACI 318 in SI units is the basis the Saudi Building 
Code concrete provisions (SBC 304) adapt; confirm the SBC edition in force for a 
permit. Every value in these tables is the calculator's own output for the stated inputs.

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

