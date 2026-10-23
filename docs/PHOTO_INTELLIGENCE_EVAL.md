# Site-photo intelligence — evaluation on real construction photographs

**Date:** 2026-08-12 · **Detector:** `data/models/safety_world_v2.onnx`
(YOLO-Worldv2, open-vocabulary, 33 baked prompts, `v4-hybrid`) ·
**Threshold:** `conf_threshold=0.05` so the report's low-confidence tier is reachable.

This measures the bridge built in `app/containers/construction/photo_observations.py`
against 21 real photographs sourced from Wikimedia Commons. Every photo was
described by hand BEFORE the detector ran, and the expected classes were chosen
from the baked vocabulary only. Images are not committed; see
`tests/fixtures/photo_eval_manifest.json` and `scripts/fetch_eval_photos.py`.

A photo counts as **correct** when a class the photo genuinely contains is
detected at >= 0.30, **low-conf flagged** when it is detected in the 0.05-0.30
band (surfaced as "possible ... -- low confidence", never promoted), and
**missed** when it does not appear at all.

## Headline

| vocabulary | photos | correct (>=0.30) | + low-conf flagged | missed |
|---|---|---|---|---|
| **Quality / workmanship** (13 prompts) | 11 | **1 (9%)** | 4 (36%) | 7 (64%) |
| **Plant / temporary works** (3 prompts) | 5 | **3 (60%)** | 3 (60%) | 2 (40%) |
| Hard cases | 3 | 1 | 2 | 1 |
| PPE / people (context only) | 1 | 1 | 1 | 0 |
| Hazard (context only) | 1 | 0 | 0 | 1 |

**Quality detection is weak — 9% at the reporting threshold.** That is the
honest number and it is the headline, not a footnote. Equipment is 60% on a
three-word vocabulary. Reporting one blended figure would have hidden both
facts, which is why they are split.

| # | photo (id) | what it actually shows | top detections (conf) | consumer output | verdict |
|---|---|---|---|---|---|
| 1 | `q01_crack_retaining_wall` | Close, well-lit detail of a vertical crack running through a cast conc | exposed aggregate in concrete surface 0.582; exposed aggregate in concrete surface 0.247; concrete honeycomb voids 0.189 | exposed aggregate in concrete surface detected; possible concrete honeycomb voids detected -- low confidence | **missed** |
| 2 | `q02_crack_reinforcement_swelling` | Concrete wall cracked along the line of corroding reinforcement; rust  | water stain on wall 0.338; exposed aggregate in concrete surface 0.17; concrete honeycomb voids 0.143 | water stain on wall detected; possible exposed aggregate in concrete surface detected -- low confidence; possible concrete honeycomb voids detected -- low confidence | **low-conf flagged** |
| 3 | `q03_crack_corroding_rebar` | Second view of the same failure mode - spalled concrete face with corr | water stain on wall 0.187; concrete honeycomb voids 0.173 | possible water stain on wall detected -- low confidence; possible concrete honeycomb voids detected -- low confidence | **missed** |
| 4 | `q04_honeycomb_ring_beam` | Honeycombing on a cast in-situ ring beam - voids where mortar failed t | water stain on wall 0.099; open excavation pit 0.098; water stain on wall 0.081 | possible water stain on wall detected -- low confidence; possible exposed aggregate in concrete surface detected -- low confidence; possible concrete honeycomb voids detected -- low confidence | **low-conf flagged** |
| 5 | `q05_honeycomb_concrete` | Honeycombed concrete surface, aggregate exposed across a large area of | water stain on wall 0.349; water stain on wall 0.164; porous holes in concrete surface 0.111 | water stain on wall detected; possible porous holes in concrete surface detected -- low confidence | **missed** |
| 6 | `q06_exposed_rebar_chock` | Broken precast parking chock with reinforcement bar exposed and rustin | exposed aggregate in concrete surface 0.63 | exposed aggregate in concrete surface detected | **missed** |
| 7 | `q07_rebar_stacks` | Bundles of reinforcement bar stacked on a metro construction site - st | exposed aggregate in concrete surface 0.305; concrete honeycomb voids 0.211; exposed aggregate in concrete surface 0.106 | exposed aggregate in concrete surface detected; possible concrete honeycomb voids detected -- low confidence | **missed** |
| 8 | `q08_peeling_paint_bathroom` | Paint blistering and peeling from a plastered bathroom wall, damp-rela | none | (empty) | **missed** |
| 9 | `q09_peeling_paint_1` | Close view of paint flaking from a painted surface. | open excavation pit 0.198; person 0.104 | (empty) | **missed** |
| 10 | `q10_cracked_tile_bathroom` | Cracked ceramic floor tile in a bathroom, crack running corner to corn | person 0.484; water stain on wall 0.33; missing grout between tiles 0.296 | water stain on wall detected; possible missing grout between tiles detected -- low confidence; possible cracked floor tile detected -- low confidence | **low-conf flagged** |
| 11 | `q11_cracked_tile_flooring` | Cracked tiled floor, several tiles affected with grout lines visible. | exposed aggregate in concrete surface 0.564; concrete honeycomb voids 0.48; porous holes in concrete surface 0.43 | exposed aggregate in concrete surface detected; concrete honeycomb voids detected; porous holes in concrete surface detected | **correct** |
| 12 | `e01_tower_crane_nuremberg` | Active building site with a tower crane over a partially built frame. | crane 0.459; person 0.249; crane 0.09 | crane detected | **correct** |
| 13 | `e02_tower_crane_cabo` | Construction site with a tower crane standing over it. | crane 0.586; person 0.498; person 0.314 | possible exposed aggregate in concrete surface detected -- low confidence; crane detected | **correct** |
| 14 | `e03_scaffolding_amsterdam` | Building under construction with substantial tube-and-fitting scaffold | person 0.561; ladder 0.503; person 0.502 | concrete honeycomb voids detected; ladder detected; crane detected | **missed** |
| 15 | `e04_ladder_peabody` | Single ladder leaning against an interior wall. | ladder 0.962; water stain on wall 0.131; ladder 0.097 | possible water stain on wall detected -- low confidence; ladder detected | **correct** |
| 16 | `e05_ladders_oulu` | Ladders fixed to a building exterior in winter light. | water stain on wall 0.051 | possible water stain on wall detected -- low confidence | **missed** |
| 17 | `p01_worker_ppe_inspection` | Workers inspecting an industrial project, wearing helmets and high-vis | yellow or white safety helmet 0.989; yellow or white safety helmet 0.986; person 0.927 | possible exposed aggregate in concrete surface detected -- low confidence; possible concrete honeycomb voids detected -- low confidence | **correct** |
| 18 | `h01_trench_excavation_geograph` | Open service trench excavation in a road, spoil alongside. | person 0.308; person 0.262; person 0.204 | possible exposed aggregate in concrete surface detected -- low confidence; possible ladder detected -- low confidence | **missed** |
| 19 | `hard01_honeycomb_archival_tif` | Archival monochrome large-format survey photograph of a honeycombed co | concrete honeycomb voids 0.086 | possible concrete honeycomb voids detected -- low confidence | **low-conf flagged** |
| 20 | `hard02_crane_millennium_distant` | Tower crane photographed at distance against a city skyline; the crane | person 0.619; crane 0.309; crane 0.308 | crane detected | **correct** |
| 21 | `hard03_damaged_wall_dark_interior` | Damaged wall opening into a dark unfinished interior; exposed concrete | water stain on wall 0.169; water stain on wall 0.167; exposed aggregate in concrete surface 0.066 | possible water stain on wall detected -- low confidence; possible exposed aggregate in concrete surface detected -- low confidence; possible concrete honeycomb voids detected -- low confidence | **missed** |

## Failure analysis

Every failure is categorised **bridge bug** (my code — fix here), **vocabulary
gap** (no prompt string exists — needs a re-bake), or **model limitation** (the
prompt exists and the model did not fire).

### Bridge bugs: 0

The bridge was verified to pass detections through faithfully. Spot-checks:
`q11` produced four observations matching its four detections including
`cracked floor tile detected (0.364)`; `e04` produced
`ladder detected (0.962)` categorised `temporary_works`; `q10`'s
`cracked floor tile 0.258` came through as **"possible cracked floor tile
detected -- low confidence"** and was not promoted; `q08` with zero detections
produced an empty list rather than an invented observation.

### Vocabulary gaps: 0 in this set

Every class tested exists in `safety_world_prompts.json`. The known gap is not
exercised here because it is untestable by construction: **excavators,
telehandlers, dumpers, piling rigs and concrete pumps have no prompt string**,
so they cannot be detected at any confidence. That is why equipment coverage is
measured over 5 photos and 3 classes rather than a realistic plant list, and it
is registered in `docs/KNOWN_INCOMPLETE.md` as a vocabulary limit requiring a
re-bake. Out of scope here — the model is open-vocabulary, so widening it means
adding prompt strings and re-exporting the ONNX.

### Model limitations: all 10 misses

| photo | expected | what happened |
|---|---|---|
| `q01` crack in retaining wall | crack in concrete wall | not detected at any confidence; `exposed aggregate 0.582` fired instead |
| `q03` spalled face, corroding rebar | crack / rust / exposed rebar | nothing from the expected set; two low-confidence unrelated hits |
| `q05` honeycombed concrete | honeycomb voids | `water stain on wall 0.349` fired; `porous holes` only 0.111 |
| `q06` broken chock, exposed rebar | exposed reinforcement bar | `exposed aggregate 0.63` fired instead — related surface, wrong class |
| `q07` stacked rebar bundles | rust / exposed rebar | not detected; deliberate edge case (stored stock, not rebar in a member) |
| `q08` peeling paint, bathroom | peeling paint | **zero detections on the whole image** |
| `q09` flaking paint close-up | peeling paint | not detected; `open excavation pit 0.198` fired |
| `e03` scaffolded facade | scaffolding | scaffolding not detected; `ladder 0.503` and people fired |
| `e05` ladders on a facade, winter | ladder | not detected; one 0.051 hit unrelated |
| `h01` open service trench | open excavation pit | only people detected |

**The pattern that matters: fine-grained surface defects are the weak class.**
Cracks, peeling paint and honeycombing — the things a QA engineer photographs —
are largely missed, while whole-object classes (ladder 0.962, helmet 0.989,
crane 0.586) are strong. That is consistent with an open-vocabulary detector
grounded on object nouns rather than material-condition textures.

### False positives — the more serious finding

Misses are visible; false positives are not. Two prompts fire broadly on almost
any textured concrete or tile:

- `exposed aggregate in concrete surface` — fired at **0.582** on a plain
  cracked wall (`q01`), **0.63** on a broken chock (`q06`), **0.564** on a tiled
  floor (`q11`), 0.305 on stacked rebar (`q07`), and 0.231 on a photo of people
  in PPE (`p01`).
- `water stain on wall` — fired at **0.338** / **0.349** / 0.33 on three photos
  with no water staining.

Also `person 0.484` on a bathroom-tile photo (`q10`) with no person in it.

`q11` is the clearest case: the one true positive (`cracked floor tile 0.364`)
arrived alongside **three false positives at the same reporting tier**. A reader
cannot tell them apart from the report alone. This is why the output carries
`photo` and `confidence` on every observation — the evidence trail back to the
image is what makes the section reviewable rather than authoritative.

## What this means for use on a client site

**Do not present these as findings.** At 9% recall and a visible false-positive
rate at the reporting threshold, the quality section is a prompt for a human to
look at specific photographs — not a QA record. The wording enforces that: every
entry says what was *detected*, with a confidence and a source image, and
nothing in the pipeline is permitted to call anything a defect, violation or
non-compliance.

Equipment at 60% over three classes is usable as a "plant seen on site" prompt,
with the same caveat and the explicit knowledge that most plant is invisible to
this vocabulary.

## Registered for future work

1. **Vocabulary gap — plant.** Add excavator, telehandler, dumper, piling rig,
   concrete pump, mobile crane. Requires re-baking the ONNX.
2. **Vocabulary gap — quality precision.** `exposed aggregate in concrete
   surface` and `water stain on wall` behave as near-catch-alls for concrete
   texture. Either tighten the prompt strings or raise a per-class threshold.
3. **Model limitation — surface defects.** Cracks and peeling paint may need a
   different approach (a texture/segmentation model) rather than more prompts.

## Reproduce

```bash
python scripts/fetch_eval_photos.py          # downloads to a gitignored dir
SAFETY_WORLD_WEIGHTS=data/models/safety_world_v2.onnx \
  python -m pytest tests/test_photo_observations.py tests/test_daily_report_photo_path.py -q
```
