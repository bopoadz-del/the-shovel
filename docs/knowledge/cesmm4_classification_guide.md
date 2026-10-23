# CESMM4 Classification Reference Guide

## What CESMM4 Is

CESMM4 is the Civil Engineering Standard Method of Measurement, 4th edition — the UK / international convention for classifying construction quantities in Bills of Quantities (BOQs). It groups items by category so that quantities can be measured and priced consistently.

## CESMM4 References Are Classifications, Not Line Items

The core principle that drives every BOQ rate lookup:

> A CESMM4 reference code (e.g. `D999.46`) is a **classification reference**, not a priced line item. Rates attach to specific **described items** under each classification — diameter, depth, material, condition — not to the classification code itself.

This means you cannot ask "what is the rate for D999.14?" and expect a single number. D999.14 is a category. The rate depends on the specific item described under that category.

### Why This Matters

Treating a CESMM4 code as a line item leads to fabricated rates. The correct mental model:

- CESMM4 code = category header
- Described item = specific quantity (e.g. "300mm HDPE culvert at 1-1.5m depth")
- Rate = SAR/m attached to the described item

## How To Look Up a Rate

1. Identify the specific item description: diameter, material, depth band, condition.
2. Find that description within the CESMM4 classification group in the BOQ.
3. Read the rate against the description.

Never use the CESMM4 code alone as a cost lookup key.

## Reference Infrastructure BOQ — D999.x Family

The D999.x series in the reference KSA infrastructure BOQ covers stormwater, drainage, and pipeline protection works. All rates are in SAR (Saudi Riyal). Most items are measured per linear metre, but not all — read the unit column.

### Verified Described Items

These three items are known from the verified BOQ page:

- **D999.46** — Protection of existing wastewater pipeline, 600mm diameter, average depth 5-6m. Rate: **10,317.00 SAR/m**. Unit: linear metre.
- **D999.47** — Protection of existing wastewater pipeline, 250mm diameter, average depth 4-6m. Rate: **10,317.00 SAR/m**. Unit: linear metre.
- **D999.48** — Protection of existing potable water pipeline, 160mm diameter, average depth 1-1.5m. Rate: **364.00 SAR/m**. Unit: linear metre.

### Unpriced Codes in the Series

Other D999.x classification codes (for example D999.14) do not carry their own unit rate. They are classification references only. Rates are attached to specific described items within each classification.

## Currency

The reference KSA infrastructure BOQ is priced in SAR (Saudi Riyal). It does not provide rates in USD, AED, or ECU. Any USD or AED conversion is a function of the exchange rate at valuation time and is not part of the source BOQ data.

## Unit Conventions

Most D999.x items are measured per **linear metre** (m). Always read the unit column of the BOQ row rather than assume a unit from the item-code prefix. Some items in the same classification family may be measured per number (Nr) — for example, allowance items such as end caps or specific chamber types.

## Common Misconceptions

- **"Each CESMM4 code has its own rate"** — wrong. Rates attach to described items.
- **"D999.x is always measured per linear metre"** — wrong. Most are, but some (allowance items, chamber types) are per Nr.
- **"BOQ rates are in AED"** — wrong. The reference BOQ rates are in SAR.
- **"Rate per kilogram or per tonne can be derived"** — wrong unless the BOQ row specifies a mass unit. D999.x items are measured by length.
