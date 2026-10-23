# EVM Thresholds and Traffic Light System

## CPI and SPI — Definitions

**CPI (Cost Performance Index)** = EV ÷ AC
- CPI > 1.0 means cost efficient — earned value exceeds actual cost.
- CPI = 1.0 means on budget.
- CPI < 1.0 means over budget — earned value is less than actual cost.

**SPI (Schedule Performance Index)** = EV ÷ PV (also expressed as BCWP ÷ BCWS)
- SPI > 1.0 means ahead of schedule.
- SPI = 1.0 means on schedule.
- SPI < 1.0 means behind schedule.

Interpretation note: an SPI of 0.8 means the project has earned only 80 cents of value for every dollar planned. An SPI below 1.0 always means the project is behind schedule.

## CPI Traffic Light System

The CPI Traffic Light is the standard executive-dashboard rule:

- **GREEN — CPI > 1.00** — Under Budget, Cost Efficient. Keep doing what works.
- **AMBER — 0.90 ≤ CPI ≤ 1.00** — Slightly Over Budget. Investigate and take action.
- **RED — CPI < 0.90** — Over Budget, Cost Inefficient. Take corrective action now.

The RED threshold is **CPI < 0.90**. Anything below 0.90 is RED.

The AMBER band is **0.90 ≤ CPI ≤ 1.00** (inclusive on both ends). A CPI of exactly 0.90 is AMBER. A CPI of exactly 1.00 is AMBER.

The GREEN band is **CPI > 1.00** (strictly greater than 1.00).

### Worked Examples

- CPI of 0.85 → RED (below the 0.90 threshold).
- CPI of 0.91 → AMBER (above 0.90, within the 0.90-1.00 band).
- CPI of 0.93 → AMBER.
- CPI of 0.95 → AMBER.
- CPI of 1.05 → GREEN (above 1.00).
- CPI of 0.72 → RED. Action required: forensic cost-variance analysis and a recovery plan. The project is materially overspending against earned value.

## SPI Traffic Light System

- **GREEN — SPI > 1.00** — Ahead of Schedule. Maintain momentum.
- **AMBER — 0.90 ≤ SPI ≤ 1.00** — Slightly Behind. Review plan and recover.
- **RED — SPI < 0.90** — Behind Schedule, At Risk. Take corrective action now.

## Combined CPI / SPI Interpretation

Both indices together describe project health:

- CPI > 1.0 AND SPI > 1.0 → Ahead and under budget. Great performance.
- CPI < 1.0 AND SPI > 1.0 → Ahead but over budget. Monitor costs.
- CPI > 1.0 AND SPI < 1.0 → On budget but behind. Recover schedule.
- CPI < 1.0 AND SPI < 1.0 → Behind and over budget. Take corrective action immediately.

## TCPI — To-Complete Performance Index

TCPI = (BAC − EV) ÷ (BAC − AC)

TCPI is the cost efficiency required for the remaining work to complete on budget. If TCPI > 1.0, the project must perform better than it has been; if TCPI < 1.0, the project can afford to underperform slightly and still hit BAC.

### Worked Example — Section 20 Overrun

A project with BAC = $50M, EV = $28.5M, AC = $32M:

- Remaining budget = BAC − AC = $50M − $32M = $18M
- Remaining work = BAC − EV = $50M − $28.5M = $21.5M
- TCPI = $21.5M ÷ $18M = **1.19**

A TCPI of 1.19 means the project must achieve 19% better cost efficiency for the rest of the work to deliver on budget. This is a meaningful signal — a TCPI much above 1.0 indicates that on-budget delivery is unrealistic and EAC should be revised.

## Forecasting

- **EAC (Estimate at Completion)** = BAC ÷ CPI — assumes future performance mirrors past.
- **ETC (Estimate to Complete)** = EAC − AC — remaining cost to finish.
- **VAC (Variance at Completion)** = BAC − EAC — projected over/under run.

## Key Rule

You cannot improve what you do not measure. The CPI/SPI traffic light is the simplest executive control on cost and schedule health.
