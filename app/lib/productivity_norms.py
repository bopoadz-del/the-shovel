"""Generic construction productivity norms — sourced kernel DATA (S-1b).

ORDERS_28OCT_0900Z: when a brief states a *quantity* for an activity but no
productivity *rate*, the schedule estimator may fall back to a GENERIC,
per-trade productivity norm so the activity gets a kernel-computed duration
instead of UNKNOWN. These are the rules this table lives under:

  * GENERIC ONLY. Every row is an industry order-of-magnitude planning norm
    keyed to a trade (excavation, concrete, blockwork, …). There are NO
    client names, NO project names, NO probe-specific values, and NO
    activity strings copied from any test or corpus. The keywords are the
    plain vocabulary of the trades, nothing case-specific. (NO HARDWIRING,
    AGENTS.md standing order.)
  * SOURCED. Every row carries a ``source`` naming the provenance of the
    figure, and the duration the estimator derives is labelled
    ``estimated — kernel …, norm <source>`` so a reader always sees the
    number is a generic default, not their own data.
  * A STATED RATE ALWAYS WINS. This table is consulted ONLY when the input
    gave a quantity and no rate. A rate stated in the brief is used verbatim
    and never overridden by a norm (see schedule.py).
  * NO NORM → UNKNOWN. A trade this table does not cover yields no estimate;
    the activity stays UNKNOWN (duration None) and the schedule is reported
    incomplete. The table never invents a rate for an unknown trade.

The rate strings are per-gang-day production rates in the exact text form the
kernel ``productivity_manpower_duration`` parses (``"<n> <unit> per gang-day"``):
paired with the stated quantity they yield a duration in working days. The
figures are deliberately round, generic planning values — provenance below.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Provenance for the generic figures below. These are standard construction
# estimating / planning references from which order-of-magnitude single-gang
# daily outputs are drawn; the rows carry generic planning values, never a
# project-measured rate.
_SOURCE = "generic construction planning norm (industry single-gang daily output; not project-specific)"

# Each row: trade keywords (plain trade vocabulary) + a generic per-gang-day
# production rate (kernel rate text) + a source. Ordered most-specific first so
# a compound trade ("reinforced concrete") matches before a broader one.
_NORMS: List[Dict[str, object]] = [
    {"keywords": ["site clearance", "clear site", "site preparation"], "rate": "500 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["tree removal", "remove tree", "removal of tree", "tree felling", "grubbing"], "rate": "20 nr per gang-day", "source": _SOURCE},
    {"keywords": ["reinforced concrete", "rc "], "rate": "35 m3 per gang-day", "source": _SOURCE},
    {"keywords": ["blinding concrete", "mass concrete", "concrete pour", "concrete placing", "concrete to", "concrete"], "rate": "40 m3 per gang-day", "source": _SOURCE},
    {"keywords": ["reinforcement", "rebar", "steel fixing", "bar bending"], "rate": "1.5 t per gang-day", "source": _SOURCE},
    {"keywords": ["formwork", "shuttering", "falsework"], "rate": "20 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["blockwork", "block wall", "masonry"], "rate": "15 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["brickwork", "brick wall"], "rate": "12 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["plaster", "render", "skim"], "rate": "25 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["paint", "decorat"], "rate": "80 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["tiling", "floor finish", "screed"], "rate": "25 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["asphalt", "carriageway", "road pavement", "paving", "kerb"], "rate": "300 m2 per gang-day", "source": _SOURCE},
    {"keywords": ["fence", "fencing", "chain link"], "rate": "40 m per gang-day", "source": _SOURCE},
    {"keywords": ["pipe laying", "pipelaying", "drainage", "culvert", "ducting"], "rate": "30 m per gang-day", "source": _SOURCE},
    {"keywords": ["piling", "bored pile", "driven pile"], "rate": "8 nr per gang-day", "source": _SOURCE},
    {"keywords": ["demolition", "demolish", "breaking out", "break out", "break up"], "rate": "30 m3 per gang-day", "source": _SOURCE},
    {"keywords": ["bulk excavation", "excavation", "excavate", "earthwork", "cut and fill"], "rate": "200 m3 per gang-day", "source": _SOURCE},
    {"keywords": ["backfill", "filling", "compaction"], "rate": "250 m3 per gang-day", "source": _SOURCE},
]


def norm_rate_for(activity_name: str) -> Optional[Dict[str, str]]:
    """Return ``{"rate", "source"}`` for the generic productivity norm whose
    trade matches ``activity_name``, or ``None`` when no generic trade matches.

    Matching is by plain keyword containment on the lower-cased name, most-
    specific row first. It never invents a rate for an unmatched trade — the
    caller then leaves the activity UNKNOWN.
    """
    name = (activity_name or "").lower().strip()
    if not name:
        return None
    for row in _NORMS:
        for kw in row["keywords"]:  # type: ignore[union-attr]
            if kw in name:
                return {"rate": str(row["rate"]), "source": str(row["source"])}
    return None
