"""A BOQ item reference is a LABEL, not a figure.

Retrieve it, display it, cite it -- never reason about it arithmetically.
``D529.2`` is the identifier of a bill item; it is not the number 529.2, and
``A.1.2.3`` is not 1.2 and 3. The grounding gate read them as figures because
they are digits in a retrieved chunk, which put label numbers into the set a
cost claim may be grounded against.

TAXONOMY, deliberately narrow, because material grades and bar sizes look
similar and their numbers ARE real (T12 is a 12 mm bar; C30/37 is a strength
class; M20, B500B likewise), as are plain decimals and clause numbers:

    identifier      letter prefix + dot-separated number   D529.2, D.589.1,
                                                           E101.4, BQ.12.4
                    three or more numeric groups           A.1.2.3, 1.2.3
    still numeric   letters + digits, no dot group         T12, C30, C30/37,
                                                           M20, B500B
    still numeric   one or two plain groups                4.2, 18.75, 142.00
"""
from __future__ import annotations

import re

BOQ_REF_CODE_RE = re.compile(
    r"""(?<![\w.])(?:
            [A-Z]{1,3}\.?\d+(?:\.\d+)+       # D529.2, D.589.1, A.1.2.3, E101.4
          | \d+(?:\.\d+){2,}                 # 1.2.3
        )(?![\w.])""",
    re.VERBOSE,
)


def is_ref_code(token: str) -> bool:
    """True when ``token`` is a BOQ item reference in its entirety."""
    return bool(BOQ_REF_CODE_RE.fullmatch((token or "").strip()))


def find_ref_codes(text: str) -> list[str]:
    """Every BOQ item reference in ``text``, in order, deduplicated.

    The retrieval layer labels a chunk with these so a consumer can tell an
    identifier from a figure without re-deriving the taxonomy.
    """
    seen: dict[str, None] = {}
    for m in BOQ_REF_CODE_RE.finditer(text or ""):
        seen.setdefault(m.group(0), None)
    return list(seen)


def strip_ref_codes(text: str) -> str:
    """``text`` with item references blanked, for callers that extract NUMBERS.

    A space replaces each code so neighbouring words never fuse ("D529.2 m" ->
    " m"). Everything else, including the row's real figures, is untouched.
    """
    return BOQ_REF_CODE_RE.sub(" ", text or "")
