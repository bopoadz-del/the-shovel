# Hand-value cases (KS-1: certified means verified)

A formula is labelled **certified** only when it declares the standard it
implements (`standard=` on `@formula`) **and** its kernel reproduces every
hand-value case listed for it here. The label is derived at run time by
`app.lib.formula_registry.certified`; it is never set by hand.

Each `*.json` file in this directory is a list of cases:

```json
[
  {
    "formula": "<registered formula name>",
    "inputs": {"<input>": 1.0},
    "expect": {"<output key>": 2.0},
    "rel_tol": 0.001,
    "working": "the hand working from the standard, step by step"
  }
]
```

* `expect` values are worked **by hand** from the published definition, never
  copied from the product's output. A kernel that disagrees is a defect; the
  expected value is never moved to match.
* `rel_tol` is optional (default 0.001).
* A malformed case, a case naming no registered formula, or a case the kernel
  does not reproduce fails `tests/test_certified_means_verified.py`.
* Adding the first case (or a declared standard) for a formula removes its
  `standard_or_hand_value` entry from `tests/fixtures/certification_gaps.json`,
  which may only shrink.
