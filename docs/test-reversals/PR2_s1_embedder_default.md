# Reversed-with-evidence: PR #2 (fix/s1-embedder-default)

Three tests were deleted from `tests/test_embedder_default_matches_baked_model.py` in 5950f17. Line numbers refer to that file at d025ff3. Kill-checks were run locally on 5950f17: each mutation was applied alone and then reverted.

## 1. `test_boot_identity_mismatch_refuses_to_serve` (deleted)
- **What it asserted.** `run_boot_identity_check()` returns a non-None result and sets `identity_serving_blocked()` to True.
- **Why it was stubbed.** At :58-69 it monkeypatched `vs.get_store` to return `_Mismatch`, whose `check_identity` returns a canned dict (`expected` m2/16 against `stored` m1/8). The real store, the real `check_identity` and the real store-open path never ran. So it could not see that the real gate, at vector_store.py:2479-2482 on d025ff3, swallowed the real mismatch error (`except Exception` → "store probe skipped" → serving NOT blocked). It stays green whether or not that error is swallowed.
- **Replaced by.**
  - `test_boot_check_blocks_serving_when_the_index_was_built_at_another_dim`: the index is seeded at D1, then a fresh process resolves the real `get_embedder()` to D2. The real `run_boot_identity_check` runs the real `check_identity`, and the test asserts `identity_serving_blocked() is True`.
  - `test_boot_check_blocks_serving_when_the_namespace_is_mapped_at_another_dim`: the same check in-process.
- **Kill-check K3.** Make the gate stop catching `EmbeddingIdentityMismatch` (the swallow restored). Both replacement tests fail; the deleted stub test would still pass.

## 2. `test_make_rag_chunk_class_is_idempotent_across_import_order` (deleted)
- **What it asserted.** Two calls to `make_rag_chunk_class("orderns", 8, "m1")`, with `_RAG_CHUNK_CLASS_CACHE.pop(ns)` in between, return the same class at dim 8.
- **Why it was weak.** At :129-143 it ran in one process and one order. It never imported in two orders. It used a hand-made cache pop to "simulate desync". It passed under lazy first-use registration, so it could not fail if lazy registration returned. It also certified the reuse path (`_mapped_rag_chunk_class_for_table`, models.py:136-150 on d025ff3). That path skipped the width guard: dim 16 returned the dim-8 class. It also raised `UnmappedClassError` in CI postgres shard 1/4.
- **Replaced by** `test_chunk_tables_register_at_import_and_build_identically_in_either_order`. Two fresh processes build namespaces in opposite orders. Each must have the configured table on `Base.metadata` straight after import, end with an identical chunk MetaData (VECTOR widths included), and raise on 4 of 4 dim/model conflicts.
- **Kill-checks.**
  - K1: remove the import-time `register_configured_rag_namespace()` (lazy registration returns). The test fails.
  - K2: make the factory reuse the class on a mismatch. The test fails, along with 2 others.

## 3. `test_real_check_identity_flags_stored_vs_embedder_dim_mismatch` (deleted)
- **What it asserted.** After a row is seeded at m1/8, `store.check_identity()` returns stored m1/8 and expected m2/16.
- **Why it was weak.** At :120-121 it produced D2 by assigning `store.model_name = "m2"; store.dim = 16` by hand, not through an embedder that resolves to D2. It called `check_identity()` directly and never went through `run_boot_identity_check()` or `identity_serving_blocked()`. So it proved the comparison, but not that a real D2 embedder at boot blocks serving.
- **Replaced by** `test_boot_check_blocks_serving_when_the_index_was_built_at_another_dim`. D2 comes from the real `get_embedder()` resolution in a fresh process, and the test asserts the mismatch dims plus `identity_serving_blocked() is True`.
- **Kill-check K3** (same as note 1). The replacement fails.
