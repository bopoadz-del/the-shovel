# Retrieval and answer-path passes — what stays, what F-DRIVER deletes

Every pass that, after the main hybrid search, scans candidate chunks (or
re-queries the store) to add, lift, fence, reserve or rewrite something.
Classified by the owner's test: **would this line still be right, unchanged,
on a different client's project?** Yes → **general**: kept, with a synthetic
test, and not rebuilt as configuration for its own sake. No → **rescue**:
deleted at F-DRIVER step 16. No new passes of either kind are added on the old
path. "Written for" names the kind of question or document, never a stored
case's identifiers.

Totals: **89 passes — 26 general, 63 rescues.**

## Retrieval (`app/core/rag/retriever.py`, in `retrieve_with_filter` order)

| # | Pass | What it does | Written for | Verdict |
|---|---|---|---|---|
| 1 | `_dual_search` / `_strip_question_wrapper` | second search with question wrapper words removed; best score per chunk | long contracts where "how many days is …" wrappers dilute the query | general |
| 2 | particulars third search (inline) | extra search with Contract Data particulars wording | particulars-field asks | rescue |
| 3 | `expand_contract_synonyms` search | adds a contract form's canonical headings for synonyms | synonym asks for one contract form | rescue |
| 4 | `_fetch_numeric_requirement_chunks` | BM25 with cover / compaction / lux vocabulary | numeric-requirement asks of three kinds | rescue |
| 5 | Master-Corpus fallback | searches the fallback corpus, disclosed, when the project is thin | empty or thin projects | general (its calculator-fixture carve-out is a rescue) |
| 6 | identifier-search fusion | exact-code search fused into the pool | any reference code | general |
| 7 | semantic-candidate identifier bonus | bonus for pooled chunks containing the code tokens | any reference code | general |
| 8 | `_identifier_context_terms` | rest-of-query overlap on chunks carrying the identifier | one BOQ-item disambiguation | rescue |
| 9 | lexical term rescue (pair co-occurrence) | bonus or fetch for chunks holding the query's term pairs | multi-word terms split across chunks | general |
| 10 | `follow_quantity_pointers` | pools the spec clause that points a figure to drawings | "per the specification" quantity asks | rescue |
| 11 | `recall_asked_quantity_chunks` | subject + unit-anchor fetch | three quantity kinds | rescue |
| 12 | `_pool_docs_named_by_query` | pulls documents whose filename matches query terms or "letter" | one letter-signatory ask | rescue |
| 13 | `recall_titled_documents` | pools documents whose name carries the asked title | "which document covers <title>" | general |
| 14 | `recall_labelled_rows` | loads particulars documents; pools rows the question's labels name | Contract Data particulars | rescue |
| 15 | `_pool_named_document_control_block` | pulls a named document's cover block | document-identity asks keyed on cover labels | rescue |
| 16 | `recall_issue_stamps` | fetches "Date + label No." stamps, elects a contract year | issue-date asks | rescue |
| 17 | `recall_composition_operands` | whole-document scan for a missing delay-damages operand | delay-damages daily amount | rescue |
| 18 | `recall_rows_deep_in_pooled_documents` | reads particulars volumes whole for a known row | particulars rows past early windows | rescue |
| 19 | `_pool_page_total_rows` | fetches a BOQ page's summary footer | BOQ page totals | rescue |
| 20 | `recall_list_continuations` | pools the next chunk when an intro ends "as follows" | lists split by the chunker | general |
| 21 | `_gk_lexical_bonus` | capped term-overlap bonus on general-knowledge chunks | general knowledge on merit | general |
| 22 | `_apply_contract_data_particulars_boost` | lifts particulars rows, penalises glossary | particulars rows ranking low | rescue |
| 23 | GK ownership boost / score margin / lexical fold | configured general-knowledge ranking knobs | general-knowledge contamination | general |
| 24 | revision currency | penalises superseded files; drops lower revisions of a drawing | stale revisions | general |
| 25 | `_apply_filename_overlap_boost` | lifts chunks whose filename shares query terms | any named-document ask | general (its "letter" tier is a rescue) |
| 26 | `_apply_source_class_preference` | lifts spec/HSE-named files, demotes others by name | one "per the specification" ask | rescue |
| 27 | `_cap_specification_class_bonus` | caps #26 on cover asks | cover asks | rescue |
| 28 | `_apply_numeric_requirement_boost` | lifts chunks stating a cover / MDD / lux figure | three quantity kinds | rescue |
| 29 | `_apply_quantity_pointer_boost` | lifts pointer clauses and targets | "per the specification" quantity asks | rescue |
| 30 | `_apply_title_filename_boost` | lifts a filename containing the asked title | "which document covers <title>" | general |
| 31 | `_apply_register_line_boost` | lifts a register line where a code precedes the title | document registers | general |
| 32 | `_apply_contract_data_filename_boost` | lifts "Contract Data" filenames per particular | particulars asks | rescue |
| 33 | `_apply_asked_particular_value_boost` | lifts chunks stating named particulars | five named particulars | rescue |
| 34 | `_apply_schedule_register_boost` | lifts "Schedule N: Not Used" rows | one schedule-register ask | rescue |
| 35 | `_apply_pcg_value_boost` | lifts the guarantee row over its form | one guarantee ask | rescue |
| 36 | `_apply_commencement_date_boost` | lifts the commencement row over a pack | one commencement ask | rescue |
| 37 | `_apply_rate_only_boost` | lifts a "Rate Only" BOQ row | one BOQ row type | rescue |
| 38 | `_apply_priced_boq_boost` | lifts a priced BOQ row | priced BOQ items | rescue |
| 39 | `_apply_part_summary_boost` | lifts a page-summary row | BOQ page totals | rescue |
| 40 | layered precedence bonus | authority / layer bonus | layered RAG | general |
| 41 | `_keep_project_layer_first` | keeps project evidence above general knowledge on project-framed asks | layer rule | general |
| 42 | noise filename filter | drops lockfiles and similar | file noise | general |
| 43 | `_ContractScope` / `elect_answer_bearing_contract` | contract-year lock plus per-ask fences | one contract-numbering scheme and ~14 asks | rescue |
| 44 | `chunk_copy_key` dedup | collapses duplicate chunk bodies | signed / unsigned copies | general |
| 45 | GK top-k cap | caps general-knowledge chunks in the final k | general-knowledge crowding | general |
| 46 | cross-encoder rerank | reorders survivors (flag, default off) | ranking quality | general |
| 47 | `reserve_matching_particulars_row` | swaps a kept slot for the asked particulars row | one particulars ask | rescue |
| 48 | `reserve_monetary_base_row` | reserves a slot for the contract amount | delay arithmetic | rescue |
| 49 | `reserve_daily_damages_operands` | forces the rate and amount into kept | delay-damages daily amount | rescue |
| 50 | `ensure_kept_can_compose_daily_damages` | swaps in operands | delay-damages daily amount | rescue |
| 51 | `ensure_kept_has_including_vat` | forces the incl-VAT amount row into kept | one amount ask | rescue |
| 52 | `reserve_contract_synonym_row` | reserves a slot for a synonym's heading row | synonym asks | rescue |

## Injection (`app/core/rag/inject.py`, `rag_inject`)

| # | Pass | What it does | Written for | Verdict |
|---|---|---|---|---|
| 53 | `build_retrieval_query` | prepends prior turns for thin follow-ups | follow-up questions | general |
| 54 | `rag_retrieval_k` | +2 slots for source-scoped quantity asks | one ask kind | rescue |
| 55 | `_drop_master_corpus_for_formula_fixture` | strips fallback chunks on calculator asks | calculator fixtures | rescue |
| 56 | identifier precision gate | a miss when no chunk holds the asked identifier | any reference code | general |
| 57 | named-contract scope filter | keeps chunks of the named contract number | one contract-numbering scheme | rescue |
| 58 | `apply_token_cap` operand protection | keeps delay-damages operands under the cap | delay-damages | rescue |
| 59 | SUPERSEDED note | warns when an excerpt is superseded | revision currency | general |
| 60 | SOURCE CLASS note | precedence note when classes mix | mixed sources | general |
| 61 | TERM EQUIVALENCE note | synonym → contract-form term hint | one contract form | rescue |
| 62 | SCHEDULE REGISTER note | "Not Used" is the answer | one schedule ask | rescue |
| 63 | PRICED BOQ / RATE ONLY note | composes a priced row | BOQ items | rescue |
| 64 | PART SUMMARY note | composes a page total | BOQ page totals | rescue |
| 65 | ACA INCLUDING VAT note | lead-with hint | one amount ask | rescue |
| 66 | DELAY DAMAGES PER DAY note | compose hint | delay-damages | rescue |
| 67 | DELAY DAMAGES OVER A PERIOD note | compose hint | delay-damages | rescue |
| 68 | TIME FOR COMPLETION note | lead-with hint and a named lookalike | one particular | rescue |
| 69 | hypothetical milestone instruction | user-supplied milestone arithmetic | one ask | rescue |
| 70 | party identity / withheld note | Engineer identity hint | one party ask | rescue |
| 71 | PARENT COMPANY GUARANTEE note | guarantee hints | one guarantee ask | rescue |
| 72 | COMMENCEMENT DATE note | commencement hints | one commencement ask | rescue |
| 73 | NAMED STANDARD ABSENT note | warns when a named code is not in the excerpts | a list of named codes | rescue |
| 74 | CONTRACT ATTRIBUTION note | names excerpts' contract numbers | one contract-numbering scheme | rescue |

## Answer path (`app/agents/runtime.py` `_postprocess_answer` and callees)

| # | Pass | What it does | Written for | Verdict |
|---|---|---|---|---|
| 75 | `_graft_asked_contract_particular` | writes a particulars row from excerpts; rescans the volume | five particulars | rescue |
| 76 | `_graft_composed_delay_damages_daily` | composes rate × amount per day | delay-damages | rescue |
| 77 | `_graft_composed_delay_damages_over_period` | composes rate × amount × days | delay-damages | rescue |
| 78 | `_graft_composed_percentage_of_aca` | percentage × contract amount | advance payment | rescue |
| 79 | `_graft_priced_boq_item` | writes a BOQ qty/amount line | BOQ items | rescue |
| 80 | `_graft_part_summary_total` | writes a page total | BOQ page totals | rescue |
| 81 | `_graft_combined_part_summary_total` | sums page totals | BOQ page totals | rescue |
| 82 | `_graft_named_community_tfc_span` | longest/shortest completion span | one project's community table | rescue |
| 83 | `_graft_rate_only_item` | states "Rate Only" | one BOQ row type | rescue |
| 84 | `_graft_honest_contract_refusal` | not-required / not-populated lines | two asks | rescue |
| 85 | `_cost_grounding_gate` | refuses money figures not traceable to excerpts or tools | any cost figure | general |
| 86 | `citation_provenance.gate` | strips attributions no evidence backs | any attribution | general |
| 87 | `_graft_named_standard_attribution` | relabels a project figure claimed as a code's | a list of named codes | rescue |
| 88 | `withhold_party_names` | replaces party names with their role | owner policy | general |
| 89 | `apply_first_line_hard_rule` | installs a figure + clause first line | three ask kinds | rescue |
