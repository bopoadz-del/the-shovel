# GOLDEN_SET_REPORT — TASK 4e pilot gate

Generated 2026-07-08 05:36 UTC from golden_set_results.jsonl.

## Score: 9/28 PASS

Gate bar: >= 90% of the golden set must PASS (>= 26/28). **Gate NOT MET.**

| id | feature | project | verdict | retrieval | latency |
|---|---|---|---|---|---|
| fresh_pier_mix_cement | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 4s |
| fresh_r9_lighting_wattage | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 3s |
| fresh_eot_notice_period | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 4s |
| fresh_s4_manhole_spacing | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 3s |
| fresh_design_review_codes | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 8s |
| fresh_excavation_rate | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 5s |
| fresh_rebar_cover | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 3s |
| fresh_wearing_course | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 3s |
| fresh_permit_to_work | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 3s |
| fresh_mv_soak_test | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 3s |
| fresh_cpi_threshold | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 4s |
| fresh_mass_concrete_curing | fresh_upload_grounding | ce1ced1a | FAIL | MISS | 30s |
| pilot_doc_qa_client_pep | process_document | master_corpus | FAIL | - | 12s |
| pilot_boq_total | boq_process | 96bd7cd1 | FAIL | - | 6s |
| pilot_spec_extraction | spec_analyze | master_corpus | FAIL | - | 7s |
| pilot_document_metadata | document_metadata | master_corpus | FAIL | - | 9s |
| pilot_wbs_generation | generate_wbs | master_corpus | PASS | - | 11s |
| pilot_manpower_histogram | resource_histogram | master_corpus | FAIL | - | 14s |
| pilot_milestones | parse_primavera_schedule | 7ce7b9d0 | PASS | - | 11s |
| pilot_s_curve | cash_flow_forecast | master_corpus | PASS | - | 16s |
| pilot_procurement_list | procurement_list_generator | master_corpus | PASS | - | 10s |
| pilot_rfp_sections | rfp_management | master_corpus | PASS | - | 13s |
| pilot_kb_mass_concrete | construction_advisor | master_corpus | PASS | - | 6s |
| pilot_qto_floor_area | drawing_qto | 7ce7b9d0 | PASS | - | 5s |
| demo_project_brief | demo_flow | master_corpus | PASS | - | 10s |
| demo_biggest_cost_items | demo_flow | master_corpus | FAIL | - | 6s |
| demo_open_risks | demo_flow | master_corpus | FAIL | - | 11s |
| demo_procure_first | demo_flow | master_corpus | PASS | - | 5s |

## Failures

### fresh_pier_mix_cement (fresh_upload_grounding)

- prompt: What is the cement content for the Pier P-217 concrete mix?
- expected (all must match): `385\s*kg`
- retrieval_expect: `P-217-pier-mix-design`
- what failed:
  - answer_expect not matched: '385\\s*kg'
  - retrieval_expect 'P-217-pier-mix-design' not in cited sources: SPEC-C900-part-1.pdf, SPEC-C900-part-1.pdf, SPEC-C900-part-1.pdf
- cited sources: SPEC-C900-part-1.pdf; SPEC-C900-part-1.pdf; SPEC-C900-part-1.pdf

What came back (verbatim excerpt):

```
I don't have the specific cement content for the Pier P-217 concrete mix in the provided reference context.
```

### fresh_r9_lighting_wattage (fresh_upload_grounding)

- prompt: What wattage LED fixture is specified for the Road R-9 lighting poles?
- expected (all must match): `45\s*W`
- retrieval_expect: `road-R9-street-lighting`
- what failed:
  - answer_expect not matched: '45\\s*W'
  - retrieval_expect 'road-R9-street-lighting' not in cited sources: PRC-946_HSE Audit and Inspection Procedure.pdf, TEM-963_Variation settlement Recommendation form.xlsx, TEM-963_Variation settlement Recommendation form.xlsx
- cited sources: PRC-946_HSE Audit and Inspection Procedure.pdf; TEM-963_Variation settlement Recommendation form.xlsx; TEM-963_Variation settlement Recommendation form.xlsx

What came back (verbatim excerpt):

```
I don't have the specific wattage for the LED fixture specified for the Road R-9 lighting poles in the provided context.
```

### fresh_eot_notice_period (fresh_upload_grounding)

- prompt: Within how many days must the Contractor give notice of an EOT claim on this project?
- expected (all must match): `21\s*days?`
- rejected (none may match): `within\s+28\s+days`; `28[-\s]day\s+(?:notice|period|time\s*bar)`
- retrieval_expect: `contract-eot-notice-period`
- what failed:
  - answer_expect not matched: '21\\s*days?'
  - retrieval_expect 'contract-eot-notice-period' not in cited sources: 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx, 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx, 2012 01 10 EXW (Show Package).docx
- cited sources: 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx; 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx; 2012 01 10 EXW (Show Package).docx

What came back (verbatim excerpt):

```
The Contractor must give a Claim Notice stating that it is a notice pursuant to Sub-Clause 20.1 to the Employer's Representative within 14 days after the Contractor became aware, or should have become aware acting in accordance with Best Industry Practices, of the event or circumstance giving rise to the claim. Additionally, within 3 Business Days of it becoming reasonably evident that the Works are delayed or are likely to be delayed beyond the Date for Taking-Over, or the Time for Taking-Over (as the case may be), the Contractor must give the Employer a written notice.
```

### fresh_s4_manhole_spacing (fresh_upload_grounding)

- prompt: What is the maximum manhole spacing on sewer line S-4?
- expected (all must match): `42\s*m\b`
- retrieval_expect: `sewer-S4-manhole-spacing`
- what failed:
  - answer_expect not matched: '42\\s*m\\b'
  - retrieval_expect 'sewer-S4-manhole-spacing' not in cited sources: boq_units_of_measurement.md, 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx, 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx
- cited sources: boq_units_of_measurement.md; 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx; 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx

What came back (verbatim excerpt):

```
I don't have the information regarding the maximum manhole spacing on sewer line S-4 in the provided context.
```

### fresh_design_review_codes (fresh_upload_grounding)

- prompt: What design review status codes are used on this project?
- expected (all must match): `CODE[-\s]?A`; `CODE[-\s]?B`; `CODE[-\s]?C`
- retrieval_expect: `design-review-status-codes`
- what failed:
  - answer_expect not matched: 'CODE[-\\s]?A'
  - answer_expect not matched: 'CODE[-\\s]?B'
  - answer_expect not matched: 'CODE[-\\s]?C'
  - retrieval_expect 'design-review-status-codes' not in cited sources: PRC-951_Design Reviews & Acceptance.pdf
- cited sources: PRC-951_Design Reviews & Acceptance.pdf

What came back (verbatim excerpt):

```
The design review status codes used on this project include the following definitions for the review status of design documents:

- **For ‘Comment’**: The nominated reviewer shall review the design documentation and mark-up and collate comments as required. All design comments shall be issued to the PMT Document Controller (or directly through the Document Control System). The PMT Design Manager shall manage commenting and accepting design documentation.

These codes guide the project team in determining the appropriate review status at all times. 

*Source: PRC-951_Design Reviews & Acceptance.pdf, chunks 6, 7.*
```

### fresh_excavation_rate (fresh_upload_grounding)

- prompt: What is the contract rate per m3 for excavation in ordinary soil?
- expected (all must match): `SAR`; `27\.50?`
- retrieval_expect: `boq-excavation-rate`
- what failed:
  - answer_expect not matched: 'SAR'
  - answer_expect not matched: '27\\.50?'
  - retrieval_expect 'boq-excavation-rate' not in cited sources: TEM-963_Variation settlement Recommendation form.xlsx, TEM-963_Variation settlement Recommendation form.xlsx
- cited sources: TEM-963_Variation settlement Recommendation form.xlsx; TEM-963_Variation settlement Recommendation form.xlsx

What came back (verbatim excerpt):

```
The contract rate for excavation in ordinary soil is as follows:

- For maximum depth not exceeding 0.25m: **2.03** per m³
- For maximum depth between 0.25m and 0.5m: **2.56** per m³
- For maximum depth between 0.5m and 1.0m: **3.41** per m³
- For maximum depth between 1.0m and 2.0m: **9.74** per m³

(Source: TEM-963_Variation settlement Recommendation form.xlsx, chunks 56, 57)
```

### fresh_rebar_cover (fresh_upload_grounding)

- prompt: What is the minimum concrete cover for substructure elements in contact with soil?
- expected (all must match): `75\s*mm`
- retrieval_expect: `rebar-cover-substructure`
- what failed:
  - answer_expect not matched: '75\\s*mm'
  - retrieval_expect 'rebar-cover-substructure' not in cited sources: SPEC-C900-part-3.pdf, SPEC-C900-part-1.pdf, SPEC-C900-part-3.pdf
- cited sources: SPEC-C900-part-3.pdf; SPEC-C900-part-1.pdf; SPEC-C900-part-3.pdf

What came back (verbatim excerpt):

```
The minimum concrete cover for substructure elements in contact with soil is 2 inches. In saltwater areas, the cover should be increased to 3 inches. This ensures adequate protection against corrosion and other environmental factors.
```

### fresh_wearing_course (fresh_upload_grounding)

- prompt: What thickness and type is the asphalt wearing course on this project?
- expected (all must match): `60\s*mm`; `polymer|PMB`
- retrieval_expect: `asphalt-wearing-course`
- what failed:
  - answer_expect not matched: '60\\s*mm'
  - answer_expect not matched: 'polymer|PMB'
  - retrieval_expect 'asphalt-wearing-course' not in cited sources: Message from MGT-C552-34 (Sect-3).pdf, boq_units_of_measurement.md, construction_kb.md
- cited sources: Message from MGT-C552-34 (Sect-3).pdf; boq_units_of_measurement.md; construction_kb.md

What came back (verbatim excerpt):

```
I don't have specific information regarding the thickness and type of the asphalt wearing course for this project in the provided context.
```

### fresh_permit_to_work (fresh_upload_grounding)

- prompt: How long is a permit-to-work valid on this project?
- expected (all must match): `12\s*hours?`
- retrieval_expect: `hse-permit-to-work`
- what failed:
  - answer_expect not matched: '12\\s*hours?'
  - retrieval_expect 'hse-permit-to-work' not in cited sources: 2012 01 10 EXW (Show Package).docx, 2012 01 17 EXW (Show Package)_MarkedUp copy.docx, 2012 01 10 EXW (Show Package).docx
- cited sources: 2012 01 10 EXW (Show Package).docx; 2012 01 17 EXW (Show Package)_MarkedUp copy.docx; 2012 01 10 EXW (Show Package).docx

What came back (verbatim excerpt):

```
I don't have information on how long a permit-to-work is valid on this project.
```

### fresh_mv_soak_test (fresh_upload_grounding)

- prompt: How long is the soak test required for the MV substation before taking-over?
- expected (all must match): `5[-\s]?days?`
- retrieval_expect: `mv-substation-soak-test`
- what failed:
  - answer_expect not matched: '5[-\\s]?days?'
  - retrieval_expect 'mv-substation-soak-test' not in cited sources: 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx, 2011.01.01 EXM CSA (Long) With Adv Pmt TEMPLATE v4.docx, 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx
- cited sources: 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx; 2011.01.01 EXM CSA (Long) With Adv Pmt TEMPLATE v4.docx; 2011.01.01 EXM Lump Sum Const Contract TEMPLATE v4.docx

What came back (verbatim excerpt):

```
I don't have the specific duration for the soak test required for the MV substation before taking-over in the provided reference context.
```

### fresh_cpi_threshold (fresh_upload_grounding)

- prompt: What CPI threshold triggers a corrective action plan on this project?
- expected (all must match): `0\.92`
- retrieval_expect: `evm-cpi-threshold`
- what failed:
  - answer_expect not matched: '0\\.92'
  - retrieval_expect 'evm-cpi-threshold' not in cited sources: PRC-942 _Construction Non- Conformance Reporting.pdf, PRC-942 _Construction Non- Conformance Reporting.pdf, PRC-946_HSE Audit and Inspection Procedure.pdf
- cited sources: PRC-942 _Construction Non- Conformance Reporting.pdf; PRC-942 _Construction Non- Conformance Reporting.pdf; PRC-946_HSE Audit and Inspection Procedure.pdf

What came back (verbatim excerpt):

```
I don't have the specific information regarding the CPI threshold that triggers a corrective action plan on this project.
```

### fresh_mass_concrete_curing (fresh_upload_grounding)

- prompt: How many days must mass concrete pours be cured on this project?
- expected (all must match): `14\s*days?`
- retrieval_expect: `mass-concrete-curing`
- what failed:
  - answer_expect not matched: '14\\s*days?'
  - retrieval_expect 'mass-concrete-curing' not in cited sources: SPEC-C900-part-1.pdf
- cited sources: SPEC-C900-part-1.pdf

What came back (verbatim excerpt):

```
Mass concrete pours must be cured for not less than 7 days using clean water or any other method which has been shown to allow continuous hydration of the cement over the curing period. *(source: SPEC-C900-part-1.pdf, chunk 28)*
```

### pilot_doc_qa_client_pep (process_document)

- prompt: what does the the client project project execution plan cover?
- expected (all must match): `execution plan`; `(scope|schedule|quality|risk|procurement|hse|safety|management|organi[sz]ation)`
- what failed:
  - answer_expect not matched: '(scope|schedule|quality|risk|procurement|hse|safety|management|organi[sz]ation)'

What came back (verbatim excerpt):

```
I searched the project documents for the the client project project execution plan but could not find specific information. Based on the retrieved context, the execution plan generally covers the planning, execution, and monitoring of the Works, ensuring that the networked activities are detailed enough to provide a meaningful measurement tool for progress. 

For more specific details about the the client project project execution plan, please ensure the relevant document is uploaded and indexed.
```

### pilot_boq_total (boq_process)

- prompt: process the bill of quantities and give me the total package value
- expected (all must match): `\b(SAR|USD|AED)\b`; `\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?|\d{7,}`
- what failed:
  - answer_expect not matched: '\\b(SAR|USD|AED)\\b'
  - answer_expect not matched: '\\d{1,3}(?:,\\d{3}){2,}(?:\\.\\d+)?|\\d{7,}'
- cited sources: 2012 01 10 EXW (Show Package).docx; 2012 01 17 EXW (Show Package)_CLEAN.docx; 2012 01 17 EXW (Show Package)_MarkedUp copy.docx

What came back (verbatim excerpt):

```
I couldn't process the Bill of Quantities (BOQ) because the PDF file is too large (26.9 MB) for the server to handle. Please upload the .xlsx or .csv version of this BOQ instead, as those formats can be processed reliably and compute the totals without memory issues.
```

### pilot_spec_extraction (spec_analyze)

- prompt: analyze the concrete specification requirements - what grades and standards apply?
- expected (all must match): `\bC\d{2}(?:/\d{2})?\b|grade`; `\b(ASTM|ACI|BS|EN|SASO|SBC|ISO)\b`
- what failed:
  - answer_expect not matched: '\\bC\\d{2}(?:/\\d{2})?\\b|grade'
- cited sources: SPEC-C900-part-1.pdf; SPEC-C900-part-1.pdf

What came back (verbatim excerpt):

```
The concrete specification requirements refer to the following standards:

1. **EN 197** - Cement; Composition, specifications and conformity criteria.
2. **EN 206** - Concrete - Performance, production, placing and compliance criteria.

Additionally, the aggregates must meet all specification requirements and pass all chemical and physical tests. The cement must be chosen to meet the specific requirements of the job, and additives and admixtures are to be used as appropriate. The blended mix of the ingredients must be within the envelope specified in the mix specification, and the curve of the mix should be as smooth as possible without gaps.


Source: SPEC-C900-part-1.pdf, chunk 35, 100.
```

### pilot_document_metadata (document_metadata)

- prompt: list the documents in this project and what type each one is
- expected (all must match): `(?m)^\s*(?:[-*•]|\d{1,3}[.)]|\|)\s*\S`; `\.(pdf|xlsx?|docx?|dwg|dxf|csv)\b|(drawing|report|spec(?:ification)?|boq|bill|schedule|plan|register)`
- what failed:
  - answer_expect not matched: '(?m)^\\s*(?:[-*•]|\\d{1,3}[.)]|\\|)\\s*\\S'
- cited sources: 2011.01.01 EXM Measured Price Const. Contract (Schedules) TEMPLATE v5.docx; 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx; 2011.01.01 EXM Measured Price Const. Contract (Schedules) TEMPLATE v5.docx

What came back (verbatim excerpt):

```
I don't have the specific list of documents in this project and their types based on the provided reference context. The context primarily discusses the structure and references related to drawing schedules and project documents but does not enumerate specific documents or their types.
```

### pilot_manpower_histogram (resource_histogram)

- prompt: produce a manpower histogram for the structure works over 12 months
- expected (all must match): `\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|\b(month|period|week)\s*[-#]?\s*\d{1,2}\b|\bM\d{1,2}\b|\bQ[1-4]\b`; `(manpower|labou?r|worker|crew|headcount|staff)`; `\d{2,}`
- what failed:
  - answer_expect not matched: '\\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\b|\\b(month|period|week)\\s*[-#]?\\s*\\d{1,2}\\b|\\bM\\d{1,2}\\b|\\bQ[1-4]\\b'
- cited sources: 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx; 2011.01.01 EXM Lump Sum Const Contract SCHEDULES TEMPLATE v4.docx; 2011.01.01 EXM Measured Price Const. Contract (Schedules) TEMPLATE v5.docx

What came back (verbatim excerpt):

```
### Manpower Histogram for Structure Works Over 12 Months

#### Headline Metrics
- **Total Activities**: 204
- **Total Duration**: 688 days (~22.9 months)
- **Critical-Path Count**: 44
- **Project Type**: Building
- **Assumptions**:
  - Rule-of-thumb activity durations; replace with project-specific data when available.
  - FS-only predecessors; no SS/FF/SF; zero lag.
  - Zone-multiplier scales repeatable activities to reach target_count.

#### Per-Phase Summary
| Phase                | Start Day | End Day | Activity Count | Critical Count |
|----------------------|-----------|---------|----------------|-----------------|
| Site Preparation      | 0         | 63      | 12             | 6               |
| Substructure          | 58        | 87      | 6              | 0               |
| Su
[... truncated ...]
```

### demo_biggest_cost_items (demo_flow)

- prompt: what are the biggest cost items on this project?
- expected (all must match): `(cost|value|SAR|USD)`; `\d{1,3}(?:,\d{3})+|\d{4,}`
- what failed:
  - answer_expect not matched: '\\d{1,3}(?:,\\d{3})+|\\d{4,}'
- cited sources: TEM-963_Variation settlement Recommendation form.xlsx; MNL-908_PMWeb Cost Management.pdf; TEM-963_Variation settlement Recommendation form.xlsx

What came back (verbatim excerpt):

```
I don't have information on the biggest cost items for this project based on the provided context.
```

### demo_open_risks (demo_flow)

- prompt: summarize the open risks on this project
- expected (all must match): `risk`; `(mitigat|impact|likelihood|priorit|contingen|action)`
- what failed:
  - answer_expect not matched: '(mitigat|impact|likelihood|priorit|contingen|action)'

What came back (verbatim excerpt):

```
I searched the project documents for open risks but could not find specific information. Based on the retrieved context, project risk management includes the following six process groups:

1. Planning risk management
2. Risk identification
3. Performing qualitative risk analysis
4. Performing quantitative risk analysis
5. Planning risk responses
6. Monitoring and controlling risks

All risks must have ownership by a person or named organization, and the owner is responsible for the ongoing management of that risk. A risk register must be maintained throughout the entire program delivery period, and all project risks must be integrated into the program risk register by the Delivery Manager(s).

To get a project-specific answer regarding open risks, ensure the relevant document is uploaded a
[... truncated ...]
```
