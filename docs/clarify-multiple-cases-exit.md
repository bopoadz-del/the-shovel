# Clarify-before-choosing: the exit mechanism (PR 2 part b)

Design note, written before the code. Companion to #495, which carries the
instruction text (PR 2 part a). This is the diag for the mechanism the owner
ordered at 15:00Z / 15:03Z (19 Nov UTC): the clarifying question is a general
behaviour, and its enforcement on the retrieval side lives at the answer exit.

## What the mechanism does

When the handed evidence supports more than one materially different answer to
the question as asked, and the question does not say which case is meant, the
exit turns the draft answer into ONE short clarifying question that names the
cases it actually found, each with its source, and the next turn answers the
case the user picks.

It generalises what `app/agents/first_line_hard_rule.py` already does for two
typed topics (compaction and concrete cover: "Retrieved context states more
than one … figure … Which document's figure is meant?"). The owner's 15:03Z
rule removes the topic boundary: it is one duty for every source the product
answers from — the client's documents, the encoded knowledge, and the
calculators. The calculators' "ask for the missing input by its label" is the
same duty and stays as it is.

## Where it lives

- **Detection is pure.** Like `app/agents/missing_input.py`, the detector is a
  function of (handed evidence, draft answer, question) with no I/O, so it can
  be tested against recorded strings rather than a mock of the retriever.

- **The exit hook** is the finalise step in `app/agents/runtime.py` where
  `apply_first_line_hard_rule(text, rag_sys_msg, messages)` is already called.
  The clarifying-question check runs beside it, after the draft has committed
  to a figure, from the same handed evidence (`rag_sys_msg` retrieval markers
  plus tool passages; the evidence records come from
  `app.agents.citation_provenance.build_evidence`).

- **The turn state is reused, not reinvented.** `app.agents.answer_exit.Turn`
  already carries `awaiting_inputs` (the calculators' missing-input state) and
  `questions`. `check_result` / `check_end_event` already emit
  `{"missing": [...]}` from `turn.awaiting_inputs`; the route observer reads
  that state (never a phrase match, S1c). The clarifying question rides out the
  same carrier, so the client renders "the agent is asking" the way it already
  renders a calculator's missing input.

## The detector (general — no list of known structures)

A **case** = one value claim in the handed passages:

- a **value**: a number with its unit, or a clause / document-id / rate
  reference, read by the typed readers already in the product; and
- its **subject**: the noun phrase the passage's own sentence, heading, or
  detail title names for that value; and
- its **source**: document name (and page/chunk) from the evidence records.

Two claims are *different cases* when their subjects differ and their values
differ materially. The subject is read from the passage's own words — never
from a list of known structures, elements, locations, depths, sizes, materials,
or stages. (A hardwired subject list would be the exact per-case branch the
standing order forbids.)

The draft has *silently picked one* when the question names no case and the
draft states exactly one of the values without acknowledging the others.

## Bounds — each one a test, not prose

1. One clarifying question per turn, never a chain.
2. Only when the cases lead to **different** answers (same value in every case:
   answer).
3. Never when the question already names the case.
4. Never to ask for something the documents state plainly (no invented
   question).
5. Options listed are only cases actually found, each with its source.
6. If the user answers "all" or does not choose: every case side by side,
   labelled.
7. A question that is merely broad is answered, not bounced back.

Case 2 vs case 3 (15:00Z): different values for *different* subjects, question
names none → ask (this mechanism). Different values for the *same* subject →
the both-values statement ("the documents differ", every value with origin),
which is #453, already on main. When the passages do not let the product tell
them apart, it behaves as case 2 (it asks).

## What was still needed before the code freezes

1. **Which reader extracts the general value+subject.** Done (M64):
   `cases_from_passages` reads a measure with the product's figure reader and
   a reference by shape; the subject is the sentence/heading that states the
   value.
2. **The replay data.** Done (M65): 0 of 53 rows on RAGQ's two newest full
   sets get a clarifying question, and every current row names its case, so 0
   is correct.
3. **Cost measurement.** Done (M65): detector 2-55 ms/turn, ~11 ms avg; a
   non-ambiguous turn is byte-for-byte unchanged.
4. **Dates are not references.** Done: the reader skips a letter-number token
   whose leading word is a month name ("Jan-25", "30-Jan-25"). General, no
   per-case list.

## Not yet frozen -- the reader over-extracts

The reader reads every figure and reference in every handed chunk (16-155
claims per row). The committed gate counts (subject, value, source) claims, so
a value repeated across chunks is committed-to once per chunk and the positive
path ("the draft silently picked one of several values") is unsatisfiable end
to end on real rows.

Collapsing claims by value (one value = one case) is the general fix, but
landing it alone regresses the replay: A6 and A12 then ask, because the reader
reads a class code ("D400") and a document id
("ABC-0000-XYZ-000001") as value claims and the draft's citation of
one of them reads as "committed to one value". The question names the case in
both, but the extracted subject (a full sentence or page heading) shares no
word with the question, so the "question names its case" bound does not fire.

The remaining general fix is to constrain the reader to the question's own
subject (keep only value claims whose subject relates to the question, with a
cleaner subject phrase) before the collapse lands. That rework is not shipped
rushed; it is the next step.

## Done: the sentence break no longer cuts a value in half (M67)

A prerequisite for the subject rework: the reader split a sentence at every
period, so a decimal measure or a dotted code read as a truncated subject
("Depth to invert is 4.", "Clause 2.", "Grade is D999."). A sentence now ends
only at . ! ? followed by whitespace or end of string, or at a newline; a
period between digits is part of the value. Replay re-run: still 0 ASK on the
53 handed rows. This corrects the subject text but does NOT yet shorten it to
its noun phrase -- that and the question-relevance filter + value collapse are
the rework that remains.

## The rework that remains (diagnosed on the A6/A12 rows, not shipped)

Collapsing by value alone regresses A6 (draft commits to the one class code
D400 while the reader also read mortar/pipe/membrane measures and drawing,
clause and coordinate references) and A12 (a schedule id read as a value).
The general fix is to make a case carry a SHORT subject (the noun phrase the
value modifies, not the whole sentence), then keep only cases relevant to the
question before collapsing: relevance = the value's kind matches the kind the
question asks for (nominal class/grade/type vs a measure) AND the subject
shares a word with the question's own subject. One step at a time, each with a
two-input test, replayed on the real rows before anything lands.

## Step 2 attempted and reverted -- the naive filter regresses (M69)

The relevance filter was implemented twice this tick and reverted both times
because it makes the negative side regress on the real rows (it turns a
named-case row into an ASK):

- "subject shares a word" is too weak. B2 ("maximum free fall of concrete")
  keeps a "2 hour" trail-run case whose subject shares only the generic word
  "maximum"; B3 keeps a cement-temperature case that shares "temperature"
  with the ask.
- marker-based ask-kind detection conflates a SOURCE with the ASK. B10 ("what
  reinforcement yield strength ... what do the contract drawing and the
  contractor's method statement each say?") is a both-values (#453) ask, but
  "drawing"/"method statement" are read as a nominal ask, so the filter
  keeps only the cited drawing ids and the draft commits to one -- a false
  ASK.

So the two conditions in the plan are right in shape but need real head-noun
extraction before they can land: the question's ASK noun must be told apart
from its qualifier ("free fall" vs "of concrete") and from its source
("drawing" / "method statement"), and "shares a word" must match that ask
noun, not any content word. That is the remaining rework; it is not shipped
rushed. #498 stays at the short-subject state (M68), which replays 0 ASK on
all 53 rows.

## Done: step 2 (relevance) + step 3 (collapse) -- general, replayed 0 ASK

The rework landed as three general mechanisms, not per-case switches:

1. **Head-noun relevance.** A case's subject is reduced to its head noun
   (the last content word before the first preposition: "cover to
   reinforcement" -> "cover", "fall of concrete" -> "fall", "temperature
   limit" -> "limit"). The question's own ask is reduced the same way. A
   case survives only when its kind (measure / clause / code) matches the
   kind the question asks for AND its head noun matches the question's. A
   generic ask ("what is the value?", "which clause applies?") narrows by
   kind only; a party/action ask ("who", "what must happen") keeps nothing.

2. **A document id is a source, not a value.** A letter-number code with two
   or more dash-separated segments ("DOC-2024-001",
   "ABC-0000-XYZ-000001") is skipped by the reader, so an answer
   citing its source no longer reads as "committed to that value". Shape
   only, never a known id.

3. **Value collapse + a one-value silent-pick gate.** Distinct evidence
   values are collapsed (one value = one case wherever it repeats), so
   "committed to exactly one" is reachable; and the draft must state exactly
   ONE relevant value overall (so a draft that already answers two structures
   side by side is never a silent pick).

Replay (fresh read-only run over RAGQ's two v5 handed-id sets, Neon
SELECT-only, SHOW transaction_read_only=on): **0 ASK on all 53 rows**,
detector total ~350 ms. Local: 87 passed in tests/test_clarify_cases.py;
mutation gate 34 run / 34 killed / 0 red; two-input lint 0 new; ruff clean;
scan_repo_hygiene 0 live names. #498 stays a draft until A1 (the bill
write) is live.

## Done: the exit is wired and the relevance filter names qualifiers (M72)

Two general changes land the detector in the answer exit, with no case name,
structure or value in code:

1. **The exit hook is live.** ``runtime._apply_clarify_cases`` runs at the
   end of ``_postprocess_answer``, beside ``apply_first_line_hard_rule``. It
   reads the handed evidence (``build_evidence(..., tool_passages=True)``,
   retrieval records only -- the client's documents and the encoded
   knowledge) into passages, calls the pure detector on the final draft, and
   when it fires replaces the draft with the ONE clarifying question and
   records it on the turn (``Turn.questions`` + a ``which_case`` marker on
   ``Turn.awaiting_inputs``, so the ``{"missing": [...]}`` carrier the route
   observer reads stays non-empty). Kill-switch ``CLARIFY_CASES=0``; a
   failure passes the answer through.

2. **A qualifier narrows the ask.** Wiring surfaced a false positive the
   replay had not: "what maximum fresh concrete temperature" (a specific
   subject) kept an "ambient temperature" case because both reduce to the
   head noun "temperature", and the exit asked. The relevance filter now
   keeps a case only when its head matches AND, when the question's subject
   names a qualifier word too, the case shares at least one of those
   qualifiers ("fresh" / "concrete" vs "ambient"). A generic ask ("what is
   the temperature?") names no qualifier and still asks across cases.

Local after the fold: 105 passed across
``tests/test_clarify_cases.py`` + ``tests/test_clarify_cases_exit.py``;
two-input lint 0 new; ruff clean; scan_hardwiring 0 new; scan_repo_hygiene
clean. The full ``_postprocess_answer``-touching suite (40 files, 3093
tests) passes with ``RAG_EMBEDDING_MODEL=fake``.

## Not in this PR

- The instruction text: #495 (part a).
- The same-subject both-values statement: #453 (merged).
