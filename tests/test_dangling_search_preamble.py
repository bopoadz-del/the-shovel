"""A promise to search is never a final answer - for ANY question type.
Live WAVE-2 re-smoke (2026-08-31, e6c250e): D1 and E1 both ended on a bare
promise and the user saw it as the answer. Two independent defects:
1. _final_text_needs_forced_retry only forced the no-tools retry for
document-deliverable or generative requests, so plain factual lookups fell
straight through.
2. _SEARCH_PREAMBLE_RE only matched "let me" / "I'll" openers, missing
"I'm searching ...", "I am running a targeted search ...", and "I need to
run the corrected tool call first".
Pinned here with the verbatim live strings.
"""
import pytest
from app.agents.runtime import _final_text_needs_forced_retry
from app.agents.runtime import _looks_like_search_preamble
D1 = "Let me search more precisely for the Engineer's Representative within the project documents."
E1 = "I'm searching the executed contract volume for the Delay Damages figure."
HIST_TAIL = "Because those inputs are absent from the retrieved excerpts, I am running a targeted search of the contract document to pull the clause details."
HIST_MOMENT = "I need to run the corrected tool call first. One moment."
WIR = "I'll pull the WIR template now."
CHECK = "Let me check the project documents."
E4_VALIDATE = "Let me validate…"
E4_VALIDATE_WASTE = "Let me validate the volume against the documented waste factor."
LIVE_DANGLING = [D1, E1, HIST_TAIL, HIST_MOMENT, WIR, CHECK, E4_VALIDATE, E4_VALIDATE_WASTE]
ANS_BOQ = "In the client Demolition BOQ, the unit rate for CESMM4 reference D 549.2 is 80.00 SAR/m for 3,504 m, total 280,320.00 SAR. (Source: Demolition BOQ.pdf, Page d/3/3)"
ANS_LEADVERB = "Search results show three matching specifications; the governing one is SECTION 012650."
ANS_MISS = "I could not confirm this reference in the indexed project sources. Please check whether the document is indexed."
ANS_SPEC = "Specification 012650 - Variation and Adjustments. Revision Date: 16 October 2019."
ANS_DLP = "The Defects Liability period is 1 year, per the Project Execution Plan."
ANS_HELLO = "Hello - how can I help you today?"
ANS_NOTOOL = "No MCP weather/fetch tool is connected to this agent. I cannot retrieve current weather data for Riyadh."
LIVE_REAL_ANSWERS = [ANS_BOQ, ANS_LEADVERB, ANS_MISS, ANS_SPEC, ANS_DLP, ANS_HELLO, ANS_NOTOOL]
Q_ENGINEER = "Under AB-2023-101, who is the Engineer's Representative in the contract?"
Q_DELAY = "Under Sub-Clause 8.8, what is the Delay Damages daily rate in SAR?"
Q_RETENTION = "What retention percentage does the contract specify?"
FACTUAL_LOOKUPS = [Q_ENGINEER, Q_DELAY, Q_RETENTION]
LONG_ANSWER = "I will summarise the retention provisions. " + "The contract states retention is held at the rate in the Schedule. " * 12
@pytest.mark.parametrize("text", LIVE_DANGLING)
def test_live_dangling_preambles_are_detected(text):
  assert _looks_like_search_preamble(text), text

@pytest.mark.parametrize("text", LIVE_REAL_ANSWERS)
def test_real_answers_are_not_preambles(text):
  assert not _looks_like_search_preamble(text), text

@pytest.mark.parametrize("question", FACTUAL_LOOKUPS)
@pytest.mark.parametrize("text", LIVE_DANGLING)
def test_factual_lookup_forces_retry_on_a_dangling_promise(text, question):
  assert _final_text_needs_forced_retry(text, user_message=question)

@pytest.mark.parametrize("text", LIVE_REAL_ANSWERS)
@pytest.mark.parametrize("question", FACTUAL_LOOKUPS)
def test_real_answers_never_force_a_retry(text, question):
  assert not _final_text_needs_forced_retry(text, user_message=question)

def test_empty_still_forces_retry():
  assert _final_text_needs_forced_retry("", user_message="anything")

def test_long_grounded_answer_is_never_a_preamble():
  assert len(LONG_ANSWER) > 500
  assert not _looks_like_search_preamble(LONG_ANSWER)
  
