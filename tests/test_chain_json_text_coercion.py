"""Chain output-unwrapping: a JSON-dict output flowing into a text-expecting
block is coerced to its primary text value.

Regression for the reported bug: a chain like translate -> chat fails with
"chat expects Text but received JSON" because translate returns a dict
({"translated": "...", ...}) with no top-level "text" key, so the
orchestrator's runtime type check classifies it as JSON and rejects it.
"""

from app.blocks.orchestrator import OrchestratorBlock


# ── stub blocks ────────────────────────────────────────────────────────────────

class _TranslateLikeBlock:
    """Step-0 stand-in for `translate`: declares a `text` output (so the chain
    passes upfront validation) but actually returns a JSON dict whose text
    lives under `translated`, never a top-level `text` key."""

    name = "translate_like"
    ui_schema = {"input": {"type": "text"}, "output": {"type": "text"}}

    async def execute(self, input_data, params=None):
        return {
            "status": "success",
            "result": {
                "original": "Good morning",
                "translated": "Buen día",
                "source_language": "en",
                "target_language": "es",
                "char_count": 12,
            },
        }


class _TextOnlyBlock:
    """Step-1 stand-in for `chat`: only accepts text. Records what it received
    so the test can assert the JSON dict was unwrapped before delivery."""

    name = "text_only"
    accepted_input_types = ["Text", "TextContent", "ChatMessage"]
    ui_schema = {"input": {"type": "text"}, "output": {"type": "text"}}

    def __init__(self):
        self.received = None

    async def execute(self, input_data, params=None):
        self.received = input_data
        return {"status": "success", "result": {"text": f"replied to {input_data}"}}


def _orchestrator():
    """An OrchestratorBlock wired with the two stub blocks, no platform deps."""
    orch = OrchestratorBlock()
    step0 = _TranslateLikeBlock()
    step1 = _TextOnlyBlock()
    instances = {"translate_like": step0, "text_only": step1}
    orch.set_platform(registry={}, instance_cache=instances, create_block_fn=lambda b: b)
    return orch, step1


# ── test ────────────────────────────────────────────────────────────────────────

async def test_json_output_unwrapped_for_text_expecting_block():
    """translate -> chat: the text block must receive the translated string,
    not have the chain rejected as a JSON/Text type mismatch."""
    orch, text_block = _orchestrator()

    result = await orch.process(
        "Good morning",
        {"steps": [{"block": "translate_like"}, {"block": "text_only"}]},
    )

    assert result["status"] == "success", result
    assert result["steps_executed"] == 2, result

    # The text-expecting block ran and received the translated text — not the
    # raw translate dict (which would have carried source_language etc.).
    received = text_block.received
    assert received is not None, "text block never executed — chain was rejected"
    assert "Buen día" in str(received), received
    assert "source_language" not in str(received), (
        f"raw JSON dict leaked into the text block: {received}"
    )
