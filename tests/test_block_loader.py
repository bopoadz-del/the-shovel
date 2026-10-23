"""The block registry loads resiliently — one broken block module does not
abort the whole registry (and therefore the app)."""

from app.blocks import BLOCK_REGISTRY, FAILED_BLOCKS, _load_blocks


def test_registry_loads_and_is_populated():
    """A healthy install loads generic blocks; construction is kit-gated."""
    from app.core.domain_kit_loader import is_virgin_boot

    assert "chat" in BLOCK_REGISTRY and "pdf" in BLOCK_REGISTRY
    assert FAILED_BLOCKS == {}, f"blocks failed to load: {FAILED_BLOCKS}"
    if is_virgin_boot():
        assert len(BLOCK_REGISTRY) >= 15
        assert "construction" not in BLOCK_REGISTRY
    else:
        assert len(BLOCK_REGISTRY) >= 30


def test_loader_isolates_a_broken_block():
    """A spec whose module cannot be imported is recorded in `failed` and the
    other blocks still load — the loader does not raise."""
    registry, failed = _load_blocks([
        ("good", "app.blocks.translate", "TranslateBlock"),
        ("broken_module", "app.blocks.does_not_exist_xyz", "Nope"),
        ("broken_class", "app.blocks.translate", "ClassThatDoesNotExist"),
        ("also_good", "app.blocks.search", "SearchBlock"),
    ])

    # The good blocks loaded.
    assert "good" in registry and "also_good" in registry
    # The broken ones were isolated, not raised.
    assert "broken_module" in failed
    assert "broken_class" in failed
    assert "broken_module" not in registry
    assert "broken_class" not in registry


def test_class_reexports_still_work():
    """`from app.blocks import XBlock` keeps working for loaded blocks."""
    from app.blocks import ChatBlock, SearchBlock

    assert ChatBlock is BLOCK_REGISTRY["chat"]
    assert SearchBlock is BLOCK_REGISTRY["search"]
    if "local_drive" in BLOCK_REGISTRY:
        from app.blocks import LocalDriveBlock

        assert LocalDriveBlock is BLOCK_REGISTRY["local_drive"]
    else:
        from app.blocks.local_drive import LocalDriveBlock

        assert LocalDriveBlock.__name__ == "LocalDriveBlock"
