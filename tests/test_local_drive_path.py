"""local_drive: paths are honored from the `path`/`folder_path`/`file_path`
params and confined to the drive root — path traversal is blocked.
"""

from app.blocks.local_drive import LocalDriveBlock


async def test_list_honors_path_and_folder_path_params(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DRIVE_ROOT", str(tmp_path))
    (tmp_path / "a.txt").write_text("x")
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "b.txt").write_text("y")

    root_listing = await LocalDriveBlock().process(
        None, {"operation": "list", "path": "."}
    )
    assert root_listing["status"] == "success", root_listing
    # Listing now returns dicts with name/is_folder/size_bytes/modified so
    # the UI can distinguish folders from files (the old shape was a list
    # of bare strings, which made every entry look like a flat file).
    names = [e["name"] for e in root_listing["files"]]
    assert "a.txt" in names
    assert "docs" in names
    docs_entry = next(e for e in root_listing["files"] if e["name"] == "docs")
    assert docs_entry["is_folder"] is True

    # A subdirectory passed via the `folder_path` param is honored.
    sub_listing = await LocalDriveBlock().process(
        None, {"operation": "list", "folder_path": "docs"}
    )
    assert sub_listing["status"] == "success", sub_listing
    sub_names = [e["name"] for e in sub_listing["files"]]
    assert sub_names == ["b.txt"]


async def test_read_and_write_within_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DRIVE_ROOT", str(tmp_path))

    written = await LocalDriveBlock().process(
        None, {"operation": "write", "file_path": "out/note.txt", "content": "hello"}
    )
    assert written["status"] == "success", written

    read = await LocalDriveBlock().process(
        None, {"operation": "read", "file_path": "out/note.txt"}
    )
    assert read["status"] == "success", read
    assert read["content"] == "hello"


async def test_path_traversal_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DRIVE_ROOT", str(tmp_path))
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("TOP SECRET")

    # Reading outside the root via .. must fail and must not leak the file.
    escaped = await LocalDriveBlock().process(
        None, {"operation": "read", "file_path": "../secret.txt"}
    )
    assert escaped["status"] == "error", escaped
    assert "SECRET" not in str(escaped)

    # Writing outside the root via .. must fail.
    wescaped = await LocalDriveBlock().process(
        None, {"operation": "write", "file_path": "../pwned.txt", "content": "x"}
    )
    assert wescaped["status"] == "error", wescaped
    assert not (tmp_path.parent / "pwned.txt").exists()

    # Listing the parent directory must fail.
    listed = await LocalDriveBlock().process(
        None, {"operation": "list", "path": "../"}
    )
    assert listed["status"] == "error", listed


async def test_nonexistent_path_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DRIVE_ROOT", str(tmp_path))
    result = await LocalDriveBlock().process(
        None, {"operation": "list", "path": "does-not-exist"}
    )
    assert result["status"] == "error", result
    assert "Not a directory" in result["error"]
