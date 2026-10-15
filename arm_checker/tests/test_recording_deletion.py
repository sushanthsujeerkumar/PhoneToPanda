"""Deleting a take archives its pair and never partially consumes a bad batch."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arm_checker.app import create_app
from arm_checker.recording import SessionStore


def save_take(store):
    store.start("demo", {"upper_id": "3", "forearm_id": "4"}, {"neutral_ready": True})
    store.append({"synthetic": True}, {"valid": True}, store.started)
    return store.stop()["id"]


def saved_bytes(directory):
    return {path.name: path.read_bytes() for path in directory.glob("*.json")}


def test_bulk_delete_archives_both_files_and_preserves_unselected_content(tmp_path):
    store = SessionStore(tmp_path)
    keep = save_take(store)
    deleted = [save_take(store), save_take(store)]
    original = saved_bytes(tmp_path)
    (tmp_path / ".gitkeep").write_text("keep this")
    (tmp_path / "notes.txt").write_text("unrelated")
    assert store.delete(deleted) == 2
    assert [item["id"] for item in store.list()] == [keep]
    assert store.last_id is None
    batches = list((tmp_path / ".trash").iterdir())
    assert len(batches) == 1
    for identity in deleted:
        for name in (f"{identity}.json", f"{identity}.meta.json"):
            assert not (tmp_path / name).exists()
            assert (batches[0] / name).read_bytes() == original[name]
    assert (tmp_path / f"{keep}.json").read_bytes() == original[f"{keep}.json"]
    assert (tmp_path / ".gitkeep").read_text() == "keep this"
    assert (tmp_path / "notes.txt").read_text() == "unrelated"


@pytest.mark.parametrize("ids", [None, [], "all", [True], ["../outside"], [".gitkeep"], ["a"] * 101])
def test_invalid_selection_cannot_mutate_saved_files(tmp_path, ids):
    store = SessionStore(tmp_path)
    save_take(store)
    before = saved_bytes(tmp_path)
    with pytest.raises(ValueError):
        store.delete(ids)
    assert saved_bytes(tmp_path) == before
    assert not (tmp_path / ".trash").exists()


@pytest.mark.parametrize("bad", ["missing", "duplicate", "bad_id", "missing_metadata", "malformed_data",
                                "malformed_metadata", "nonfinite_data", "wrong_metadata_id"])
def test_whole_batch_is_validated_before_any_file_moves(tmp_path, bad):
    store = SessionStore(tmp_path)
    first, second = save_take(store), save_take(store)
    ids = [first, second]
    if bad == "missing":
        ids[-1] = "missing"
    elif bad == "duplicate":
        ids[-1] = first
    elif bad == "bad_id":
        ids[-1] = "../outside"
    elif bad == "missing_metadata":
        store.path(second).with_suffix(".meta.json").unlink()
    elif bad == "malformed_data":
        store.path(second).write_text("not JSON")
    elif bad == "malformed_metadata":
        store.path(second).with_suffix(".meta.json").write_text("not JSON")
    elif bad == "nonfinite_data":
        store.path(second).write_text(json.dumps({"metadata": {"id": second}, "frames": [float("nan")]}))
    else:
        store.path(second).with_suffix(".meta.json").write_text(json.dumps({"id": "someone_else"}))
    before = saved_bytes(tmp_path)
    with pytest.raises(ValueError):
        store.delete(ids)
    assert saved_bytes(tmp_path) == before
    assert not (tmp_path / ".trash").exists()


@pytest.mark.parametrize("target", ["data", "metadata", "trash"])
def test_symlinks_are_rejected_without_touching_external_files(tmp_path, target):
    store = SessionStore(tmp_path / "sessions")
    identity = save_take(store)
    outside = tmp_path / "outside"
    if target == "trash":
        outside.mkdir()
        (store.directory / ".trash").symlink_to(outside, target_is_directory=True)
    else:
        path = store.path(identity)
        if target == "metadata":
            path = path.with_suffix(".meta.json")
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        store.delete([identity])
    assert outside.exists()
    if target == "trash":
        assert list(outside.iterdir()) == []
    else:
        assert json.loads(outside.read_text())
        assert path.is_symlink()


def test_active_recording_blocks_archive(tmp_path):
    store = SessionStore(tmp_path)
    identity = save_take(store)
    before = saved_bytes(tmp_path)
    store.start("demo", {}, {})
    with pytest.raises(ValueError, match="Stop recording"):
        store.delete([identity])
    assert store.current is not None
    assert saved_bytes(tmp_path) == before
    assert not (tmp_path / ".trash").exists()


def test_failed_paired_move_rolls_back_all_preceding_moves(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    identities = [save_take(store), save_take(store)]
    before = saved_bytes(tmp_path)
    fail_path = store.path(identities[1]).with_suffix(".meta.json")
    original_rename = Path.rename

    def fail_last_move(path, destination):
        if path == fail_path:
            raise OSError("Simulated disk error")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_last_move)
    with pytest.raises(ValueError, match="restored"):
        store.delete(identities)
    assert saved_bytes(tmp_path) == before
    assert store.last_id == identities[-1]
    assert list((tmp_path / ".trash").iterdir()) == []


def test_failed_rollback_retains_data_and_reports_recovery_folder(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    identity = save_take(store)
    data = store.path(identity)
    metadata = data.with_suffix(".meta.json")
    original_data = data.read_bytes()
    original_rename = Path.rename

    def fail_second_move_and_recovery(path, destination):
        if path == metadata or ".trash" in path.parts:
            raise OSError("Simulated unavailable destination")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_second_move_and_recovery)
    with pytest.raises(ValueError, match="Some files remain in") as error:
        store.delete([identity])
    archived = list((tmp_path / ".trash").glob(f"*/{identity}.json"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == original_data
    assert str(archived[0].parent) in str(error.value)
    assert metadata.exists()
    assert store.last_id == identity


def test_delete_api_exits_deleted_replay_and_requires_explicit_ids(tmp_path):
    app = create_app(sessions_dir=tmp_path)
    checker = app.state.checker
    identity = save_take(checker.store)
    checker.start_replay(identity)
    client = TestClient(app)
    for body in ({}, {"all": True}, {"ids": [identity], "all": True}, {"ids": []}):
        assert client.post("/api/sessions/delete", json=body).status_code == 400
        assert checker.store.path(identity).exists()
    response = client.post("/api/sessions/delete", json={"ids": [identity]})
    assert response.status_code == 200
    assert response.json() == {"deleted": 1, "sessions": []}
    assert checker.replay is None
    assert checker.store.last_id is None
    assert client.get("/api/state").json()["source"] == "demo"
    assert client.get(f"/api/sessions/{identity}/download").status_code == 404


def test_delete_response_ignores_unrelated_malformed_metadata(tmp_path):
    app = create_app(sessions_dir=tmp_path)
    identity = save_take(app.state.checker.store)
    unrelated = tmp_path / "notes.meta.json"
    unrelated.write_text("[]")
    response = TestClient(app).post("/api/sessions/delete", json={"ids": [identity]})
    assert response.status_code == 200
    assert response.json() == {"deleted": 1, "sessions": []}
    assert unrelated.read_text() == "[]"


def test_delete_api_preserves_other_replay_and_blocks_active_recording(tmp_path):
    app = create_app(sessions_dir=tmp_path)
    checker = app.state.checker
    keep, delete = save_take(checker.store), save_take(checker.store)
    checker.start_replay(keep)
    client = TestClient(app)
    assert client.post("/api/sessions/delete", json={"ids": [delete]}).status_code == 200
    assert checker.replay["metadata"]["id"] == keep
    checker.replay = None
    checker.store.start("demo", {}, {})
    response = client.post("/api/sessions/delete", json={"ids": [keep]})
    assert response.status_code == 400
    assert checker.store.path(keep).exists()
