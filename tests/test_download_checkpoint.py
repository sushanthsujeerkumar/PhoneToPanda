"""Release installs are verified before publication into their final folder."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from urllib.request import Request

import pytest

from scripts import download_checkpoint as download


def file_info(contents):
    return {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}


def write_archive(path, entries):
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as package:
        for name, contents, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(contents)
                package.addfile(member, io.BytesIO(contents))
            else:
                member.linkname = "../../outside"
                package.addfile(member)


@pytest.fixture
def release(tmp_path):
    checkpoint = f"{download.PREFIX}/original"
    notices = f"{download.PREFIX}/original.licenses"
    contents = {
        checkpoint + "/model.safetensors": b"test weights",
        checkpoint + "/config.json": b'{"type": "smolvla"}\n',
        notices + "/NOTICE.txt": b"Test notice\n",
    }
    archive = tmp_path / "release.tar.gz"
    write_archive(archive, [(name, data, tarfile.REGTYPE) for name, data in contents.items()])
    filename = "phone-to-panda-original-20261015.tar.gz"
    entry = {
        "checkpoint_path": checkpoint,
        "notices_path": notices,
        "archive": {
            "filename": filename,
            "url": f"https://github.com/{download.REPOSITORY}/releases/download/{download.TAG}/{filename}",
            **file_info(archive.read_bytes()),
        },
        "files": {name: file_info(data) for name, data in contents.items()},
        "checkpoint_files_sha256": {
            name[len(checkpoint) + 1:]: file_info(data)["sha256"]
            for name, data in contents.items() if name.startswith(checkpoint + "/")
        },
    }
    manifest = {
        "schema_version": 1, "repository": download.REPOSITORY,
        "release_tag": download.TAG, "models": {"original": entry},
    }
    manifest_path = tmp_path / "manifest.json"
    root = tmp_path / "repository"
    root.mkdir()
    fixture = SimpleNamespace(root=root, manifest=manifest, manifest_path=manifest_path,
                              archive=archive, entry=entry, contents=contents)
    save_manifest(fixture)
    return fixture


def save_manifest(release, refresh_archive=False):
    if refresh_archive:
        release.entry["archive"].update(file_info(release.archive.read_bytes()))
    release.manifest_path.write_text(json.dumps(release.manifest))


def install(release):
    return download.install_checkpoint(root=release.root, manifest_path=release.manifest_path,
                                       archive=release.archive)


def assert_no_install_or_temporary_files(release):
    parent = release.root / download.PREFIX
    assert not parent.exists() or list(parent.iterdir()) == []


def test_local_archive_installs_verified_files_and_preserves_existing_target(release):
    result = install(release)
    assert result == release.root / release.entry["checkpoint_path"]
    for name, contents in release.contents.items():
        assert (release.root / name).read_bytes() == contents
    assert {path.name for path in result.parent.iterdir()} == {"original", "original.licenses"}
    (result / "user-note.txt").write_text("keep this")
    with pytest.raises(FileExistsError, match="Preserving"):
        install(release)
    assert (result / "user-note.txt").read_text() == "keep this"
    assert release.archive.exists()


def test_outer_archive_corruption_cleans_up_without_touching_source(release):
    with release.archive.open("ab") as stream:
        stream.write(b"unexpected bytes")
    before = release.archive.read_bytes()
    with pytest.raises(ValueError, match="size"):
        install(release)
    assert_no_install_or_temporary_files(release)
    assert release.archive.read_bytes() == before


@pytest.mark.parametrize("case", ["wrong_hash", "missing", "extra", "duplicate"])
def test_bad_archive_inventory_never_leaves_partial_checkpoint(release, case):
    entries = [(name, data, tarfile.REGTYPE) for name, data in release.contents.items()]
    if case == "wrong_hash":
        name, data, kind = entries[0]
        entries[0] = name, b"x" * len(data), kind
    elif case == "missing":
        entries.pop()
    elif case == "extra":
        entries.append((release.entry["checkpoint_path"] + "/unlisted.txt", b"extra", tarfile.REGTYPE))
    else:
        entries.append(entries[0])
    write_archive(release.archive, entries)
    save_manifest(release, refresh_archive=True)
    with pytest.raises(ValueError):
        install(release)
    assert_no_install_or_temporary_files(release)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.DIRTYPE])
def test_link_and_nonregular_archive_members_are_rejected(release, kind):
    entries = [(name, data, tarfile.REGTYPE) for name, data in release.contents.items()]
    entries[0] = entries[0][0], b"", kind
    write_archive(release.archive, entries)
    save_manifest(release, refresh_archive=True)
    with pytest.raises(ValueError, match="nonregular"):
        install(release)
    assert_no_install_or_temporary_files(release)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "a/../../outside", "a\\outside",
                                  "C:/outside", "a//outside", "a/./outside", "."])
def test_archive_path_traversal_and_noncanonical_paths_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        download.safe_relative_path(name)
    archive = tmp_path / "unsafe.tar.gz"
    write_archive(archive, [(name, b"payload", tarfile.REGTYPE)])
    with pytest.raises(ValueError):
        download.extract_verified(archive, tmp_path / "staging", {name: file_info(b"payload")})
    assert not (tmp_path / "staging").exists()


def test_manifest_refuses_changed_inventory_or_download_location(release):
    release.entry["files"][release.entry["checkpoint_path"] + "/config.json"]["sha256"] = "0" * 64
    save_manifest(release)
    with pytest.raises(ValueError, match="evaluation identity"):
        install(release)
    release.entry["archive"]["url"] = "http://example.com/file.tar.gz"
    save_manifest(release)
    with pytest.raises(ValueError, match="download URL"):
        install(release)
    assert_no_install_or_temporary_files(release)


def test_destination_parent_symlinks_are_refused(release, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (release.root / "checkpoints").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        install(release)
    assert list(outside.iterdir()) == []


def test_atomic_install_failure_rolls_back_notices_and_staging(release, monkeypatch):
    rename = download.os.rename

    def fail_checkpoint(source, destination):
        if Path(destination).name == "original":
            raise OSError("simulated filesystem failure")
        return rename(source, destination)

    monkeypatch.setattr(download.os, "rename", fail_checkpoint)
    with pytest.raises(OSError, match="simulated"):
        install(release)
    assert_no_install_or_temporary_files(release)


class Response(io.BytesIO):
    def __init__(self, data, url="https://release-assets.githubusercontent.com/asset"):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url


def test_https_download_checks_hash_and_does_not_overwrite_existing_file(tmp_path):
    data = b"archive payload"
    entry = {"archive": {"url": "https://github.com/release.tar.gz", **file_info(data)}}
    destination = tmp_path / "download.tar.gz"
    opener = SimpleNamespace(open=lambda request, timeout: Response(data))
    download.download_archive(entry, destination, opener)
    assert destination.read_bytes() == data
    destination.write_bytes(b"existing file")
    with pytest.raises(FileExistsError):
        download.download_archive(entry, destination, opener)
    assert destination.read_bytes() == b"existing file"


@pytest.mark.parametrize("data", [b"short", b"expected payload plus extra", b"x" * len(b"expected payload")])
def test_failed_download_removes_partial_archive(tmp_path, data):
    entry = {"archive": {"url": "https://github.com/release.tar.gz", **file_info(b"expected payload")}}
    destination = tmp_path / "download.tar.gz"
    with pytest.raises(ValueError):
        download.download_archive(entry, destination, SimpleNamespace(open=lambda request, timeout: Response(data)))
    assert not destination.exists()


def test_download_and_redirects_cannot_downgrade_to_http(tmp_path):
    entry = {"archive": {"url": "https://github.com/release.tar.gz", **file_info(b"data")}}
    opener = SimpleNamespace(open=lambda request, timeout: Response(b"data", "http://example.com/asset"))
    with pytest.raises(ValueError, match="HTTPS"):
        download.download_archive(entry, tmp_path / "download", opener)
    with pytest.raises(ValueError, match="HTTPS"):
        download.HTTPSOnlyRedirect().redirect_request(
            Request("https://github.com/release.tar.gz"), None, 302, "Found", {}, "http://example.com/asset")
    assert list(tmp_path.iterdir()) == []


def test_committed_release_manifest_matches_frozen_evaluation_identities():
    root = Path(__file__).resolve().parents[1]
    for model in download.MODELS:
        entry = download.load_model(root / "checkpoints/release_20261015.json", model)
        identity = root / entry["evaluation_identity"]["path"]
        assert hashlib.sha256(identity.read_bytes()).hexdigest() == entry["evaluation_identity"]["sha256"]
        assert json.loads(identity.read_text())["checkpoint_files_sha256"] == entry["checkpoint_files_sha256"]
