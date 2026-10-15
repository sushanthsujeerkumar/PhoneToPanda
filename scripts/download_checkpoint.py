"""Download one exact evaluated PhoneToPanda checkpoint from this repo's release.

Uses only the Python standard library. Archive and file hashes come from the
versioned release manifest; existing checkpoint folders are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("original", "continuation", "augmented")
REPOSITORY = "sushanthsujeerkumar/PhoneToPanda"
TAG = "imu-pilot-20261015"
PREFIX = "checkpoints/pilot_20261015"
CHUNK = 1024 * 1024


def safe_relative_path(value: str) -> PurePosixPath:
    """Accept canonical relative POSIX names; no Windows or parent traversal."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("Invalid archive path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != value or ":" in value:
        raise ValueError("Unsafe archive path")
    return path


def validate_file_info(info: dict):
    if (not isinstance(info, dict) or not isinstance(info.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None
            or isinstance(info.get("bytes"), bool) or not isinstance(info.get("bytes"), int)
            or not 0 <= info["bytes"] < 2 * 1024**3):
        raise ValueError("Invalid release file size or SHA256")


def load_model(manifest_path: Path, model: str) -> dict:
    if model not in MODELS:
        raise ValueError("Unknown checkpoint model")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("repository") != REPOSITORY
            or manifest.get("release_tag") != TAG):
        raise ValueError("Unexpected checkpoint release manifest")
    entry = manifest.get("models", {}).get(model)
    if not isinstance(entry, dict):
        raise ValueError("Checkpoint is missing from the release manifest")
    checkpoint, notices = f"{PREFIX}/{model}", f"{PREFIX}/{model}.licenses"
    if entry.get("checkpoint_path") != checkpoint or entry.get("notices_path") != notices:
        raise ValueError("Unexpected checkpoint destination")
    archive = entry.get("archive", {})
    validate_file_info(archive)
    filename = f"phone-to-panda-{model}-20261015.tar.gz"
    expected_url = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{filename}"
    if archive.get("filename") != filename or archive.get("url") != expected_url:
        raise ValueError("Unexpected release download URL")
    files = entry.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Release manifest has no file inventory")
    for name, info in files.items():
        path = safe_relative_path(name)
        if not any(path.is_relative_to(PurePosixPath(base)) and str(path) != base
                   for base in (checkpoint, notices)):
            raise ValueError("Release file is outside its checkpoint or licenses folder")
        validate_file_info(info)
    if sum(info["bytes"] for info in files.values()) > 5 * 1024**3:
        raise ValueError("Unpacked checkpoint exceeds the supported size")
    if not any(name.startswith(notices + "/") for name in files):
        raise ValueError("Release manifest omits the license notices")
    checkpoint_hashes = {name[len(checkpoint) + 1:]: info["sha256"]
                         for name, info in files.items() if name.startswith(checkpoint + "/")}
    if checkpoint_hashes != entry.get("checkpoint_files_sha256") or "model.safetensors" not in checkpoint_hashes:
        raise ValueError("Checkpoint file inventory differs from its evaluation identity")
    return entry


def verify_file(path: Path, info: dict):
    if path.stat().st_size != info["bytes"]:
        raise ValueError("File size does not match the release manifest")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != info["sha256"]:
        raise ValueError("SHA256 does not match the release manifest")


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme != "https":
            raise ValueError("Refusing a download redirect away from HTTPS")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def download_archive(entry: dict, destination: Path, opener=None):
    archive = entry["archive"]
    if urllib.parse.urlsplit(archive["url"]).scheme != "https":
        raise ValueError("Checkpoint downloads require HTTPS")
    opener = opener or urllib.request.build_opener(HTTPSOnlyRedirect())
    request = urllib.request.Request(archive["url"], headers={"User-Agent": "PhoneToPanda-checkpoint-downloader"})
    created = False
    try:
        with opener.open(request, timeout=60) as response:
            if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                raise ValueError("Checkpoint download left HTTPS")
            received, next_progress = 0, 64 * CHUNK
            with destination.open("xb") as output:
                created = True
                while chunk := response.read(CHUNK):
                    received += len(chunk)
                    if received > archive["bytes"]:
                        raise ValueError("Download is larger than the release manifest permits")
                    output.write(chunk)
                    if received >= next_progress:
                        print(f"Downloaded {received / CHUNK:.0f} / {archive['bytes'] / CHUNK:.0f} MiB", flush=True)
                        next_progress += 64 * CHUNK
        verify_file(destination, archive)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def extract_verified(archive: Path, staging: Path, files: dict):
    """Copy only named regular files into a private staging directory.

    Do not use extractall: links, special entries and duplicate names are refused.
    Each streamed member must match both its declared size and its file hash.
    """
    seen = set()
    with tarfile.open(archive, "r|gz") as package:
        for member in package:
            relative = safe_relative_path(member.name)
            if (not member.isfile() or member.issparse() or member.name in seen
                    or member.name not in files):
                raise ValueError("Archive contains an unexpected, linked or nonregular entry")
            info = files[member.name]
            if member.size != info["bytes"]:
                raise ValueError("Archive member size differs from the release manifest")
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum, count = hashlib.sha256(), 0
            with package.extractfile(member) as source, target.open("xb") as output:
                while chunk := source.read(CHUNK):
                    count += len(chunk)
                    checksum.update(chunk)
                    output.write(chunk)
            if count != info["bytes"] or checksum.hexdigest() != info["sha256"]:
                raise ValueError("Archive member SHA256 differs from the release manifest")
            target.chmod(0o644)
            seen.add(member.name)
    if seen != set(files):
        raise ValueError("Archive is missing files from the release manifest")


def install_checkpoint(model: str = "original", root: Path = ROOT,
                       manifest_path: Path | None = None, archive: Path | None = None) -> Path:
    root = Path(root).resolve()
    entry = load_model(manifest_path or ROOT / "checkpoints/release_20261015.json", model)
    parent = root
    for part in PurePosixPath(PREFIX).parts:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError("Checkpoint destination parents must not be symbolic links")
        parent.mkdir(exist_ok=True)
    destination, notices = root / entry["checkpoint_path"], root / entry["notices_path"]
    for path in (destination, notices):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Preserving existing folder: {path}")
    lock = parent / f".{model}.download.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{model}-download-", dir=parent) as folder:
            staging = Path(folder)
            if archive is None:
                selected = staging / entry["archive"]["filename"]
                download_archive(entry, selected)
            else:
                selected = Path(archive).resolve()
                verify_file(selected, entry["archive"])
            contents = staging / "contents"
            extract_verified(selected, contents, entry["files"])
            # Keep the checkpoint invisible until every file has been verified.
            # Notices install first; roll them back if final installation fails.
            for path in (destination, notices):
                if path.exists() or path.is_symlink():
                    raise FileExistsError(f"Preserving existing folder: {path}")
            os.rename(contents / entry["notices_path"], notices)
            try:
                os.rename(contents / entry["checkpoint_path"], destination)
            except BaseException:
                shutil.rmtree(notices)
                raise
    finally:
        lock.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="original")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded release archive; all checks still apply")
    args = parser.parse_args(argv)
    try:
        destination = install_checkpoint(model=args.model, archive=args.archive)
    except urllib.error.HTTPError as error:
        parser.exit(1, f"Release download failed (HTTP {error.code}). Check that the release asset is available.\n")
    except (OSError, ValueError, tarfile.TarError, urllib.error.URLError) as error:
        parser.exit(1, f"Checkpoint was not installed: {error}\n")
    print(f"Verified checkpoint installed: {destination}")
    print(f"License and modification notices: {destination}.licenses")


if __name__ == "__main__":
    main()
