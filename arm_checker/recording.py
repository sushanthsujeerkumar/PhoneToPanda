"""Local JSON sessions keep raw measurements, calibration and displayed estimates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import stat
import time
import uuid


def _reject_nonfinite(value):
    raise ValueError(f"Session JSON contains a non-finite number: {value}")


class SessionStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.current = None
        self.last_id = None

    def status(self):
        return {"active": self.current is not None,
                "frames": len(self.current["frames"]) if self.current else 0,
                "session_id": self.current["metadata"]["id"] if self.current else self.last_id}

    def start(self, source, config, calibration):
        if self.current:
            raise ValueError("A recording is already running.")
        now = datetime.now(timezone.utc)
        self.started = time.monotonic()
        self.current = {"metadata": {
            "id": now.strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8],
            "created_utc": now.isoformat(), "schema_version": 1,
            "source": source, "synthetic": source == "demo", "config": dict(config),
            "calibration": calibration,
            "clock_note": "Device microseconds and host monotonic seconds are different clocks. Camera synchronization is not performed here.",
            "pose_note": "Two-segment estimate relative to a fixed shoulder; not measured absolute hand position or robot commands.",
        }, "frames": []}

    def append(self, frame, pose, received_at):
        if not self.current or received_at < self.started:
            return
        self.current["frames"].append({"host_elapsed_s": received_at - self.started,
                                       "host_monotonic_s": received_at,
                                       "device_frame": frame, "pose": pose})

    def stop(self):
        if not self.current:
            raise ValueError("Start a recording first.")
        session = self.current
        meta = session["metadata"]
        meta.update(duration=max(0., time.monotonic() - self.started), frames=len(session["frames"]),
                    valid_pose_frames=sum(bool(f["pose"].get("valid")) for f in session["frames"]))
        destination = self.path(meta["id"])
        temporary = destination.with_suffix(".partial")
        temporary.write_text(json.dumps(session, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(destination)
        destination.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        self.last_id = meta["id"]
        self.current = None
        return meta

    def path(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            raise ValueError("Invalid session identifier.")
        return self.directory / f"{session_id}.json"

    def delete(self, session_ids):
        """Move complete saved sessions to a private recovery folder as one batch."""
        if self.current:
            raise ValueError("Stop recording before deleting saved sessions.")
        if not isinstance(session_ids, list) or not 1 <= len(session_ids) <= 100:
            raise ValueError("Choose between 1 and 100 saved session IDs.")
        # Validate every ID before inspecting or moving any files.
        paths = [self.path(identity) for identity in session_ids]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Choose each saved session only once.")
        root = self.directory.resolve()
        if self.directory.is_symlink() or not root.is_dir():
            raise ValueError("The session directory must be a regular directory.")

        files = []
        for identity, path in zip(session_ids, paths):
            pair = (path, path.with_suffix(".meta.json"))
            contents = []
            for candidate in pair:
                try:
                    info = candidate.lstat()
                    if (not stat.S_ISREG(info.st_mode) or candidate.resolve().parent != root):
                        raise ValueError("Saved session files must be regular files inside the session directory.")
                    contents.append(json.loads(candidate.read_text(), parse_constant=_reject_nonfinite))
                except (OSError, UnicodeError, ValueError) as error:
                    raise ValueError(f"Cannot delete {identity}: a session file is missing, malformed or unsafe.") from error
                files.append((candidate, info.st_dev, info.st_ino))
            data, metadata = contents
            if (not isinstance(data, dict) or not isinstance(metadata, dict)
                    or not isinstance(data.get("metadata"), dict)
                    or data["metadata"].get("id") != identity or metadata.get("id") != identity
                    or not isinstance(data.get("frames"), list)):
                raise ValueError(f"Cannot delete {identity}: the saved data and metadata do not match this session.")

        trash = root / ".trash"
        if trash.is_symlink() or (trash.exists() and not trash.is_dir()):
            raise ValueError("The session recovery folder must be a regular directory.")
        batch = trash / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex)
        moved = []
        try:
            trash.mkdir(mode=0o700, exist_ok=True)
            batch.mkdir(mode=0o700)
            for path, device, inode in files:
                # Reject a replacement that appeared after batch validation.
                current = path.lstat()
                if (not stat.S_ISREG(current.st_mode)
                        or (current.st_dev, current.st_ino) != (device, inode)):
                    raise OSError("A saved session file changed while preparing deletion.")
                destination = batch / path.name
                path.rename(destination)
                moved.append((path, destination))
        except OSError as error:
            rollback_failed = False
            for original, archived in reversed(moved):
                try:
                    # Never overwrite a newly created file during recovery.
                    if original.exists() or original.is_symlink():
                        raise OSError("The original filename is occupied.")
                    archived.rename(original)
                except OSError:
                    rollback_failed = True
            try:
                batch.rmdir()
            except OSError:
                pass
            if rollback_failed:
                raise ValueError(f"Deletion could not finish. Some files remain in {batch}; no files were erased. Restore them before retrying.") from error
            raise ValueError("Deletion could not finish; moved files were restored. No sessions were deleted.") from error
        if self.last_id in session_ids:
            self.last_id = None
        return len(session_ids)

    def list(self):
        sessions = []
        for path in sorted(self.directory.glob("*.meta.json"), reverse=True)[:100]:
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                item = json.loads(path.read_text())
                if (not isinstance(item, dict)
                        or self.path(item.get("id")).with_suffix(".meta.json") != path):
                    continue
                sessions.append({key: item.get(key) for key in ("id", "created_utc", "duration", "frames", "source", "synthetic", "valid_pose_frames")})
            except (ValueError, OSError):
                continue
        return sessions
