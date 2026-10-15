"""The pilot viewer exposes only its own replay and leaves observations intact."""
import json
from http.server import ThreadingHTTPServer
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from scripts.view_imu_pilot import ReplayView, make_handler


@pytest.fixture
def viewer(tmp_path):
    view = ReplayView(tmp_path / "episode.npz")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(view))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield view, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_viewer_uses_real_callback_observations_without_modifying_them(viewer):
    view, base = viewer
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/frame.jpg")
    assert error.value.code == 503
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    original = image.copy()
    view.observe(SimpleNamespace(steps=5, control_hz=20), {
        "observation.images.image": image, "observation.images.image2": image,
    }, "carry_imu")
    np.testing.assert_array_equal(image, original)
    with urlopen(base + "/state") as response:
        state = json.load(response)
    assert state["status"] == "running" and state["frames"] == 5
    assert state["simulation_seconds"] == .25 and state["success"] is None
    with urlopen(base + "/frame.jpg?frame=5") as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.read().startswith(b"\xff\xd8")
    view.finish({"success": False, "failure": "No firm grasp", "video": None})
    with urlopen(base + "/state") as response:
        state = json.load(response)
    assert state["success"] is False and state["failure"] == "No firm grasp"


def test_server_only_serves_selected_artifacts_and_supports_video_seeking(viewer):
    view, base = viewer
    view.output.with_suffix(".mp4").write_bytes(b"0123456789")
    # Existing files are not exposed until the replay report declares its video.
    for path in ("/replay.mp4", "/../episode.mp4", "/episode.npz", "/state/../episode.mp4"):
        with pytest.raises(HTTPError) as error:
            urlopen(base + path)
        assert error.value.code == 404
    view.finish({"success": True, "video": str(view.output.with_suffix(".mp4"))})
    with urlopen(Request(base + "/replay.mp4", headers={"Range": "bytes=2-5"})) as response:
        assert response.status == 206 and response.read() == b"2345"
        assert response.headers["Content-Range"] == "bytes 2-5/10"
    with urlopen(Request(base + "/replay.mp4", headers={"Range": "bytes=-3"})) as response:
        assert response.status == 206 and response.read() == b"789"
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/replay.mp4", headers={"Range": "bytes=20-25"}))
    assert error.value.code == 416
