# Validation notes

## 15 October 2026: publication check

All **111 Python tests** and **10 browser-timer tests** passed. These cover the
packet reader, calibration, session storage and recoverable deletion, plus the
five-second countdown and automatic save timer. The current 11-page illustrated
guide was rendered and inspected page by page.

One selected physical recording contains 1,501 valid serial poses over about
15 seconds. Its [paired phone/IMU inspection and limitations](../docs/imu_pilot_20261015.md)
are documented separately. This is not a calibrated accuracy or latency result.
The app's forward-direction instructions now explicitly require both table +X
arrows to point toward the direction the wearer will face while recording.

## 11 October 2026: initial software checks

These checks used synthetic data. No mounted hardware accuracy, drift or
physical latency result is claimed.

## Automated checks

On Linux with Python 3.12.3, all **63 tests passed**:

```sh
arm_checker/.venv/bin/python -m pytest -q arm_checker/tests
```

Coverage includes corrupt and oversized packets, quaternion conventions,
known joint angles, calibration order and sample counts, sensor assignments,
frozen timestamps, stale input, device resets, session recording and replay.
The test runner reported one upstream Starlette/AnyIO deprecation warning.

A separate synthetic pseudo-terminal check exercised the actual serial reader,
AM packet decoder, calibration and recorder together:

| Check | Observed result |
| --- | --- |
| Table capture | 2.037 seconds |
| Neutral capture | 3.025 seconds |
| Recorded poses | 25 of 25 valid |
| Device timestamps | Distinct and increasing |
| Known elbow bend | 90 degrees recovered |
| Corrupt packets | Zero in this clean synthetic stream |
| Disconnect | Cleared calibration and invalidated geometry |

No physical serial port was opened for that check. It verifies software data
flow, not the accuracy of an IMU or the mounting procedure.

## Browser and document review

The browser flow was checked through table alignment, neutral capture,
record/stop, replay and return to the source. With no compatible USB receiver,
the live view reported disconnection and disabled recording. Demo data and
saved demo sessions remained labelled synthetic. WebSocket updates, camera
views and the help guide were checked locally. Desktop and narrow layouts
were inspected; browser compatibility beyond this Chromium environment has
not been established.

All nine PDF pages were rendered and visually inspected. The guide is a
proposed capture procedure, not a completed camera/IMU comparison.

## Next physical check

Verify both sensor IDs separately, perform table and mounted calibration,
then compare the display with a straight arm, an elbow bend, a forward lift
and a sideways lift. Check inward/outward motion and repeated neutral returns.
Record any mismatch before collecting the pilot. The earlier rig's reported
direction reversal has not been physically diagnosed by these software tests.
