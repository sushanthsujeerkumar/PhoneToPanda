# Camera calibration

`camera.json` contains the calibration used for all five accepted recordings.
The chessboard had 25 mm squares. Calibration used 24 views and gave a
reprojection RMS error of 0.6634 pixels.

The original recording was `IMG_2090.MOV`. Its hash is recorded in the JSON.
The video is kept in the local working copy but excluded from Git. The supplied
parameters are enough to reproduce trajectory extraction for these recordings.

Record and calculate a new calibration if you change the camera lens, zoom,
focus or image resolution. Do not reuse these parameters for a different camera.
