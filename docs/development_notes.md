# Development notes

This log records the order of the main changes and experiments. Dates come from
the saved project records; exact event times are not reconstructed.

## 3 October 2026

The project started with a single pick-and-place task and a phone-video pipeline.
The selected method was to measure object motion, replay it with a simulated
Panda and train SmolVLA on successful robot demonstrations. Recording instructions
and an initial project structure were prepared.

The first practical check was to turn one recording into a successful physical
replay before attempting a larger training run. An IMU was considered and left
out to keep the method focused on visual tracking.

## 4 October 2026

Camera calibration, marker tracking, event labels and robot replays were tested
with several recording batches. Early videos had glare and long gaps in object
tracking. An exploratory fit using relaxed tracking limits completed training
but achieved 0/3 placements. That data was kept separate from the accepted batch.

A later six-recording batch produced successful robot demonstrations, but its
500-update and 5,000-update policies each failed their live evaluation trial.
Checks of the exported images, state/action arrays and saved normalization found
no mismatch in those checks. Lower training loss alone did not solve the grasping
problem.

A one-recording diagnostic tested wrist control. The full learned action output
achieved 0/3 placements at its training start. Replacing rotational outputs with
downward wrist feedback gave 1/3. This was a small test at a familiar start, but
it motivated using the simpler hybrid controller in the next batch.

The four-reference-marker batch used measured 50 mm reference squares and a
30 mm object marker. Five recordings passed the unchanged tracking limits;
`IMG_2097` was rejected at 89.80% measured samples. Twenty successful replays
provided 4,471 training frames. The 5,000-update model achieved 3/5 placements on
new simulator starts. A local NVIDIA kernel-module problem was also resolved so
CUDA training could continue.

The additional-data experiment reused those five recordings to make 40 more
successful replays. After another 1,000 training updates, the adapted model scored
14/20 against the original model's 15/20 on paired starts. The original model was
kept as the default. The extra replays and failed policy trials remain part of the
experiment record.

The project was then reorganised as `PhoneToPanda`, with shorter command names,
relative paths and a smaller set of documentation files. The original experiment
was preserved separately.

## Verification status

The original working project completed source and shell checks, geometry and
annotation checks, a successful setup replay, dataset export, pretrained training
and checkpoint reload. The completed policy evaluations are recorded in
[results](results.md).

The reorganised copy was checked on **4 October 2026, 14:56 BST** in a
new Python environment on the same Ubuntu computer, reusing cached downloads.
The checks gave these results:

- All 44 automated tests passed, including contact-physics checks.
- Tracking `IMG_2092` again produced exactly the preserved measured samples.
- All 20 robot replays succeeded; export produced the same 4,471-frame count.
- Running the dataset command again reused the validated files.
- A two-update training test changed a tiny random model's weights and reloaded
  its checkpoint. This checks the training interface, not pick-and-place ability.
- The saved main policy completed the desktop demo at seed 1000 in 208 steps.
  This repeats a known start and is not added to the reported policy scores.
- The source recordings and copied model weights match their original hashes.

The new desktop launcher initially used a graphics setting that conflicted with
robosuite. It was corrected to use EGL for policy-camera rendering while MuJoCo
opens its own desktop window. The old browser viewer was removed.

These checks did not repeat the full 5,000-update training or test installation on
a clean operating system. The saved models remain local and excluded from Git;
a public checkpoint download has not been published. The detailed check record
is in [reorganization_check.json](../results/reorganization_check.json).

The next useful experiment would examine the missed grasps with matched data and
training budgets. Adding more simulated replays from the same five recordings
did not improve the current paired test.

## Preparing the public repository — 4 October 2026

A screen recording of the desktop demo was added at the start of the README.
The source recording has no audio. It was converted to a smaller MP4 for browser
playback, keeping the full session at normal speed. These interactive attempts
are separate from the saved evaluation trials.

The phone recordings contained location metadata. The public copies were changed
to contain only the existing video stream, without audio or private metadata.
All 2,106 decoded frames and their timing were checked against the private
originals. The tracking, replay, dataset and checkpoint records were updated to
use the public file hashes. Dataset reuse still gives 20 episodes and 4,471 frames;
model weights and robot demonstration arrays are unchanged.
