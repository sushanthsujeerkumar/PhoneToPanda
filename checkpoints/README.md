# Saved models

The [15 October release](https://github.com/sushanthsujeerkumar/PhoneToPanda/releases/tag/imu-pilot-20261015)
contains three exact evaluated checkpoints. Each download is about **797 MB**
(760 MiB), unpacking to about **1.20 GB**. These large files are release assets,
separate from the Git history.

| Model | What it contains |
| --- | --- |
| `original` | The original vision model, trained for 5,000 updates. |
| `continuation` | The same model continued for 500 updates on the original 20 demonstrations. |
| `augmented` | The same model continued for 500 updates on those 20 demonstrations plus four simulated replays derived from one paired video and IMU take. |

All three policies take simulator images and robot state as input. The augmented
model uses the IMU take to help construct training demonstrations; it does not
receive live IMU input. See the [pilot report](../docs/imu_pilot_20261015.md) for
the comparison, failures, and limitations.

## Download and run

From the repository root, install the original model with Python 3.11 or newer:

```bash
python3 scripts/download_checkpoint.py
```

After completing the environment setup in the [main README](../README.md), run it
with an explicit checkpoint path:

```bash
bash demo.sh --checkpoint checkpoints/pilot_20261015/original
```

The other two models use the same commands with their names:

```bash
python3 scripts/download_checkpoint.py --model continuation
bash demo.sh --checkpoint checkpoints/pilot_20261015/continuation

python3 scripts/download_checkpoint.py --model augmented
bash demo.sh --checkpoint checkpoints/pilot_20261015/augmented
```

The downloaded folders include weights, model and training configuration,
processor state, normalization statistics, and training provenance. Running a
policy does not require the original recordings or training datasets. On first
load, the runtime may download the upstream SmolVLM2 configuration, processor,
and tokenizer from Hugging Face unless they are already cached.

The downloader uses only Python's standard library. It verifies the archive and
every extracted file against [the release manifest](release_20261015.json), rejects
unsafe archive entries, and installs only after all checks pass. An existing
checkpoint or license folder is preserved; choose another checkout or move your
existing folder yourself before installing again. Allow at least 2.5 GB of free
space for one download and its unpacked checkpoint, in addition to the runtime.

You can also download an asset from the release page yourself and pass its path;
the same integrity checks apply:

```bash
python3 scripts/download_checkpoint.py --model original \
  --archive /path/to/phone-to-panda-original-20261015.tar.gz
```

## Provenance and licenses

The manifest records each file's SHA256 and the SHA256 of its frozen evaluation
identity under [`results/imu_pilot_20261015/evaluation/`](../results/imu_pilot_20261015/evaluation/).
Checkpoint bytes are unchanged from evaluation. Some saved configuration and
provenance fields therefore retain historical local paths; these paths are not
required to run the downloaded policy. Optimizer states and training datasets
are not included, so these are inference checkpoints rather than full training
resumption snapshots.

The weights are fine-tuned derivatives of
[SmolVLA](https://huggingface.co/lerobot/smolvla_base), whose vision-language base is
[SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).
Both model cards specify Apache-2.0. Each archive includes the Apache license,
LeRobot's third-party license notices, source references, and a modification notice
in a sibling folder such as `checkpoints/pilot_20261015/original.licenses/`.

The earlier local `expanded/` model and the
[11 October study](../docs/improvement_20261011.md) checkpoints are not part of this
release. Their measured results and saved metadata remain in the repository.
