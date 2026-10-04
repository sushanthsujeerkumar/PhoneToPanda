# Saved models

The working copy has two local model folders:

- `main/`: the original 5,000-update model. It scored 15/20 in the paired test.
- `expanded/`: the extra 1,000-update model. It scored 14/20 in that test.

The model files are excluded from Git because each is about 1.2 GB. They have not
been uploaded. The public repository includes the recordings and training code
needed to reproduce them.

You can reproduce the main training run from the supplied recordings instead.
The training instructions save a new model under `runs/train/`. The demo launcher
can use that checkpoint when `main/` is absent. Check the main README for commands.
