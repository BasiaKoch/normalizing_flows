# M2 Deep Learning Coursework Submission

This repository is the submission for the M2 Deep Learning normalizing-flows coursework.

## What To Run

The submission is self-contained in [coursework.ipynb](coursework.ipynb). On a clean CPU-only machine, install the dependencies from [pyproject.toml](pyproject.toml) and run the notebook from top to bottom.

The notebook regenerates all required artifacts:

- [results.json](results.json)
- [figs/Figure1c.pdf](figs/Figure1c.pdf)
- [figs/Figure2a.pdf](figs/Figure2a.pdf)
- [figs/Figure2c.pdf](figs/Figure2c.pdf)
- [figs/Figure3b.pdf](figs/Figure3b.pdf)
- [checkpoints/flow_full.pt](checkpoints/flow_full.pt)
- [logs/training_curves.json](logs/training_curves.json)

The written response is included in [results.json](results.json) under the `writeup` key as a single Markdown string. No separate report PDF is required.

## Repository Layout

- [coursework.ipynb](coursework.ipynb): full implementation, training, analysis, and artifact export
- [pyproject.toml](pyproject.toml): Python dependencies
- [data/](data): provided train, validation, and test CSV files
- [figs/](figs): generated PDF figures
- [checkpoints/](checkpoints): generated trained-model checkpoint
- [logs/](logs): generated training curves JSON
- [scripts/run_notebook.sh](scripts/run_notebook.sh): helper script used to execute the notebook non-interactively

## Notes

- The notebook is written for CPU execution only.
- Random seeds are set inside the notebook for reproducibility.
- No normalizing-flow library is used; the affine coupling flow is implemented directly in PyTorch.
