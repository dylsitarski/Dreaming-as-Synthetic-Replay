# Team workflow

## Branching
- `main` must always run. Don't push to it directly.
- Create one branch per task, named like `<name>/<short-topic>` (for example `alex/mixup-dream`).
- Open a pull request and get one teammate's review before merging.

## Issues
- Every task gets a GitHub Issue. Link the PR to it with `Closes #N`.
- Use labels such as `infra`, `experiment`, `dream-strategy`, `baseline`, `writing`, `literature`.

## Experiments
- Every run is defined by a config file in `configs/`. Never hard-code hyperparameters in a script.
- Always fix and record the random seed. Report the mean ± std over at least 3 seeds.
- Save metrics as CSV/JSON in `results/<experiment-name>/`. Large checkpoints stay out of git.

## Code style
- Python 3.10 or newer, PyTorch.
- Format with `black` and lint with `ruff` before you push.

## Progress log
The professor asked to be kept posted. After each meeting, add a short entry to `docs/progress-log.md` covering what was tried, the results, and the new references.
