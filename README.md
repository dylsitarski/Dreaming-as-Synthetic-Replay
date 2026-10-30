# Dreaming as Synthetic Replay

COMS 5730 (Machine Learning), Group 2, Fall 2026

Can a separate "dream" phase improve how well a neural network generalizes? In this phase the network trains on distorted and recombined versions of its training data. The idea comes from Erik Hoel's *Overfitted Brain Hypothesis*.

**Research question:** What properties must synthetic replay have to improve generalization?

**Hypothesis:** As dream distortion increases, test performance of an overfit model first **improves** and then **degrades**, once the distortion stops carrying meaningful information. If so, there is an optimal distortion strength.

## Experimental setup (planned)

1. **Wake phase:** train a classifier with a dataset size and model capacity chosen so that it measurably overfits.
2. **Dream phase:** continue training on synthetic replay produced by different strategies:
   - noise (Gaussian, salt-and-pepper, …)
   - transformations (affine, elastic, blur, …)
   - recombinations (interpolation / mixup-style blends, patch mixing, …)
3. **Measure** test accuracy and the generalization gap against **distortion strength**, for each strategy.
4. **Compare** against standard baselines: no regularization, standard data augmentation, dropout, weight decay.
5. **Optional:** check whether dreaming helps more when training data is limited.

**Datasets:** MNIST, for sanity checks only. **Fashion-MNIST and/or CIFAR-10** for the real evaluation, which the professor requires.

## Repository layout

```
docs/          proposal, professor feedback, references, meeting notes
src/dreamreplay/  library code (data, models, dream strategies, training loop)
configs/       experiment configs (one file per run/sweep)
experiments/   scripts that launch sweeps
notebooks/     exploration & plotting
results/       metrics/plots (large artifacts are git-ignored)
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow.
