# Progress log

The professor asked to be kept updated on what we have tried and which references we are using. Add the newest entry at the top.

## 2026-11-08: Regularization baselines (dropout, weight decay)
- Same setup as the wake-phase baseline plus one regularizer at a time, 3 seeds per setting: dropout in the MLP head (0.25, 0.5, 0.75) and decoupled weight decay with AdamW (0.03, 0.1, 0.3, 1, 10). The strength to report is chosen by mean validation accuracy, not test accuracy.
- Neither improves test accuracy. Best dropout (0.25): 85.3% ± 0.2%. Best weight decay (0.1): 85.2% ± 0.3%. No regularization: 85.3% ± 0.5%. All dropout rates and weight decay up to 0.3 are within noise of each other. Stronger weight decay hurts (83.4% at 1, 76.1% at 10).
- They do fix overconfidence. Final test loss falls from 0.99 to 0.43 with dropout 0.75 and to 0.61 with weight decay 1. With dropout 0.75, test loss reaches its minimum only at epoch 56, versus epoch 7 without regularization.
- The generalization gap misleads on its own. Weight decay 10 has the smallest gap (2.9 points) only because it cannot fit the training set, and it has the worst test accuracy.
- What this means for the dream phase: in this setup, overfitting shows up as overconfidence rather than lost accuracy, and limiting capacity does not recover accuracy. If dreaming raises test accuracy, it is doing something dropout and weight decay cannot. Standard data augmentation, which also adds new training inputs, is the key baseline still to run.
- New reference: Loshchilov & Hutter (2019) on decoupled weight decay (AdamW).

## 2026-10-30: Wake-phase baseline
- Small CNN (2 conv layers, width 32, no dropout or weight decay) trained with Adam (lr 1e-3) for 60 epochs on a class-balanced 2,000-image Fashion-MNIST subset.
- Results over 3 seeds: train accuracy 100%, test accuracy 85.3% ± 0.5%, generalization gap 14.7% ± 0.5%. Train accuracy reaches 100% by about epoch 30. From epoch 10 to 60, test loss roughly doubles (about 0.48 to 1.0) while test accuracy stays flat, so the model clearly overfits.
- Checkpoints (`results/baseline_fmnist/seed*/wake.pt`, not committed) will be the starting point for the dream phase.

## 2026-10-30
- Repository initialized. Proposal and professor feedback added to `docs/`.
- Feedback to address:
  - Survey standard data augmentation for vision, and consider new augmentations inspired by the visual cortex during sleep.
  - Compare against regularizers such as dropout.
  - Be aware of augmentation for generative models (e.g., GANs). Experiments with them are not required.
  - MNIST is fine for sanity checks, but evaluation must use Fashion-MNIST or CIFAR-10.
