# Progress log

The professor asked to be kept updated on what we have tried and which references we are using. Add the newest entry at the top.

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
