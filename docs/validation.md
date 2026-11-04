# Implementation checks

These checks verify code paths, not research performance. No Kaggle training
result or benchmark score is supplied in this repository.

* Downloaded and inspected the actual public Kaggle archive: 1,512 image files,
  1,594 CSV rows, and 1,505 unique file hashes after duplicate removal. The
  preparer completed with 1,050/229/226 train/validation/test images and every
  class present in each split. See `kaggle_audit.json`. This is a data audit only,
  not real-data training.
* `python -m pytest -q`: 12 passed on CPU, including a real randomly initialised
  YOLOv8s backbone forward/backward pass.
* Tiny synthetic image fixture: all four Experiment 2 adaptation methods, scratch
  training, all three DINO adaptation methods, frozen-feature probing, validation
  selection of k-NN parameters, test artifact export and Experiment 1 inference
  completed. Synthetic outputs are deliberately NOT committed or shown as results.
* ActiveGCD: base phase, query selection, verified-label mapping, EMA, active
  training, test evaluation and query exports completed for random and adaptive
  strategies with a tiny test backbone and synthetic images.
* Streamlit AppTest: empty and populated artifact views rendered without exceptions.
* Loaded the pinned official DINO ViT-B/16 weights and completed a 224px forward
  pass through the actual ActiveGCD head: feature/projection/logit dimensions
  768/256/7. Confirmed only transformer block 11 is trainable.
* Downloaded actual author CarDD v2 checkpoint: confirmed segmentation task and
  six expected category names. All four feature extractor variants produced
  512-dimensional vectors. Backbone trainable counts: full 5,079,712; LoRA 54,528;
  residual expansion 66,112; frozen 0 (classification head counted separately).

The remote CUDA environment, real-data convergence, class imbalance, annotation
quality, inference speed on the target GPU and model generalisation still require
the remote experiments. Scheduler templates require site-specific resource names.
