# Third-party research and assets

* `carrepair/vendor/activegcd/model.py`: verbatim from ActiveGCD commit
  66598129185020a8cb8842696f8ea660e59d5861. MIT licence preserved beside it.
  Query and label-mapping logic in `carrepair/activegcd.py` is adapted from that
  repository's Adaptive-Novel sampler and `utils_al/train_recipes_ema.py`.
* ActiveGCD builds on SimGCD and DINO; cite the respective papers in presentations.
* The external DINO backbone is loaded through the official Facebook Research
  repository, pinned to 7c446df5b9f45747937fb0d72314eb9f7b66930a. No weights are
  redistributed here. Its upstream licence applies.
* Ultralytics is a runtime dependency with its own AGPL/commercial licence terms.
* The author-provided CarDD checkpoint and original CarDD images have separate
  terms. Kaggle data has its own provenance/licensing; no images are redistributed.

This repository contains experimental glue and a documented research adaptation;
it does not claim authorship of the underlying architectures or research methods.
