# Experiment 5B: ActiveGCD on Kaggle

Research: Ma et al., **Active Generalized Category Discovery**, CVPR 2024.
Paper: https://arxiv.org/abs/2403.04272
Official code: https://github.com/mashijie1028/ActiveGCD
Pinned source commit: `66598129185020a8cb8842696f8ea660e59d5861`.

## Retained research components

* DINO-pretrained ViT-B/16, 224px inputs, 768-dimensional features.
* Original `DINOHead_feature`, SimGCD classification/contrastive/distillation
  objectives and entropy regularisation, vendored from `model.py` with MIT licence.
* Last transformer block trainable (`grad_from_block: 11`); earlier blocks frozen.
* Adaptive-Novel selection: per-provisional-novel-slot quota, high-margin samples
  before `adaptive_round`, low-margin samples thereafter; no oracle labels enter
  the query function. Default original switch round is configurable, not adapted
  using our test scores.
* EMA model and Hungarian label mapping from **initial and queried labelled
  training images only**; separate supervised active-learning updates.

## Kaggle protocol

Exclude the miscellaneous CSV `unknown` class. Initially label four categories:
bumper dent, door dent, bumper scratch, door scratch. Three initially unlabelled
categories are glass shatter, headlamp and tail lamp. Class names must match the
actual CSV (`head_lamp`, `tail_lamp`, etc.); inspect preparation audit first.

Within the training split, use one fixed group-aware 80% subset of known-class
images as the initial labelled pool. Remaining known-class images plus all novel
training images form the discovery pool. This fixed reserve provides a mixed
known/novel discovery pool for AGCD; it is not a training-percentage ablation.
All class counts/protocol choices are explicit. All seven output slots exist
before annotation: **the total class count is supplied**. Unknown names/labels
are hidden from the learner until query selection; the set of candidate class
names appears in the config for dataset preparation, not as classifier text input.

Run random and adaptive strategies using identical seeds, initial pools, backbone,
number of rounds and annotation budgets. The selected label is revealed by a
simulated oracle only after selection; every reveal is saved in a round CSV.
Do not add test images to discovery, supervised learning or label mapping.

This is pool-based discovery: novel training images are available without labels
at base training. It does not model strictly unseen streaming categories appearing
only after training, nor estimate an unbounded number of categories. DINO
pretraining may contain related visual concepts. Do not claim semantic novelty
relative to every pretraining image.

## Port deviations

The runner is a Kaggle adaptation, **not a bit-for-bit paper reproduction**:

1. It guarantees labelled and unlabelled examples in each mixed minibatch to
   avoid empty supervised losses on this small dataset, rather than relying on
   upstream weighted sampling.
2. If a query cannot be filled with predicted-novel samples, it fills the remainder
   using the same margin ordering and records the fallback count. The upstream
   sampler asserts in this case. This does not use hidden labels.
3. It uses fixed final-epoch EMA checkpoints and never reports the maximum test
   epoch or chooses a checkpoint using test labels. Test metrics are computed
   once per phase/round; hyperparameters and number of epochs must be fixed first.
4. Semantic test accuracy uses only mappings learned from verified training labels.
   Unmapped outputs stay `novel_slot_N`. Offline cluster ACC (Hungarian alignment),
   ARI and NMI may use test labels strictly as evaluation; those mappings are never
   fed back into training or presentation category names.
5. CPU support, configurable learning rate and small batch sizes are added.
6. Phase checkpoints are written atomically; exact interrupted round resume is not
   currently supported. Use a fresh output root for a restart.

Outputs include per-round old/novel semantic accuracy, offline clustering metrics,
novel categories observed in queried labels, annotation counts, test predictions,
verified label-to-slot mappings, checkpoint states and query fallback counts.
Do not compare this adapted protocol's scores directly to published benchmark
tables as if they were a reproduction.
