# E6 AGE / E7 DP-BOA: Open-World Vehicle Damage Discovery

## Attribution and status

Both experiments are **carABL adaptations, NOT faithful end-to-end reproductions**. E6 follows the Gaussian expansion mechanism in the [AGE author repository](https://github.com/Ashengl/AGE) (ICLR 2026) with a frozen DINO ViT-B/16 encoder and known-only logistic-regression head instead of upstream full training. E7 is an **independent NIW/Dirichlet-process posterior-predictive implementation inspired by [DP-BOA](https://arxiv.org/abs/2607.13504)** (ECCV 2026), not official author code. The [official DP-BOA implementation](https://github.com/algpy/DP-BOA) is now available (verified 7 November 2026). This branch retains its existing independent baseline: it uses fixed `kappa0`/`nu_extra` and covariance pooled across known support, rather than the paper's empirical-Bayes prior estimated from within-class covariances and between-class means. Hyperparameters require validation; these results must not be called official AGE/DP-BOA reproductions.

## Strict shared protocol

1. Reuse the immutable, group-disjoint data/prepared/manifest.csv. Only 4 known categories are initially used: bumper_dent, door_dent, bumper_scratch, door_scratch.
2. Select an 80% group-aware fraction of the **known training data** for supervised model fitting. Known validation images alone calibrate the AGE confidence thresholds. No images or labels from new categories enter training, PCA fitting or threshold calibration.
3. Use the SAME held-out shuffled test stream of 144 known+novel images per seed for E6 and E7. Miscellaneous unknown category is excluded as an unstructured catch-all. The three evaluation-only novel categories are glass_shatter, head_lamp, tail_lamp; **neither their names nor their count are given to the learner**.
4. Shared frozen pretrained DINO ViT-B/16 features (ImageNet normalization, 224px); PCA fitted only to known support; known-only logistic regression on the original embeddings for E6. E7 uses the same PCA features with its independent online decision model.
5. **Prequential protocol**: for each incoming image, first decide known category/new or existing novel cluster, then update unsupervised cluster statistics. Held-out labels are never used by either algorithm. Stream order is shuffled deterministically by seed.
6. Seeds are 0/1/2. Feature files are cached per seed with a metadata fingerprint. Classifier features may include prior semantic knowledge from public DINO pretraining; this does not establish universal semantic novelty.

There is no encoder fine-tuning/epoch schedule in E6/E7. The only supervised fit
is the known-only logistic probe (`max_iter=2000`); E7's decisions use the NIW
model rather than that probe. The top-level `epochs`, `lr`, LoRA settings and
E5B active-learning rounds do not control these runs. `open_world` sets the
support fraction, PCA dimension and online hyperparameters. `batch_size` and
`workers` control frozen feature extraction; `seeds`, `device`, `data_root`,
`manifest` and `output` retain their usual meanings. An older custom YAML needs
the `open_world` block copied from `configs/default.yaml`; use a local config copy
for E6/E7 and retain the old YAML with the earlier results.

## Algorithms

**E6 AGE adaptation:** Known-only classifier probabilities and known validation confidence thresholds decide familiar vs rejected. Rejected images are scored under Gaussian prototypes; if the most likely prototype belongs to a known category, create a novel Gaussian; if it belongs to a discovered novel category, update that Gaussian and its covariance. This mimics the decision logic in the AGE author's project_utils/AGE.py, without reproducing the entire upstream visual training pipeline.

**E7 DP-BOA-inspired adaptation:** For each existing cluster compute log(count) plus NIW posterior predictive multivariate Student-t log likelihood. For a new cluster compute log(alpha) plus NIW base-prior predictive log likelihood. Compare existing-versus-new evidence; create a new cluster only if the birth score wins. Means/scatters update online on unlabeled features. This is an independently coded baseline and should not be labeled an exact paper reproduction.

## Verification scope

The automated suite checks known-only partitioning, online births/assignments,
NIW sufficient statistics and decisions before updates, plus offline-only label
alignment. A synthetic-image integration test exercises real image loading,
PCA/probe fitting, both online algorithms, shared feature-cache reuse, run
provenance and complete paired-seed aggregation. Its encoder is a tiny local
fixture; no pretrained weights or benchmark results are fabricated. Submission
tests cover custom configs/seeds, repeated launches, failed-output protection,
the shell alias, and unchanged E1–E5 plans. The earlier `smoke --activegcd`
command covers E1–E5 only; it does not establish real DINO/GPU E6/E7 execution.

Local validation of this workflow update on 7 November 2026: **34 tests passed**
under Ubuntu 22.04, Python 3.10.12, CPU PyTorch 2.6.0 / torchvision 0.21.0,
SciPy 1.15.3 and scikit-learn 1.7.2. Python compilation and Bash syntax checks
also passed. Real pretrained DINO inference, CUDA and scheduler execution remain
to be checked on Artemis with the existing data and environment.

## Run on Artemis

Use the same existing checkout, `carabl` environment and prepared data as E1–E5.
The path below is the established path from `docs/open_set.md`. Use your existing
SSH login/alias; no new environment, dataset split or workflow is needed.
Switch/update only when no jobs using this checkout are queued or running.
If `git status` shows local edits, retain them and reconcile them before switching.

```bash
cd /mnt/nfs2/engdes/jw840/carABL
git status --short
git fetch origin
git switch experiments/e6-age-e7-dpboa-20261107
git pull --ff-only origin experiments/e6-age-e7-dpboa-20261107
source scripts/artemis_env.sh
mkdir -p logs
python -m pip freeze > logs/e6-e7-environment-before.txt
python -m pip install --no-deps -e .
python -m pip check
OMP_NUM_THREADS=1 python -m pytest -q
```

The editable install registers this checkout without upgrading the existing
PyTorch/dependencies. Do not recreate `carabl`, reinstall CUDA/PyTorch or run a
dependency upgrade for E6/E7. Dependencies are already declared in the project;
if imports/checks fail, resolve that missing dependency before submitting.
`scripts/artemis_env.sh` still loads `CUDA/12.4.0`, `Miniconda3/4.12.0`, switches
via base to `carabl`, and sets `PYTHONNOUSERSITE=1`.

Both methods reuse E5B's pinned DINO ViT-B/16 source and Torch cache. They do not
require `weights/cardd-v2.pt` or an E3/E5 checkpoint. On a login/data-transfer node
that permits downloads, pre-cache the encoder on CPU (no dataset fitting):

```bash
python -c 'import torch; from carrepair.common import read_config; cfg=read_config("configs/default.yaml"); torch.hub.load(cfg["open_world"]["dino_repo"], "dino_vitb16", trust_repo=True).eval(); print("DINO cached at", torch.hub.get_dir())'
sinfo -p gpu -o "%P %a %l %G"
```

The cache must be readable at the same path on compute nodes. Keep the usual
`TORCH_HOME` if already set. Pre-caching also includes the pinned Torch Hub source.

Preview, then submit through the SAME E1–E5 planner and GPU wrapper:

```bash
bash scripts/submit_all.sh --experiments 6 7 --registry logs/submissions_e6_e7.json
bash scripts/submit_all.sh --experiments 6 7 --registry logs/submissions_e6_e7.json --submit
squeue -u "$USER"
```

The default config produces six jobs: E6/E7 × seeds 0/1/2. The default E1–E5
selection remains 136 jobs. Explicitly selecting `1 2 3 4 5b 6 7` produces 142.
`scripts/submit_open_world.sh` is now only a compatibility alias for the same
planner with `--experiments 6 7`; **it previews by default and requires `--submit`**.
Pass `--config PATH` for your established custom YAML, and use that same config
for aggregation. Seeds/output/paths come from the YAML; they are not hard-coded.
Use an E6/E7 registry to retain the older submission record unchanged.

Resources are inherited from `scripts/slurm_job.sh`: one GPU, 8 CPUs, 32 GB RAM,
partition `gpu`, 24 hours. Override with planner `--partition NAME --time TIME`
if required by your allocation. Logs are `logs/carABL-e6-sN-JOBID.log` and
`logs/carABL-e7-sN-JOBID.log`; IDs and commands are in the registry and
`logs/experiment_plan.csv`. Check actual job completion, not just an activation banner:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
tail -n 100 logs/carABL-e6-s0-JOBID.log
```

Repeating the same launcher/registry reuses recorded job IDs and skips completed
outputs; it does not automatically retry failed jobs. E6/E7 do **not** support
mid-stream resume. Incomplete unrecorded output blocks submission even with
`--resume`. After checking job state, use a deliberate fresh output root/config
and registry for a restart. Do not remove successful results. The shared feature
cache is locked per seed and rejects incompatible metadata rather than silently
reusing it. Cache v2 includes data-root/path and extraction-library versions;
older caches are retained but rejected and need a deliberate fresh output root.

Single jobs still use the prior convention (do not also queue the same run via
the planner):

```bash
sbatch scripts/slurm_job.sh open-world --config configs/default.yaml --method e6 --seed 0
sbatch scripts/slurm_job.sh open-world --config configs/default.yaml --method e7 --seed 0
```

Inside an existing GPU allocation, run the corresponding `python -m carrepair
open-world ...` command directly. CPU debugging uses `--device cpu --output
outputs_debug` in a compute allocation. Do not run experiment fitting on a login node.

## Aggregate and download only new outputs

After all six jobs succeed, in the same environment/config:

```bash
python -m carrepair open-world --config configs/default.yaml --aggregate
cat outputs/open_world_comparison/mean_std.csv
tar --exclude='*.pt' --exclude='*.lock' -czf carABL_e6_e7_results.tar.gz \
    -C outputs exp6 exp7 open_world_comparison
```

Aggregation requires both methods for every configured seed, matching config,
manifest, paired stream order, feature-cache hash and SciPy/scikit-learn versions.
It refuses incomplete/mixed comparisons. Earlier E6/E7 runs without these
fingerprints remain untouched; keep/report them separately rather than mixing
them into this checked comparison. `metrics.json` is written atomically last as
the completion marker. The archive contains only E6/E7 reports/predictions/
provenance and their comparison. Dataset images, old results, pretrained weights
and `open_world_feature_cache/` stay on Artemis. Keep the feature cache there for
repeatability. All paths above use the default `output: outputs`; substitute your
configured output root if different.

From your own computer, using your existing SSH host or alias:

```bash
scp YOUR_ARTEMIS_SSH_ALIAS:/mnt/nfs2/engdes/jw840/carABL/carABL_e6_e7_results.tar.gz .
```

This is the same tar/download pattern as the earlier open-set experiment. Do not
rerun E1–E5 or download their results again. The existing general `export` and
Streamlit viewer do not render E6/E7's distinct streaming metrics; use these CSVs.

## HPC assumptions to verify on site

Local CPU tests cannot verify Artemis's NVIDIA driver, CUDA-enabled PyTorch,
module names, account/QoS access, memory/wall-time adequacy, network access or
shared-filesystem `flock`/atomic-rename semantics. The wrapper checks CUDA access
in the actual allocation. Verify the current partition limit with `sinfo`; the
repository's three-day GPU maximum is prior supplied information, not a live
cluster measurement. The [Sussex Artemis service page](https://www.sussex.ac.uk/its/services/research/highperformance)
confirms Slurm, but does not publish these exact module versions or limits.
The established NFS path and SSH alias must still be valid for your account.

## Results and meaning

Outputs: outputs/exp6/seedN and outputs/exp7/seedN each contain known_support.csv, known_validation.csv, stream_without_labels.csv, stream_predictions.csv, per_category.csv, metrics.json, protocol.json and provenance.json. Outputs/open_world_comparison/per_seed.csv and mean_std.csv are produced by --aggregate. Test prediction files contain ground-truth and offline aligned labels **for scoring only**, never for training.

Metrics: known semantic accuracy; novel detection recall; false novel referral on known samples; hindsight Hungarian cluster accuracy overall and for novel images; NMI; ARI; estimated number of novel clusters; novel count absolute error; category-level recall and prediction-as-novel rate. Novel cluster IDs are operationally unnamed; Hungarian accuracy uses test ground truth **only for offline evaluation** and must never be presented as deployed semantic accuracy.

## Critical limitations

* The documented prior manifest has only a 144-image test stream (83 known, 61 novel), not a deployment evaluation. These counts are not hard-coded: inspect each run's actual saved counts. Order, prior parameters, confidence thresholds and features can cause over/under-splitting.
* Test-time unsupervised updates are expressly permitted, but this is different from frozen test evaluation. No human labeling is assumed.
* No actual GPU results are claimed until jobs complete. Hyperparameters have not been tuned on an independent open-world development dataset.
* Do not compare these E6/E7 scores directly to E5B as if the training, available unlabeled data, or annotation budget were equivalent. E5B receives all novel unlabelled training images and fixes 7 output slots.
