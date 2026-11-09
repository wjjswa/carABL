# E14 without Hugging Face approval: DINOv2 + SigLIP2

Use this replacement plan when the original DINOv3 access request is pending.
All three model repositories are public and ungated; anonymous model metadata,
config/processor files and weight-file access were checked on 9 November 2026.
No `hf auth login` or access-approval step is required. This is a different
backbone comparison from the original DINOv3 plan; label results **E14 ungated
(DINOv2/SigLIP2)** rather than reporting them as DINOv3 results.

| Variant | Model | Trainable parameters |
|---|---|---|
| E14-A | `facebook/dinov2-small`, ViT-S/14 | Linear six-class head; encoder frozen |
| E14-B | `facebook/dinov2-small`, ViT-S/14 | Last four encoder blocks, final norm, head |
| E14-C | `facebook/dinov2-base`, ViT-B/14 | Last four encoder blocks, final norm, head |
| E14-D | `google/siglip2-base-patch16-224` | Last four vision blocks, final norm, head |

These preserve E11's exact 1050/229/226 image splits, six root classes, seeds
0/1/2, 300 maximum epochs, validation macro-F1 selection/patience 30, and the
existing same-seed E9-B experts for hard/soft eight-class evaluation.
DINOv2 uses its published processor (resize shortest side to 256 and centre
crop to 224, then ImageNet normalization). SigLIP2 uses its published 224px
processor. DINOv2's patch size is 14 rather than the original DINOv3's 16.
Source revisions are pinned in `configs/e14_ungated.yaml` and source files are
SHA256-fingerprinted by the training pipeline.

## 1. Update the existing worktree

Run these on Artemis after the E13 jobs have left the queue. The update retains
the original E13 configuration and outputs.

```bash
cd /mnt/nfs2/engdes/jw840/carABL-e13e14
git fetch origin
git merge --ff-only origin/experiments/e13-e14-resolution-foundation-20261107
source scripts/artemis_env.sh
python -m pip install 'transformers==4.57.6' 'safetensors>=0.4,<1'
```

If you previously queued any DINOv3 E14 jobs, inspect `squeue -u jw840` and
their logs/registry before running a different E14 plan. Do not alter their
configuration or outputs. This replacement explicitly uses a new output and
registry so existing runs are never reused as DINOv2 results.

## 2. Create a NEW config from the completed E11 setup

```bash
python scripts/create_e13e14_artemis_config.py \
  --e11 ../carABL-e11/configs/e11_artemis.yaml \
  --template configs/e14_ungated.yaml \
  --output configs/e14_ungated_artemis.yaml
```

The generated YAML inherits E11's actual manifest, images, CarDD checkpoint,
E9-B paths and completed baseline. It writes results into
`outputs/e14_ungated`, leaving `outputs/e13e14` and E13's registry unchanged.
If this new YAML already exists from a previous invocation, reuse it rather
than overwriting it. The generator refuses missing original E11 inputs.

## 3. Download public models without authentication

```bash
python scripts/download_e14_models.py \
  --config configs/e14_ungated_artemis.yaml \
  --variants e14a e14c e14d --anonymous
```

E14-A/B share one DINOv2-S snapshot. `--anonymous` explicitly ignores cached
Hugging Face tokens, including expired or unapproved tokens. Downloads run
on the network-enabled login node, outside GPU jobs. Ungated means no access
approval is needed; Artemis must still be able to reach Hugging Face and its
weight download service. A network failure is separate from an access error.

## 4. Check real inputs and models, then submit

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m carrepair.e13e14_preflight \
  --config configs/e14_ungated_artemis.yaml \
  --variants e14a e14b e14c e14d --model-smoke
```

Only after `PREFLIGHT PASSED`, submit **24 jobs**: 12 root training plus
12 matching-seed dependent evaluation jobs, with no expert retraining:

```bash
bash scripts/submit_e13e14.sh \
  --config configs/e14_ungated_artemis.yaml \
  --variants e14a e14b e14c e14d \
  --registry logs/e14_ungated_submissions.json --submit
```

Preview the plan by omitting `--submit`. Preflight covers a small CPU
forward/backward and strict expert loading, not CUDA capacity or accuracy.
The recorded-job and retry rules in [the original guide](e13e14.md) apply.

```bash
squeue -u jw840 -o "%.12i %.45j %.2t %R"
sacct -u jw840 -S 2026-11-09 --format=JobID,JobName%40,State,ExitCode
```

## 5. Aggregate and download

Once all 24 jobs complete successfully:

```bash
python -m carrepair.e13e14 aggregate \
  --config configs/e14_ungated_artemis.yaml \
  --variants e14a e14b e14c e14d

tar --exclude='*.pt' --exclude='*.tmp' \
  -czf /mnt/nfs2/engdes/jw840/carABL_e14_ungated_results.tar.gz \
  -C outputs e14_ungated
```

The results include mean ± sample standard deviation over three seeds,
per-class scores, hard/soft predictions, learning curves, and paired image
comparisons with E11. Download the archive with your usual `scp`/SFTP client.
These are experiments on the repeatedly inspected historical test set; no
new claim of independent generalization is implied.

## Validation

The actual public pretrained DINOv2-S snapshot was downloaded anonymously and
passed frozen and last-four-block 224px CPU forward/backward checks. This checks
model compatibility; it is not training or accuracy evaluation on the benchmark.
The DINOv2 adapter is tested with real small local Transformers model classes:
official processor style, frozen/last-four-block gradient boundaries and exact
checkpoint reload. Tests also verify the separate ungated config, pinned
revisions and anonymous downloader arguments. Original DINOv3 and SigLIP2
support is retained for the earlier configuration. No E14 benchmark results
are claimed before the Artemis runs finish.
