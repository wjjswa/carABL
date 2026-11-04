# carABL — vehicle damage experiments

Remote-training code and a Streamlit results viewer for the ABL interview project.
No dataset images, pretrained weights or fabricated experiment results are committed.
Training is intended for your remote GPU platform; the automated checks use tiny
synthetic fixtures and random-weight models only.

## Agreed experiment matrix

| Experiment | Initialisation / backbone | Training | Evaluation |
|---|---|---|---|
| 1 | Original CarDD YOLOv8s v2 segmentation checkpoint | None on target | Fixed Kaggle test images; save original, boxes, masks and structured predictions |
| 2 | CarDD-pretrained YOLOv8s backbone + pooled linear head | Full, convolutional LoRA, CNN block expansion, frozen-backbone linear probing; 20/40/60/80/100% labelled training pool | Trained head and k-NN using the same labelled fraction; fixed test set |
| 3 | Same CarDD-pretrained backbone | DINO teacher–student adaptation on **all training images without labels**, with full / LoRA / block expansion | Freeze final EMA teacher features; fit linear heads and k-NN reference banks using 20/40/60/80/100% labels; fixed test set |
| 4 | Same YOLO backbone architecture, **random weights** + linear head | Train entire backbone and head using 20/40/60/80/100% labels | Trained head and k-NN; fixed test set |
| 5B | Official ActiveGCD architecture: DINO-pretrained ViT-B/16, SimGCD head | Single fixed known/novel split; last ViT block + head; repeated random or Adaptive-Novel queries and verified-label mapping | Fixed-epoch test results after each annotation round, old/novel semantic accuracy and offline clustering metrics |

**There is no Experiment 5A.** The default matrix has three seeds. For each seed,
Exp2 has 20 classification fits, Exp3 has 3 DINO fits + 15 linear probes, Exp4 has
5 scratch fits, and Exp5B has 2 strategies. Every classification evaluation also
performs k-NN, without training another backbone. The full matrix is substantial:
run single configurations first before submitting all jobs.

## Remote setup

Your previous workflow used Artemis and Conda. The project does not assume an SSH
credential, GPU partition or scheduler. Use your usual login and GPU allocation;
do not train on the login node. Run from the repository root.

```bash
git clone https://github.com/wjjswa/carABL.git
cd carABL
module load CUDA/12.4.0
module load Miniconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
# Only create this environment once; skip creation if carabl already exists.
conda create -n carabl python=3.11 -y
conda activate carabl
# Keep your platform's compatible CUDA-enabled PyTorch, or install one appropriate
# for its driver. Example for a CUDA 12.4-compatible installation:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
bash scripts/setup_remote.sh
```

`scripts/slurm_job.sh` and `scripts/sge_job.sh` are **scheduler templates**; edit
resource directives to match your platform. Single-run commands below also work
inside an interactive GPU allocation. CPU runs are supported via `--device cpu`.
This code is single-process/single-GPU; launch separate jobs for different seeds.

All job/setup scripts source `scripts/artemis_env.sh`, which loads
`CUDA/12.4.0` and `Miniconda3/4.12.0`, initialises Conda and activates
`carabl`. It disables user-site packages so the earlier Python 3.9 user install
cannot interfere. It does not update the shared Conda base. Set
`CARABL_CONDA_ENV` only if you intentionally use a different project environment.

From the repository root, after preparing the dataset/checkpoint:

```bash
git pull
sbatch scripts/slurm_job.sh supervised --experiment 2 --method full --fraction .2 --seed 0
```

Adjust the Slurm resource/partition flags to your Artemis allocation. Job scripts
verify GPU access before starting. The setup script permits CPU-only checks on
the login node. To run the entire matrix within an existing GPU allocation, use
`bash scripts/run_experiments.sh`; this may exceed a single job's wall time.

## Obtain and inspect the target dataset

Download the [specified Kaggle dataset](https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment)
using your existing Kaggle access and unzip under `data/kaggle`. Do not commit
credentials, images, checkpoints or output directories.

```bash
# Optional, if the Kaggle CLI and credentials are already configured:
kaggle datasets download -d hamzamanssor/car-damage-assessment -p data/kaggle --unzip
python -m carrepair prepare --root data/kaggle --csv data/kaggle/data.csv --out data/prepared
```

The preparer detects common path/label column names; override with
`--path-column image --label-column classes` if needed. Folder-per-class datasets
can omit `--csv`. A supplied vehicle-ID column can be used via `--group-column`.

It audits readable images, removes exact duplicates, groups near-duplicate images
by perceptual hash, and creates a fixed approximately 70/15/15 group-stratified
train/validation/test manifest. It fails if any class lacks independent groups in
any split. Inspect `audit.json`, class counts, related-vehicle grouping and label
quality before training. Hashing cannot prove source independence or find every
related vehicle. Audit CarDD overlap separately if its images are available.

**Never overwrite an established split.** The preparer refuses an existing output.
If an official split is available, create a manifest with columns
`id,path,label,group,split`, relative image paths, and `train/val/test` values,
preserving vehicle groups; set `manifest` in the config to that file. All runs
share it. Run metadata records its hash. Training percentages apply only to the
training pool; the fixed labelled validation set is additional. Group-aware nested
subsets may differ slightly from the requested percentage; exact counts are saved.

## Configure once

Copy `configs/default.yaml` to a local file and edit `data_root`, `manifest`,
`checkpoint`, `output`, `device`, batch sizes and workers. All paths can be
absolute on your remote platform. No workstation-specific paths are embedded.

```bash
python -m carrepair download-model
python -m carrepair matrix --config configs/default.yaml  # print only
```

The model download is `abdullahg7/cardd-yolov8s/v2.0/best.pt`; its SHA-256 is saved.
Only load trusted model checkpoints. Experiments 2/3 reuse backbone modules 0–9
through SPPF; the code verifies that boundary. The neck and segmentation head are
not trained for these image-classification experiments.

## Run individual experiments

```bash
# Exp1: unchanged model, images + boxes/masks + mapped image-level scores
python -m carrepair direct --config configs/default.yaml

# Exp2: repeat for full/lora/block/linear and .2/.4/.6/.8/1.0
python -m carrepair supervised --experiment 2 --method lora --fraction .2 --seed 0

# Exp3: adapt once per method/seed on ALL unlabelled training images
python -m carrepair dino --method lora --seed 0
# Then evaluate the SAME frozen final teacher with each labelled fraction
python -m carrepair probe --method lora --fraction .2 --seed 0

# Exp4: random initialisation; no pretrained weights are loaded
python -m carrepair supervised --experiment 4 --method full --fraction .2 --seed 0

# Exp5B: same fixed split, backbone, query budget, and training for both strategies
python -m carrepair activegcd --strategy random --seed 0
python -m carrepair activegcd --strategy adaptive --seed 0
```

Pass `--config PATH` to every command when using a custom config. Use `--resume`
for interrupted Exp2/3/4 runs. Exp3 uses `dino_last.pt`; probing resumes separately.
Exp5B saves phase checkpoints and annotation records but currently requires a
fresh output root to restart; exact mid-round resume is not implemented. Keep
completed runs, and never silently overwrite them. For all experiments:

```bash
python -m carrepair matrix --config configs/default.yaml --run
python -m carrepair export --output outputs
```

## Evaluation and presentation

Classification reports include macro-F1, accuracy, per-class precision/recall,
confusion matrices, per-image predictions, confidence scores, training histories
and trainable parameter counts. k and voting temperature are selected on
validation data only. Its reference bank contains the same labelled subset used
by that run; test/validation images never enter the bank. In Exp3, backbone
adaptation uses more unlabelled images at small label budgets; interpret gains
accordingly. Frozen-backbone Exp2 is the no-DINO representation baseline.

Exp1 merges bumper/door dent to `dent`, bumper/door scratch to `scratch`, and
head/tail lamps to `lamp_broken`. The CSV's `unknown` class is excluded from its
four-class mapped scoring. CarDD `crack` and `tire_flat` have no matching target
label; all instance outputs are still saved. The highest confidence compatible
detection gives the mapped top-1 result; no compatible detection is an abstention.
These image-level proxy scores do not establish localisation performance.
**No box/mask mAP is claimed without independently annotated target boxes/masks.**

```bash
python -m carrepair export --output outputs
python -m streamlit run app.py --server.address 127.0.0.1
```

The viewer shows label-efficiency curves, predictions, Exp1 box/mask overlays,
k-NN neighbours, annotation examples, ActiveGCD rounds and run metadata. Set the
image/results roots in its sidebar when relocating artifacts. It starts with an
empty-state message until actual results exist. For remote presentation, use your
normal SSH forwarding, e.g. `ssh -L 8501:localhost:8501 YOUR_REMOTE_HOST`, then
open `http://localhost:8501`. Keep the server bound to localhost.

## Method definitions and limitations

* Full: update all backbone parameters and the linear head.
* LoRA: frozen original backbone, train low-rank branches on ungrouped 1×1
  convolutions (rank/alpha configured), plus the classification head.
* Block expansion: frozen original CNN, append an identity-initialised residual
  1×1 bottleneck block after SPPF, plus the classification head. This is an explicit
  CNN adaptation baseline, **not** a reproduction of transformer block expansion.
* Linear probing: freeze backbone **and BatchNorm running statistics**; train the
  linear head. During Exp3 probing all adapted modules remain frozen as well.
* DINO: original-style multi-crop EMA teacher/student self-distillation with
  centring/sharpening, implemented for the CNN. This is a documented adaptation,
  not DINOv2 or a claim to reproduce ImageNet DINO training. Validate against
  representation collapse and inspect per-class behaviour before interpreting it.

See [docs/activegcd.md](docs/activegcd.md) for the research port and its assumptions.
Image-level classification gains in Exp2–4 do not prove improved masks or boxes.
Dataset and model licences must be checked for use beyond this research prototype.

## Verification

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python -m compileall -q carrepair app.py
```

The test suite covers leakage rejection, grouped nested subsets, label-free DINO
data, teacher gradients, frozen BatchNorm, identity-initialised adapters,
k-NN selection, ActiveGCD sampling/mapping and a real YOLO forward/backward pass.
Remote GPU training quality and Kaggle performance require running the jobs.
