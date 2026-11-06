# Seven known damage classes and assessor referral

This experiment compares full fine-tuning from the CarDD YOLOv8s v2 checkpoint with training the same backbone and classification head from random weights. Both models predict seven damage classes. The dataset's miscellaneous `unknown` photos are reserved for the final evaluation. Earlier eight-class checkpoints cannot be reused for this comparison.

## Submit on Artemis

Use your existing carABL checkout, Conda environment, checkpoint and prepared data. The login node submits a single GPU job that runs the two models for seeds 0, 1 and 2 sequentially. Each of the six fits trains for 100 epochs, giving 600 training epochs in total.

```bash
cd /mnt/nfs2/engdes/jw840/carABL
git pull --ff-only origin main
bash scripts/submit_open_set.sh
squeue -u "$USER"
```

Slurm prints the job ID. Use it in `tail -f logs/open-set-JOBID.log`. The job loads the existing Artemis modules and the `carabl` environment. Allow roughly 0.6 GB for the six runs and their checkpoints. The script defaults to `configs/default.yaml`. You may pass the path to your own configuration as its first argument. The default paths are `data/prepared/manifest.csv`, `data/kaggle` and `weights/cardd-v2.pt`.

Re-run the submission command after an interruption. It checks the run configuration and file hashes, skips completed runs and resumes a saved training checkpoint. One writer can use an output directory at a time. When changing the protocol or inputs, choose a fresh `output` in the configuration. No earlier experiment needs to be repeated.

For an existing GPU allocation, the equivalent training command is:

```bash
python -m carrepair open-set --config configs/default.yaml --resume
```

## Data and decision rule

| Split in the existing manifest | Used for | Images |
|---|---|---:|
| Known training photos only | Training all backbone and head parameters and building a reference feature bank | 668 |
| Known validation photos only | Best checkpoint by macro-F1 and selection of two referral cutoffs | 147 |
| Known test photos | Exact-label accuracy, wrong automatic labels and unnecessary referrals | 144 |
| Miscellaneous unknown test photos | Correct referrals and false acceptance as named damage | 82 |

The 382 training and 82 validation rows labelled `unknown` are excluded. All runs share this split. The dedicated setting `open_set_epochs: 100` controls the duration of each fit. Both use the same augmentation, optimiser and cosine schedule spanning those 100 epochs. Scratch uses the YOLOv8s segmentation architecture specification to initialise fresh weights. Only its backbone and a new seven-class head enter the classifier. Pretraining image overlap with Kaggle has not been established.

For each fitted model, the trained head proposes a damage class. The referral score is the cosine similarity of the photo's feature vector to its fifth-nearest known training photo. Both feature extraction and comparison use the best validation checkpoint with deterministic evaluation preprocessing. Cutoffs are the 10th and 20th percentiles of similarity on the known validation photos (nominal 90% and 80% known acceptance). A score below a cutoff sends the photo to a human. Each model gets cutoffs for the same two target acceptance rates. The mixed test set is scored after calibration.

## Read and download the results

```bash
cat outputs/open_set/comparison.csv
tar --exclude='*.pt' --exclude='.run.lock' -czf carABL_open_set_results.tar.gz -C outputs open_set
```

| Output under `outputs/open_set/` | Contents |
|---|---|
| `comparison.csv` | Mean and sample SD across seeds, by model and target coverage. Rates are fractions. |
| `per_seed.csv` | Counts and rates at each cutoff for each fit, including known-only accuracy before referral and unknown-detection AUROC. |
| `{finetune,scratch}_seedN/predictions.csv` | Test image ID, path, true label, proposed known class, maximum softmax score, similarity and both referral outcomes. |
| `{finetune,scratch}_seedN/calibration.csv` | Known validation images and similarity scores used to select the cutoffs. |
| Each run directory | Known-only training manifest, class names, provenance, history, best and last checkpoints and final metrics. |

For the interview, show **known correct automatic labels / all known photos**, **unknown referrals / all unknown photos**, **known referrals / all known photos**, and **all referrals / all test photos**, alongside known-only accuracy before referral. The two settings show the tradeoff between catching unfamiliar cases and assessor workload. A referral records a request for review. Correctly naming a novel category would require additional annotation. The miscellaneous test bucket is a proxy for unfamiliar damage. A separate repair-centre test set is needed to measure transfer across sites.
