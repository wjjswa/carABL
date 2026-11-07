# All experiments and one-command Slurm submission

The smoke test has passed on Artemis. The agreed scientific protocol stays the
same. There is no Experiment 5A. This launcher only submits jobs to Slurm;
it does not run training on the login node.

## Complete default matrix

| Group | Models/methods | Labelled fractions | Seeds | Slurm jobs |
|---|---|---|---|---:|
| Exp1 direct transfer | Original CarDD YOLOv8s v2 detection + segmentation | No target training | Deterministic inference | 1 |
| Exp2 supervised | Full, LoRA, CNN block expansion, frozen-backbone linear probing | 20/40/60/80/100% | 0,1,2 | 60 |
| Exp3 DINO adaptation | Full, LoRA, CNN block expansion | All training images; labels hidden | 0,1,2 | 9 |
| Exp3 downstream probes | Frozen adapted features; linear head and k-NN | 20/40/60/80/100% | 0,1,2 | 45 |
| Exp4 scratch | Randomly initialised YOLO backbone; full supervised training | 20/40/60/80/100% | 0,1,2 | 15 |
| Exp5B ActiveGCD | ViT-B/16, random versus adaptive queries | One fixed known/novel protocol | 0,1,2 | 6 |
| Total | | | | **136** |

E6 AGE and E7 DP-BOA-inspired discovery are explicit additions on the experiment
branch, using this same launcher and environment. The default 136-job selection
above is unchanged. Select only the six new jobs with
`bash scripts/submit_all.sh --experiments 6 7 --registry logs/submissions_e6_e7.json`
(preview); append `--submit` to queue them. Explicitly selecting
`--experiments 1 2 3 4 5b 6 7` gives 142 jobs. See [E6/E7 instructions](open_world.md)
for configuration, aggregation, cache checks and the results-only archive.

Each Exp2, Exp3 probe and Exp4 job also evaluates k-NN using the same labelled
subset as its head. Do not submit extra k-NN jobs. The 45 Exp3 probe jobs reuse
the 9 adapted backbones, rather than repeating DINO at every label budget.

## Methods and training schedule

* Exp1 uses the complete unchanged six-class checkpoint, saves boxes/masks and
  scores only compatible mapped image labels; it does not claim target mask mAP.
* Exp2 has 15 jobs for each of full, LoRA, block and linear (5 budgets × 3 seeds).
  Default supervised training is 50 epochs. All run on the target CSV's eight
  classes; pretrained detection/segmentation heads are not used for classification.
* Exp3 has 3 adaptation jobs and 15 probes for each method. Adaptation uses all
  target training images without labels for 100 epochs. Probes train only their
  linear heads for 50 epochs. k-NN banks use the same 20–100% labelled subsets.
* Exp4 trains the same classifier architecture with random weights for 50 epochs.
* Exp5B runs each strategy three times. Defaults: 100 base epochs, 5 active
  rounds × 15 epochs, querying 12 labels per round (60 additional annotations).
  Initially labelled categories are bumper/door dent and bumper/door scratch.
  Glass shatter and head/tail lamp labels are initially hidden. The miscellaneous
  `unknown` category is excluded; the total category count is supplied. See
  `activegcd.md` for the Kaggle port, fixed known-class reserve and assumptions.

Every full experiment keeps the established dataset split. For Exp1–4 the
prepared split contains 1,050 training, 229 validation and 226 test images.
Exp5B excludes the miscellaneous category. The training fractions are nested
group-aware subsets of training only, with identical subsets across methods;
the fixed labelled validation set is additional to those budgets.

## Preview every command (no jobs submitted)

From the repository root on the login node:

```bash
git pull
bash scripts/submit_all.sh
```

This prints the commands, saves `logs/experiment_plan.csv`, and shows the DINO
dependencies. Placeholder job IDs in a dry run are not executable dependencies;
use the launcher with `--submit` to replace them with actual returned Slurm IDs.

## Submit all

```bash
bash scripts/submit_all.sh --submit
```

Defaults: partition `gpu`; one GPU, 8 CPUs and 32 GB RAM inherited from the
wrapper; 24-hour wall time per training job; 30 minutes for Exp1 inference.
Slurm controls when queued jobs run, subject to your allocation and queue limits.
If a submission is rejected, the launcher stops and retains all earlier returned
IDs. Re-running with the SAME registry reuses those IDs rather than queuing them
again. Completed outputs are skipped. The count of newly submitted jobs may
therefore be less than 136.

Customise time/partition or select experiment groups if needed:

```bash
bash scripts/submit_all.sh --experiments 2 4 --submit
bash scripts/submit_all.sh --experiments 3 --time 2-00:00:00 --submit
```

Use `--config PATH` to select a custom YAML. A different protocol/output root
requires a different `--registry` path; an existing registry rejects config/data
fingerprint changes. Existing run provenance is checked before submission.

## Dependencies and duplicate jobs

Each probe has `afterok:ID` for the matching DINO method/seed. If DINO fails,
Slurm will not run its probes. A completed `teacher_features.pt` permits probes
without a dependency. Inspect failed dependencies before retrying them.

The launcher records jobs in `logs/submissions.json` and never automatically
resubmits a recorded job, even if it failed. This protects against duplicates on
repeated launches; it is NOT an automatic recovery system. It cannot identify
manually submitted jobs that are still pending and have created no output yet.
Do not queue the same method/fraction/seed manually and through the launcher.

If a manually submitted run already completed, it is skipped. If incomplete
output exists but is not recorded, submission stops before enqueuing any jobs.
Check that no corresponding job is running before using `--resume` for Exp2/3/4.
ActiveGCD does not support partial-round resume; restart it deliberately with a
fresh output root/config and registry. Do not delete successful outputs to retry
an unrelated failed job.

Keep code/config/manifest/checkpoint unchanged while the jobs are queued or
running; jobs read those files when they start. Do not repeatedly submit the
entire matrix using a new registry as this bypasses duplicate-job tracking.

## Monitor and present

```bash
squeue -u "$USER"
```

Each log is `logs/carABL-EXPERIMENT_KEY-JOBID.log`. For a particular returned ID:

```bash
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

After all jobs finish:

```bash
python -m carrepair export --output outputs
python -m streamlit run app.py --server.address 127.0.0.1
```

No automatic report job or extra GPU job is submitted. k-NN and linear scores,
image predictions and annotation-round results are already saved by the jobs.
