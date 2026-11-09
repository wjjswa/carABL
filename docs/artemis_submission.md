# Artemis submission guide

## What is fixed

All four shell entry points source scripts/artemis_env.sh:
slurm_job.sh, sge_job.sh, run_experiments.sh, setup_remote.sh.
The helper loads CUDA/12.4.0 and Miniconda3/4.12.0; switches via base to
carabl; rebuilds PATH; clears shell command hashing; verifies Python >=3.10
and that its prefix matches the activated Conda environment. The base switch
is shell activation only: it does not update the shared base installation.

The Slurm wrapper uses unbuffered Python logs. It now explicitly selects
partition gpu and requests 1-00:00:00 (24 hours), within the supplied 3-day
GPU partition maximum. Smoke jobs should override --time=00:30:00.
The setup script includes Kaggle and six, addressing the missing-six error.
None of these edits changes experiment methods or training fractions.

## Files that must exist on Artemis

| File/directory | Purpose | Submit separately? |
|---|---|---|
| scripts/slurm_job.sh | Single Slurm wrapper for every GPU experiment | Yes, with the command arguments below |
| scripts/artemis_env.sh | Shared module/Conda activation | No; the wrapper sources it |
| carrepair/ including vendor/activegcd/ | All Python training, inference, evaluation, smoke and upstream loss code | No; keep the entire cloned package |
| configs/default.yaml or your own YAML | Paths, methods, fractions, seeds, epochs and batch sizes | No; pass --config if using a different file |
| data/kaggle/data.csv and image/ | Downloaded target dataset | No |
| data/prepared/manifest.csv | Locked train/validation/test split | No; do not regenerate between experiments |
| weights/cardd-v2.pt | Pretrained checkpoint for experiments 1–3 | No |
| pyproject.toml | Install dependencies/package | No |
| tests/ | Automated checks | No; run pytest |
| app.py | Streamlit artifact viewer | No training submission needed |
| README.md, docs/ | Usage and protocol reference | No |

The ActiveGCD ViT weights download into the normal Torch cache on first use.
Keep cache access available on the compute node; the smoke test checks this.

## Other scripts

| Script | When to use |
|---|---|
| scripts/setup_remote.sh | Install/check dependencies in existing carabl; safe for CPU checks on login |
| scripts/run_experiments.sh | Entire matrix sequentially inside an already allocated GPU session/job |
| scripts/sge_job.sh | Legacy SGE template; not used for your current Slurm submissions |

Clone/pull the complete repository, rather than uploading individual Python
files. Dataset/weights/results are separate and are not downloaded by git pull.

## Before full training

The current smoke job uses its own output directory. Let it finish before
submitting the full matrix. Require SMOKE TEST PASSED and a successful exit.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 100 carABL-JOB_ID.log
```

Replace JOB_ID. An activation/GPU banner alone is not a completed smoke test.

Inspect the current GPU partition limit:

```bash
sinfo -p gpu -o "%P %a %l %G"
```

Use an allowed time long enough for a particular training run. The 30-minute
smoke request is not a guarantee that full DINO/ActiveGCD training will finish.
The rejected 24-hour request used the default general partition, which permits
only 8 hours. The same request is within the documented gpu partition limit.
The running general-partition smoke job does not need to be changed.

## Commands for each experiment

Run sbatch from the repository root. The pattern is:

```bash
sbatch --time=HH:MM:SS scripts/slurm_job.sh COMMAND ARGUMENTS
```

Replace HH:MM:SS with a permitted time. Add site-specific partition/resource
flags as required. Below are the COMMAND ARGUMENTS appended to the wrapper:

| Task | COMMAND ARGUMENTS |
|---|---|
| Smoke check including 5B | smoke --activegcd |
| Exp1 unchanged model | direct --config configs/default.yaml |
| Exp2 supervised adaptation | supervised --experiment 2 --method full --fraction .2 --seed 0 |
| Exp3 DINO adaptation | dino --method full --seed 0 |
| Exp3 frozen evaluation | probe --method full --fraction .2 --seed 0 |
| Exp4 train from scratch | supervised --experiment 4 --method full --fraction .2 --seed 0 |
| Exp5B random query baseline | activegcd --strategy random --seed 0 |
| Exp5B adaptive query method | activegcd --strategy adaptive --seed 0 |
| Exp6 AGE adaptation | open-world --method e6 --seed 0 |
| Exp7 DP-BOA-inspired adaptation | open-world --method e7 --seed 0 |

If changing the config, add --config YOUR_CONFIG to every training/evaluation
command, not only the first. There is no Experiment 5A.

* Exp2 methods: full, lora, block, linear. Fractions: .2, .4, .6, .8, 1.
* Exp3 adaptation methods: full, lora, block; one adaptation per method/seed.
  After it finishes, run five probes for that same method/seed. Each probe
  evaluates both its linear head and k-NN; no separate k-NN job is required.
* Exp4 uses full only, with all five fractions.
* Default seeds: 0, 1, 2. Each method/fraction/seed has its own output directory.
* Exp5B uses random/adaptive with one fixed data protocol, no percentage sweep.

## DINO/probe job dependencies

A probe cannot start before teacher_features.pt exists for the corresponding
DINO method/seed. Either wait for that job to finish, or submit the probe with:

```bash
sbatch --dependency=afterok:DINO_JOB_ID --time=HH:MM:SS \
  scripts/slurm_job.sh probe --method full --fraction .2 --seed 0
```

Replace DINO_JOB_ID and time, and match method/seed/config to the adaptation job.
Independent Exp2/Exp4/Exp5B runs can be separate jobs, subject to your allocation.

## Results and presentation

| Task | Output directory |
|---|---|
| Smoke | outputs_smoke/TIMESTAMP/ |
| Exp1 | outputs/exp1/ |
| Exp2 | outputs/exp2/ |
| Exp3 | outputs/exp3/ |
| Exp4 | outputs/exp4/ |
| Exp5B | outputs/exp5b/ |
| Exp6 | outputs/exp6/seedN/ |
| Exp7 | outputs/exp7/seedN/ |
| Exp6/7 comparison | outputs/open_world_comparison/ |

E6/E7 use the existing `carabl` environment and the same Slurm planner:
`bash scripts/submit_all.sh --experiments 6 7 --registry logs/submissions_e6_e7.json`
previews six jobs; append `--submit` to queue them. E1–E5 defaults, configs,
checkpoints and output paths remain unchanged. See [open_world.md](open_world.md)
for the exact branch update, DINO pre-cache and results-only download commands.

Once real experiments finish:

```bash
python -m carrepair export --output outputs
python -m streamlit run app.py --server.address 127.0.0.1
```

Use your SSH tunnel to view Streamlit. Smoke scores must not be mixed into
research figures. The full-matrix driver runs sequentially and may exceed one
allocation's wall time; separate experiment jobs are preferable for scheduling.

## Supplied Artemis partition limits

| Partition | Maximum runtime | Slurm time value |
|---|---|---|
| general | 8 hours | 08:00:00 |
| short | 2 hours | 02:00:00 |
| long | 8 days | 8-00:00:00 |
| verylong | 30 days | 30-00:00:00 |
| gpu | 3 days | 3-00:00:00 |

These are the limits supplied from the Artemis documentation. Partition access,
GPU availability and any account/QoS restrictions remain enforced by Slurm.
The listed CPU/RAM slot allowances are not instructions to request all resources.
Our single-GPU job retains 8 CPUs and 32 GB RAM.

Smoke submission:

```bash
sbatch --partition=gpu --time=00:30:00 --job-name=carABL-smoke \
  scripts/slurm_job.sh smoke --activegcd
```

Example full training submission (inherits gpu + 24 hours):

```bash
sbatch scripts/slurm_job.sh supervised --experiment 2 --method full --fraction .2 --seed 0
```

Keep the DINO/probe dependency order described above; the entire experiment
matrix is not guaranteed to fit in one 24-hour allocation.
