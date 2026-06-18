# Gaze Clean-to-Clean Upper-bound Experiment

Purpose:

```text
Clean MediaPipe FaceMesh train -> Clean MediaPipe FaceMesh test -> Gaze task only
```

This is the FaceMesh-style landmark upper-bound baseline for the Gaze task.

## Run

Use the same environment style as AblationB:

```bash
cd /data/shared/scuppy/hyi/Ablation/Ablation_ex_Gaze_clean_to_clean
chmod +x run_gaze_clean_to_clean.sh
./run_gaze_clean_to_clean.sh
```

The script uses:

```bash
PYTHON=/data/shared/envs/scuppy/bin/python
```

To override:

```bash
PYTHON=/your/env/bin/python ./run_gaze_clean_to_clean.sh
```

Alternative:

```bash
chmod +x train/gaze_clean_to_clean_single.sh
./train/gaze_clean_to_clean_single.sh
```

## Output

```text
artifacts/gaze_clean_to_clean_seed42/
├─ best.pt
├─ last.pt
├─ metrics_gaze_only.csv
├─ result_gaze_clean_to_clean.csv
├─ summary_gaze_clean_to_clean.json
└─ train.log
```
