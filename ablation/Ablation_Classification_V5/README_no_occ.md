# no_occ_original only package

이 zip은 `Classification_model_V5_transformer` 프로젝트 루트에 덮어 풀어서 사용한다.
기존 파일을 거의 건드리지 않고 아래 파일을 추가한다.

```text
configs/v5_no_occ_original_mediapipe_seed42.yaml
src/evaluation/eval_only_save_predictions.py
train/run_no_occ_train_test_export.sh
train/run_no_occ_eval_export_only.sh
tools/no_occ_roc_pr_summary.py
```

## 목적

`no_occ_original` baseline만 다시 학습하고, test_clean/test_masked 평가 후 ROC/PR용 prediction CSV를 저장한다.

- OCC 사용 안 함: `occ.enabled: false`
- fusion: `concat`
- train/val/test split: fixed manifest의 clean+masked 사용
- checkpoint: `artifacts/v5_no_occ_original_mediapipe_seed42/best.pt`
- output prediction CSV:
  - `test_clean_predictions.csv`
  - `test_masked_predictions.csv`
  - `test_clean_action_clip_predictions.csv` 등 head별 파일

## 설치/배치

```bash
cd /data/shared/scuppy/hyi/Classification_model_V5_transformer
unzip /path/to/no_occ_original_train_test_export.zip -d .
chmod +x train/run_no_occ_train_test_export.sh train/run_no_occ_eval_export_only.sh
```

## 학습부터 test/export까지 한 번에

```bash
cd /data/shared/scuppy/hyi/Classification_model_V5_transformer
bash train/run_no_occ_train_test_export.sh
```

Python을 명시하려면:

```bash
PYTHON=/data/shared/envs/scuppy/bin/python bash train/run_no_occ_train_test_export.sh
```

또는:

```bash
PYTHON=/home/hyi/Code/.venv/bin/python bash train/run_no_occ_train_test_export.sh
```

## 이미 학습된 best.pt가 있을 때 eval/export만

```bash
bash train/run_no_occ_eval_export_only.sh
```

## 생성 확인

```bash
find artifacts/v5_no_occ_original_mediapipe_seed42 -maxdepth 1 -name '*predictions.csv' | sort
```

## ROC/PR summary만 빠르게 생성

```bash
python tools/no_occ_roc_pr_summary.py \
  --run-dir artifacts/v5_no_occ_original_mediapipe_seed42
```

## 기존 통합 ROC/PR 노트북에 포함시키기

노트북의 prediction root에 아래가 포함되어 있으면 자동으로 잡힌다.

```python
Path('/data/shared/scuppy/hyi/Classification_model_V5_transformer/artifacts')
```

생성되는 run dir:

```text
/data/shared/scuppy/hyi/Classification_model_V5_transformer/artifacts/v5_no_occ_original_mediapipe_seed42
```
