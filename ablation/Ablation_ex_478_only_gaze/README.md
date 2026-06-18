# Ablation_ex_478_only_gaze

목적: **FaceMesh 478 full node 기반, gaze task만 학습/평가**하는 실험 코드.

- 학습 variant: `clean`
- 평가/test variant: `clean`
- task: `gaze` only
- action / hands / talk head는 모델 출력, loss, metric에서 제외
- 실행 config: `configs/clean_only_train_clean_test.yaml` 하나만 사용

## 서버 배치

```bash
mkdir -p /data/shared/scuppy/hyi/Ablation/Ablation_ex_478_only_gaze
# zip 압축을 이 경로에 풀기
cd /data/shared/scuppy/hyi/Ablation/Ablation_ex_478_only_gaze
```

## 실행

```bash
bash train/run_single_yaml.sh
```

또는 직접 실행:

```bash
/data/shared/envs/scuppy/bin/python -m src.training.train --config configs/clean_only_train_clean_test.yaml
```

## 다른 YAML을 지정해야 할 때

```bash
CONFIG=configs/clean_only_train_clean_test.yaml bash train/run_single_yaml.sh
```

## 출력

기본 저장 위치:

```text
/data/shared/scuppy/hyi/Ablation/Ablation_ex_478_only_gaze/artifacts/clean_only_gaze_only_train_clean_test_seed42
```

주요 결과:

```text
train.log
metrics.csv
history.json
best.pt
last.pt
summary.json
```
