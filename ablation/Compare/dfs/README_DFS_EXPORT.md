# DFS prediction export for ROC/PR

이 패치는 DFS baseline에서 AUROC/AUPRC 계산용 prediction probability CSV를 생성한다.
학습은 다시 하지 않고 `best.pt`를 로드해 `test_clean`, `test_masked`만 forward한다.

## 배치 위치

`dfs.zip`을 서버에 풀어서 다음 구조가 되게 둔다.

```text
/data/shared/scuppy/baselines/dfs/
├── dfs_multitask.py
├── train_dfs_multitask.py
├── experiments/dfs_fixed_clean_masked_seed42.yaml
├── runs/dfs_fixed_clean_masked_seed42/best.pt
└── runs/dfs_fixed_clean_masked_seed42/config.json
```

그 다음 이 zip의 파일을 DFS_ROOT에 넣는다.

```bash
cp export_dfs_predictions.py /data/shared/scuppy/baselines/dfs/export_dfs_predictions.py
cp run_dfs_export_predictions.sh /data/shared/scuppy/baselines/dfs/run_dfs_export_predictions.sh
chmod +x /data/shared/scuppy/baselines/dfs/run_dfs_export_predictions.sh
```

## 실행

```bash
DFS_ROOT=/data/shared/scuppy/baselines/dfs \
RUN_DIR=/data/shared/scuppy/baselines/dfs/runs/dfs_fixed_clean_masked_seed42 \
PYTHON=/data/shared/envs/scuppy/bin/python \
bash /data/shared/scuppy/baselines/dfs/run_dfs_export_predictions.sh
```

## 출력

```text
runs/dfs_fixed_clean_masked_seed42/test_clean_predictions.csv
runs/dfs_fixed_clean_masked_seed42/test_masked_predictions.csv
runs/dfs_fixed_clean_masked_seed42/test_clean_action_clip_predictions.csv
runs/dfs_fixed_clean_masked_seed42/test_masked_action_clip_predictions.csv
...
```

CSV 컬럼:

```text
sample_id, clip_id, split, head, level, n_windows, y_true, y_pred, prob_0, prob_1, ...
```

이 파일들이 생성되면 기존 ROC/PR 취합 노트북이 자동으로 AUROC/AUPRC를 계산할 수 있다.
