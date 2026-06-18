# Fusion 실험 교체 방식

이 리팩터링의 목표는 학습 루프를 건드리지 않고 `model.fusion.kind`만 바꿔 실험을 교체하는 것이다.

## 현재 지원 방식

```yaml
model:
  fusion:
    kind: concat
```

```yaml
model:
  fusion:
    kind: concat_condition
    occ_hidden_dim: 64
    occ_dropout: 0.1
```

```yaml
model:
  fusion:
    kind: task_gated_late
    gate_hidden_channels: 128
    gate_dropout: 0.2
    gate_feature_scale: 0.25
    init_bias:
      action: 0.0
      gaze: 0.0
      hands: 0.0
      talk: 0.0
```

```yaml
model:
  fusion:
    kind: task_region_gated_late
    gate_hidden_channels: 128
    gate_dropout: 0.2
    gate_feature_scale: 0.25
    region_num_heads: 4
    region_num_layers: 1
    region_dropout: 0.1
    region_ff_mult: 2
    init_bias:
      action: 0.0
      gaze: 0.0
      hands: 0.0
      talk: 0.0
```

## 새 방법 추가 위치

1. `src/models/fusion/task_feature_fusion.py`에 새 class 추가
2. `src/models/fusion/factory.py`에 `kind` 등록
3. YAML에서 `model.fusion.kind`만 변경

`src/training/runner.py`, `src/training/builders.py`, `src/models/multitask_classifier.py`는 새 실험마다 수정하지 않는 것을 원칙으로 한다.
