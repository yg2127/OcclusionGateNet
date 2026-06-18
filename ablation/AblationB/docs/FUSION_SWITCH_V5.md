# V5 Fusion Switch Guide

`configs/region_transformer.yaml`에서 아래 블록만 교체하면 된다.

```yaml
model:
  fusion:
    kind: task_region_scalar_gated_late
```

## V5 권장 비교 순서

```text
1. concat
2. concat_condition
3. task_gated_late
4. task_region_gated_late
5. task_region_scalar_gated_late
6. explicit_region_scalar_mask_gate
7. occ_token_region_transformer
8. occ_attention_bias
```

## 해석 기준

- `concat`: 기본 shared feature baseline
- `concat_condition`: OCC를 단순 condition으로 붙인 baseline
- `task_gated_late`: task별 pose/face reliability 조절
- `task_region_gated_late`: face region gate만 사용
- `task_region_scalar_gated_late`: V5 핵심, region gate + scalar gate
- `explicit_region_scalar_mask_gate`: V5 핵심 + 명시적 OCC region reliability mask
- `occ_token_region_transformer`: OCC token을 Transformer 입력에 추가
- `occ_attention_bias`: OCC를 attention score에 직접 bias로 반영
