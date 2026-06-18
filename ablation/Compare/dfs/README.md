# DFS DMD baseline

Paper: **Multi-modality action recognition based on dual feature shift in vehicle cabin monitoring** (`arXiv:2401.14838`).

The paper proposes DFS for multi-modality driver action recognition. It uses RGB, IR, and depth clips from Drive&Act and applies:

- modality feature interaction: channel shift between modalities
- neighbour feature propagation: temporal shift between adjacent frames
- shared middle feature extraction stages for efficiency
- final average fusion and action classification

## Adaptation to this project

The DMD project does not use RGB/IR/depth video clips in the current multi-task pipeline. Instead, this baseline treats existing preprocessed streams as modalities:

- body pose sequence
- face/facemesh sequence
- optional head-pose sequence

The output heads follow the project label space:

- `action`: 11 classes
- `gaze`: 9 classes, excluding `not_valid`
- `hands`: 4 classes
- `talk`: 2 classes

Implemented class:

```python
from dfs_multitask import DFSDMDMultitaskClassifier
```

Expected forward inputs:

```python
out = model(x_body, x_face, x_head_pose=None)
```

Output:

```python
{
    "action": Tensor[N, 11],
    "gaze": Tensor[N, 9],
    "hands": Tensor[N, 4],
    "talk": Tensor[N, 2],
}
```
