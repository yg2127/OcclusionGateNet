# DMD-paper baseline adaptation for current multi-task labels

## Why this exists

The ECCV 2020 DMD paper demonstrates a **fusion-study baseline family** on
`dBehaviourMD`, comparing:

- single stream
- early fusion
- late fusion
- score fusion

That original experiment is action-only and uses the paper's own real-time
system setup. Our current project, however, evaluates a broader DMD label
space:

- `action`: 11 classes
- `gaze`: 9 classes
- `hands`: 4 classes
- `talk`: 2 classes

So this baseline keeps the **same comparison idea** while adapting the outputs
to the current multi-task project labels.

## Inputs used here

The current project already standardizes window inputs as:

- `x_body`: preprocessed DMD pose tensor `(C_pose, T, V_pose)`
- `x_face`: preprocessed DMD face tensor `(C_face, T, V_face)`
- `x_head_pose`: optional head-pose tensor `(2, T, 3)`

This baseline therefore uses the same inputs for a fair comparison on the same
split and same data builder.

## Implemented fusion modes

- `single_body`
- `single_face`
- `single_head_pose`
- `early_fusion`
- `late_fusion`
- `score_fusion`

Recommended starting point for paper-style comparison:

- `late_fusion`

Recommended ablation table:

1. `single_body`
2. `single_face`
3. `early_fusion`
4. `late_fusion`
5. `score_fusion`

## Important caveat

This is **DMD-paper-inspired**, not a strict historical reproduction. The
paper's demo system and the current project have different final label spaces
and different preprocessing assets. The purpose of this baseline is to preserve
the paper's **fusion comparison direction** while making the experiment directly
comparable to the current multi-task DMD models.
