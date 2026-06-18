# Drive&Act label analysis and DMD adaptation

## Drive&Act labeling in the paper

Drive&Act uses a three-level hierarchical vocabulary for driver behavior recognition.

| Level | Paper label type | Count | Meaning | DMD adaptation |
|---|---:|---:|---|---|
| Level 1 | Scenario / task | 12 | High-level task context such as eating/drinking, work on laptop, take-over, entertainment | Not used as a separate head because DMD labels are action-centric rather than scenario-centric |
| Level 2 | Fine-grained activity | 34 | Semantic driver activities such as drinking, eating, reading magazine, working on laptop, talking on phone | Mapped primarily to DMD `action` |
| Level 3 | Atomic action unit | 5 actions + 17 objects + 14 locations, 372 observed triplets | Low-level triplets `{Action, Object, Location}` such as reaching for / phone / front area | Replaced by multi-head DMD labels: `action`, `hands`, `talk`, `gaze` |
| Additional | Driving context | left/right/both hands, automation status, take-over timestamp | Dense context annotations | Mapped to DMD `hands`; gaze context is added from DMD |

The paper benchmarks both end-to-end 3D CNN models such as C3D, P3D, and I3D, and a body-pose/car-interior model. The most relevant model for the current project is the body-pose/car-interior family because DMD preprocessing already provides body pose, face landmarks, head pose, and hand-on-wheel labels.

## DMD label heads used in our project

| Head | Classes | Source in DMD builder | Drive&Act analogue |
|---|---:|---|---|
| `action` | 11 | `driver_actions/*` from distraction data | Fine-grained activity |
| `hands` | 4 | `hands_using_wheel/*` or `hands_on_wheel/*` | Additional steering-with-hands context |
| `talk` | 2 | overlap with `talking/talking` | Fine-grained `talking on phone` / communication cue |
| `gaze` | 9 | `gaze_zone/*`, excluding `not_valid` | Driver attention / interior context cue |
| `gaze_weak` | 2 | `looking_road` vs `not_looking_road`, distraction data only | Weak attention context, not a separate paper label |

## Adjusted abnormal behavior grouping

Drive&Act has many labels that do not exist in DMD, such as opening backpack, reading magazine, taking off sunglasses, or putting laptop into backpack. The closest DMD-compatible grouping is:

| DMD action class | Normal / abnormal | Drive&Act closest labels |
|---|---|---|
| `safe_drive` | Normal | sitting still, take over steering, driving preparation when not distracting |
| `texting_right` | Abnormal | interacting with phone, writing |
| `texting_left` | Abnormal | interacting with phone, writing |
| `phonecall_right` | Abnormal | talking on phone |
| `phonecall_left` | Abnormal | talking on phone |
| `radio` | Abnormal / secondary task | using multimedia display, pressing automation button |
| `drinking` | Abnormal / secondary task | drinking, opening bottle, closing bottle |
| `reach_side` | Abnormal | reaching for, fetching an object, placing an object, front area / console locations |
| `reach_backseat` | Abnormal | reaching for, fetching an object, placing an object, backseat locations |
| `hair_and_makeup` | Abnormal | taking off sunglasses, putting on sunglasses, clothing/accessory actions |
| `talking_to_passenger` | Abnormal / social distraction | talking on phone, communication-like activity |

## Final label design for implementation

The implemented model keeps the project's existing label heads:

```text
action: 11-class fine-grained driver behavior
hands: 4-class steering-hand state
talk: 2-class talking state
gaze: 9-class gaze zone
```

This is intentionally not a direct 34-class Drive&Act classifier. Instead, it is a Drive&Act-inspired multi-stream model adapted to the DMD label space.

## Model adaptation

Drive&Act paper model:

```text
Temporal body stream + Spatial body stream + Car-interior stream
late fusion
activity classifier
```

DMD-adapted implementation:

```text
Temporal pose stream
+ Spatial pose stream
+ Face/head-pose context stream
+ late fusion
+ 4 task heads: action, gaze, hands, talk
```

The car-interior stream cannot be replicated exactly because DMD does not provide the Drive&Act object/location distance primitives. The replacement context stream uses face landmarks and optional head pose as a proxy for gaze, attention, and driver state.
