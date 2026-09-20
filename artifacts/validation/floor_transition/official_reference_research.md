# Official AgiBot recovery-reference research

Checked 2026-09-20. Read-only research: documentation, HTTP headers, ZIP directory
metadata, and a few small YAML files. No vendor policy or executable was
downloaded, installed, executed, or integrated into this project. No vendor
recovery result is claimed. Approximately 378 KB of ZIP metadata/configuration
was read through bounded HTTP byte ranges; complete packages were not fetched.

## Verified primary sources and packages

- [Simulation pipeline overview](https://x2-aimdk.agibot.com/en/latest/sim_rl/index.html):
  separates user-policy RL deployment from the official MC simulation pipeline.
- [MC simulation overview](https://x2-aimdk.agibot.com/en/latest/sim_index.html):
  identifies `lx2501_3_t2d5` (X2 T2.5), with built-in stand/walk/get-up skills.
- [MC setup instructions](https://x2-aimdk.agibot.com/en/latest/sim_contents.html):
  x86_64, Ubuntu 22.04, ROS 2 Humble/Fast DDS; 16 GiB RAM recommended; GPU optional.
  A separate Docker deployment is documented. It launches `sim_mujoco/bin/start_sim.sh -s`
  and `mc/bin/em_run.sh`, with high-level SDK commands.
- [SDK acquisition](https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html) and
  [actual public download manifest](https://x2-aimdk.agibot.com/sdk-downloads.json).
  The toolbar obtains the manifest through
  [the site's navigation JavaScript](https://x2-aimdk.agibot.com/lang-ver-nav.js).

The following links returned HTTP 200 to HEAD and support byte ranges:

| Manifest label | Exact official package | Bytes |
| --- | --- | ---: |
| `mc-x86-v1.0.0` | [mc-x86-v1.0.0-20260522.zip](https://x2-aimdk.agibot.com/downloads/mc-x86-v1.0.0-20260522.zip) | 225,367,933 |
| `sim_mujoco-x86-v1.0.0` | [sim_mujoco-x86-v1.0.0-20260522.zip](https://x2-aimdk.agibot.com/downloads/sim_mujoco-x86-v1.0.0-20260522.zip) | 99,278,626 |
| `X2 RL Deploy extras v1.1` | [aimdk-extra-x2_rl_deploy-v1.1.zip](https://x2-aimdk.agibot.com/downloads/aimdk-extra-x2_rl_deploy-v1.1.zip) | 67,474,415 |

The manifest also advertises SDK v1.1.0 as
`aimdk-sdk-v1_1_0-aimdk-aarch64-faf4c945-artifacts.zip`, 282,622,293 bytes. This
package was not inspected. Do not assume its prebuilt aarch64 components run on
the local x86 machine; follow the documented source build or matching delivery.

## What the MC archive actually contains

The ZIP directory has 804 entries, including compiled executables/shared
libraries and pretrained ONNX policies. Paths below are relative to:

```text
mc-x86-v1.0.0-20260522/bin/mc_param/src/robot/lx2501_3_t2d5/
```

The inspected `action_setting.yaml` maps:

| Action | Runner | Configuration |
| --- | --- | --- |
| `LIE_UP_DEFAULT` | `rl_play` | `rl/ground_init_config.yaml` |
| `GROUNDUP` | `rl_bmimic` | `rl/ground_up_config.yaml` |
| `GET_UP_DEFAULT` | `rl_play` | `rl/prone_init_config.yaml` |
| `PRONE_UP_DEFAULT` | `rl_bmimic` | `rl/prone_up_config.yaml` |
| `SIT_UP_DEFAULT` | `rl_bmimic` | `rl/sit_up_config.yaml` |

`ground_up_config.yaml` selects **`rl_models/policy_b_lie_up.onnx`**, whose archive
entry is 1,319,616 bytes. Other bundled policy names include `policy_up.onnx`,
`policy_paqi.onnx`, and `policy_stanup_delong.onnx`; their exact roles were not
verified from configuration. Policy bytes were not read.

The inspected ground-up configuration specifies BMIMIC, CPU ONNX, `direct_output`
enabled, 20 ms policy period, 29 joint/action values, declared `obs_nums=151`, and
`frame_stack=5`. It provides observation scales, nominal positions and PD gains.
Phase counts are 0–310, followed by `STAND_DEFAULT`; the nominal count duration
would be 6.2 seconds at 20 ms, but runtime phase semantics were not verified.

Its joint observation/action sequence interleaves sides:

```text
left hip pitch, right hip pitch, waist yaw,
left hip roll, right hip roll, waist pitch,
left hip yaw, right hip yaw, waist roll,
left knee, right knee,
left shoulder pitch, right shoulder pitch,
left ankle pitch, right ankle pitch,
left shoulder roll, right shoulder roll,
left ankle roll, right ankle roll,
left shoulder yaw, right shoulder yaw,
left elbow, right elbow,
left wrist yaw, right wrist yaw,
left wrist pitch, right wrist pitch,
left wrist roll, right wrist roll.
```

These correspond to the standard `_joint` names in the configuration. Head
joints are excluded. The configuration does not fully specify every assembled
observation feature or the effect of `direct_output`; those require the actual
inference implementation/graph contract.

The ground-up PD gains below follow that named ordering (left/right share values):

| Joint group | KP | KD |
| --- | ---: | ---: |
| Hip pitch | 120 | 5 |
| Hip roll / yaw | 100 | 4 |
| Knee | 150 | 5 |
| Ankle pitch / roll | 40 | 2 |
| Waist yaw | 40.1792 | 2.5579 |
| Waist pitch / roll | 200 | 2 |
| Shoulder pitch / roll / yaw | 50 | 3 |
| Elbow | 50 | 3 |
| Wrist yaw | 20 | 2 |
| Wrist pitch / roll | 5 | 0.5 |

Its nominal pose has hip pitch −0.312, knee 0.669, ankle pitch −0.363,
shoulder pitch 0.2, shoulder roll left +0.2/right −0.2, and elbow −0.3 radians;
other listed joints are zero. The configured per-action scale is 0.25, but whether
that scale applies with `direct_output=true` requires checking the runner.

`ground_init_config.yaml` uses a different, grouped 29-joint ordering, declares
76 observations and six history frames, and transitions to `GROUNDUP`. Its four
planner `ref_pos` arrays are all zero, with a 5-second wait; it also contains one
asymmetric default posture. **These are not an exposed, complete get-up joint
trajectory.** No explicit ground-recovery CSV/NPZ reference was identified in
this bounded inspection. The useful confirmed recovery asset is the vendor's
pretrained ONNX and configuration, not source training code.

Initialization uses different gains: hip pitch/knee 200/5, hip roll/yaw 100/2.5,
ankles 40/2, waist 200/4, shoulder pitch 100/2, shoulder roll/yaw and elbow 100/1,
and wrists 20/1 (KP/KD). This distinction matters for reproducing the
`LIE_UP_DEFAULT` initialization → `GROUNDUP` inference → `STAND_DEFAULT` sequence.

The [separate RL deployment example](https://x2-aimdk.agibot.com/en/latest/rl_index.html)
is documented as a dance policy and inference example, without training. Its
[deployment contract](https://x2-aimdk.agibot.com/en/latest/rl_contents.html)
uses ROS joint/IMU topics and C++ `RlDefault()` adaptation. It is not itself the
get-up solution discovered in the MC package.

## Compatibility and licensing limits

Our model is X2 Ultra v1.3.0 at official URDF revision
`60c5de582c523cd188f563819e62d34cfdc3d2d0`, with 31 controlled joints and a 106-value
observation. The MC directory says X2 T2.5, but its actual model hashes were not
compared. Different joint ordering, 29 versus 31 actions, history/observation
schema, PD gains, action interpretation, collision geometry, and actuator
controller make the vendor actor **not directly compatible** with our current
policy interface. Shared robot family/name is insufficient validation.

The [official URDF repository license](https://github.com/AgibotTech/agibot_x2_urdf/blob/main/LICENSE)
is Mulan PSL v2. That license is evidence for the URDF repository only. No explicit
LICENSE, COPYING, or README entry was found in the inspected MC ZIP directory;
reuse/redistribution/training-data terms for its binaries and policies remain
**unverified**. Public download availability is not proof that the URDF license
covers the vendor controller. Do not label these assets as our own PPO or copy
them into the submitted trained-policy artifacts.

## Practical next method

First, separately inspect the vendor package terms and reproduce its get-up in
the matching official simulator, if those terms permit the intended use. Record
the actual contact sequence, joint states/commands, base trajectory and gains.
This provides evidence of how the missing floor transition works. It remains a
vendor-policy demonstration and must be reported separately.

A lightweight Python ONNX diagnostic appears technically plausible: the selected
get-up model is only approximately 1.32 MB, joint names/gains are exposed, and the
current local MuJoCo simulator can host a separately labeled controller. It would
avoid installing the full vendor executable stack. However, first inspect the
ONNX input/output shapes and metadata and verify BMIMIC observation assembly,
history initialization, phase inputs/reference features and output semantics.
Do not assume that `obs_nums=151` and `frame_stack=5` imply a single 755-element
input. Exact runner semantics and model compatibility remain unresolved; this
is a potential diagnostic method, not a ready-to-run adapter. Keep vendor policy
bytes outside committed deliverables while license terms remain unclear.

Then map an appropriately permitted reference into the pinned model by joint
name and check the complete physical rollout under our existing position, speed,
effort and clean-standing audits. Only validated demonstrations should become
imitation data, followed by our PPO refinement and independent supine evaluation.
If vendor asset reuse is unsuitable, the documented contact sequence can instead
guide independently authored trajectory optimization. More PPO computation
alone does not resolve the currently missing feasible floor-transition reference.
