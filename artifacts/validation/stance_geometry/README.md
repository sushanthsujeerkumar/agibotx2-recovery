# Stance geometry audit: preserved checkpoint 9251

Read-only audit of `artifacts/submission/stable_crossed_stance_009251/actor.pt`, SHA256 `2d5196394f5107205bf166d59b91b7e1044fd45f53c78639bdf3b7d013129043`. One independent CPU episode (seed 1001) continued for all 15 seconds. `geometry.json` preserves the sampled configurations and calculated geometry, including the nominal standing keyframe. No model parameters or running training processes were changed.

## Finding

The observed stance is **inward-twisted, staggered, supported on foot edges, and braced by substantial foot-to-foot force**. Negative separation of the foot collision centres does not by itself prove the legs or ankle joints have crossed. At 15 seconds:

| Measurement | Policy | Nominal standing |
|---|---:|---:|
| Foot collision-centre width, horizontal pelvis lateral axis | −0.0495 m | +0.2743 m |
| Ankle-origin width on the same axis | +0.0442 m | +0.2743 m |
| Knee-origin width on the same axis | +0.1353 m | +0.2746 m |
| Foot-centre fore-aft separation | −0.1441 m | approximately 0 m |
| Left / right foot heading relative to pelvis | −70.8° / +59.4° | 0° / 0° |
| Left / right sole tilt from horizontal | 26.2° / 58.5° | 0° / 0° |
| Foot-to-foot normal force | 195.9 N | no contact |
| Foot-to-foot contact penetration | 0.373 mm | no contact |

Both ankle origins and both knee origins retain the expected left/right ordering. The foot-centre ordering reverses because the feet point inward and are tilted; a collision centre lies 0.036 m forward and 0.0494 m below its ankle origin. The tilted pelvis-y measurement is −0.0427 m, whereas the horizontal lateral measurement is −0.0495 m. Use a horizontal heading frame when describing stance width on the floor.

The roughly 196 N foot-to-foot normal force is substantial relative to the robot's approximately 412 N weight. This is load-bearing mutual bracing, not merely incidental contact. Both feet still have actual floor contacts, so the result can satisfy the existing recovery criterion without providing a neutral standing posture.

## Model checks

All 31 generated joint axes and position ranges match the pinned official URDF. The opposite-signed hip-yaw values are allowed by the official bounds. The two foot collision boxes have `contype=1` and `conaffinity=1`; there is no left-foot/right-foot exclusion. Actual solver contacts independently confirm that foot-to-foot collision is active. The small reported penetration is a compliant contact response, not evidence of a disabled collider.

In the compiled nominal standing configuration, both foot collision local **+x** axes point forward with the pelvis, and both local **+z** axes point vertically upward. The official visual foot meshes extend farther in +x (toe) than −x (heel), so these are appropriate heading and sole-normal conventions. The nominal signed width is +0.274300 m, with approximately zero fore-aft offset.

The collision boxes are simplified geometry, not exact visual meshes; this audit does not establish that every visual mesh surface is free of overlap. There is no evidence here of a reversed joint axis, incorrect foot-axis convention, or missing foot-pair collision causing this stance. It is an undesirable posture permitted by the current reward and success definition.

## Suggested refinement

Keep the original recovery result and its definition unchanged, and report a separate clean-stance measure. Apply additional posture rewards smoothly during the near-standing phase, using pelvis height and torso uprightness as the gate. Do not require both foot contacts to enable the penalties, because lifting a foot could otherwise disable them. Permit crossing and varied contacts during the ground-recovery phase.

- Reward positive foot-centre lateral width near the nominal 0.274 m, with a tolerant target band such as 0.20–0.34 m.
- Reward alignment of each horizontal foot +x direction with horizontal pelvis +x; use `1 - cosine(heading error)` to avoid angle wrapping.
- Reward each sole +z direction approaching world +z. A separate clean-stance threshold of at most 20° sole tilt would reject the present edge-supported result.
- Apply a targeted hip-yaw posture penalty, averaged over those two joints rather than diluted across all 31 joints.
- Also monitor fore-aft staggering (nominal zero; an initial clean-stance tolerance around 0.10 m is reasonable) and foot-to-foot contact force. Width alone cannot exclude a strongly staggered stance or mutual bracing.

These bands and thresholds are proposed simulation evaluation choices, not manufacturer requirements. Use moderate reward weights and retain the existing checkpoint so a refinement cannot erase the demonstrated recovery result.
