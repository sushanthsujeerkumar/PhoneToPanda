# IMU-assisted demonstration pilot

Four controller replays succeeded. The learned policies were then tested separately on ten matching new simulator starts.

| Model | Narrow | Wide | Total |
| --- | ---: | ---: | ---: |
| Original model | 3/5 | 3/5 | 6/10 |
| Continued training | 3/5 | 3/5 | 6/10 |
| IMU-assisted augmentation | 3/5 | 2/5 | 5/10 |

![All planned outcomes](comparison.png)

The original checkpoint remains the default. These small pilot counts do not establish general improvement.

Compared with original, augmentation fixes 0 failed starts and regresses on 1 successful starts.
Compared with continuation, augmentation fixes 0 failed starts and regresses on 1 successful starts.

- One new human recording; four robot replays reuse its path.
- Ten new simulator starts per model, one scene and one training seed.
- Timing is provisional; shoulder is fixed; physical tracking accuracy is unmeasured.
- The policy uses simulator images and robot state, not IMUs.
- This tests demonstration augmentation, not vision/IMU sensor fusion.
- Every planned result is reported; no model is selected using these starts.

[Method and reproduction](../../docs/imu_pilot_20261015.md) · [Raw comparison](summary.json)
