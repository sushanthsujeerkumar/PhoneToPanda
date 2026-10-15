# Development diagnostics

Reference: **original_h5**. Five source recordings; one custom simulated task.

| Model | Group | Placements | 95% Wilson interval | Lifts | Failed to lift | Lifted, not placed |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| original_h5 | new_narrow | 10/10 | 72.2%–100.0% | 10/10 | 0 | 0 |
| original_h5 | new_wide | 6/10 | 31.3%–83.2% | 6/10 | 4 | 0 |
| original_h5 | combined | 16/20 | Not pooled | 16/20 | 4 | 0 |
| continuation2000_h5 | new_narrow | 9/10 | 59.6%–98.2% | 10/10 | 0 | 1 |
| continuation2000_h5 | new_wide | 6/10 | 31.3%–83.2% | 6/10 | 4 | 0 |
| continuation2000_h5 | combined | 15/20 | Not pooled | 16/20 | 4 | 1 |
| uniform2000_h5 | new_narrow | 10/10 | 72.2%–100.0% | 10/10 | 0 | 0 |
| uniform2000_h5 | new_wide | 6/10 | 31.3%–83.2% | 7/10 | 3 | 1 |
| uniform2000_h5 | combined | 16/20 | Not pooled | 17/20 | 3 | 1 |
| targeted2000_h5 | new_narrow | 7/10 | 39.7%–89.2% | 7/10 | 3 | 0 |
| targeted2000_h5 | new_wide | 5/10 | 23.7%–76.3% | 5/10 | 5 | 0 |
| targeted2000_h5 | combined | 12/20 | Not pooled | 12/20 | 8 | 0 |

All compared starts have exactly matching initial cube and robot states.

| Candidate vs reference | Group | Improved | Regressed | Both placed | Both failed | Exact paired p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| continuation2000_h5 | new_narrow | 0 | 1 | 9 | 0 | 1 |
| continuation2000_h5 | new_wide | 1 | 1 | 5 | 3 | 1 |
| continuation2000_h5 | combined | 1 | 2 | 14 | 3 | 1 |
| uniform2000_h5 | new_narrow | 0 | 0 | 10 | 0 | 1 |
| uniform2000_h5 | new_wide | 1 | 1 | 5 | 3 | 1 |
| uniform2000_h5 | combined | 1 | 1 | 15 | 3 | 1 |
| targeted2000_h5 | new_narrow | 0 | 3 | 7 | 0 | 0.25 |
| targeted2000_h5 | new_wide | 2 | 3 | 3 | 2 | 1 |
| targeted2000_h5 | combined | 2 | 6 | 10 | 2 | 0.2891 |

Exact p values use a two-sided McNemar/binomial test of discordant placements; no multiple-comparison adjustment is applied.

![Placement counts and uncertainty](placement_results.png)

![Initial positions and outcomes](start_positions.png)

- Development results are diagnostics and selection evidence, not a held-out final test.
- One custom simulation scene, five source recordings, fixed cube and target; no new human recordings or real-robot evidence.
- Hybrid control: learned XYZ/gripper with fixed downward wrist feedback.
- Lift means cube centre exceeded 0.97 m. Failed after lift combines dropped-object and unsuccessful-placement outcomes; it does not diagnose which occurred.
- Wilson intervals describe each start group's marginal binomial proportion; they are not confidence intervals on the paired difference.
- Combined rates use fixed per-group trial quotas (new_narrow: 50.0%, new_wide: 50.0%). No pooled binomial interval is reported for this fixed stratified design.
- Intervals and exact paired p values compare these fixed checkpoints over simulator starts. One training seed provides no estimate of optimizer-seed or training-run variability.
- Development comparisons used for candidate selection are exploratory. Final candidate identities must be locked before final outcomes are observed; final outcomes must not select a new winner.
- Repeated starts in this scene do not establish broad task generalization. A selected rollout video illustrates one seed; aggregate counts provide the performance comparison.
- Recorded wall times may include concurrent workloads and are not a latency benchmark.
