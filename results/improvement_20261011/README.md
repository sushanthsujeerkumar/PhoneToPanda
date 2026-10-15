# Existing-recording study: 11 October 2026

The original checkpoint was selected using development results before final testing and remains the default. All adaptations are secondary comparisons. Five source recordings were reused; no new human or IMU recordings were collected.

| Model | Development | Final narrow | Final wide | Final total |
| --- | ---: | ---: | ---: | ---: |
| `original_h5` | 16/20 | 25/30 | 18/30 | **43/60** |
| `continuation2000_h5` | 15/20 | 24/30 | 20/30 | **44/60** |
| `uniform2000_h5` | 16/20 | 26/30 | 18/30 | **44/60** |
| `targeted2000_h5` | 12/20 | 20/30 | 18/30 | **38/60** |
| `uniform4000_h5` | 16/20 | 24/30 | 18/30 | **42/60** |
| `recovery2000_h5` | 13/20 | Not advanced | Not advanced | Not advanced |

The narrow and wide groups contain 30 matched starts each. All 300 planned final episodes completed, and the compared models began each paired episode at exactly the same recorded cube and robot state. There is one training seed and one fixed simulation scene; these counts do not establish real-robot or new-object generalization.

## Read and watch

- [Full methods, findings and reproduction instructions](../../docs/improvement_20261011.md).
- [Frozen final comparison](final/summary.md), [machine-readable final results](final/summary.json).
- [Overview chart](final_overview.png), [overview input/output hashes](final_overview.json).
- [All six development candidates](development/summary.md), [matched 2k comparison](matched_development/summary.md).
- [Action-horizon diagnostic](action_timing_diagnostic/summary.md): five versus one action per observation on old known-wide cases.
- [Improved example, narrow seed 6128](../../media/final_uniform_improved_seed6128.mp4): original versus uniform2k, a secondary contrast.
- [Expert recovery collection example](../../media/recovery_example.mp4): scripted teaching data, not a learned-policy success claim.

Illustration selection follows [the recorded presentation rule](illustration_plan.json); [the selection receipt](illustration_selection.json) identifies every available improved/regressed seed and the chosen examples. Each paired MP4 has a same-name JSON sidecar with input hashes and verified starting states. Videos follow simulation time, with the last frame held after a shorter episode finishes; they are not inference-speed comparisons.

## Evidence and checks

- [Study plan](study_plan.json), [selection decision](selection_decision.json), [frozen candidate lock](candidate_lock.json).
- [Replay coverage](replay_coverage.png), [complete numeric export audit](dataset_audit.json), [sampled image audit](image_export_audit.json).
- [Original final failure analysis](final_original_failure_analysis.json): every failed seed, commanded closure, sampled contacts and reference-point gaps; a descriptive post-test analysis.
- [Data collection provenance](data/), [checkpoint metadata and normalization verification](training/).
- [Final verification](final_verification.json): 137 tests passed, frozen source/configuration and original-data identities checked, and relocated evidence reproduced all summaries.
- [Evaluated source](evaluated_source.tar.gz) and [source manifest](evaluated_source_manifest.json): 49 files preserved before the later comments and documentation pass. All 25 pre-final frozen files matched their recorded hashes. This keeps the original evaluation code available as the project develops.
- [Readability checks](readability_verification.json): all 31 Python files retain the same executable syntax, and all 137 tests pass after the comments pass. The historical study verification above remains unchanged.
- [Evidence archive](study_evidence.tar.gz): 51.3 MB, 536 original files, 140 development and 300 final action traces. [File hashes](evidence_manifest.json), [archive receipt](evidence_archive.json).

Archive SHA256: `22a47ef35023d0861b5164c335e3c93b5e6abd3728941dc7fee5408aee03a40f`.

The archive stores paths relative to the study directory. It includes plans, selection sidecars, manifests, complete reports, every recorded action trace and the canonical source MP4s for the paired illustrations. Model weights and generated training datasets are omitted; the original reference is now available in the [model release](../../checkpoints/README.md), while the other 11 October candidate weights remain local. Summary regeneration requires no weights or simulator. Rerunning the policy itself requires the exact checkpoints and environment.

```bash
mkdir /tmp/phonetopanda-evidence
tar -xzf results/improvement_20261011/study_evidence.tar.gz -C /tmp/phonetopanda-evidence
.venv/bin/python scripts/summarize_study.py \
  --study /tmp/phonetopanda-evidence --phase final_test \
  --names original_h5 continuation2000_h5 uniform2000_h5 targeted2000_h5 uniform4000_h5 \
  --output /tmp/phonetopanda-final-summary
```

Published final starts are reproduction cases, not unseen evidence for future tuning. Historical 4 October results remain [separate](../../docs/results.md).
