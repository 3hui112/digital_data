# H2 follow-up checks — decision rules fixed before any of these runs (2026-10-27)

Context: a validation-only pilot E1 on the Phase 1 checkpoints (seeds 0–2) showed ~0 drop for 3–5 / 5–8 Hz removal
in all models, so the proposal's H2 (3–8 Hz) is inconclusive by its own rule. The 1–2 Hz band showed the largest
selective-minus-LTI difference in that pilot. Both checks below use validation only; the official test set is not touched.

Common definitions (same as the proposal draft "Team6_proposal (1).pdf"):
- Filter: scipy.signal.butter(4, [lo, hi], btype="bandstop", fs=50, output="sos"), applied with sosfiltfilt along time
  (default odd padding) to all 9 raw channels, before the original training-fitted normalizer.
- Dynamic-activity F1 = mean of WALK, UP, DOWN class F1 from the six-class predictions.
- drop = metric(reference input) − metric(band-stopped input), same checkpoint, same validation windows.
- q_s = (selective drop − LTI drop) for seed s; δ = 1 percentage point.
- Support: q_s > δ in all seeds. Falsified: q_s < −δ in all seeds. Otherwise inconclusive.
- Decision metric is dynamic-activity F1; macro-F1 and accuracy are reported but do not decide.

## Check A — confirm the pilot-derived 1–2 Hz hypothesis on independent seeds
- H2': selective Mamba shows a larger dynamic-activity F1 drop than LTI Mamba when 1–2 Hz is removed.
- New models: selective and LTI, starter defaults, seeds 3, 4, 5 (never used in the pilot).
- Reference input: unfiltered validation. Band: 1–2 Hz only.
- All seven bands are also evaluated and reported, but only 1–2 Hz decides.

## Check B — the original 3–8 Hz question under a condition where low frequencies are unavailable
- H2'': when 0.3–3 Hz is removed from both training and validation data, selective Mamba relies more than LTI on 3–8 Hz.
- New models: selective and LTI retrained on 0.3–3 Hz band-stopped data (one sosfiltfilt pass, band 0.3–3 Hz),
  normalizer fitted on the unfiltered training subset, checkpoint chosen on identically filtered validation,
  starter defaults otherwise, seeds 0, 1, 2.
- Reference input: 0.3–3 Hz-removed validation. Test inputs: reference plus an additional 3–5 Hz or 5–8 Hz band-stop.
- q_s = mean over {3–5, 5–8} of (selective drop − LTI drop).
