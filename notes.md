# Product thesis

TRIBE Compare Lab is for a marketing lead at a 20–200 person company or a solo founder choosing between two short video ads before committing budget. The app does one job: turn one or two uploaded videos into a fast structural read with six descriptive metrics, an overlaid curve view, and numeric observations that make differences legible without prescribing what to do next.

# Claims vs. non-claims

The app claims that it provides a descriptive, TRIBE-based comparison of two creatives and shows where they differ structurally and by how much. It does not claim to predict virality, CTR, ROAS, sales, or any business outcome. It does not claim to measure real cognition or neuroscience. It is not a replacement for audience testing; it is a pre-spend triage tool.

# Structural metric definitions

- Opening: mean activation over the first 20% of windows.
- Middle: mean activation over the middle 60% of windows.
- Closing: mean activation over the last 20% of windows.
- Peak: max activation across all windows.
- Spread: max minus min of the curve.
- Consistency: mean divided by standard deviation with an epsilon guard.

# TRIBE v2 constraints

The live path currently supports video and returns metrics derived from the model's response curve only. Temporal metrics are shown only when a temporal curve exists. The compare table assigns a winner only when the symmetric percentage gap exceeds the tie threshold. Until test-retest drift is measured in Hour 0, the fallback tie threshold is 7%, wired through `TRIBE_TIE_THRESHOLD_PCT` and documented as a temporary basis. Fixture mode is available through `TRIBE_FIXTURE_MODE=1`, with optional fallback via `TRIBE_FIXTURE_FALLBACK=1`, so the demo path remains reliable if live inference is too slow or unavailable.
