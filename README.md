# mlb-win-predictor

## Known Data Limitations
- 2 games (0.015% of dataset) are missing starting pitcher data: 
  2021-08-10 LAA/TOR game 2, and 2022-05-10 DET/ATH game 2. 
  Statcast has no pitch-tracking data for these specific games 
  (confirmed via full-month re-pull, not a caching or pipeline bug) — 
  a known, occasional gap in Statcast's public dataset. These rows 
  will have null pitcher-feature values and should be dropped or 
  handled explicitly during model training.

## EDA Findings

Leakage sanity checks performed on the final feature table: home team
win% entering game and starting pitcher WHIP entering game were each
plotted against actual game outcomes. Both showed real-but-modest
separation in the theoretically correct direction (higher win% and
lower WHIP correlate with home wins), with substantial distributional
overlap — consistent with legitimate signal, not data leakage.

Confirmed real-world plausibility: overall home-field win rate (53.1%)
matches published MLB home-field advantage figures; run differential
vs. win rate shows the expected strong positive relationship.

## Additional Known Limitations

- Small-sample noise in early-season rolling stats: pitchers/teams with
  very few games or starts recorded can produce extreme values (e.g. a
  WHIP as high as ~21 from a single short outing, or a win% of exactly
  0% or 100% from 1-2 games played). Visually confirmed in EDA
  (histograms show a small tail/spike from this effect). Not corrected
  for, since tree-based models (Decision Tree, Random Forest, XGBoost)
  are relatively robust to this kind of outlier compared to linear
  models — a deliberate scope decision, not an oversight.
- The Athletics' 2025 franchise rename (OAK -> ATH) means team-level
  aggregate views (e.g. home win rate by team) require combining both
  labels for accurate team-level display; the underlying per-season
  modeling data correctly keeps them separate by season.