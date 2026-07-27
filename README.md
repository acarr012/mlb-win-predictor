# mlb-win-predictor

## Known Data Limitations
- 2 games (0.015% of dataset) are missing starting pitcher data: 
  2021-08-10 LAA/TOR game 2, and 2022-05-10 DET/ATH game 2. 
  Statcast has no pitch-tracking data for these specific games 
  (confirmed via full-month re-pull, not a caching or pipeline bug) — 
  a known, occasional gap in Statcast's public dataset. These rows 
  will have null pitcher-feature values and should be dropped or 
  handled explicitly during model training.