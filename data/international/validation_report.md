# International / World Cup model validation

Status: **VALIDATED_FORECAST_ONLY**. This is a calibrated outcome forecast, not a validated
betting strategy. There are no historical pre-match odds in the source, so the
report makes no ROI claim and the live pipeline must not create coupon/P&L bets.

## Frozen design

- Source: Mart Jürisoo international results (CC0), mirrored by OpenFootball
- Source commit: `23449460b67a975bcf84d1042472c4f8da507f9c`
- Source SHA-256: `126d983c1e0f6849f1d75222da93aa0e8559ce3fb32ab17da02c526c4b69288e`
- Model selection/calibration: 2010-01-01 through 2017-12-31
- Untouched holdout: 2018-01-01 through 2025-12-31
- Point-in-time rule: predict every date batch before applying any result from it
- Selected parameters: `{"draw_base": 0.3, "draw_decay": 0.9, "home_advantage": 100.0, "k_factor": 40.0, "min_team_matches": 8, "temperature": 0.85}`

## Honest holdout

| Scope | Matches | Accuracy | Brier (lower) | Log loss (lower) | Top-label ECE |
|---|---:|---:|---:|---:|---:|
| Elo model, all internationals | 7,797 | 59.66% | 0.5213 | 0.8886 | 1.83% |
| Fixed prior baseline | 7,797 | 47.71% | 0.6337 | 1.0508 | 0.19% |
| World Cup 2018 + 2022 only | 128 | 53.12% | 0.6032 | 1.0144 | 6.33% |
| World Cup fixed prior baseline | 128 | 41.41% | 0.6461 | 1.0658 | 0.18% |

## Fail-closed gates

- PASS — `enough_holdout_matches`
- PASS — `enough_world_cup_matches`
- PASS — `beats_prior_accuracy`
- PASS — `beats_prior_brier`
- PASS — `beats_prior_log_loss`
- PASS — `calibration_within_limit`
- PASS — `world_cup_beats_prior_accuracy`
- PASS — `world_cup_beats_prior_brier`
- PASS — `world_cup_beats_prior_log_loss`
- PASS — `world_cup_calibration_within_limit`
- PASS — `point_in_time_batching`

## Limitations

- No historical pre-match odds; ROI and betting edge are not validated.
- Source full-time scores may include extra time, so this is not a 90-minute betting settlement backtest.
- No lineups, injuries, player availability, travel or current FIFA ranking snapshots.
