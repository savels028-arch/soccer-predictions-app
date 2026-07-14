# Overnight Optimization Report

Generated: 2026-05-11

> **SUPERSEDED 2026-07-14 — not a live betting recommendation.** This report
> is preserved as a historical experiment log. Its headline profits were
> selected from older sweeps and do not pass the newer leakage-resistant,
> locked-policy promotion gates. The current full research run is
> `NO_PROMOTION`: no 1X2, O/U 2.5, Asian-handicap or coupon strategy is approved
> for actionable bets. See `research/README.md` and the latest artifact under
> `data/research/runs/` for the current evidence.

## Penge Først

Målet er ikke at ramme flest kampe. Målet er at tjene penge.

En strategi kan godt ramme under halvdelen og stadig tjene penge, hvis oddsene er høje nok. Eksempel: ved odds 4.00 skal vi kun ramme mere end 25% for at have en fordel. Ved odds 1.25 skal vi ramme mere end 80%, ellers taber vi penge.

Det vi leder efter i historikken er derfor:

- hvor ville vi have tjent flest penge fra 10,000 DKK.
- hvor ofte var strategien positiv år efter år.
- hvor stort fald kunne vi have ramt undervejs.
- om historikken viser en gentagelig fordel, ikke bare en heldig lille sample.

Historisk bedste in-sample kandidat i denne nu forældede rapport:

- Historik-edge kuponer.
- Brug markedsfavoritten, men kun når samme hjemmehold mod samme udehold tidligere havde et tydeligt mønster.
- Krav: mindst 2 tidligere indbyrdes kampe og mindst 60% historisk støtte.
- Krav: v2 confidence mindst 65%.
- Kupon: max 6 kampe, max 2 fra samme liga.
- Start: 10,000 DKK.
- Slut: 29,366 DKK.
- Profit: +19,366 DKK.
- Drawdown: 2,708 DKK.
- Kupon-hit: 31.82%.
- Resultatet er ikke længere godkendt som en fremadrettet strategi.

Historisk bedste rene historik-kandidat i denne rapport:

- Samme hjemmehold mod samme udehold.
- Spil kun når der var mindst 3 tidligere indbyrdes kampe, og mindst 80% endte med samme 1X2-resultat.
- Start: 10,000 DKK.
- Slut: 15,290 DKK.
- Profit: +5,290 DKK.
- Drawdown: 1,637 DKK.
- Det viser, at historikken har værdi, men den har ikke slået den bedste pengestrategi endnu.

Historisk mere rolig historik-kupon i denne rapport:

- Samme hjemme-vs-ude historik-bekræftelse, max 3 kampe på kuponen.
- Start: 10,000 DKK.
- Slut: 16,658 DKK.
- Profit: +6,658 DKK.
- Drawdown: 783 DKK.
- Coupon hit: 54.14%.
- Den tjener mindre, men har langt mindre fald undervejs og er nemmere at stole på end max-profit strategien.

## Commands Run

- `venv/bin/python -m pytest tests/test_prediction_pipeline_regressions.py tests/test_web_refresh_regressions.py`
- `venv/bin/python backtest.py --walk-forward`
- `venv/bin/python backtest.py --optimize data/backtest_predictions.json`
- `venv/bin/python backtest.py --history-edge data/backtest_predictions.json`
- `venv/bin/python backtest.py --h2h-coupon-criteria-csv --start-season 2000 --end-season 2025 --first-test-season 2012 --validation-start-season 2022`

## Walk-Forward Model Result

- Dataset: 22,495 walk-forward predictions per model, seasons 2019-2025.
- v1 accuracy: 51.85%.
- v2 accuracy: 46.21%.
- v1 Brier: 0.5917.
- v2 Brier: 0.7279.
- v1 log loss: 0.9928.
- v2 log loss: 1.2715.
- v2 stays disabled. It did not beat v1 stably on accuracy, Brier, or log loss.

## Recommended Single Strategy

- Model: v1.
- Bet style: market underdog.
- Filter: Serie A (`SA`), draw only, model edge >= 2%, no confidence minimum.
- Bankroll simulation: 10,000 -> 17,255.
- Profit: +7,255 DKK.
- ROI: +20.97%.
- Max drawdown: 1,010 DKK.
- Bets: 346.
- Hit rate: 36.13%.
- Note: this is not "bet everything opposite"; it is a narrow draw-underdog strategy where lower hit rate is compensated by higher odds.

## Latest Full Strategy Sweep

Re-run on 2026-05-12 with all available historical walk-forward model predictions:

- Historical source file: `data/backtest_predictions.json`.
- Scope: walk-forward seasons 2019-2025, 22,495 predictions per model.
- Tested single styles: model pick, least likely, market underdog.
- Tested singles: 22,176 candidates across v1/v2.
- Tested coupons: 17,170 candidates across v1/v2.
- Best single stayed:
  - `v1 single style=market_underdog conf>=none edge>=2% outcome=draw leagues=league:SA`.
  - 346 bets, 125 wins, 36.13% hit rate.
  - 10,000 -> 17,255, profit +7,255 DKK, ROI +20.97%, max drawdown 1,010 DKK.
- Best coupon stayed:
  - `v1 coupon conf>=50% edge>=2% max=6 sort=edge_x_confidence max_per_league=3 leagues=top_leagues`.
  - 369 coupons, 77 winners, 20.87% coupon hit rate.
  - 10,000 -> 23,632, profit +13,632 DKK, ROI +36.94%, max drawdown 3,440 DKK.
- v2 remains disabled:
  - v1 accuracy 51.85%, Brier 0.5917, log loss 0.9928.
  - v2 accuracy 46.21%, Brier 0.7279, log loss 1.2715.

## Source Backtest Status

- The new external sites can be used live from now on, but cannot yet be backtested through 2019-2025 because the project has not archived their historical daily picks.
- Local SQLite only has 7 historical `prediction_results` rows from AI-site sources, which is not enough for a serious source strategy.
- `PredictionPitch` exposes public current/upcoming predictions and value-bet metrics, but its public frontend API only exposes `limit` and `valueBetsOnly`, not a multi-year historical date range.
- `WinFulltime` exposes a short rolling date window through its public JSON endpoint, not multi-year history.
- Next required lift: persist every scraped source pick daily with odds-at-scrape and result, then run this same optimizer by source once enough samples exist.

## Historical Pattern Backtest

Added and ran `backtest.py --history-patterns data/backtest_predictions.json` on 2026-05-12.

- Source matches: 22,495 walk-forward historical matches.
- Tested pattern families:
  - exact directed H2H outcome: same home team vs same away team.
  - unordered pair dominant result: same two teams, either venue.
  - home team's home-outcome tendency.
  - away team's away-outcome tendency.
  - home/away team's any-venue win/draw/loss tendency.
  - no-draw H2H with market favourite.
- Tested 840 pattern candidates.
- Best historical pattern:
  - `directed_h2h_outcome min_matches=3 min_rate=80% max_odds=4.0`.
  - Meaning: only bet when the exact same home-vs-away fixture had at least 3 prior meetings and at least 80% of those meetings ended with the same 1X2 outcome.
  - Bankroll: 10,000 -> 15,290.
  - Profit: +5,290 DKK.
  - ROI: +4.91%.
  - Bets: 1,078.
  - Hit rate: 61.41%.
  - Max drawdown: 1,637 DKK.
- Season result:
  - 2022: -138 DKK, 57.98% hit.
  - 2023: +391 DKK, 58.56% hit.
  - 2024: +1,009 DKK, 61.79% hit.
  - 2025: +4,028 DKK, 64.86% hit.
- Interpretation: useful signal, but not stronger than the best single strategy or coupon strategy. It should be used as a filter/confirmation signal, not as the main betting engine.

## Recommended Coupon Strategy

- Model: v1.
- Filter: top leagues only (`BL1`, `DED`, `FL1`, `PD`, `PL`, `PPL`, `SA`).
- Confidence >= 50%.
- Edge >= 2%.
- Max 6 picks.
- Max 3 picks per league.
- Sort by edge x confidence.
- Bankroll simulation: 10,000 -> 23,632.
- Profit: +13,632 DKK.
- ROI: +36.94%.
- Max drawdown: 3,440 DKK.
- Coupons: 369.
- Coupon hit rate: 20.87%.

## Extra Deep Search After H2H Patterns

- Tested additional label strategies:
  - normal model pick.
  - max model-implied edge across home/draw/away.
  - best non-model edge, i.e. fade/contrarian candidates.
  - market favourite.
  - market underdog.
  - forced home/draw/away picks.
- Tested additional filters:
  - all leagues, top leagues, and selected single leagues.
  - odds bands from 1.2 to 4.0.
  - exact H2H confirmation with no future leakage.
- Result: no new strategy beat the current best profit strategy (`+13,632 DKK`).
- Best new conservative coupon candidate:
  - Model: v1 model pick.
  - Filter: all leagues, confidence >= 50%, selected odds 1.2-2.5.
  - H2H confirmation: same home-vs-away fixture had at least 3 prior matches and at least 80% same outcome.
  - Coupon: max 3 legs, max 2 per league, sort by edge x confidence.
  - Bankroll simulation: 10,000 -> 16,658.
  - Profit: +6,658 DKK.
  - Max drawdown: 783 DKK.
  - Coupons: 157.
  - Coupon hit rate: 54.14%.
- Interpretation: this is not the best growth strategy, but it is a better stability/high-hit candidate. It is useful as an alternate low-drawdown mode or as a confirmation filter, not as the main profit-maximizing config.

## Money-First Historical Edge Backtest

Added and ran `backtest.py --history-edge data/backtest_predictions.json` on 2026-05-13.

- Tested 21,798 single strategies.
- Tested 2,700 coupon strategies from the strongest historical filters.
- Best single:
  - v2 model pick.
  - Exact home-vs-away H2H had at least 2 prior matches and at least 75% same outcome.
  - Confidence >= 60%.
  - Bankroll: 10,000 -> 19,461.
  - Profit: +9,461 DKK.
  - Bets: 1,553.
  - Hit rate: 68.0%.
  - Max drawdown: 1,605 DKK.
- Best coupon:
  - v2 market favourite.
  - Exact home-vs-away H2H had at least 2 prior matches and at least 60% same outcome.
  - Confidence >= 65%.
  - Max 6 coupon legs, max 2 per league, sort by edge x confidence.
  - Bankroll: 10,000 -> 29,366.
  - Profit: +19,366 DKK.
  - Coupons: 440.
  - Coupon hit: 31.82%.
  - Max drawdown: 2,708 DKK.
- Profit by season:
  - 2021: +2,144 DKK.
  - 2022: +1,713 DKK.
  - 2023: +1,556 DKK.
  - 2024: +2,160 DKK.
  - 2025: +11,793 DKK.
- Interpretation: this is the strongest money result so far. It uses v2 only inside this historical-edge rule; v2 should still not be switched on globally until this is validated against fresh data.

## Long CSV Historical Edge Backtest From 2000

Added and ran `backtest.py --history-edge-csv --start-season 2000 --end-season 2025` on 2026-05-13.

- Source: Football-Data CSV archive.
- Seasons: 2000/01 through 2025/26.
- Source matches loaded: 91,366.
- Usable matches with 1X2 odds: 84,357.
- This is a pure history/odds test. It does not include v1/v2 model confidence.
- Best robust single by score:
  - League: Portugal (`PPL`).
  - Bet: market favourite only.
  - Rule: exact same home-vs-away fixture had at least 6 prior matches and at least 67% same result.
  - Bankroll: 10,000 -> 13,036.
  - Profit: +3,036 DKK.
  - Bets: 749.
  - Hit rate: 76.23%.
  - Max drawdown: 1,030 DKK.
- Highest-profit eligible single found in the top results:
  - League: Spain (`PD`).
  - Bet: historical dominant outcome.
  - Rule: at least 3 prior exact home-vs-away matches, at least 80% same outcome, historical edge >= 0%.
  - Bankroll: 10,000 -> 15,467.
  - Profit: +5,467 DKK.
  - Bets: 778.
  - Hit rate: 65.68%.
  - Max drawdown: 1,321 DKK.
- Best coupon candidate:
  - Bankroll: 10,000 -> 12,578.
  - Profit: +2,578 DKK.
  - Coupons: 104.
  - Coupon hit: 55.77%.
  - Max drawdown: 581 DKK.
  - Not accepted as robust because several seasons had too few coupons.
- Interpretation:
  - The pure historical edge exists back to 2000, but it is modest.
  - The big 2019-2025 v2 coupon result should be treated as promising, not proven.
  - The safest forward test is singles first, not aggressive coupons.
  - The exact v2-confidence version cannot be honestly tested back to 2000 without generating long-range walk-forward model predictions first.

## Strategy Zoo Backtest From 2000

Added and ran `backtest.py --strategy-zoo-csv --start-season 2000 --end-season 2025` on 2026-05-13.

- Scope:
  - 91,366 historical matches loaded.
  - 84,357 usable matches with 1X2 odds.
  - 18 strategy families.
  - 62,362 single-bet candidates.
  - 930 coupon candidates.
- Best robust single strategy:
  - Strategy: direct H2H history in top leagues.
  - Rule: at least 10 prior exact home-vs-away matches, at least 75% same outcome, odds 1.20-2.00.
  - Bankroll: 10,000 -> 14,805.
  - Profit: +4,805 DKK.
  - Bets: 1,052.
  - Hit rate: 73.48%.
  - ROI: +4.6%.
  - Max drawdown: 1,590 DKK.
- Highest-profit single strategy:
  - Strategy: direct H2H history in La Liga.
  - Rule: at least 3 prior exact home-vs-away matches, at least 80% same outcome, historical edge >= 0%.
  - Bankroll: 10,000 -> 15,467.
  - Profit: +5,467 DKK.
  - Bets: 778.
  - Hit rate: 65.68%.
  - ROI: +7.0%.
  - Max drawdown: 1,321 DKK.
- Best coupon strategy:
  - Strategy: 2-leg coupons from direct H2H top-league picks.
  - Rule: at least 10 prior exact home-vs-away matches, at least 75% same outcome, edge >= 2%, odds 1.20-2.50, sort by confidence.
  - Bankroll: 10,000 -> 15,761.52.
  - Profit: +5,761.52 DKK.
  - Coupons: 209.
  - Coupon hit rate: 55.98%.
  - ROI: +27.6%.
  - Max drawdown: 571.75 DKK.
- Interpretation:
  - The strongest repeated signal is not "more sources" by itself. It is old direct H2H dominance plus reasonable odds, especially in top leagues.
  - The best coupon result beats the best single result on bankroll and drawdown, but it has fewer observations than the best single strategy.
  - This is a better candidate than the earlier broad CSV historical-edge run, but it should still be paper-traded live before staking real money.
  - The best long-history strategy starts in 2012 because the rule needs at least 10 prior exact H2H meetings before it is allowed to bet.

## Strategy Zoo Walk-Forward Validation

Added `backtest.py --strategy-zoo-walk-forward-csv` on 2026-05-13. This mode chooses the strategy from past seasons only, then tests it on the next season. It is stricter than picking the best rule on the full dataset.

- Full walk-forward from 2005:
  - Singles: 10,000 -> 9,560.
  - Single profit: -440 DKK.
  - Single bets: 939.
  - Single hit rate: 64.1%.
  - Coupons: 10,000 -> 12,961.
  - Coupon profit: +2,961 DKK.
  - Coupons played: 341.
  - Coupon hit rate: 39.3%.
  - Coupon max drawdown: 4,018 DKK.
  - Interpretation: profitable, but the early immature-history period has too much drawdown.
- Mature-history walk-forward from 2012:
  - Command run: `venv/bin/python backtest.py --strategy-zoo-walk-forward-csv --start-season 2000 --end-season 2025 --first-test-season 2012 --min-train-bets 150`
  - Singles: 10,000 -> 8,499.
  - Single profit: -1,501 DKK.
  - Single bets: 537.
  - Single hit rate: 70.0%.
  - Coupons: 10,000 -> 14,352.
  - Coupon profit: +4,352 DKK.
  - Coupons played: 161.
  - Coupon hit rate: 55.9%.
  - Coupon ROI: +27.0%.
  - Coupon max drawdown: 626 DKK.
  - Best repeated coupon family in recent folds: direct H2H / favorite-agrees-with-H2H.
- Interpretation:
  - The single-bet strategy is not good enough out-of-sample, even with high hit rate.
  - This older coupon result looked promising, but it did not survive the stricter 2026-07-14 locked-policy promotion gates.
  - There is no current forward coupon recommendation; the live product must abstain.
  - Real-money staking remains disabled unless a future forward-only run passes every promotion gate with verified at-pick odds.

## H2H Coupon Criteria Sweep

Added and ran `backtest.py --h2h-coupon-criteria-csv --start-season 2000 --end-season 2025 --first-test-season 2012 --validation-start-season 2022` on 2026-05-13.

- Purpose: test extra profit knobs against history instead of guessing.
- Train period: 2012-2021.
- Validation period: 2022-2025.
- Source matches: 91,366.
- Usable matches with 1X2 odds: 84,357.
- Filters evaluated: 3,911.
- Coupon candidates evaluated: 546.
- Criteria tested:
  - direct H2H vs market-favourite/H2H agreement.
  - top leagues and per-league filters.
  - Bet365 odds, Bet365 closing odds, average odds, and average closing odds.
  - prior H2H count, H2H hit rate, historical edge, odds band.
  - season timing by league match number.
  - recent-form support from each team's last five completed matches.
  - coupon legs, sort mode, max picks per league, and max combined coupon odds.
- Best strategy selected only from train period:
  - `favorite_direct_h2h_agree`, top leagues, average closing odds.
  - Rule: at least 12 prior exact H2H matches, at least 75% same outcome, edge >= 2%, odds 1.20-2.50.
  - Coupon: 2 legs, sort by confidence, max 1 per league, max combined odds 5.0.
  - Train: 10,000 -> 12,050, profit +2,050 DKK, 57 coupons, 61.4% hit, 352 DKK max drawdown.
  - Validation: 10,000 -> 10,728, profit +728 DKK, 47 coupons, 53.2% hit, 527 DKK max drawdown.
- Best validation candidate:
  - `direct_h2h`, top leagues, average odds.
  - Rule: at least 12 prior exact H2H matches, at least 75% same outcome, edge >= 5%, odds 1.20-2.00.
  - Coupon: up to 3 legs, sort by confidence, max 1 per league, max combined odds 4.0.
  - Validation: 10,000 -> 11,657, profit +1,657 DKK, 35 coupons, 60.0% hit, 300 DKK max drawdown.
- Best live-like closing-odds candidate:
  - Bet365 closing odds, `favorite_direct_h2h_agree`/`direct_h2h`, top leagues.
  - Rule: at least 12 prior exact H2H matches, at least 75% same outcome, edge >= 5%, odds 1.20-2.50.
  - Validation: +1,603 DKK, 45 coupons, 55.6% hit.
- Recent-form filter result:
  - No recent-form candidate reached the top validation list.
  - The last-five-games form filter did not improve this H2H coupon strategy.
- Decision:
  - Do not replace live config from this sweep.
  - The older mature H2H coupon result is superseded and is not a forward recommendation after the stricter 2026-07-14 run returned `NO_PROMOTION`.
  - Closing odds may be tracked as a diagnostic, but actionable evaluation requires verified pre-match odds captured at pick time.

## Over/Under 2.5 Market Test

Added and ran `backtest.py --over-under25-walk-forward-csv` on 2026-06-03.

- Purpose: test whether moving from 1X2 into goal markets creates a better money edge.
- Scope:
  - Full archive run: 91,366 source matches, 29,403 usable matches with Bet365 O/U 2.5 odds.
  - Newer odds-rich run: 24,624 source matches, 24,604 usable matches with Bet365 O/U 2.5 odds.
- Strategy families tested:
  - market favourite and market underdog on O/U 2.5.
  - Poisson total-goal probability.
  - Poisson edge against no-vig market probability.
  - recent team over/under rates.
  - league over/under rates.
  - pair over/under history.
  - market/Poisson agreement.
- Full archive no-hindsight result:
  - Candidates evaluated: 204,624.
  - Bankroll: 10,000 -> 8,240.
  - Profit: -1,760 DKK.
  - ROI: -5.57%.
  - Bets: 316.
  - Hit rate: 56.01%.
  - Max drawdown: 2,979 DKK.
- Newer odds-rich no-hindsight result:
  - Candidates evaluated: 178,091.
  - Bankroll: 10,000 -> 2,308.
  - Profit: -7,692 DKK.
  - ROI: -9.72%.
  - Bets: 791.
  - Hit rate: 49.43%.
  - Max drawdown: 7,725 DKK.
- Decision:
  - Reject O/U 2.5 for live betting for now.
  - Do not add O/U 2.5 to the coupon engine just because competitors market goal tips.
  - Goal markets stay in research mode until we can beat this backtest with no-hindsight selection and controlled drawdown.

## Config Changes Applied Locally

- These entries describe the 2026-05 experiment state and are retained only for audit history.
- The current config disables the legacy H2H coupon and paper-trading strategies because the latest locked-policy run returned `NO_PROMOTION`.
- `run_pipeline.py` must publish `ABSTAIN` unless a future registered strategy passes the promotion gates.
- `run_pipeline.py` now respects coupon `allowed_leagues`.
- Legacy `historical_h2h_coupon` code is not an approved live betting path.
- `AB_TEST["enabled"]` remains `False`.

## Why We Do Not Hit More

- Raw all-league betting is still negative. More data sources do not help if they repeat the same market signal or add noisy late information.
- v1 confidence is reasonably calibrated, but edge alone is weak across all leagues.
- Draw predictions have low hit rate, but can still be profitable in narrow high-odds spots.
- Some profitable-looking strategies are league-specific, so using them globally destroys ROI.
- Coupons can produce high profit, but the hit rate is naturally lower because every leg must win.

## Free Prediction Site Crawlers

- Existing crawler sources: `ai-goalie.com`, `betswithbots.com`, `soccertips.ai`, `footballpredictions.ai`, `predictionpitch.com`, `winfulltime.com`.
- New sources added on 2026-05-12:
  - `predictionpitch.com`: public JSON endpoint with 1X2 probabilities, odds, BTTS, O/U, and value-bet flags.
  - `winfulltime.com`: public JSON endpoint with 1X2 probabilities and tip labels.
- Live test on 2026-05-12 returned 690 predictions:
  - `ai-goalie.com`: 574.
  - `predictionpitch.com`: 85.
  - `winfulltime.com`: 15.
  - `betswithbots.com`: 9.
  - `footballpredictions.ai`: 7.
  - `soccertips.ai`: blocked/non-ready in this run (`HTTP 202` on prediction page).
- Fixed crawler status reporting so it falls back from `HEAD` to `GET`; the old status check incorrectly reported working sites as down.
- The old file comment claimed `OddAlerts.com`, but there was no OddAlerts scraper implemented. The comment is now corrected.

## GitHub / OSS Candidates Checked

- `penaltyblog`: useful next dependency for Dixon-Coles/Poisson models, implied odds, overround removal, and football-data scraping.
- `soccerdata`: useful for richer team/player context from Club Elo, ESPN, FBref, Football-Data, FotMob, Sofascore, SoFIFA, Understat, and WhoScored.
- `socceraction`: useful later if we add event-data based VAEP/xT features.
- `soccer-xg`: useful later if we ingest event streams and train our own xG model.
- Recommendation: add `penaltyblog` first only if Python compatibility is clean in this environment; add `soccerdata` behind an optional enrichment command because scraping dependencies and website drift can be brittle.

## Missing Data That Should Be Next

- Confirmed starting lineups.
- Injuries and suspensions.
- Player strength or player rating deltas.
- Opening odds vs closing odds and closing-line value.
- xG, shot, event, and tracking data.
- EPV or possession-value features if event/tracking data becomes available.

## Internet Notes

- Football-Data documents opening and closing odds columns, including closing home-draw-away odds and market average/max prices: https://www.football-data.co.uk/data
- Expected-goals research supports using xG and richer player/team ability features to reduce goal-result noise: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282295
- Player-adjusted xG work points directly at player quality as a missing signal: https://www.sciencedirect.com/science/article/pii/S2773186323000282
- EPV vs xG research suggests event/tracking features can improve pre-match outcome prediction: https://www.frontiersin.org/articles/10.3389/fspor.2025.1713852/full
- Betting line movement research supports tracking opening-to-closing market movement instead of only settled odds: https://pubsonline.informs.org/doi/10.1287/mnsc.2022.00456

## Push/Deploy

No push or deploy was run.
