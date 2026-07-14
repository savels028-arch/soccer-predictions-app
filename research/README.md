# AIBets strategy-research

Den hjemlige strategy zoo er adskilt fra den forecast-only landsholdsmodel.
VM-modellens dataproveniens, validering og begrænsninger er dokumenteret i
[`data/international/README.md`](../data/international/README.md) og den frosne
[`validation_report.md`](../data/international/validation_report.md). Der laves
ingen VM-ROI-påstand, fordi den internationale kilde ikke har historiske
pre-match odds.

Denne mappe er AIBets' isolerede research-lag. Formålet er at finde strategier,
der skaber positiv ROI på odds, uden at bruge information fra fremtiden. En høj
prediction-procent er ikke nok: profit, usikkerhed, stabilitet og closing-line
value (CLV) vurderes separat.

Research-resultater ændrer ikke live-predictions automatisk. En kandidat skal
først bestå de faste gates nedenfor og derefter køre fremadrettet i shadow mode.

## Kør en reproducerbar analyse

Kør kommandoerne fra projektets rod med projektets virtualenv:

```bash
# Kontroller datasæt, dubletter, datointerval og odds-dækning
venv/bin/python -m research.run_zoo audit \
  --start-season 1993 --end-season 2024

# Fuld strategy zoo for 1X2 og over/under 2,5 mål
venv/bin/python -m research.run_zoo backtest \
  --start-season 1993 --end-season 2024 \
  --first-test-season 2012 --last-test-season 2024 \
  --policy-lock-season 2022 \
  --markets 1x2,ou25 --rebuild-features

# Korrekt afregnet Asian Handicap-negativ kontrol
venv/bin/python -m research.run_asian_handicap_benchmark \
  --end-season 2024 --include-proxies \
  --output data/research/runs/asian_handicap_blind_canonical_1993_2024.json
```

Brug `--leagues PL PD ...` til et afgrænset ligasæt, `--no-boosting` til en
hurtigere baseline og `--rebuild-features` når feature-logikken er ændret. Se
alle valg med `venv/bin/python -m research.run_zoo backtest --help`.

### Offentligt Strategy Zoo-overblik

Den faste, offentlige strategioversigt genereres separat fra modeludvælgelsen:

```bash
venv/bin/python -m research.run_pattern_zoo
```

Kommandoen skriver det validerede artifact
[`data/strategy_zoo_public.json`](../data/strategy_zoo_public.json). Det viser
33 på forhånd definerede H2H-, 1X2-, uafgjort-, mål- og exact-score-regler år
for år. Reglerne bliver ikke genvalgt ud fra samme års resultat. Kampe med samme
starttid isoleres fra hinanden, og kun komplette Bet365-opening-markeder kan
skabe P&L. Manglende odds giver derfor aldrig syntetisk profit.

Det aktuelle artifact dækker 115.840 evaluerede kampe fra 1993/94 til og med
2025/26. Alle ti 2025/26-filer blev hentet og atomisk valideret efter sæsonens
afslutning; der er derfor ingen karantænesatte kampe i den aktuelle udgave.

Den samlede status er `NO_CONFIRMED_BETTING_EDGE`. Tre af 141 H2H-kandidater
bestod en separat win-rate-test med Benjamini–Hochberg-korrektion ved q<=0,05:
Ajax–Waalwijk, PSV–Waalwijk og Feyenoord–Zwolle. Det bekræfter et gentageligt
resultatmønster mod en 50%-nulhypotese, ikke profit til de tilbudte odds. Alle
tre står derfor fortsat uden bekræftet betting-edge eller garanti.

Den lokale uge-træning genopbygger artifactet efter en bestået data-refresh.
Hvis genopbygningen fejler, bevares det seneste validerede artifact i stedet
for at publicere et halvt eller ukomplet resultat.

Hver kørsel skriver et manifest med SHA256 for datasæt, feature-cache og hver
research-kildefil, samlet code-fingerprint samt Git dirty/untracked-status.
Feature-cache-nøglen afledes af datasæt, parser/feature-kode og den eksplicitte
feature-konfiguration, så ændret logik ikke kan genbruge en gammel cache. Der
skrives desuden fold-resultater, alle bets, en opsummering, promotion-gates, den
låste strategi, en registry-audit og en læsbar rapport til
`data/research/runs/<run-id>/`. Feature-cache og runs er genererede artefakter og
er derfor ikke versionsstyret.

## Datasæt og begrænsninger

Den aktuelle canonical cache bruger dataset-id `7e131786cd7a35f0ae84` fra 328
lokale Football-Data CSV-filer:

- 115.840 valide, unikke klubkampe fra 23. juli 1993 til 24. maj 2026.
- Ti turneringer: Premier League, Championship, Bundesliga, 2. Bundesliga,
  La Liga, Serie A, Ligue 1, Eredivisie, Primeira Liga og belgisk 1. division.
- Åbningsodds findes for 91.451 1X2-kampe (78,9 %), 78.753 O/U 2,5-kampe
  (68,0 %) og 78.907 Asian Handicap-kampe (68,1 %). Closing odds findes for
  49.364 1X2-kampe og 24.760 O/U-kampe.
- Kampe uden komplette odds opdaterer stadig historiske hold- og ligafeatures,
  men kan ikke blive til afregnede bets.

Den fulde modelkørsel dokumenteret længere nede er fortsat et frosset,
reproducerbart run på det tidligere dataset-id `722889f3145638357ee9` gennem
2024/25. Dens holdout-tal omskrives ikke bagudrettet, blot fordi den canonical
cache nu også indeholder 2025/26. En ny fuld walk-forward-kørsel skal have sit
eget run-id og manifest.

Dette datasæt indeholder ikke landsholds- eller VM-kampe, ægte xG, skader,
lineups, vejr eller bookmaker-limits. Historiske bookmakerfelter har også
skiftet format over tid. Resultaterne kan derfor ikke bruges som bevis for, at
samme odds og indsats kunne opnås live.

## Metode uden fremtidslæk

Alle features beregnes ved kampstart og bruger kun tidligere afsluttede kampe.
Kampe med samme starttid behandles som én batch, så de ikke kan se hinandens
resultater. Closing odds bruges kun til efterfølgende CLV-måling, aldrig som
modelinput. CLV beregnes kun, når opening og closing kan knyttes til samme
kilde-familie; en Pinnacle-opening må eksempelvis ikke sammenlignes med en
Bet365-closing.

For hver ydre testsæson `S` bruges denne tidslinje:

1. Modellerne trænes til og med `S-2`.
2. Første halvdel af `S-1` kalibrerer sandsynlighederne.
3. Anden halvdel af `S-1` vælger model, side, minimum-edge, confidence og
   oddsinterval.
4. Den urørte sæson `S` måler resultatet med flat stake på 1 unit.

Strategy zoo'en sammenligner bookmaker-markedet, ligaprior, Elo, den simple
Poisson-baseline, en tidsvægtet og regulariseret Dixon–Coles-model, logistisk
regression, histogram gradient boosting og markedsblends. Rå og
temperaturkalibrerede sandsynligheder testes. Isotonic calibration er valgfri.
Der trækkes som standard 1 % fra oddsens profitdel for at simulere dårligere
eksekvering. Dixon–Coles fit'es kun på foldens træningsrækker, afviser
predictions ved eller før trænings-cutoff og giver ukendte hold en neutral
liga-baseline.

Artefaktets evalueringsantal er policy-slice-evalueringer: hver konkret
`StrategySpec` evalueres én gang i hver fold og hvert prisspor, hvor den
forekommer. Det er ikke et antal unikke, uafhængige strategier på tværs af
folds. Det endelige run indeholder disse eksakte summer fra `fold_results.json`:

| Marked og spor | Adaptiv selection | Fixed-policy udvikling | I alt |
|---|---:|---:|---:|
| 1X2, executable | 748.800 | 144.000 | 892.800 |
| 1X2, proxy upper bound | 748.800 | 144.000 | 892.800 |
| O/U 2,5, executable | 374.400 | 72.000 | 446.400 |
| O/U 2,5, proxy upper bound | 561.600 | 108.000 | 669.600 |
| **Alle policy-slices** | **2.433.600** | **468.000** | **2.901.600** |

`selection.evaluated_strategy_specs` summeres over 13 ydre folds. Fixed-policy-
genberegningerne summeres kun over de ti udviklingsfolds 2012–2021; fra 2022
er policyen allerede låst. Tallene må ikke omtales som 2,9 millioner unikke
strategier.

Der findes to pris-spor:

- `executable`: komplette, sammenhængende priser fra en bookmakerkilde, som
  potentielt kunne være valgt før kampstart.
- `proxy_upper_bound`: gennemsnits- eller maksimumspriser. De viser kun et
  optimistisk loft og kan aldrig promoveres.

Den almindelige ydre test må gerne vælge en ny policy i hver sæson og er derfor
diagnostisk. Den strengeste test samler kun udviklingsresultater fra 2012–2021,
låser én uændret bettingregel før 2022-sæsonen og evaluerer den på de tre
urørte holdout-sæsoner 2022–2024. Holdout-perioden påvirker hverken policyvalg,
side, edge-grænse, confidence-grænse eller oddsinterval. Selve
prognosemodellen genfit'es fortsat sekventielt i hver fold på kun historiske
data; testen er derfor en låst bettingpolicy, ikke en frossen modelbinær.

## Promotion-gates

En fast policy kan kun låses fra udviklingsperioden, hvis den har mindst 300
bets, fem sæsoner, positiv ROI og mindst 55 % profitable sæsoner. Blandt de
egnede kandidater prioriteres den højeste konservative 90 % ROI-grænse.

Den låste `executable`-policy skal derefter opfylde alle disse krav på den urørte
holdout-periode:

- mindst 300 bets og positiv ROI;
- 95 % block-bootstrap interval med nedre ROI-grænse over nul;
- mindst 95 % estimeret sandsynlighed for positiv ROI;
- mindst tre holdout-sæsoner med bets og mindst 60 % profitable sæsoner;
- mindst 100 same-source closing-observationer, closing-dækning på mindst 50 %
  af alle holdout-bets og positiv gennemsnitlig CLV.

Selv et bestået historisk resultat giver kun status
`PROMOTABLE_TO_SHADOW`. Det er ikke en automatisk betting-anbefaling.
Registry-laget genberegner gates og afviser ugyldige strategier, proxy/max-odds
og manipuleret evidens. Det har med vilje ingen live-aktiveringsfunktion.

## Asian Handicap som negativ kontrol

Den fulde blinde sanitytest afregner whole-, half- og quarter-lines korrekt,
inklusive push, half-win og half-loss. Den vælger ingen strategi og er derfor
ikke en champion-test. På det samme komplette 1993–2024-datasæt taber blind
betting på begge sider ved alle testede priskilder:

| Kilde | Spor | Bets pr. side | Hjemme-ROI | Ude-ROI |
|---|---|---:|---:|---:|
| Bet365 | executable | 26.239 | -3,87 % | -1,37 % |
| Pinnacle | executable | 21.199 | -3,54 % | -0,98 % |
| Markedsgennemsnit | proxy | 70.330 | -3,55 % | -3,74 % |
| Markedsmaksimum | proxy | 70.329 | -0,92 % | -1,01 % |

Artefaktet ligger i
[`data/research/runs/asian_handicap_blind_canonical_1993_2024.json`](../data/research/runs/asian_handicap_blind_canonical_1993_2024.json).
Resultatet er en negativ kontrol: en fremtidig AH-model skal dokumentere en
stabil fordel ud over bookmaker-marginen, ikke blot slå en tilfældig
classifier. Gennemsnits- og maksimumspriserne er ikke eksekverbare live.

## Aktuelt ærligt resultat

Den auditerede fulde kørsel
[`20260714T183718Z_722889f3145638357ee9_full`](../data/research/runs/20260714T183718Z_722889f3145638357ee9_full/report.md)
brugte 1 % odds-haircut, 2.000 block-bootstrap-resamples og kun komplette
sæsoner. Registry-resultatet er **`NO_PROMOTION`**; ingen markeder blev
registreret og automatisk live-aktivering er `false`.

| Marked og test | Bets | Hit-rate | Profit | ROI | 95 % ROI-interval | Profitable sæsoner | Beslutning |
|---|---:|---:|---:|---:|---:|---:|---|
| 1X2, adaptiv executable | 1.659 | 42,4 % | -118,7 u | -7,15 % | -12,25 % til -2,30 % | 2/13 | Afvist |
| 1X2, adaptiv proxy | 2.011 | 38,2 % | -112,4 u | -5,59 % | -10,98 % til -0,08 % | 5/13 | Proxy; afvist |
| 1X2, låst executable 2022–2024 | 249 | 62,2 % | +1,9 u | +0,77 % | -9,24 % til +10,63 % | 2/3 | Afvist |
| 1X2, låst proxy 2022–2024 | 29 | 31,0 % | +1,0 u | +3,30 % | -45,04 % til +48,51 % | 1/3 | Proxy; afvist |
| O/U 2,5, adaptiv executable | 663 | 56,6 % | +10,2 u | +1,54 % | -5,20 % til +7,58 % | 3/5 | Afvist |
| O/U 2,5, adaptiv proxy | 2.288 | 52,1 % | -35,4 u | -1,55 % | -5,68 % til +2,91 % | 6/13 | Proxy; afvist |
| O/U 2,5, låst executable 2022–2024 | 0 | — | 0,0 u | 0,00 % | Ingen bets | 0/0 | Ingen kvalificeret policy |
| O/U 2,5, låst proxy 2022–2024 | 109 | 59,6 % | +11,1 u | +10,22 % | -3,72 % til +24,19 % | 2/3 | Proxy; afvist |

Den låste 1X2-policy er den klareste læring. Hjemmehold med rå
markedssandsynlighed på mindst 55 % og primære odds 1,50–2,50 gav +6,35 % ROI
på 419 udviklingsbets i 2012–2021. Den samme bettingregel gav kun +0,77 % på
249 urørte bets i 2022–2024. Den klarede hverken minimum 300 bets,
usikkerhedsgrænsen, 95 % sandsynlighed for positiv ROI eller CLV-kravene.

Det adaptive O/U-resultat er positivt, men er ikke evidens for en profitabel
live-strategi. Intervallet krydser nul, sandsynligheden for positiv ROI er kun
66,85 %, og same-source closing-dækningen er 39,5 % mod kravet på 50 %. Den
gennemsnitlige same-source CLV er +0,59 % på 262 bets, men kun 47,7 % af dem har
positiv CLV. Ingen fast eksekverbar O/U-policy bestod udviklingskravene. Den
bedste diagnostiske udviklingsregel faldt til -14,10 % ROI på 98 holdout-bets.

### Metodiske forbehold

- Football-Data har ikke et præcist tidsstempel for hver historisk oddsquote.
  En gemt pris beviser derfor ikke, at samme pris og limit kunne opnås live.
- En 1 % odds-haircut er en stresstest, ikke en fuld model for limits, slippage,
  afviste indsatser, likviditet eller bookmakerbegrænsninger.
- Block-bootstrap bevarer lokal kronologi, men kan undervurdere korrelation
  mellem samtidige kampe og ligaer. Konfidensintervallerne er ikke garantier.
- Den faste test låser bettingreglen, mens prognosemodellen genfit'es
  sekventielt. Resultatet må ikke beskrives som validering af én uændret
  modelbinær.
- CLV-gates fejler lukket, når der er under 100 sammenlignelige same-source
  closes eller under 50 % dækning. Det forhindrer, at manglende data tolkes som
  positiv evidens.
- Millioner af policy-slice-evalueringer skaber en stor multiple-testing-
  byrde. Nested walk-forward og den låste holdout reducerer risikoen for
  overtilpasning, men fjerner den ikke.
- Datasættet indeholder ikke landsholds- eller VM-kampe, ægte xG, skader,
  lineups, vejr eller bookmaker-limits. Landsholdsmodellen er et separat
  forecast-only spor uden ROI-påstand.

## Fagligt grundlag

- [Football-Data: historiske resultater og odds](https://www.football-data.co.uk/data.php)
- [Dixon og Coles: dynamisk Poisson-model til fodbold](https://doi.org/10.1111/1467-9876.00065)
- [Evaluering af fodboldprognoser på økonomisk værdi](https://doi.org/10.1016/j.ijforecast.2003.12.007)
- [Publiceret O/U-modellering med profit som mål](https://doi.org/10.1016/j.ijforecast.2019.02.008)
- [Scikit-learn: tidsserie-validering](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Scikit-learn: sandsynlighedskalibrering](https://scikit-learn.org/stable/modules/calibration.html)
- [Hvorfor kalibrering skal måles separat fra accuracy](https://arxiv.org/abs/2303.06021)

ROI er historisk profit divideret med historisk indsats. Ingen backtest kan
garantere fremtidigt afkast. Brug kun resultaterne til forskning og ansvarlig
paper tracking.
