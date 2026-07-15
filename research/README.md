# AIBets strategy-research

Den hjemlige strategy zoo er adskilt fra den forecast-only landsholdsmodel.
VM-modellens dataproveniens, validering og begrænsninger er dokumenteret i
[`data/international/README.md`](../data/international/README.md) og den frosne
[`validation_report.md`](../data/international/validation_report.md). En
separat, checksum-pinnet VM-workbook giver navngivne historiske 90-minutters
1X2-odds for 2014, 2018, 2022 og den foreløbige 2026-turnering. De tal er
bagudskuende evidens, aldrig en valideret fremadrettet edge. EM har fortsat
ingen verificeret oddsserie og viser derfor ingen ROI.

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
46 på forhånd definerede H2H-, 1X2-, uafgjort-, mål- og exact-score-regler år
for år. Før strategierne vises et rent sæsongrundkort med hjemme/uafgjort/ude,
Over/Under 0,5–5,5, eksakte måltotaler og holdfavoritternes W/D/L. Artifactet
måler også hvad der gentager sig på tværs af sæsoner, og adskiller en vinder
valgt efter sæsonen fra et walk-forward-valg baseret kun på tidligere sæsoner.
Reglerne bliver ikke genvalgt ud fra samme års resultat. Kampe med samme starttid
isoleres fra hinanden, og kun komplette Bet365-opening-markeder kan skabe P&L.
Manglende odds giver derfor aldrig syntetisk profit.

Det aktuelle artifact dækker 115.460 evaluerede kampe fra 1993/94 til og med
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

### Edge Atlas og EM/VM

Edge Atlas viser grundkort, ROI og usikkerhed for hver liga og sæson. Det
adskiller konsekvent: hvad der skete; hvilken fast strategi der ser bedst ud
med facit i hånden; og hvad en regel valgt udelukkende på tidligere sæsoner
faktisk leverede. EM- og VM-fanerne holder klubdata, kvalifikation og
slutrunder adskilt.

Begge offentlige atlasfiler bygges og kopieres til deploy-repositoryet med én
fail-closed kommando:

```bash
venv/bin/python scripts/publish_research_atlases.py \
  --world-cup-xlsx /tmp/WorldCup2026.xlsx
```

Kommandoen kræver den fulde canonical klubcache, det manifesterede
landsholdssnapshot og den reviewede VM-workbook med SHA256
`b14a24e218f25ffaa0037718471187bced6b835e2293b6e94cb2ce2a76ad544b`.
Alle source-checksums valideres før deploy-assets overskrives. VM-workbooken
indeholder 64 kampe i hver af 2014, 2018 og 2022 samt 100 af 104 kampe i 2026
gennem 12. juli; 2026-resultater er derfor tydeligt markeret foreløbige.

Hver kørsel skriver et manifest med SHA256 for datasæt, feature-cache og hver
research-kildefil, samlet code-fingerprint samt Git dirty/untracked-status.
Feature-cache-nøglen afledes af datasæt, parser/feature-kode og den eksplicitte
feature-konfiguration, så ændret logik ikke kan genbruge en gammel cache. Der
skrives desuden fold-resultater, alle bets, en opsummering, promotion-gates, den
låste strategi, en registry-audit og en læsbar rapport til
`data/research/runs/<run-id>/`. Feature-cache og runs er genererede artefakter og
er derfor ikke versionsstyret.

## Datasæt og begrænsninger

Den aktuelle canonical cache bruger canonical dataset-id `b182289f8a9733b01da6`
fra 328 lokale Football-Data CSV-filer:

- 115.460 valide, unikke klubkampe fra 23. juli 1993 til 24. maj 2026.
- 380 rækker er eksplicit afvist, fordi `Div`-feltet modsiger ligaen i
  filnavnet. De stammer fra en historisk Portugal-URL, der leverer den spanske
  1993/94-fil, og må derfor ikke tælles som Primeira Liga.
- Dataset-id'et hashes de normaliserede rækker, som analysen faktisk bruger;
  den separate råfil-identitet er `7e131786cd7a35f0ae84`. En parserrettelse
  kan derfor ikke længere genbruge samme dataset-id som et forurenet run.
- Ti turneringer: Premier League, Championship, Bundesliga, 2. Bundesliga,
  La Liga, Serie A, Ligue 1, Eredivisie, Primeira Liga og belgisk 1. division.
- Den offentlige coverage-kontrakt kræver præcis 327 valide liga-sæson-par.
  De eneste undtagelser er Belgien 1993/94 og 1994/95, som Football-Data ikke
  udgiver, samt Primeira Liga 1993/94, hvor den publicerede P1-fil i stedet
  indeholder de 380 spanske SP1-rækker, som vi afviser.
- Åbningsodds findes for 91.451 1X2-kampe (79,2 %), 78.753 O/U 2,5-kampe
  (68,2 %) og 78.907 Asian Handicap-kampe (68,3 %). Closing odds findes for
  49.364 1X2-kampe og 24.760 O/U-kampe.
- Kampe uden komplette odds opdaterer stadig historiske hold- og ligafeatures,
  men kan ikke blive til afregnede bets.

Den fulde modelkørsel dokumenteret længere nede er et frosset, reproducerbart
run på det korrigerede dataset-id `3528b6fd6613e6e7a94c`: 111.927 kampe gennem
2024/25. 2025/26 holdes uden for modelvalget og fungerer som næste
fremadrettede periode; run-id og manifest ændres ved hver ny kørsel.

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
[`20260715T133227Z_3528b6fd6613e6e7a94c_full`](../data/research/runs/20260715T133227Z_3528b6fd6613e6e7a94c_full/report.md)
brugte 1 % odds-haircut, 2.000 block-bootstrap-resamples og kun komplette
sæsoner. Registry-resultatet er **`NO_PROMOTION`**; ingen markeder blev
registreret og automatisk live-aktivering er `false`.

| Marked og test | Bets | Hit-rate | Profit | ROI | 95 % ROI-interval | Profitable sæsoner | Beslutning |
|---|---:|---:|---:|---:|---:|---:|---|
| 1X2, adaptiv executable | 1.897 | 45,4 % | -115,7 u | -6,10 % | -9,92 % til -2,37 % | 2/13 | Afvist |
| 1X2, adaptiv proxy | 1.830 | 40,5 % | -29,7 u | -1,62 % | -7,74 % til +4,28 % | 8/13 | Proxy; afvist |
| 1X2, låst executable 2022–2024 | 249 | 62,2 % | +1,9 u | +0,77 % | -9,24 % til +10,63 % | 2/3 | Afvist |
| 1X2, låst proxy 2022–2024 | 29 | 31,0 % | +1,0 u | +3,30 % | -45,04 % til +48,51 % | 1/3 | Proxy; afvist |
| O/U 2,5, adaptiv executable | 662 | 56,3 % | +7,4 u | +1,11 % | -5,80 % til +7,63 % | 3/5 | Afvist |
| O/U 2,5, adaptiv proxy | 2.149 | 51,5 % | -33,7 u | -1,57 % | -5,97 % til +2,71 % | 4/13 | Proxy; afvist |
| O/U 2,5, låst executable 2022–2024 | 0 | — | 0,0 u | 0,00 % | Ingen bets | 0/0 | Ingen kvalificeret policy |
| O/U 2,5, låst proxy 2022–2024 | 11 | 45,5 % | -1,1 u | -10,27 % | -61,03 % til +35,57 % | 1/3 | Proxy; afvist |

Den låste 1X2-policy er den klareste læring. Hjemmehold med rå
markedssandsynlighed på mindst 55 % og primære odds 1,50–2,50 gav +6,35 % ROI
på 419 udviklingsbets i 2012–2021. Den samme bettingregel gav kun +0,77 % på
249 urørte bets i 2022–2024. Den klarede hverken minimum 300 bets,
usikkerhedsgrænsen, 95 % sandsynlighed for positiv ROI eller CLV-kravene.

Det adaptive O/U-resultat er positivt, men er ikke evidens for en profitabel
live-strategi. Intervallet krydser nul, sandsynligheden for positiv ROI er kun
61,75 %, og same-source closing-dækningen er 39,3 % mod kravet på 50 %. Den
gennemsnitlige same-source CLV er +0,59 % på 260 bets, men kun 47,3 % af dem har
positiv CLV. Ingen fast eksekverbar O/U-policy bestod udviklingskravene. Den
bedste diagnostiske udviklingsregel faldt til -14,10 % ROI på 98 holdout-bets.

En særskilt audit af 36 faste COD/opposite-varianter fandt én watchlist-regel,
som var positiv i alle tre tidsblokke: i La Liga, spil hjemmeholdet når
udeholdets Wheatcroft-COD er mindst 0,875. Med Bet365-priser gav den +14,45 %
på 530 udviklingsbets (2005–2017), +2,95 % på 191 valideringsbets (2018–2024)
og +50,34 % på 35 bets i 2025/26. Det er stadig **ikke en bekræftet edge**:
valideringsintervallet er -32,58 % til +38,49 %, 2025/26-intervallet er
-10,27 % til +110,96 %, og udviklingsfundet består ikke korrektionen for de
mange testede regler. Reglen er derfor tilføjet som synlig, årlig
opposite-COD-strategi i Edge Atlas, men forbliver i paper-tracking.

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
- Klubdatasættet indeholder ikke landshold, ægte xG, skader, lineups, vejr eller
  bookmaker-limits. Landshold er et separat forecast-spor; VM-workbookens
  historiske ROI er kun hindsight, og EM-ROI fejler lukket uden oddsdata.

## Fagligt grundlag

- [Football-Data: historiske resultater og odds](https://www.football-data.co.uk/data.php)
- [Dixon og Coles: dynamisk Poisson-model til fodbold](https://doi.org/10.1111/1467-9876.00065)
- [Evaluering af fodboldprognoser på økonomisk værdi](https://doi.org/10.1016/j.ijforecast.2003.12.007)
- [Publiceret O/U-modellering med profit som mål](https://doi.org/10.1016/j.ijforecast.2019.02.008)
- [Wheatcroft: chance-of-defeat og maksimumsodds](https://doi.org/10.1515/jqas-2019-0009)
- [Angelini og De Angelis: oddsaggregation og markedseffektivitet](https://doi.org/10.1016/j.ijforecast.2018.07.008)
- [White: Reality Check for multiple testing](https://doi.org/10.1111/1468-0262.00152)
- [Hansen: Superior Predictive Ability-test](https://doi.org/10.1198/073500105000000063)
- [Scikit-learn: tidsserie-validering](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Scikit-learn: sandsynlighedskalibrering](https://scikit-learn.org/stable/modules/calibration.html)
- [Hvorfor kalibrering skal måles separat fra accuracy](https://arxiv.org/abs/2303.06021)

ROI er historisk profit divideret med historisk indsats. Ingen backtest kan
garantere fremtidigt afkast. Brug kun resultaterne til forskning og ansvarlig
paper tracking.
