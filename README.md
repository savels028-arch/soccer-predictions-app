# ⚽ Soccer Predictions Pro

> AI-powered fodbold-forudsigelser med Machine Learning - Desktop App

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-XGBoost%20|%20Neural%20Net%20|%20Ensemble-orange.svg)

## 🎯 Features

| Feature | Beskrivelse |
|---------|-------------|
| 📊 **Dashboard** | Oversigt over dagens kampe fra alle top-ligaer |
| 🎯 **AI Predictions** | ML predictions fra multiple modeller |
| 📈 **Sammenligning** | Sammenlign XGBoost, Neural Net, Random Forest og Ensemble |
| 🔴 **Live Scores** | Real-time kampresultater med auto-refresh |
| 💡 **Forslag** | Intelligente value-bet forslag baseret på sandsynlighed |
| 🧠 **ML Training** | Træn modeller på historisk data direkte i appen |

## 🏟️ Understøttede Ligaer

- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
- 🇪🇸 La Liga
- 🇩🇪 Bundesliga
- 🇮🇹 Serie A
- 🇫🇷 Ligue 1
- 🏆 Champions League
- 🏆 Europa League
- 🇳🇱 Eredivisie
- 🇵🇹 Primeira Liga
- 🇧🇷 Série A (Brasilien)

## 🚀 Hurtig Start

### 1. Klon repository

```bash
git clone https://github.com/dit-username/soccer-predictions-app.git
cd soccer-predictions-app
```

### 2. Installér dependencies

```bash
# Opret virtual environment (anbefalet)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Installér packages
pip install -r requirements.txt
```

### 3. Start appen

```bash
python main.py
```

> **✅ Ingen API nøgler nødvendige!** Appen bruger gratis, åbne API'er (ESPN, TheSportsDB, OpenLigaDB) som ikke kræver registrering. Du får ægte kampdata fra alle top-ligaer ud af boksen!

### (Valgfrit) Ekstra API nøgler

Hvis du vil have endnu mere data, kan du tilføje valgfrie API nøgler:

```bash
# Football-Data.org (valgfrit)
export FOOTBALL_DATA_API_KEY="din-nøgle-her"

# API-Football (valgfrit)
export API_FOOTBALL_KEY="din-nøgle-her"
```

`API_FOOTBALL_KEY` aktiverer pipeline-data for lineups, skader/suspensioner,
spiller-rating, fixture-statistik og xG når API'en leverer det. Pipeline gemmer
også odds snapshots og pick snapshots, så vi kan måle closing-line value i stedet
for kun hitrate.

## 💻 Brug

### GUI Mode (standard)

```bash
python main.py              # Start med GUI
python main.py --train      # Træn modeller og start GUI
```

### CLI Mode

```bash
python main.py --no-gui             # Kør predictions i terminal
python main.py --no-gui --train     # Træn + predictions i terminal
```

### Live prediction pipeline

```bash
python run_pipeline.py                    # Full pipeline
python run_pipeline.py --odds-only        # Gem odds movement snapshots
python run_pipeline.py --context-only     # Hent lineups/skader/spillerdata
python run_pipeline.py --watch            # Kør løbende hvert 15. minut
```

### Privat lokal drift uden Vercel/GitHub Actions

Hvis siden kun skal bruges privat, kan din Mac selv opdatere data og du kan åbne
siden lokalt.

```bash
# Start hjemmesiden lokalt
scripts/start-local-site.sh

# Kør en fuld dataopdatering manuelt
scripts/run-local-pipeline.sh

# Installer automatisk lokal opdatering via macOS launchd
scripts/install-local-scheduler.sh
```

Hvis projektet ligger under `~/Desktop`, kan macOS blokere `launchd` med
`Operation not permitted`. Brug i stedet en driftskopi uden for Desktop:

```bash
mkdir -p ~/AIBets
rsync -a --exclude .git --exclude deploy/node_modules --exclude deploy/.next \
  ./ ~/AIBets/soccer-predictions-app/
cd ~/AIBets/soccer-predictions-app
scripts/install-local-scheduler.sh
```

Den lokale scheduler kører når Mac'en er vågen:

- daglig full pipeline kl. 08:30
- odds snapshot kl. 12:00
- resultatevaluering kl. 23:15
- retraining mandag kl. 07:00

Stop scheduler igen med:

```bash
scripts/uninstall-local-scheduler.sh
```

## 🔬 Reproducerbar strategi-research

Det isolerede research-lag tester 1X2 og over/under 2,5 med point-in-time
features, nested walk-forward, en låst tre-sæsoners holdout og faste
promotion-gates. Det ændrer aldrig live-predictions automatisk. Se
[metode, CLI, datadækning og aktuelle resultater](research/README.md).

### Med API nøgler

```bash
python main.py --fd-key "DIN_NØGLE" --af-key "DIN_NØGLE"
```

## 🧠 ML Modeller

### XGBoost
- Gradient boosting med 200 estimators
- Optimeret til multi-class classification
- Typisk accuracy: 45-55%

### Neural Network
- 3-layer neural network (128→64→32)
- Dropout + BatchNorm for regularisering
- Fallback til sklearn MLP hvis TensorFlow ikke er installeret

### Random Forest
- 300 decision trees med max_depth=10
- Robust og svær at overfitte
- God til feature importance analyse

### Poisson Model
- Statistisk model baseret på forventede mål
- Bruger team attack/defense styrke
- Uafhængig af ML-træning

### Ensemble
- Vægtet kombination af alle modeller
- Weights: XGBoost 40%, Neural Net 35%, Random Forest 25%
- Typisk den mest præcise

## 📊 Features brugt til predictions

| Feature | Beskrivelse |
|---------|-------------|
| Win/Draw/Loss % | Holdets overordnede resultater |
| Hjemme/Ude specifikke stats | Performance hjemme vs. ude |
| Mål gennemsnit | Scorede og indkasserede mål per kamp |
| Form | Seneste 5 kampes resultater |
| Head-to-Head | Historiske indbyrdes kampe |
| Odds | Bookmaker odds konverteret til sandsynligheder |
| Clean sheets | Procentdel af kampe uden mål imod |
| Points per game | Gennemsnitlige point per kamp |

## 📁 Projektstruktur

```
soccer-predictions-app/
├── main.py                      # 🚀 App entry point
├── requirements.txt             # 📦 Dependencies
├── config/
│   ├── __init__.py
│   └── settings.py              # ⚙️ App konfiguration
├── research/                    # 🔬 Leakage-free strategi-research og CLI
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── football_data_client.py   # 🌐 Football-Data.org API
│   │   ├── api_football_client.py    # 🌐 API-Football client
│   │   └── data_aggregator.py        # 📊 Data sammenlægning
│   ├── database/
│   │   ├── __init__.py
│   │   └── db_manager.py            # 💾 SQLite database
│   ├── predictions/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py    # 🔧 Feature engineering
│   │   ├── models.py                 # 🧠 ML modeller
│   │   └── prediction_engine.py      # 🎯 Prediction orchestration
│   └── gui/
│       ├── __init__.py
│       ├── widgets.py                # 🎨 Custom widgets
│       └── app_window.py             # 🖥️ Main GUI window
├── data/
│   ├── db/                           # 💾 SQLite database filer
│   ├── models/                       # 🧠 Gemte ML modeller
│   ├── research/                     # 🔬 Genererede research-runs og feature-cache
│   └── cache/                        # 📦 API cache
└── logs/                             # 📋 Log filer
```

## ⚠️ Disclaimer

> Denne app er kun til **underholdning og uddannelsesformål**.
> Forudsigelser er baseret på statistiske modeller og garanterer ikke resultater.
> Gambling kan være vanedannende - spil ansvarligt.

## 📜 License

MIT License - Se [LICENSE](LICENSE) filen.

## 🤝 Bidrag

Pull requests er velkomne! For større ændringer, åbn venligst et issue først.

---

Lavet med ❤️ og Python 🐍
