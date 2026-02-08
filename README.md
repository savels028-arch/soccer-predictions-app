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

### 3. (Valgfrit) Sæt API nøgler

```bash
# Football-Data.org (gratis: https://www.football-data.org/client/register)
export FOOTBALL_DATA_API_KEY="din-nøgle-her"

# API-Football (gratis: https://www.api-football.com/)
export API_FOOTBALL_KEY="din-nøgle-her"
```

> **Note:** Appen fungerer uden API nøgler med realistiske demo-data!

### 4. Start appen

```bash
python main.py
```

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
