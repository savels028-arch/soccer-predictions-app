#!/usr/bin/env python3
"""
⚽ Soccer Predictions Pro - Desktop Application
Main entry point. Launch the app from here.

Usage:
    python main.py
    python main.py --train       # Train models on startup
    python main.py --no-gui      # Run predictions without GUI
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOG_SETTINGS, FOOTBALL_DATA_API_KEY, API_FOOTBALL_KEY
from src.database.db_manager import DatabaseManager
from src.api.football_data_client import FootballDataClient
from src.api.api_football_client import ApiFootballClient
from src.api.data_aggregator import DataAggregator
from src.predictions.prediction_engine import PredictionEngine


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, LOG_SETTINGS["level"]),
        format=LOG_SETTINGS["format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_SETTINGS["file"]), encoding="utf-8"),
        ]
    )


def create_app_components():
    """Initialize all application components."""
    logger = logging.getLogger(__name__)

    # Database
    logger.info("Initializing database...")
    db = DatabaseManager()

    # API Clients
    fd_client = None
    af_client = None

    if FOOTBALL_DATA_API_KEY:
        fd_client = FootballDataClient(FOOTBALL_DATA_API_KEY)
        logger.info("Football-Data.org client initialized")
    else:
        logger.info("No Football-Data.org API key set - using demo data")

    if API_FOOTBALL_KEY:
        af_client = ApiFootballClient(API_FOOTBALL_KEY)
        logger.info("API-Football client initialized")
    else:
        logger.info("No API-Football key set - using demo data")

    # Data Aggregator
    data_aggregator = DataAggregator(db, fd_client, af_client)

    # Prediction Engine
    prediction_engine = PredictionEngine(db, data_aggregator)

    return db, data_aggregator, prediction_engine


def run_gui(db, data_aggregator, prediction_engine):
    """Launch the GUI application."""
    from src.gui.app_window import SoccerPredictionsApp

    app = SoccerPredictionsApp(db, data_aggregator, prediction_engine)
    app.run()


def run_cli(db, data_aggregator, prediction_engine, train: bool = False):
    """Run predictions in CLI mode."""
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 60)
    print("⚽ Soccer Predictions Pro - CLI Mode")
    print("=" * 60)

    # Fetch matches
    print("\n📊 Fetching today's matches...")
    matches = data_aggregator.fetch_todays_matches()
    print(f"   Found {len(matches)} matches")

    # Train if requested
    if train:
        print("\n🏋️  Training ML models...")
        results = prediction_engine.train_models(
            callback=lambda t, d: print(f"   [{t}] {d}")
        )
        print(f"\n📊 Training results:")
        for name, acc in results.items():
            print(f"   {name}: {acc:.1%}")

    # Generate predictions
    print("\n🎯 Generating predictions...")
    for match in matches[:10]:
        home = match.get("home_team_name", "?")
        away = match.get("away_team_name", "?")
        status = match.get("status", "SCHEDULED")

        predictions = prediction_engine.predict_match(match)
        ensemble = next((p for p in predictions if p.get("model_name") == "ensemble"), None)

        print(f"\n   ⚽ {home} vs {away} [{status}]")
        if ensemble:
            print(f"      Prediction: {ensemble['predicted_outcome']} "
                  f"({ensemble['confidence']:.0%})")
            print(f"      Home: {ensemble['home_win_prob']:.0%} | "
                  f"Draw: {ensemble['draw_prob']:.0%} | "
                  f"Away: {ensemble['away_win_prob']:.0%}")
            if ensemble.get("value_rating", 0) > 0:
                print(f"      💎 Value: {ensemble['value_rating']:.2f}")
            if ensemble.get("suggestion"):
                print(f"      💡 {ensemble['suggestion']}")

    print("\n" + "=" * 60)
    print(f"📦 Database: {db.get_match_count()} matches, {db.get_prediction_count()} predictions")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="⚽ Soccer Predictions Pro - AI-powered football predictions"
    )
    parser.add_argument("--no-gui", action="store_true",
                       help="Run in CLI mode without GUI")
    parser.add_argument("--train", action="store_true",
                       help="Train ML models on startup")
    parser.add_argument("--fd-key", type=str, default="",
                       help="Football-Data.org API key")
    parser.add_argument("--af-key", type=str, default="",
                       help="API-Football API key")

    args = parser.parse_args()

    # Set API keys from arguments
    if args.fd_key:
        os.environ["FOOTBALL_DATA_API_KEY"] = args.fd_key
    if args.af_key:
        os.environ["API_FOOTBALL_KEY"] = args.af_key

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("⚽ Soccer Predictions Pro Starting...")
    logger.info("=" * 50)

    # Initialize components
    db, data_aggregator, prediction_engine = create_app_components()

    try:
        if args.no_gui:
            run_cli(db, data_aggregator, prediction_engine, train=args.train)
        else:
            if args.train:
                print("🏋️  Training models before launching GUI...")
                prediction_engine.train_models()
            run_gui(db, data_aggregator, prediction_engine)
    except KeyboardInterrupt:
        print("\n👋 Afslutter...")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
