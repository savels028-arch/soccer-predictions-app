"""
Soccer Predictions Desktop App - Main Window
Full-featured dark-themed GUI with navigation sidebar.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import logging
import threading
from datetime import datetime, date
from typing import Dict, List, Optional

from .widgets import (
    StyledFrame, CardFrame, StyledLabel, HeaderLabel, SubHeaderLabel,
    MutedLabel, AccentButton, SecondaryButton, NavButton,
    ProbabilityBar, ConfidenceMeter, ScrollableFrame, StatusBar,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import GUI_SETTINGS, LEAGUES

logger = logging.getLogger(__name__)
C = GUI_SETTINGS["colors"]


class SoccerPredictionsApp:
    """Main application window."""

    def __init__(self, db_manager, data_aggregator, prediction_engine):
        self.db = db_manager
        self.data = data_aggregator
        self.engine = prediction_engine

        # Create main window
        self.root = tk.Tk()
        self.root.title(GUI_SETTINGS["title"])
        self.root.geometry(f"{GUI_SETTINGS['width']}x{GUI_SETTINGS['height']}")
        self.root.minsize(GUI_SETTINGS["min_width"], GUI_SETTINGS["min_height"])
        self.root.configure(bg=C["bg_dark"])

        # Try to set icon (optional)
        try:
            self.root.iconbitmap("")
        except:
            pass

        # State
        self.current_page = "dashboard"
        self.nav_buttons = {}
        self.matches_cache = []
        self.predictions_cache = {}
        self.auto_refresh_id = None

        # Build UI
        self._build_ui()

        # Load initial data
        self.root.after(500, self._initial_load)

    def _build_ui(self):
        """Build the complete UI layout."""
        # Main container
        self.main_container = StyledFrame(self.root)
        self.main_container.pack(fill="both", expand=True)

        # ── Sidebar ──
        self._build_sidebar()

        # ── Content Area ──
        self.content_frame = StyledFrame(self.main_container)
        self.content_frame.pack(side="left", fill="both", expand=True)

        # ── Status Bar ──
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(side="bottom", fill="x")

        # ── Pages (stacked frames) ──
        self.pages = {}
        self._build_dashboard_page()
        self._build_predictions_page()
        self._build_comparison_page()
        self._build_live_page()
        self._build_suggestions_page()
        self._build_settings_page()

        # Show dashboard by default
        self._show_page("dashboard")

    def _build_sidebar(self):
        """Build navigation sidebar."""
        sidebar = StyledFrame(self.main_container, bg=C["bg_medium"], width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Logo/Title
        logo_frame = StyledFrame(sidebar, bg=C["bg_medium"])
        logo_frame.pack(fill="x", pady=(20, 30))

        tk.Label(logo_frame, text="⚽", font=("Segoe UI", 28),
                bg=C["bg_medium"], fg=C["accent"]).pack()
        tk.Label(logo_frame, text="Soccer Predictions", font=("Segoe UI", 13, "bold"),
                bg=C["bg_medium"], fg=C["text_primary"]).pack()
        tk.Label(logo_frame, text="Pro Edition", font=("Segoe UI", 9),
                bg=C["bg_medium"], fg=C["accent"]).pack()

        # Separator
        tk.Frame(sidebar, height=1, bg=C["border"]).pack(fill="x", padx=15, pady=5)

        # Navigation buttons
        nav_items = [
            ("dashboard",   "📊  Dashboard",     "Oversigt & dagens kampe"),
            ("predictions", "🎯  AI Predictions", "ML model predictions"),
            ("comparison",  "📈  Sammenligning",  "Sammenlign modeller"),
            ("live",        "🔴  Live Scores",    "Live kampresultater"),
            ("suggestions", "💡  Forslag",        "Betting forslag"),
            ("settings",    "⚙️  Indstillinger",  "API keys & træning"),
        ]

        for page_id, text, tooltip in nav_items:
            btn = NavButton(sidebar, text=text,
                           command=lambda p=page_id: self._show_page(p))
            btn.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[page_id] = btn

        # Bottom info
        spacer = StyledFrame(sidebar, bg=C["bg_medium"])
        spacer.pack(fill="both", expand=True)

        tk.Frame(sidebar, height=1, bg=C["border"]).pack(fill="x", padx=15, pady=5)

        info_frame = StyledFrame(sidebar, bg=C["bg_medium"])
        info_frame.pack(fill="x", padx=15, pady=10)

        self.db_count_label = tk.Label(info_frame, text="📦 0 kampe i database",
                                        font=("Segoe UI", 8), bg=C["bg_medium"],
                                        fg=C["text_muted"])
        self.db_count_label.pack(anchor="w")

        self.model_status_label = tk.Label(info_frame, text="🤖 Modeller: Ikke trænet",
                                            font=("Segoe UI", 8), bg=C["bg_medium"],
                                            fg=C["text_muted"])
        self.model_status_label.pack(anchor="w")

        self.last_update_label = tk.Label(info_frame, text="🕐 Sidst opdateret: -",
                                           font=("Segoe UI", 8), bg=C["bg_medium"],
                                           fg=C["text_muted"])
        self.last_update_label.pack(anchor="w")

    def _show_page(self, page_id: str):
        """Switch to a page."""
        self.current_page = page_id

        # Update nav buttons
        for pid, btn in self.nav_buttons.items():
            btn.set_active(pid == page_id)

        # Hide all pages, show selected
        for pid, frame in self.pages.items():
            frame.pack_forget()

        if page_id in self.pages:
            self.pages[page_id].pack(in_=self.content_frame, fill="both", expand=True)

        self.status_bar.set_status(f"Side: {page_id.title()}")

    # ═══════════════════════════════════════════
    # DASHBOARD PAGE
    # ═══════════════════════════════════════════
    def _build_dashboard_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["dashboard"] = page

        # Scrollable content
        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header_frame = StyledFrame(container)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))

        HeaderLabel(header_frame, text="📊 Dashboard - Dagens Kampe").pack(side="left")
        self.dash_date_label = MutedLabel(header_frame,
                                           text=datetime.now().strftime("%A %d. %B %Y"))
        self.dash_date_label.pack(side="right", padx=10)

        # Refresh button
        btn_frame = StyledFrame(container)
        btn_frame.pack(fill="x", padx=20, pady=(0, 10))
        AccentButton(btn_frame, text="🔄 Opdater data",
                    command=self._refresh_dashboard).pack(side="left")
        SecondaryButton(btn_frame, text="📅 Kommende kampe",
                       command=self._load_upcoming).pack(side="left", padx=10)

        # Stats cards row
        stats_frame = StyledFrame(container)
        stats_frame.pack(fill="x", padx=20, pady=10)

        self.stat_cards = {}
        for stat_id, title, icon in [
            ("total_matches", "Kampe i dag", "⚽"),
            ("live_matches", "Live nu", "🔴"),
            ("predictions", "Predictions", "🎯"),
            ("value_bets", "Value bets", "💎"),
        ]:
            card = CardFrame(stats_frame)
            card.pack(side="left", fill="x", expand=True, padx=5)

            tk.Label(card, text=icon, font=("Segoe UI", 22),
                    bg=C["bg_card"], fg=C["accent"]).pack()
            tk.Label(card, text="0", font=("Segoe UI", 20, "bold"),
                    bg=C["bg_card"], fg=C["text_primary"],
                    ).pack()
            tk.Label(card, text=title, font=("Segoe UI", 9),
                    bg=C["bg_card"], fg=C["text_secondary"]).pack()
            self.stat_cards[stat_id] = card

        # Matches list
        self.matches_container = StyledFrame(container)
        self.matches_container.pack(fill="both", expand=True, padx=20, pady=10)

        self.dash_loading_label = StyledLabel(self.matches_container,
                                               text="⏳ Indlæser kampe...",
                                               font=("Segoe UI", 12))
        self.dash_loading_label.pack(pady=40)

    def _populate_dashboard(self, matches: List[Dict]):
        """Populate dashboard with match data."""
        # Clear loading
        for widget in self.matches_container.winfo_children():
            widget.destroy()

        if not matches:
            StyledLabel(self.matches_container,
                       text="Ingen kampe fundet for i dag",
                       font=("Segoe UI", 12)).pack(pady=40)
            return

        # Update stat cards
        live_count = sum(1 for m in matches if m.get("status") in ("IN_PLAY", "HALFTIME", "LIVE"))
        self._update_stat_card("total_matches", str(len(matches)))
        self._update_stat_card("live_matches", str(live_count))

        # Group by league
        by_league = {}
        for m in matches:
            lc = m.get("league_code", "Other")
            by_league.setdefault(lc, []).append(m)

        for league_code, league_matches in sorted(by_league.items()):
            league_info = LEAGUES.get(league_code, {})
            emoji = league_info.get("emoji", "🏟️")
            name = league_info.get("name", league_code)

            # League header
            league_header = StyledFrame(self.matches_container)
            league_header.pack(fill="x", pady=(15, 5))
            SubHeaderLabel(league_header,
                          text=f"{emoji} {name}").pack(side="left")

            # Match cards
            for match in league_matches:
                self._create_match_card(self.matches_container, match)

    def _create_match_card(self, parent, match: Dict):
        """Create a single match card."""
        card = CardFrame(parent)
        card.pack(fill="x", pady=3)

        # Main row
        main_row = StyledFrame(card, bg=C["bg_card"])
        main_row.pack(fill="x")

        # Time / Status
        status = match.get("status", "SCHEDULED")
        match_date_str = match.get("match_date", "")
        try:
            match_time = datetime.fromisoformat(match_date_str.replace("Z", "+00:00"))
            time_str = match_time.strftime("%H:%M")
        except:
            time_str = "--:--"

        if status in ("IN_PLAY", "LIVE", "1H", "2H"):
            elapsed = match.get("extra_data", {}).get("elapsed", "")
            status_text = f"🔴 LIVE {elapsed}'"
            status_color = C["accent_red"]
        elif status == "HALFTIME":
            status_text = "🟡 HALFTIME"
            status_color = C["accent_yellow"]
        elif status == "FINISHED":
            status_text = "✅ FINISHED"
            status_color = C["accent_green"]
        else:
            status_text = f"🕐 {time_str}"
            status_color = C["text_secondary"]

        tk.Label(main_row, text=status_text, font=("Segoe UI", 9, "bold"),
                bg=C["bg_card"], fg=status_color, width=14, anchor="w").pack(side="left")

        # Home team
        home = match.get("home_team_name", "Unknown")
        tk.Label(main_row, text=home, font=("Segoe UI", 11, "bold"),
                bg=C["bg_card"], fg=C["text_primary"],
                width=25, anchor="e").pack(side="left", padx=(10, 5))

        # Score
        hs = match.get("home_score")
        aws = match.get("away_score")
        if hs is not None and aws is not None:
            score_text = f" {hs} - {aws} "
            score_color = C["text_primary"]
        else:
            score_text = " vs "
            score_color = C["text_muted"]

        tk.Label(main_row, text=score_text, font=("Segoe UI", 13, "bold"),
                bg=C["bg_card"], fg=score_color).pack(side="left", padx=5)

        # Away team
        away = match.get("away_team_name", "Unknown")
        tk.Label(main_row, text=away, font=("Segoe UI", 11, "bold"),
                bg=C["bg_card"], fg=C["text_primary"],
                width=25, anchor="w").pack(side="left", padx=(5, 10))

        # Odds
        odds_frame = StyledFrame(main_row, bg=C["bg_card"])
        odds_frame.pack(side="right")

        for label, value in [("1", match.get("home_odds")),
                             ("X", match.get("draw_odds")),
                             ("2", match.get("away_odds"))]:
            if value:
                odds_text = f"{label}: {value:.2f}"
                tk.Label(odds_frame, text=odds_text, font=("Segoe UI", 8),
                        bg=C["bg_card"], fg=C["text_muted"],
                        padx=5).pack(side="left")

        # Predict button for scheduled matches
        if status == "SCHEDULED":
            predict_btn = SecondaryButton(main_row, text="🎯 Predict",
                                          command=lambda m=match: self._quick_predict(m))
            predict_btn.pack(side="right", padx=5)

    def _update_stat_card(self, stat_id: str, value: str):
        """Update a stat card value."""
        if stat_id in self.stat_cards:
            card = self.stat_cards[stat_id]
            children = card.winfo_children()
            if len(children) >= 2:
                children[1].config(text=value)

    # ═══════════════════════════════════════════
    # PREDICTIONS PAGE
    # ═══════════════════════════════════════════
    def _build_predictions_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["predictions"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header_frame = StyledFrame(container)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header_frame, text="🎯 AI Predictions").pack(side="left")

        # Controls
        ctrl_frame = StyledFrame(container)
        ctrl_frame.pack(fill="x", padx=20, pady=(0, 10))

        AccentButton(ctrl_frame, text="🧠 Generér predictions",
                    command=self._generate_predictions).pack(side="left")
        SecondaryButton(ctrl_frame, text="🏋️ Træn modeller",
                       command=self._start_training).pack(side="left", padx=10)

        self.pred_status_label = StyledLabel(ctrl_frame, text="",
                                              font=("Segoe UI", 10))
        self.pred_status_label.pack(side="left", padx=20)

        # Training progress
        self.training_frame = CardFrame(container)
        self.training_progress_label = tk.Label(self.training_frame, text="",
                                                  font=("Segoe UI", 10),
                                                  bg=C["bg_card"], fg=C["accent"])
        self.training_progress_label.pack(padx=10, pady=5)

        # Predictions container
        self.pred_container = StyledFrame(container)
        self.pred_container.pack(fill="both", expand=True, padx=20, pady=10)

        StyledLabel(self.pred_container,
                   text="Klik 'Generér predictions' for at starte",
                   font=("Segoe UI", 12)).pack(pady=40)

    def _populate_predictions(self, all_predictions: Dict[str, List[Dict]]):
        """Populate predictions page."""
        for widget in self.pred_container.winfo_children():
            widget.destroy()

        if not all_predictions:
            StyledLabel(self.pred_container,
                       text="Ingen predictions genereret endnu",
                       font=("Segoe UI", 12)).pack(pady=40)
            return

        pred_count = 0
        value_count = 0

        for match_key, predictions in all_predictions.items():
            if not predictions:
                continue

            # Match header card
            card = CardFrame(self.pred_container)
            card.pack(fill="x", pady=5)

            # Title
            tk.Label(card, text=f"⚽ {match_key}", font=("Segoe UI", 12, "bold"),
                    bg=C["bg_card"], fg=C["text_primary"], anchor="w").pack(fill="x")

            # Get consensus
            ensemble_pred = None
            for p in predictions:
                if p.get("model_name") == "ensemble":
                    ensemble_pred = p
                    break

            if ensemble_pred:
                # Main prediction row
                pred_row = StyledFrame(card, bg=C["bg_card"])
                pred_row.pack(fill="x", pady=(5, 0))

                outcome = ensemble_pred.get("predicted_outcome", "")
                confidence = ensemble_pred.get("confidence", 0)
                confidence_pct = f"{confidence:.0%}"

                outcome_colors = {
                    "Home Win": C["win_color"],
                    "Draw": C["draw_color"],
                    "Away Win": C["lose_color"],
                }

                tk.Label(pred_row, text=f"Prediction: {outcome}",
                        font=("Segoe UI", 11, "bold"), bg=C["bg_card"],
                        fg=outcome_colors.get(outcome, C["text_primary"])).pack(side="left")

                # Confidence meter
                ConfidenceMeter(pred_row, confidence=confidence,
                               size=45).pack(side="left", padx=15)

                # Probability bar
                ProbabilityBar(pred_row,
                              home_prob=ensemble_pred.get("home_win_prob", 0.33),
                              draw_prob=ensemble_pred.get("draw_prob", 0.33),
                              away_prob=ensemble_pred.get("away_win_prob", 0.34),
                              width=250).pack(side="left", padx=10)

                # Value rating
                value = ensemble_pred.get("value_rating", 0)
                if value > 0:
                    value_count += 1
                    value_color = C["high_value"] if value > 0.5 else C["medium_value"]
                    tk.Label(pred_row, text=f"💎 Value: {value:.2f}",
                            font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                            fg=value_color).pack(side="right")

            # Individual model predictions
            models_frame = StyledFrame(card, bg=C["bg_card"])
            models_frame.pack(fill="x", pady=(5, 0))

            for pred in predictions:
                model_name = pred.get("model_name", "")
                if model_name == "ensemble":
                    continue

                model_label = StyledFrame(models_frame, bg=C["bg_card"])
                model_label.pack(side="left", padx=8)

                tk.Label(model_label, text=f"📐 {model_name.replace('_', ' ').title()}",
                        font=("Segoe UI", 8), bg=C["bg_card"],
                        fg=C["text_muted"]).pack()
                tk.Label(model_label, text=pred.get("predicted_outcome", "?"),
                        font=("Segoe UI", 9, "bold"), bg=C["bg_card"],
                        fg=C["text_secondary"]).pack()

            # Suggestion
            suggestion = (ensemble_pred or predictions[0]).get("suggestion", "")
            if suggestion:
                tk.Label(card, text=f"💡 {suggestion}",
                        font=("Segoe UI", 9), bg=C["bg_card"],
                        fg=C["accent_yellow"], anchor="w", wraplength=800).pack(fill="x", pady=(5, 0))

            pred_count += 1

        self._update_stat_card("predictions", str(pred_count))
        self._update_stat_card("value_bets", str(value_count))

    # ═══════════════════════════════════════════
    # COMPARISON PAGE
    # ═══════════════════════════════════════════
    def _build_comparison_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["comparison"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        header_frame = StyledFrame(container)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header_frame, text="📈 Model Sammenligning").pack(side="left")

        AccentButton(header_frame, text="🔄 Opdater",
                    command=self._refresh_comparison).pack(side="right")

        self.comparison_container = StyledFrame(container)
        self.comparison_container.pack(fill="both", expand=True, padx=20, pady=10)

        StyledLabel(self.comparison_container,
                   text="Træn modellerne først for at se sammenligning",
                   font=("Segoe UI", 12)).pack(pady=40)

    def _populate_comparison(self):
        """Populate model comparison page."""
        for widget in self.comparison_container.winfo_children():
            widget.destroy()

        performances = self.engine.get_model_comparison()

        # Performance cards
        cards_frame = StyledFrame(self.comparison_container)
        cards_frame.pack(fill="x", pady=10)

        for perf in performances:
            card = CardFrame(cards_frame)
            card.pack(side="left", fill="x", expand=True, padx=5)

            name = perf.get("model_name", "Unknown").replace("_", " ").title()
            accuracy = perf.get("accuracy", 0)
            is_trained = perf.get("is_trained", False)

            # Model name
            tk.Label(card, text=f"🤖 {name}", font=("Segoe UI", 12, "bold"),
                    bg=C["bg_card"], fg=C["accent"]).pack()

            # Accuracy
            if accuracy > 0:
                color = C["accent_green"] if accuracy > 0.5 else C["accent_yellow"]
                tk.Label(card, text=f"{accuracy:.1%}", font=("Segoe UI", 24, "bold"),
                        bg=C["bg_card"], fg=color).pack()
                tk.Label(card, text="Accuracy", font=("Segoe UI", 9),
                        bg=C["bg_card"], fg=C["text_muted"]).pack()
            else:
                status_text = "✅ Trænet" if is_trained else "❌ Ikke trænet"
                tk.Label(card, text=status_text, font=("Segoe UI", 11),
                        bg=C["bg_card"], fg=C["text_secondary"]).pack(pady=10)

            # Stats
            total = perf.get("total_predictions", 0)
            correct = perf.get("correct_predictions", 0)
            if total > 0:
                tk.Label(card, text=f"{correct}/{total} korrekte",
                        font=("Segoe UI", 9), bg=C["bg_card"],
                        fg=C["text_muted"]).pack()

        # Comparison table
        if any(p.get("accuracy", 0) > 0 for p in performances):
            table_frame = CardFrame(self.comparison_container)
            table_frame.pack(fill="x", pady=15)

            tk.Label(table_frame, text="📊 Detaljeret Sammenligning",
                    font=("Segoe UI", 13, "bold"), bg=C["bg_card"],
                    fg=C["text_primary"]).pack(anchor="w", pady=(0, 10))

            # Table header
            header_row = StyledFrame(table_frame, bg=C["bg_card"])
            header_row.pack(fill="x")

            headers = ["Model", "Accuracy", "Predictions", "Korrekte", "Status"]
            for h in headers:
                tk.Label(header_row, text=h, font=("Segoe UI", 10, "bold"),
                        bg=C["bg_card"], fg=C["accent"],
                        width=18, anchor="center").pack(side="left")

            tk.Frame(table_frame, height=1, bg=C["border"]).pack(fill="x", pady=5)

            # Table rows
            for perf in sorted(performances, key=lambda x: x.get("accuracy", 0), reverse=True):
                row = StyledFrame(table_frame, bg=C["bg_card"])
                row.pack(fill="x", pady=1)

                name = perf.get("model_name", "").replace("_", " ").title()
                acc = perf.get("accuracy", 0)
                total = perf.get("total_predictions", 0)
                correct = perf.get("correct_predictions", 0)
                is_trained = perf.get("is_trained", False)

                values = [
                    name,
                    f"{acc:.1%}" if acc > 0 else "-",
                    str(total),
                    str(correct),
                    "✅" if is_trained or acc > 0 else "❌",
                ]

                for v in values:
                    tk.Label(row, text=v, font=("Segoe UI", 10),
                            bg=C["bg_card"], fg=C["text_primary"],
                            width=18, anchor="center").pack(side="left")

    # ═══════════════════════════════════════════
    # LIVE SCORES PAGE
    # ═══════════════════════════════════════════
    def _build_live_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["live"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        header_frame = StyledFrame(container)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header_frame, text="🔴 Live Scores").pack(side="left")

        self.live_auto_label = MutedLabel(header_frame, text="Auto-refresh: 30s")
        self.live_auto_label.pack(side="right")

        ctrl_frame = StyledFrame(container)
        ctrl_frame.pack(fill="x", padx=20, pady=(0, 10))
        AccentButton(ctrl_frame, text="🔄 Opdater nu",
                    command=self._refresh_live).pack(side="left")

        self.live_container = StyledFrame(container)
        self.live_container.pack(fill="both", expand=True, padx=20, pady=10)

        StyledLabel(self.live_container,
                   text="⏳ Indlæser live kampe...",
                   font=("Segoe UI", 12)).pack(pady=40)

    def _populate_live(self, matches: List[Dict]):
        """Populate live scores page."""
        for widget in self.live_container.winfo_children():
            widget.destroy()

        live_matches = [m for m in matches if m.get("status") in
                       ("IN_PLAY", "HALFTIME", "LIVE", "1H", "2H", "HT", "PAUSED")]

        if not live_matches:
            frame = StyledFrame(self.live_container)
            frame.pack(pady=40)
            tk.Label(frame, text="📺", font=("Segoe UI", 40),
                    bg=C["bg_dark"], fg=C["text_muted"]).pack()
            StyledLabel(frame, text="Ingen live kampe lige nu",
                       font=("Segoe UI", 14)).pack()
            MutedLabel(frame, text="Live kampe vises automatisk når de starter").pack()
            return

        for match in live_matches:
            card = CardFrame(self.live_container)
            card.pack(fill="x", pady=5)

            # League
            lc = match.get("league_code", "")
            league_info = LEAGUES.get(lc, {})
            emoji = league_info.get("emoji", "🏟️")
            league_name = league_info.get("name", lc)

            tk.Label(card, text=f"{emoji} {league_name}",
                    font=("Segoe UI", 9), bg=C["bg_card"],
                    fg=C["text_muted"]).pack(anchor="w")

            # Score row
            score_row = StyledFrame(card, bg=C["bg_card"])
            score_row.pack(fill="x", pady=5)

            home = match.get("home_team_name", "?")
            away = match.get("away_team_name", "?")
            hs = match.get("home_score", 0)
            aws = match.get("away_score", 0)
            elapsed = match.get("extra_data", {}).get("elapsed", "")

            tk.Label(score_row, text=home, font=("Segoe UI", 14, "bold"),
                    bg=C["bg_card"], fg=C["text_primary"],
                    width=22, anchor="e").pack(side="left")

            # Animated score
            score_frame = StyledFrame(score_row, bg=C["accent_red"])
            score_frame.pack(side="left", padx=10)
            tk.Label(score_frame, text=f"  {hs} - {aws}  ",
                    font=("Segoe UI", 16, "bold"),
                    bg=C["accent_red"], fg="white").pack(padx=5, pady=3)

            tk.Label(score_row, text=away, font=("Segoe UI", 14, "bold"),
                    bg=C["bg_card"], fg=C["text_primary"],
                    width=22, anchor="w").pack(side="left")

            # Elapsed
            if elapsed:
                tk.Label(score_row, text=f"⏱️ {elapsed}'",
                        font=("Segoe UI", 11, "bold"), bg=C["bg_card"],
                        fg=C["accent_yellow"]).pack(side="right")

    # ═══════════════════════════════════════════
    # SUGGESTIONS PAGE
    # ═══════════════════════════════════════════
    def _build_suggestions_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["suggestions"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        header_frame = StyledFrame(container)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header_frame, text="💡 Intelligente Forslag").pack(side="left")

        ctrl_frame = StyledFrame(container)
        ctrl_frame.pack(fill="x", padx=20, pady=(0, 10))
        AccentButton(ctrl_frame, text="🔄 Generér forslag",
                    command=self._generate_suggestions).pack(side="left")

        # Info card
        info_card = CardFrame(container)
        info_card.pack(fill="x", padx=20, pady=5)
        tk.Label(info_card, text="⚠️ DISCLAIMER: Forudsigelser er kun til underholdning. "
                "Gambling kan være vanedannende. Spil ansvarligt.",
                font=("Segoe UI", 9), bg=C["bg_card"], fg=C["accent_yellow"],
                wraplength=800).pack()

        self.suggestions_container = StyledFrame(container)
        self.suggestions_container.pack(fill="both", expand=True, padx=20, pady=10)

        StyledLabel(self.suggestions_container,
                   text="Klik 'Generér forslag' for at se value bets",
                   font=("Segoe UI", 12)).pack(pady=40)

    def _populate_suggestions(self, all_predictions: Dict[str, List[Dict]]):
        """Populate suggestions page with value bets."""
        for widget in self.suggestions_container.winfo_children():
            widget.destroy()

        if not all_predictions:
            StyledLabel(self.suggestions_container,
                       text="Ingen forslag tilgængelige",
                       font=("Segoe UI", 12)).pack(pady=40)
            return

        suggestions = []
        for match_key, predictions in all_predictions.items():
            for pred in predictions:
                if pred.get("model_name") == "ensemble" and pred.get("value_rating", 0) > 0.05:
                    suggestions.append({**pred, "match_key": match_key})

        # Sort by value rating
        suggestions.sort(key=lambda x: x.get("value_rating", 0), reverse=True)

        if not suggestions:
            frame = StyledFrame(self.suggestions_container)
            frame.pack(pady=40)
            tk.Label(frame, text="🔍", font=("Segoe UI", 40),
                    bg=C["bg_dark"], fg=C["text_muted"]).pack()
            StyledLabel(frame, text="Ingen value bets fundet i dag",
                       font=("Segoe UI", 14)).pack()
            MutedLabel(frame, text="Prøv igen når der er flere kampe").pack()
            return

        # Summary
        summary_card = CardFrame(self.suggestions_container)
        summary_card.pack(fill="x", pady=5)
        tk.Label(summary_card, text=f"💎 {len(suggestions)} Value Bets Fundet",
                font=("Segoe UI", 14, "bold"), bg=C["bg_card"],
                fg=C["accent_green"]).pack(anchor="w")

        # Suggestion cards
        for i, sug in enumerate(suggestions, 1):
            card = CardFrame(self.suggestions_container)
            card.pack(fill="x", pady=5)

            value = sug.get("value_rating", 0)
            confidence = sug.get("confidence", 0)

            # Rating stars
            if value > 0.5:
                stars = "⭐⭐⭐"
                value_label = "HIGH VALUE"
                value_color = C["high_value"]
            elif value > 0.2:
                stars = "⭐⭐"
                value_label = "MEDIUM VALUE"
                value_color = C["medium_value"]
            else:
                stars = "⭐"
                value_label = "LOW VALUE"
                value_color = C["low_value"]

            # Header row
            header = StyledFrame(card, bg=C["bg_card"])
            header.pack(fill="x")

            tk.Label(header, text=f"#{i}", font=("Segoe UI", 14, "bold"),
                    bg=C["bg_card"], fg=C["accent"]).pack(side="left")

            tk.Label(header, text=f"⚽ {sug['match_key']}",
                    font=("Segoe UI", 12, "bold"), bg=C["bg_card"],
                    fg=C["text_primary"]).pack(side="left", padx=10)

            tk.Label(header, text=f"{stars} {value_label}",
                    font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                    fg=value_color).pack(side="right")

            # Details row
            details = StyledFrame(card, bg=C["bg_card"])
            details.pack(fill="x", pady=5)

            tk.Label(details, text=f"Prediction: {sug.get('predicted_outcome', '')}",
                    font=("Segoe UI", 10), bg=C["bg_card"],
                    fg=C["text_secondary"]).pack(side="left")

            ConfidenceMeter(details, confidence=confidence, size=40).pack(side="left", padx=15)

            ProbabilityBar(details,
                          home_prob=sug.get("home_win_prob", 0.33),
                          draw_prob=sug.get("draw_prob", 0.33),
                          away_prob=sug.get("away_win_prob", 0.34),
                          width=200).pack(side="left", padx=10)

            tk.Label(details, text=f"Value: {value:.2f}",
                    font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                    fg=value_color).pack(side="right")

            # Suggestion text
            suggestion_text = sug.get("suggestion", "")
            if suggestion_text:
                tk.Label(card, text=f"💡 {suggestion_text}",
                        font=("Segoe UI", 9), bg=C["bg_card"],
                        fg=C["accent_yellow"], anchor="w",
                        wraplength=800).pack(fill="x")

    # ═══════════════════════════════════════════
    # SETTINGS PAGE
    # ═══════════════════════════════════════════
    def _build_settings_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["settings"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        header_frame = StyledFrame(container)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header_frame, text="⚙️ Indstillinger").pack(side="left")

        # API Keys section
        api_card = CardFrame(container)
        api_card.pack(fill="x", padx=20, pady=10)

        tk.Label(api_card, text="🔑 API Nøgler", font=("Segoe UI", 13, "bold"),
                bg=C["bg_card"], fg=C["accent"]).pack(anchor="w", pady=(0, 10))

        tk.Label(api_card, text="Football-Data.org API Key:",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["text_secondary"]).pack(anchor="w")
        self.fd_api_entry = tk.Entry(api_card, font=("Consolas", 10), width=50,
                                      bg=C["bg_dark"], fg=C["text_primary"],
                                      insertbackground=C["text_primary"])
        self.fd_api_entry.pack(fill="x", pady=(2, 10))
        MutedLabel(api_card, text="Gratis på: https://www.football-data.org/client/register (opret konto)").pack(anchor="w")

        tk.Label(api_card, text="API-Football Key:",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["text_secondary"]).pack(anchor="w", pady=(10, 0))
        self.af_api_entry = tk.Entry(api_card, font=("Consolas", 10), width=50,
                                      bg=C["bg_dark"], fg=C["text_primary"],
                                      insertbackground=C["text_primary"])
        self.af_api_entry.pack(fill="x", pady=(2, 10))
        MutedLabel(api_card, text="Gratis på: https://dashboard.api-football.com/register").pack(anchor="w")

        SecondaryButton(api_card, text="💾 Gem API nøgler",
                       command=self._save_api_keys).pack(pady=10)

        # Training section
        train_card = CardFrame(container)
        train_card.pack(fill="x", padx=20, pady=10)

        tk.Label(train_card, text="🧠 ML Model Træning", font=("Segoe UI", 13, "bold"),
                bg=C["bg_card"], fg=C["accent"]).pack(anchor="w", pady=(0, 10))

        tk.Label(train_card, text="Vælg ligaer til træning:",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["text_secondary"]).pack(anchor="w")

        self.league_vars = {}
        leagues_frame = StyledFrame(train_card, bg=C["bg_card"])
        leagues_frame.pack(fill="x", pady=5)

        for code, info in list(LEAGUES.items())[:6]:
            var = tk.BooleanVar(value=True)
            self.league_vars[code] = var
            cb = tk.Checkbutton(leagues_frame, text=f"{info['emoji']} {info['name']}",
                               variable=var, font=("Segoe UI", 10),
                               bg=C["bg_card"], fg=C["text_primary"],
                               selectcolor=C["bg_dark"],
                               activebackground=C["bg_card"],
                               activeforeground=C["text_primary"])
            cb.pack(anchor="w")

        btn_frame = StyledFrame(train_card, bg=C["bg_card"])
        btn_frame.pack(fill="x", pady=10)

        AccentButton(btn_frame, text="🏋️ Start Træning",
                    command=self._start_training).pack(side="left")

        self.settings_train_label = tk.Label(btn_frame, text="",
                                              font=("Segoe UI", 10),
                                              bg=C["bg_card"], fg=C["accent"])
        self.settings_train_label.pack(side="left", padx=20)

        # Database section
        db_card = CardFrame(container)
        db_card.pack(fill="x", padx=20, pady=10)

        tk.Label(db_card, text="📦 Database", font=("Segoe UI", 13, "bold"),
                bg=C["bg_card"], fg=C["accent"]).pack(anchor="w", pady=(0, 10))

        self.db_info_label = tk.Label(db_card, text="",
                                       font=("Segoe UI", 10), bg=C["bg_card"],
                                       fg=C["text_secondary"])
        self.db_info_label.pack(anchor="w")

        SecondaryButton(db_card, text="🗑️ Ryd cache",
                       command=self._clear_cache).pack(pady=5)

        # About
        about_card = CardFrame(container)
        about_card.pack(fill="x", padx=20, pady=10)

        tk.Label(about_card, text="ℹ️ Om Soccer Predictions Pro",
                font=("Segoe UI", 13, "bold"), bg=C["bg_card"],
                fg=C["accent"]).pack(anchor="w", pady=(0, 5))
        tk.Label(about_card, text="Version 1.0.0 | Python | Machine Learning | Desktop App",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["text_secondary"]).pack(anchor="w")
        tk.Label(about_card, text="Models: XGBoost, Neural Network, Random Forest, Poisson, Ensemble",
                font=("Segoe UI", 9), bg=C["bg_card"],
                fg=C["text_muted"]).pack(anchor="w")

    # ═══════════════════════════════════════════
    # DATA OPERATIONS
    # ═══════════════════════════════════════════
    def _initial_load(self):
        """Load initial data on startup."""
        self.status_bar.set_status("⏳ Indlæser data...")
        self._update_sidebar_info()

        def load():
            try:
                matches = self.data.fetch_todays_matches()
                self.matches_cache = matches
                self.root.after(0, lambda: self._populate_dashboard(matches))
                self.root.after(0, lambda: self.status_bar.set_status("✅ Data indlæst"))

                # Also load live
                live = self.data.fetch_live_matches()
                self.root.after(0, lambda: self._populate_live(live))

            except Exception as e:
                logger.error(f"Initial load error: {e}")
                self.root.after(0, lambda: self.status_bar.set_status(f"❌ Fejl: {e}"))

        threading.Thread(target=load, daemon=True).start()

        # Start auto-refresh
        self._auto_refresh()

    def _auto_refresh(self):
        """Auto-refresh data periodically."""
        interval_ms = GUI_SETTINGS["refresh_interval_seconds"] * 1000

        def refresh():
            if self.current_page == "live":
                self._refresh_live()
            self._update_sidebar_info()
            self.auto_refresh_id = self.root.after(interval_ms, refresh)

        self.auto_refresh_id = self.root.after(interval_ms, refresh)

    def _refresh_dashboard(self):
        """Refresh dashboard data."""
        self.status_bar.set_status("🔄 Opdaterer...")

        def refresh():
            try:
                matches = self.data.fetch_todays_matches(force_refresh=True)
                self.matches_cache = matches
                self.root.after(0, lambda: self._populate_dashboard(matches))
                self.root.after(0, lambda: self.status_bar.set_status("✅ Opdateret"))
                self.root.after(0, self._update_sidebar_info)
            except Exception as e:
                self.root.after(0, lambda: self.status_bar.set_status(f"❌ Fejl: {e}"))

        threading.Thread(target=refresh, daemon=True).start()

    def _load_upcoming(self):
        """Load upcoming matches."""
        self.status_bar.set_status("📅 Indlæser kommende kampe...")

        def load():
            try:
                matches = self.data.fetch_upcoming_matches(days=7)
                self.matches_cache.extend(matches)
                all_matches = self.matches_cache
                self.root.after(0, lambda: self._populate_dashboard(all_matches))
                self.root.after(0, lambda: self.status_bar.set_status("✅ Kommende kampe indlæst"))
            except Exception as e:
                self.root.after(0, lambda: self.status_bar.set_status(f"❌ Fejl: {e}"))

        threading.Thread(target=load, daemon=True).start()

    def _refresh_live(self):
        """Refresh live scores."""
        def refresh():
            try:
                matches = self.data.fetch_live_matches()
                self.root.after(0, lambda: self._populate_live(matches))
            except Exception as e:
                logger.error(f"Live refresh error: {e}")

        threading.Thread(target=refresh, daemon=True).start()

    def _generate_predictions(self):
        """Generate AI predictions for today's matches."""
        if not self.matches_cache:
            messagebox.showinfo("Info", "Ingen kampe at forudsige. Indlæs kampe først.")
            return

        self.pred_status_label.config(text="⏳ Genererer predictions...")
        self.status_bar.set_status("🧠 Kører ML modeller...")

        def generate():
            try:
                scheduled = [m for m in self.matches_cache
                            if m.get("status") in ("SCHEDULED", "TIMED", "NOT_STARTED")]
                if not scheduled:
                    scheduled = self.matches_cache

                predictions = self.engine.predict_all_matches(scheduled)
                self.predictions_cache = predictions

                self.root.after(0, lambda: self._populate_predictions(predictions))
                self.root.after(0, lambda: self.pred_status_label.config(
                    text=f"✅ {len(predictions)} predictions genereret"))
                self.root.after(0, lambda: self.status_bar.set_status("✅ Predictions klar"))

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                self.root.after(0, lambda: self.pred_status_label.config(
                    text=f"❌ Fejl: {e}"))

        threading.Thread(target=generate, daemon=True).start()

    def _quick_predict(self, match: Dict):
        """Quick prediction for a single match."""
        self.status_bar.set_status(f"🎯 Predicting {match.get('home_team_name')} vs {match.get('away_team_name')}...")

        def predict():
            try:
                predictions = self.engine.predict_match(match)
                match_key = f"{match['home_team_name']} vs {match['away_team_name']}"
                self.predictions_cache[match_key] = predictions

                # Show result
                ensemble = next((p for p in predictions if p.get("model_name") == "ensemble"), None)
                if ensemble:
                    result_text = (f"Prediction: {ensemble['predicted_outcome']} "
                                 f"({ensemble['confidence']:.0%} confidence)")
                    self.root.after(0, lambda: self.status_bar.set_status(f"🎯 {result_text}"))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "🎯 Quick Prediction",
                        f"⚽ {match_key}\n\n"
                        f"Prediction: {ensemble['predicted_outcome']}\n"
                        f"Confidence: {ensemble['confidence']:.0%}\n"
                        f"Home Win: {ensemble['home_win_prob']:.0%}\n"
                        f"Draw: {ensemble['draw_prob']:.0%}\n"
                        f"Away Win: {ensemble['away_win_prob']:.0%}\n\n"
                        f"💡 {ensemble.get('suggestion', '')}"
                    ))
            except Exception as e:
                self.root.after(0, lambda: self.status_bar.set_status(f"❌ {e}"))

        threading.Thread(target=predict, daemon=True).start()

    def _start_training(self):
        """Start ML model training."""
        selected_leagues = [code for code, var in self.league_vars.items() if var.get()]
        if not selected_leagues:
            selected_leagues = ["PL", "PD", "BL1", "SA", "FL1"]

        if self.engine.is_training:
            messagebox.showinfo("Info", "Træning er allerede i gang...")
            return

        self.training_frame.pack(fill="x", padx=20, pady=5)
        self.training_progress_label.config(text="⏳ Starter træning...")
        self.status_bar.set_status("🏋️ Træner ML modeller...")

        def callback(event_type, data):
            if event_type == "status":
                self.root.after(0, lambda: self.training_progress_label.config(text=f"⏳ {data}"))
                self.root.after(0, lambda: self.status_bar.set_status(f"🏋️ {data}"))
            elif event_type == "progress":
                self.root.after(0, lambda: self.training_progress_label.config(text=f"🔄 {data}"))
            elif event_type == "done":
                result_text = " | ".join([f"{k}: {v:.1%}" for k, v in data.items()])
                self.root.after(0, lambda: self.training_progress_label.config(
                    text=f"✅ Træning færdig! {result_text}"))
                self.root.after(0, lambda: self.status_bar.set_status("✅ Modeller trænet!"))
                self.root.after(0, self._update_sidebar_info)
                self.root.after(0, self._populate_comparison)
                if hasattr(self, 'settings_train_label'):
                    self.root.after(0, lambda: self.settings_train_label.config(
                        text=f"✅ Trænet! {result_text}"))
            elif event_type == "error":
                self.root.after(0, lambda: self.training_progress_label.config(
                    text=f"❌ Fejl: {data}"))
                self.root.after(0, lambda: self.status_bar.set_status(f"❌ Træningsfejl"))

        self.engine.train_models_async(selected_leagues, callback)

    def _generate_suggestions(self):
        """Generate betting suggestions."""
        if not self.predictions_cache:
            self._generate_predictions()
            self.root.after(2000, lambda: self._populate_suggestions(self.predictions_cache))
        else:
            self._populate_suggestions(self.predictions_cache)

    def _refresh_comparison(self):
        """Refresh model comparison."""
        self._populate_comparison()

    def _save_api_keys(self):
        """Save API keys to environment."""
        fd_key = self.fd_api_entry.get().strip()
        af_key = self.af_api_entry.get().strip()

        if fd_key:
            os.environ["FOOTBALL_DATA_API_KEY"] = fd_key
        if af_key:
            os.environ["API_FOOTBALL_KEY"] = af_key

        messagebox.showinfo("✅ Gemt",
                          "API nøgler er gemt for denne session.\n"
                          "Sæt dem som miljøvariabler for permanent brug.")

    def _clear_cache(self):
        """Clear API cache."""
        self.db.clear_expired_cache()
        messagebox.showinfo("✅ Cache ryddet", "Udløbet cache er ryddet.")

    def _update_sidebar_info(self):
        """Update sidebar info labels."""
        try:
            match_count = self.db.get_match_count()
            self.db_count_label.config(text=f"📦 {match_count} kampe i database")

            trained = self.engine.is_trained
            status = "✅ Trænet" if trained else "❌ Ikke trænet"
            self.model_status_label.config(text=f"🤖 Modeller: {status}")

            now = datetime.now().strftime("%H:%M:%S")
            self.last_update_label.config(text=f"🕐 Sidst opdateret: {now}")

            # Update database info on settings page
            if hasattr(self, 'db_info_label'):
                pred_count = self.db.get_prediction_count()
                self.db_info_label.config(
                    text=f"Kampe: {match_count} | Predictions: {pred_count} | "
                         f"Database: {self.db.db_path}")
        except Exception:
            pass

    # ═══════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════
    def run(self):
        """Start the application."""
        logger.info("Starting Soccer Predictions App")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """Handle window close."""
        if self.auto_refresh_id:
            self.root.after_cancel(self.auto_refresh_id)
        self.db.close()
        self.root.destroy()
