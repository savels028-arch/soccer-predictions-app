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
        self._build_ai_sites_page()
        self._build_danske_spil_page()
        self._build_comparison_page()
        self._build_live_page()
        self._build_suggestions_page()
        self._build_history_page()
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
            ("ai_sites",    "🌐  AI Sites",       "Predictions fra 4 AI-sider"),
            ("danske_spil", "🇩🇰  Danske Spil",   "Spil hos Danske Spil"),
            ("comparison",  "📈  Sammenligning",  "Sammenlign modeller"),
            ("live",        "🔴  Live Scores",    "Live kampresultater"),
            ("suggestions", "💡  Forslag",        "Betting forslag"),
            ("history",     "📜  Historik",       "Historisk nøjagtighed"),
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
            extra = match.get("extra_data", {})
            if isinstance(extra, dict):
                elapsed = extra.get("elapsed", "")
            else:
                elapsed = ""
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
            extra = match.get("extra_data", {})
            elapsed = extra.get("elapsed", "") if isinstance(extra, dict) else ""

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
    # AI SITES PAGE  (external AI predictions)
    # ═══════════════════════════════════════════
    def _build_ai_sites_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["ai_sites"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header = StyledFrame(container)
        header.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header, text="🌐 AI Sites – Consensus Predictions").pack(side="left")

        # Controls
        ctrl = StyledFrame(container)
        ctrl.pack(fill="x", padx=20, pady=(0, 10))
        AccentButton(ctrl, text="🔄  Hent fra AI-sider",
                     command=self._fetch_ai_site_predictions).pack(side="left")
        self.ai_sites_status = StyledLabel(ctrl, text="", font=("Segoe UI", 10))
        self.ai_sites_status.pack(side="left", padx=20)

        # Source info
        info_card = CardFrame(container)
        info_card.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(info_card, text="Kilder: AI-Goalie.com  ·  BetsWithBots.com  ·  SoccerTips.ai  ·  FootballPredictions.ai",
                 font=("Segoe UI", 9), bg=C["bg_card"], fg=C["text_muted"]).pack(padx=10, pady=5)

        # Predictions container
        self.ai_sites_container = StyledFrame(container)
        self.ai_sites_container.pack(fill="both", expand=True, padx=20, pady=10)

        StyledLabel(self.ai_sites_container,
                    text="Klik '🔄 Hent fra AI-sider' for at hente predictions",
                    font=("Segoe UI", 12)).pack(pady=40)

    def _fetch_ai_site_predictions(self):
        """Fetch AI predictions in background thread."""
        self.ai_sites_status.configure(text="⏳ Henter fra 4 AI-sider… (ca. 30 sek)")
        self.status_bar.set_status("Henter AI predictions…")

        def _worker():
            try:
                consensus = self.data.fetch_ai_predictions(force_refresh=True)
                self.root.after(0, lambda: self._populate_ai_sites(consensus))
            except Exception as e:
                self.root.after(0, lambda: self.ai_sites_status.configure(
                    text=f"❌ Fejl: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_ai_sites(self, consensus: List[Dict]):
        """Populate AI Sites page with consensus data."""
        for w in self.ai_sites_container.winfo_children():
            w.destroy()

        if not consensus:
            StyledLabel(self.ai_sites_container,
                        text="Ingen AI predictions fundet i dag",
                        font=("Segoe UI", 12)).pack(pady=40)
            self.ai_sites_status.configure(text="⚠️ Ingen data")
            return

        # Stats
        multi = [c for c in consensus if c["num_sources"] >= 2]
        self.ai_sites_status.configure(
            text=f"✅ {len(consensus)} kampe  ·  {len(multi)} med 2+ kilder"
        )
        self.status_bar.set_status(f"AI Sites: {len(consensus)} predictions hentet")

        # Sort: multi-source first, then by confidence
        consensus.sort(key=lambda x: (x["num_sources"],
                                       x.get("consensus_confidence") or 0),
                        reverse=True)

        for match in consensus[:80]:  # limit display
            card = CardFrame(self.ai_sites_container)
            card.pack(fill="x", pady=4)

            home = match.get("home_team", "?")
            away = match.get("away_team", "?")
            n_src = match.get("num_sources", 0)
            league = match.get("league", "")
            ko = match.get("kickoff_time", "")

            # Top row: match title + source count
            top = StyledFrame(card, bg=C["bg_card"])
            top.pack(fill="x")

            title_text = f"⚽ {home}  vs  {away}"
            tk.Label(top, text=title_text, font=("Segoe UI", 11, "bold"),
                     bg=C["bg_card"], fg=C["text_primary"]).pack(side="left")

            meta_parts = []
            if league:
                meta_parts.append(league)
            if ko:
                meta_parts.append(f"🕐 {ko}")
            meta_parts.append(f"📡 {n_src} kilder")
            meta_text = "  ·  ".join(meta_parts)
            tk.Label(top, text=meta_text, font=("Segoe UI", 9),
                     bg=C["bg_card"], fg=C["text_muted"]).pack(side="right")

            # Probability row
            prob_row = StyledFrame(card, bg=C["bg_card"])
            prob_row.pack(fill="x", pady=(4, 0))

            h_pct = match.get("avg_home_win_pct")
            d_pct = match.get("avg_draw_pct")
            a_pct = match.get("avg_away_win_pct")

            if h_pct is not None:
                tk.Label(prob_row, text=f"1: {h_pct:.0f}%", font=("Segoe UI", 10, "bold"),
                         bg=C["bg_card"], fg=C["win_color"]).pack(side="left", padx=(0, 10))
            if d_pct is not None:
                tk.Label(prob_row, text=f"X: {d_pct:.0f}%", font=("Segoe UI", 10, "bold"),
                         bg=C["bg_card"], fg=C["draw_color"]).pack(side="left", padx=(0, 10))
            if a_pct is not None:
                tk.Label(prob_row, text=f"2: {a_pct:.0f}%", font=("Segoe UI", 10, "bold"),
                         bg=C["bg_card"], fg=C["lose_color"]).pack(side="left", padx=(0, 10))

            # Probability bar (if we have valid data)
            if h_pct and a_pct:
                dp = d_pct if d_pct else 0
                total = h_pct + dp + a_pct
                if total > 0:
                    ProbabilityBar(prob_row,
                                   home_prob=h_pct / total,
                                   draw_prob=dp / total,
                                   away_prob=a_pct / total,
                                   width=220).pack(side="left", padx=15)

            # Winner prediction
            winner = match.get("consensus_winner")
            conf = match.get("consensus_confidence")
            if winner:
                w_map = {"1": f"🏠 {home}", "2": f"✈️ {away}", "X": "🤝 Uafgjort"}
                w_color = {"1": C["win_color"], "2": C["lose_color"], "X": C["draw_color"]}
                w_text = f"Prediction: {w_map.get(winner, winner)}"
                if conf:
                    w_text += f"  ({conf:.0f}%)"
                tk.Label(prob_row, text=w_text, font=("Segoe UI", 10, "bold"),
                         bg=C["bg_card"], fg=w_color.get(winner, C["accent"])).pack(side="right")

            # Extra row: BTTS / O-U / sources
            extras = []
            btts = match.get("btts_consensus")
            ou = match.get("over_under_consensus")
            if btts:
                extras.append(f"BTTS: {btts}")
            if ou:
                extras.append(f"O/U 2.5: {ou}")
            sources = match.get("sources", [])
            if sources:
                extras.append(f"Fra: {', '.join(sources)}")

            if extras:
                extra_text = "  ·  ".join(extras)
                tk.Label(card, text=extra_text, font=("Segoe UI", 8),
                         bg=C["bg_card"], fg=C["text_muted"],
                         anchor="w").pack(fill="x", pady=(3, 0))

    # ═══════════════════════════════════════════
    # DANSKE SPIL PAGE – Konsensus + Spilbare
    # ═══════════════════════════════════════════
    def _build_danske_spil_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["danske_spil"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header = StyledFrame(container)
        header.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header, text="🇩🇰 Konsensus & Danske Spil").pack(side="left")

        # Controls
        ctrl = StyledFrame(container)
        ctrl.pack(fill="x", padx=20, pady=(0, 10))
        AccentButton(ctrl, text="🔄  Analysér alle kilder",
                     command=self._fetch_danske_spil).pack(side="left")
        self.ds_status = StyledLabel(ctrl, text="", font=("Segoe UI", 10))
        self.ds_status.pack(side="left", padx=20)

        # Info card
        info_card = CardFrame(container)
        info_card.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(info_card,
                 text="💡 Sammenligner automatisk AI-sites (4 kilder) og ML-modeller. "
                      "Viser kampe hvor kilderne er ENIGE om udfald, og hvilke der kan spilles hos Danske Spil.",
                 font=("Segoe UI", 9), bg=C["bg_card"], fg=C["text_muted"],
                 wraplength=900).pack(padx=10, pady=5)

        # Disclaimer
        disc_card = CardFrame(container)
        disc_card.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(disc_card,
                 text="⚠️ DISCLAIMER: Forudsigelser er kun til underholdning. "
                      "Gambling kan være vanedannende. Spil ansvarligt. 18+ | stopspillet.dk | rofus.nu",
                 font=("Segoe UI", 9), bg=C["bg_card"], fg=C["accent_yellow"],
                 wraplength=900).pack()

        # Summary cards row
        self.ds_summary_frame = StyledFrame(container)
        self.ds_summary_frame.pack(fill="x", padx=20, pady=(0, 10))

        # Results container
        self.ds_container = StyledFrame(container)
        self.ds_container.pack(fill="both", expand=True, padx=20, pady=10)

        StyledLabel(self.ds_container,
                    text="Klik '🔄 Analysér alle kilder' for at starte",
                    font=("Segoe UI", 12)).pack(pady=40)

    def _fetch_danske_spil(self):
        """Hent alle kilder, beregn konsensus, match med Danske Spil."""
        self.ds_status.configure(text="⏳ Henter fra alle kilder… (ca. 15-30 sek)")
        self.status_bar.set_status("🔄 Henter AI-sites + ML + Danske Spil…")

        def _worker():
            try:
                result = self.data.build_consensus_with_danske_spil(
                    prediction_engine=self.engine,
                    matches=self.matches_cache if self.matches_cache else None,
                    force_refresh=True,
                )
                self.root.after(0, lambda: self._populate_danske_spil(result))
            except Exception as e:
                logger.error(f"Consensus+DS fetch error: {e}")
                self.root.after(0, lambda: self.ds_status.configure(
                    text=f"❌ Fejl: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _populate_danske_spil(self, result: Dict):
        """Populér konsensus + Danske Spil siden."""
        # Clear
        for w in self.ds_container.winfo_children():
            w.destroy()
        for w in self.ds_summary_frame.winfo_children():
            w.destroy()

        stats = result.get("stats", {})
        playable = result.get("playable", [])
        agreed = result.get("agreed", [])
        all_matches = result.get("all_consensus", [])

        # ── Summary cards ──
        summaries = [
            ("🌐 AI-sites", str(stats.get("ai_predictions", 0)), C["accent"]),
            ("🤖 ML Modeller", str(stats.get("ml_predictions", 0)), C["accent"]),
            ("🤝 Enige", str(stats.get("agreed", 0)), C["accent_green"]),
            ("🇩🇰 Spilbare", str(stats.get("playable", 0)),
             C["accent_green"] if stats.get("playable", 0) > 0 else C["accent_red"]),
        ]
        for title, value, color in summaries:
            card = CardFrame(self.ds_summary_frame)
            card.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)
            tk.Label(card, text=title, font=("Segoe UI", 9),
                     bg=C["bg_card"], fg=C["text_muted"]).pack(anchor="w")
            tk.Label(card, text=value, font=("Segoe UI", 22, "bold"),
                     bg=C["bg_card"], fg=color).pack(anchor="w", pady=(5, 0))

        self.ds_status.configure(
            text=f"✅ {stats.get('agreed', 0)} enige · "
                 f"{stats.get('playable', 0)} spilbare hos DS · "
                 f"{stats.get('ds_events', 0)} DS kampe"
        )
        self.status_bar.set_status(
            f"Konsensus: {stats.get('playable', 0)} spilbare kampe hos Danske Spil"
        )

        # ── SEKTION 1: Spilbare hos Danske Spil (kilderne er enige) ──
        if playable:
            sec_hdr = StyledFrame(self.ds_container)
            sec_hdr.pack(fill="x", pady=(10, 5))
            SubHeaderLabel(sec_hdr,
                          text=f"🎯 Spilbare hos Danske Spil — kilderne er enige ({len(playable)})").pack(side="left")

            for entry in playable:
                self._create_consensus_card(self.ds_container, entry, show_ds=True)

        # ── SEKTION 2: Enige men ikke hos DS ──
        agreed_not_playable = [x for x in agreed if not x.get("danske_spil")]
        if agreed_not_playable:
            sec_hdr2 = StyledFrame(self.ds_container)
            sec_hdr2.pack(fill="x", pady=(20, 5))
            SubHeaderLabel(sec_hdr2,
                          text=f"🤝 Kilderne er enige — ikke hos Danske Spil ({len(agreed_not_playable)})").pack(side="left")

            for entry in agreed_not_playable[:30]:
                self._create_consensus_card(self.ds_container, entry, show_ds=False)

        # ── SEKTION 3: Kun 1 kilde / uenige ──
        rest = [x for x in all_matches if not x.get("agreed_outcome")]
        if rest:
            sec_hdr3 = StyledFrame(self.ds_container)
            sec_hdr3.pack(fill="x", pady=(20, 5))

            # Collapsible: vis kun header, ikke alle kort
            show_rest_var = tk.BooleanVar(value=False)
            rest_toggle = SecondaryButton(
                sec_hdr3,
                text=f"▶ Vis kampe med kun 1 kilde / uenige ({len(rest)})",
                command=lambda: self._toggle_rest_section(rest, rest_frame, rest_toggle, show_rest_var),
            )
            rest_toggle.pack(side="left")

            rest_frame = StyledFrame(self.ds_container)
            rest_frame.pack(fill="x")

        if not playable and not agreed_not_playable:
            StyledLabel(self.ds_container,
                        text="Ingen konsensus-kampe fundet. Prøv at opdatere eller vent på nye kampe.",
                        font=("Segoe UI", 12)).pack(pady=40)

    def _toggle_rest_section(self, rest, frame, button, var):
        """Toggle vis/skjul af uenige kampe."""
        if var.get():
            for w in frame.winfo_children():
                w.destroy()
            button.config(text=f"▶ Vis kampe med kun 1 kilde / uenige ({len(rest)})")
            var.set(False)
        else:
            for entry in rest[:50]:
                self._create_consensus_card(frame, entry, show_ds=bool(entry.get("danske_spil")))
            button.config(text=f"▼ Skjul kampe med kun 1 kilde / uenige ({len(rest)})")
            var.set(True)

    def _create_consensus_card(self, parent, entry: Dict, show_ds: bool):
        """Opret et kort for en konsensus-kamp."""
        card = CardFrame(parent)
        card.pack(fill="x", pady=4)

        home = entry.get("home_team", "?")
        away = entry.get("away_team", "?")
        ds = entry.get("danske_spil")
        agreed = entry.get("agreed_outcome")
        all_agree = entry.get("all_agree", False)
        sources = entry.get("sources", [])

        # ── Top row: kamp + enigheds-badge ──
        top = StyledFrame(card, bg=C["bg_card"])
        top.pack(fill="x")

        agree_icon = "🎯" if all_agree and ds else "🤝" if agreed else "⚪"
        title = f"{agree_icon} {home}  vs  {away}"
        tk.Label(top, text=title, font=("Segoe UI", 12, "bold"),
                 bg=C["bg_card"], fg=C["text_primary"]).pack(side="left")

        # Meta info
        meta_parts = []
        league = entry.get("league", "")
        if league:
            meta_parts.append(league)
        kickoff = entry.get("kickoff_time", "")
        if kickoff:
            try:
                dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                meta_parts.append(f"🕐 {dt.strftime('%d/%m %H:%M')}")
            except Exception:
                if ":" in str(kickoff):
                    meta_parts.append(f"🕐 {kickoff}")
        if meta_parts:
            tk.Label(top, text="  ·  ".join(meta_parts), font=("Segoe UI", 9),
                     bg=C["bg_card"], fg=C["text_muted"]).pack(side="right")

        # ── Source breakdown row ──
        src_row = StyledFrame(card, bg=C["bg_card"])
        src_row.pack(fill="x", pady=(5, 0))

        outcome_labels = {
            "HOME_WIN": ("Hjemme", C["win_color"]),
            "AWAY_WIN": ("Ude", C["lose_color"]),
            "DRAW": ("Uafgjort", C["draw_color"]),
        }

        for src in sources:
            src_frame = StyledFrame(src_row, bg=C["bg_card"])
            src_frame.pack(side="left", padx=(0, 15))

            src_icon = "🌐" if src["type"] == "ai_consensus" else "🤖"
            src_name = src.get("name", "?")
            pred = src.get("prediction", "?")
            conf = src.get("confidence")
            pred_label, pred_color = outcome_labels.get(pred, (pred, C["text_secondary"]))

            conf_text = ""
            if conf:
                conf_text = f" ({conf:.0f}%)" if conf > 1 else f" ({conf:.0%})"

            # Source name + num_sources for AI
            name_text = f"{src_icon} {src_name}"
            if src.get("num_sources"):
                name_text += f" ({src['num_sources']} sider)"

            tk.Label(src_frame, text=name_text, font=("Segoe UI", 9),
                     bg=C["bg_card"], fg=C["text_muted"]).pack(anchor="w")
            tk.Label(src_frame, text=f"{pred_label}{conf_text}",
                     font=("Segoe UI", 10, "bold"),
                     bg=C["bg_card"], fg=pred_color).pack(anchor="w")

            # BTTS + O/U for AI
            extras = []
            if src.get("btts"):
                extras.append(f"BTTS: {src['btts']}")
            if src.get("over_under"):
                extras.append(f"O/U: {src['over_under']}")
            if extras:
                tk.Label(src_frame, text=" · ".join(extras), font=("Segoe UI", 8),
                         bg=C["bg_card"], fg=C["text_muted"]).pack(anchor="w")

        # ── Agreement badge ──
        if agreed:
            agree_label, agree_color = outcome_labels.get(agreed, (agreed, C["accent"]))
            badge_frame = StyledFrame(src_row, bg=C["bg_card"])
            badge_frame.pack(side="right")
            agree_text = "ALLE ENIGE" if all_agree else "FLERTAL"
            tk.Label(badge_frame, text=f"✅ {agree_text}: {agree_label}",
                     font=("Segoe UI", 11, "bold"),
                     bg=C["bg_card"], fg=agree_color).pack()

        # ── Danske Spil odds ──
        if ds and show_ds:
            ds_row = StyledFrame(card, bg=C["bg_card"])
            ds_row.pack(fill="x", pady=(5, 0))

            tk.Label(ds_row, text="🇩🇰 Danske Spil Odds:",
                     font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                     fg=C["accent"]).pack(side="left")

            h_odds = ds.get("home_odds")
            d_odds = ds.get("draw_odds")
            a_odds = ds.get("away_odds")

            if h_odds:
                tk.Label(ds_row, text=f"  1: {h_odds:.2f}",
                         font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                         fg=C["win_color"]).pack(side="left", padx=(10, 0))
            if d_odds:
                tk.Label(ds_row, text=f"  X: {d_odds:.2f}",
                         font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                         fg=C["draw_color"]).pack(side="left", padx=(5, 0))
            if a_odds:
                tk.Label(ds_row, text=f"  2: {a_odds:.2f}",
                         font=("Segoe UI", 10, "bold"), bg=C["bg_card"],
                         fg=C["lose_color"]).pack(side="left", padx=(5, 0))

            # Extra odds
            extras = []
            ou_o, ou_u = ds.get("over_25_odds"), ds.get("under_25_odds")
            btts_y, btts_n = ds.get("btts_yes_odds"), ds.get("btts_no_odds")
            if ou_o and ou_u:
                extras.append(f"O2.5: {ou_o:.2f} / U2.5: {ou_u:.2f}")
            if btts_y and btts_n:
                extras.append(f"BTTS Ja: {btts_y:.2f} / Nej: {btts_n:.2f}")
            if extras:
                tk.Label(ds_row, text="  |  ".join(extras), font=("Segoe UI", 9),
                         bg=C["bg_card"], fg=C["text_secondary"]).pack(side="right")

            # ── Value beregning ──
            if agreed and h_odds and d_odds and a_odds:
                # Find bedste probability for det enige udfald
                ai = entry.get("ai_consensus")
                ml = entry.get("ml_ensemble")

                h_pct = d_pct = a_pct = None
                if ai:
                    h_pct = ai.get("avg_home_win_pct")
                    d_pct = ai.get("avg_draw_pct")
                    a_pct = ai.get("avg_away_win_pct")
                elif ml:
                    h_pct = ml.get("home_win_prob")
                    d_pct = ml.get("draw_prob")
                    a_pct = ml.get("away_win_prob")

                if h_pct is not None and a_pct is not None:
                    value_info = self._calc_ds_value(
                        h_pct, d_pct, a_pct,
                        h_odds, d_odds, a_odds, home, away
                    )
                    if value_info:
                        val_row = StyledFrame(card, bg=C["bg_card"])
                        val_row.pack(fill="x", pady=(3, 0))
                        tk.Label(val_row, text=value_info,
                                 font=("Segoe UI", 10, "bold"),
                                 bg=C["bg_card"], fg=C["accent_green"]).pack(side="left")

            # Deeplink
            deeplink = ds.get("deeplink")
            if deeplink:
                tk.Label(card, text=f"🔗 {deeplink}", font=("Segoe UI", 8),
                         bg=C["bg_card"], fg=C["text_muted"],
                         cursor="hand2", anchor="w").pack(fill="x", pady=(3, 0))

    def _calc_ds_value(self, h_pct, d_pct, a_pct, h_odds, d_odds, a_odds, home, away):
        """Beregn value bets baseret på AI-sandsynlighed vs DS-odds."""
        try:
            # Normalisér til 0-1 hvis procenter
            hp = h_pct / 100.0 if h_pct > 1 else h_pct
            dp = (d_pct / 100.0 if d_pct > 1 else d_pct) if d_pct else 0
            ap = a_pct / 100.0 if a_pct > 1 else a_pct

            values = []
            if h_odds > 0:
                ev_h = hp * h_odds
                if ev_h > 1.05:
                    values.append(f"💚 {home} VALUE (EV: {ev_h:.2f})")
            if d_odds > 0 and dp > 0:
                ev_d = dp * d_odds
                if ev_d > 1.10:
                    values.append(f"🟡 Uafgjort VALUE (EV: {ev_d:.2f})")
            if a_odds > 0:
                ev_a = ap * a_odds
                if ev_a > 1.05:
                    values.append(f"💚 {away} VALUE (EV: {ev_a:.2f})")

            return " | ".join(values) if values else None
        except Exception:
            return None

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
    # HISTORY PAGE – Historisk nøjagtighed
    # ═══════════════════════════════════════════
    def _build_history_page(self):
        page = StyledFrame(self.content_frame)
        self.pages["history"] = page

        scroll = ScrollableFrame(page)
        scroll.pack(fill="both", expand=True)
        container = scroll.scrollable_frame

        # Header
        header = StyledFrame(container)
        header.pack(fill="x", padx=20, pady=(20, 10))
        HeaderLabel(header, text="📜 Historisk Nøjagtighed").pack(side="left")

        # Refresh
        ctrl = StyledFrame(container)
        ctrl.pack(fill="x", padx=20, pady=(0, 10))
        AccentButton(ctrl, text="🔄  Opdater historik",
                     command=self._fetch_history).pack(side="left")
        self.hist_status = StyledLabel(ctrl, text="", font=("Segoe UI", 10))
        self.hist_status.pack(side="left", padx=20)

        # Info
        info_card = CardFrame(container)
        info_card.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(info_card,
                 text="💡 Viser hvor mange af dine predictions der gik hjem, fordelt på model og liga. "
                      "Data beregnes ved at sammenligne predicted_outcome med de faktiske slutresultater.",
                 font=("Segoe UI", 9), bg=C["bg_card"], fg=C["text_muted"],
                 wraplength=900).pack(padx=10, pady=5)

        # ── Summary cards row ──
        self.hist_summary_frame = StyledFrame(container)
        self.hist_summary_frame.pack(fill="x", padx=20, pady=(0, 10))

        # ── Model accuracy section ──
        SubHeaderLabel(container, text="🤖 Nøjagtighed per Model").pack(anchor="w", padx=20, pady=(10, 5))
        self.hist_models_frame = StyledFrame(container)
        self.hist_models_frame.pack(fill="x", padx=20, pady=(0, 10))

        # ── League accuracy section ──
        SubHeaderLabel(container, text="🏆 Nøjagtighed per Liga").pack(anchor="w", padx=20, pady=(10, 5))
        self.hist_leagues_frame = StyledFrame(container)
        self.hist_leagues_frame.pack(fill="x", padx=20, pady=(0, 10))

        # ── Recent verified predictions ──
        SubHeaderLabel(container, text="📋 Seneste Verificerede Predictions").pack(anchor="w", padx=20, pady=(10, 5))
        self.hist_recent_frame = StyledFrame(container)
        self.hist_recent_frame.pack(fill="x", padx=20, pady=(0, 20))

    def _fetch_history(self):
        """Hent historisk nøjagtighed fra databasen."""
        self.hist_status.config(text="⏳ Beregner historik...")
        self.status_bar.set_status("📜 Beregner historisk nøjagtighed...")

        def fetch():
            try:
                accuracy_data = self.db.get_prediction_accuracy()
                self.root.after(0, lambda: self._populate_history(accuracy_data))
            except Exception as e:
                logger.error("Historik fejl: %s", e)
                self.root.after(0, lambda: self.hist_status.config(text=f"❌ Fejl: {e}"))

        threading.Thread(target=fetch, daemon=True).start()

    def _populate_history(self, data: Dict):
        """Populer historik-siden med data."""
        # Clear existing
        for frame in (self.hist_summary_frame, self.hist_models_frame,
                      self.hist_leagues_frame, self.hist_recent_frame):
            for w in frame.winfo_children():
                w.destroy()

        by_model = data.get("by_model_crosscheck", [])
        by_league = data.get("by_league", [])
        recent = data.get("recent_predictions", [])
        model_perf = data.get("model_performance", [])

        # ── Compute totals ──
        total_preds = sum(m.get("total", 0) for m in by_model)
        total_correct = sum(m.get("correct", 0) for m in by_model)
        overall_acc = (total_correct / total_preds * 100) if total_preds > 0 else 0

        # ── Summary cards ──
        summaries = [
            ("📊 Total Predictions", str(total_preds), C["accent"]),
            ("✅ Korrekte", str(total_correct), C["accent_green"]),
            ("❌ Forkerte", str(total_preds - total_correct), C["accent_red"]),
            ("🎯 Samlet Nøjagtighed", f"{overall_acc:.1f}%",
             C["accent_green"] if overall_acc >= 50 else C["accent_yellow"] if overall_acc >= 35 else C["accent_red"]),
        ]

        for title, value, color in summaries:
            card = CardFrame(self.hist_summary_frame)
            card.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)
            tk.Label(card, text=title, font=("Segoe UI", 9),
                     bg=C["bg_card"], fg=C["text_muted"]).pack(anchor="w")
            tk.Label(card, text=value, font=("Segoe UI", 22, "bold"),
                     bg=C["bg_card"], fg=color).pack(anchor="w", pady=(5, 0))

        if total_preds == 0:
            self.hist_status.config(text="ℹ️ Ingen verificerede predictions endnu. "
                                        "Kør predictions på kampe, vent til de er færdige, og opdater igen.")
            no_data = CardFrame(self.hist_models_frame)
            no_data.pack(fill="x", pady=5)
            tk.Label(no_data,
                     text="📭 Ingen historiske data endnu.\n\n"
                          "Predictions gemmes automatisk når du bruger AI Predictions eller ML modeller.\n"
                          "Når kampene er færdigspillet, kan nøjagtigheden beregnes her.\n\n"
                          "Tip: Træn modellerne under ⚙️ Indstillinger, lav predictions, "
                          "og kom tilbage efter kampene er slut.",
                     font=("Segoe UI", 11), bg=C["bg_card"], fg=C["text_secondary"],
                     justify="left", wraplength=700).pack(padx=20, pady=20)

            # Show model_performance from training if available
            if model_perf:
                SubHeaderLabel(self.hist_models_frame,
                              text="🏋️ Model Træningsnøjagtighed (test-data)").pack(anchor="w", pady=(10, 5))
                for mp in model_perf:
                    card = CardFrame(self.hist_models_frame)
                    card.pack(fill="x", pady=3)
                    row = StyledFrame(card, bg=C["bg_card"])
                    row.pack(fill="x")
                    name = mp.get("model_name", "?")
                    acc = mp.get("accuracy", 0) * 100
                    total_t = mp.get("total_predictions", 0)
                    correct_t = mp.get("correct_predictions", 0)
                    emoji = "🟢" if acc >= 50 else "🟡" if acc >= 35 else "🔴"
                    tk.Label(row, text=f"{emoji} {name}", font=("Segoe UI", 12, "bold"),
                             bg=C["bg_card"], fg=C["text_primary"], width=20, anchor="w").pack(side="left")
                    tk.Label(row, text=f"{acc:.1f}%", font=("Segoe UI", 14, "bold"),
                             bg=C["bg_card"],
                             fg=C["accent_green"] if acc >= 50 else C["accent_yellow"]).pack(side="left", padx=10)
                    tk.Label(row, text=f"({correct_t}/{total_t} på test-data)",
                             font=("Segoe UI", 9), bg=C["bg_card"], fg=C["text_muted"]).pack(side="left")

            self.status_bar.set_status("📜 Historik indlæst (ingen verificerede endnu)")
            return

        # ── Model accuracy bars ──
        for model in sorted(by_model, key=lambda m: m.get("correct", 0) / max(m.get("total", 1), 1), reverse=True):
            card = CardFrame(self.hist_models_frame)
            card.pack(fill="x", pady=3)

            row = StyledFrame(card, bg=C["bg_card"])
            row.pack(fill="x")

            name = model.get("model_name", "?")
            total = model.get("total", 0)
            correct = model.get("correct", 0)
            acc = (correct / total * 100) if total > 0 else 0
            avg_conf = model.get("avg_confidence", 0)
            if avg_conf:
                avg_conf *= 100

            emoji = "🟢" if acc >= 50 else "🟡" if acc >= 35 else "🔴"
            color = C["accent_green"] if acc >= 50 else C["accent_yellow"] if acc >= 35 else C["accent_red"]

            tk.Label(row, text=f"{emoji} {name}", font=("Segoe UI", 12, "bold"),
                     bg=C["bg_card"], fg=C["text_primary"], width=20, anchor="w").pack(side="left")
            tk.Label(row, text=f"{acc:.1f}%", font=("Segoe UI", 16, "bold"),
                     bg=C["bg_card"], fg=color).pack(side="left", padx=10)
            tk.Label(row, text=f"{correct}/{total} korrekte", font=("Segoe UI", 10),
                     bg=C["bg_card"], fg=C["text_secondary"]).pack(side="left", padx=10)
            if avg_conf:
                tk.Label(row, text=f"(avg konfidens: {avg_conf:.0f}%)", font=("Segoe UI", 9),
                         bg=C["bg_card"], fg=C["text_muted"]).pack(side="left")

            # Progress bar
            bar_frame = StyledFrame(card, bg=C["bg_card"])
            bar_frame.pack(fill="x", pady=(5, 0))
            bar_canvas = tk.Canvas(bar_frame, height=10, bg=C["bg_dark"], highlightthickness=0)
            bar_canvas.pack(fill="x")
            bar_canvas.update_idletasks()
            w = max(bar_canvas.winfo_width(), 400)
            filled = int(w * acc / 100)
            bar_canvas.create_rectangle(0, 0, filled, 10, fill=color, outline="")

        # ── League accuracy ──
        for league in sorted(by_league, key=lambda l: l.get("correct", 0) / max(l.get("total", 1), 1), reverse=True):
            lc = league.get("league_code", "?")
            league_info = LEAGUES.get(lc, {})
            league_name = league_info.get("name", lc)
            league_emoji = league_info.get("emoji", "🏟️")
            total = league.get("total", 0)
            correct = league.get("correct", 0)
            acc = (correct / total * 100) if total > 0 else 0
            color = C["accent_green"] if acc >= 50 else C["accent_yellow"] if acc >= 35 else C["accent_red"]

            card = CardFrame(self.hist_leagues_frame)
            card.pack(fill="x", pady=2)
            row = StyledFrame(card, bg=C["bg_card"])
            row.pack(fill="x")

            tk.Label(row, text=f"{league_emoji} {league_name}", font=("Segoe UI", 11, "bold"),
                     bg=C["bg_card"], fg=C["text_primary"], width=25, anchor="w").pack(side="left")
            tk.Label(row, text=f"{acc:.1f}%", font=("Segoe UI", 13, "bold"),
                     bg=C["bg_card"], fg=color).pack(side="left", padx=10)
            tk.Label(row, text=f"({correct}/{total})", font=("Segoe UI", 9),
                     bg=C["bg_card"], fg=C["text_muted"]).pack(side="left")

        # ── Recent verified predictions ──
        if recent:
            # Table header
            hdr = CardFrame(self.hist_recent_frame)
            hdr.pack(fill="x", pady=(0, 2))
            hdr_row = StyledFrame(hdr, bg=C["bg_card"])
            hdr_row.pack(fill="x")
            for txt, w in [("Kamp", 28), ("Prediction", 12), ("Resultat", 8), ("✓/✗", 4)]:
                tk.Label(hdr_row, text=txt, font=("Segoe UI", 9, "bold"),
                         bg=C["bg_card"], fg=C["text_muted"], width=w, anchor="w").pack(side="left")

            for pred in recent[:50]:
                card = CardFrame(self.hist_recent_frame)
                card.pack(fill="x", pady=1)
                row = StyledFrame(card, bg=C["bg_card"])
                row.pack(fill="x")

                home = pred.get("home_team", "?")
                away = pred.get("away_team", "?")
                hs = pred.get("home_score", "?")
                aws = pred.get("away_score", "?")
                predicted = pred.get("predicted_outcome", "?")
                is_correct = pred.get("is_correct", 0)

                outcome_map = {"HOME_WIN": "Hjemme", "DRAW": "Uafgjort", "AWAY_WIN": "Ude"}
                pred_text = outcome_map.get(predicted, predicted)

                match_text = f"{home} vs {away}"
                result_text = f"{hs}-{aws}"
                check = "✅" if is_correct else "❌"
                check_color = C["accent_green"] if is_correct else C["accent_red"]

                tk.Label(row, text=match_text, font=("Segoe UI", 10),
                         bg=C["bg_card"], fg=C["text_primary"], width=28, anchor="w").pack(side="left")
                tk.Label(row, text=pred_text, font=("Segoe UI", 10),
                         bg=C["bg_card"], fg=C["accent"], width=12, anchor="w").pack(side="left")
                tk.Label(row, text=result_text, font=("Segoe UI", 10, "bold"),
                         bg=C["bg_card"], fg=C["text_primary"], width=8, anchor="w").pack(side="left")
                tk.Label(row, text=check, font=("Segoe UI", 12),
                         bg=C["bg_card"], fg=check_color, width=4).pack(side="left")

        self.hist_status.config(text=f"✅ {total_preds} predictions verificeret — {overall_acc:.1f}% korrekte")
        self.status_bar.set_status(f"📜 Historik: {overall_acc:.1f}% nøjagtighed ({total_correct}/{total_preds})")

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

        tk.Label(api_card, text="🔑 API Nøgler (VALGFRIT - appen virker uden!)",
                font=("Segoe UI", 13, "bold"),
                bg=C["bg_card"], fg=C["accent"]).pack(anchor="w", pady=(0, 5))

        tk.Label(api_card, text="✅ Appen bruger gratis API'er (ESPN, TheSportsDB, OpenLigaDB) uden registrering.\n"
                 "      API nøgler herunder er helt valgfrie og kun for ekstra data.",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["accent_green"]).pack(anchor="w", pady=(0, 10))

        tk.Label(api_card, text="Football-Data.org API Key (valgfrit):",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["text_secondary"]).pack(anchor="w")
        self.fd_api_entry = tk.Entry(api_card, font=("Consolas", 10), width=50,
                                      bg=C["bg_dark"], fg=C["text_primary"],
                                      insertbackground=C["text_primary"])
        self.fd_api_entry.pack(fill="x", pady=(2, 10))
        MutedLabel(api_card, text="Valgfrit: https://www.football-data.org/client/register").pack(anchor="w")

        tk.Label(api_card, text="API-Football Key (valgfrit):",
                font=("Segoe UI", 10), bg=C["bg_card"],
                fg=C["text_secondary"]).pack(anchor="w", pady=(10, 0))
        self.af_api_entry = tk.Entry(api_card, font=("Consolas", 10), width=50,
                                      bg=C["bg_dark"], fg=C["text_primary"],
                                      insertbackground=C["text_primary"])
        self.af_api_entry.pack(fill="x", pady=(2, 10))
        MutedLabel(api_card, text="Valgfrit: https://dashboard.api-football.com/register").pack(anchor="w")

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
