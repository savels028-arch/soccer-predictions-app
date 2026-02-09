"""
AI Prediction Scraper - Henter forudsigelser fra eksterne AI-sider
Scraper data fra 5 gratis AI fodbold-prediction-sider:
  1. AI-Goalie.com           - Win%, odds, team logos → alt attrs
  2. BetsWithBots.com        - 1X2 %, predicted score, best odds
  3. SoccerTips.ai            - 1X2%, BTTS, Over/Under, H2H
  4. FootballPredictions.ai  - 1X2 tip + schema.org JSON-LD
  5. OddAlerts.com           - AI football predictions page

NO API KEYS REQUIRED - all data from public web pages.
"""
import requests
import re
import json
import logging
import time
from datetime import date
from typing import Optional, List, Dict, Any
from html import unescape

logger = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    """Fast HTML tag removal."""
    return re.sub(r'<[^>]+>', ' ', html)


class PredictionScraper:
    """
    Scrapes AI football predictions from multiple free websites.
    Combines predictions from 5 working sources for consensus analysis.
    """

    SOURCES = [
        "ai-goalie.com",
        "betswithbots.com",
        "soccertips.ai",
        "footballpredictions.ai",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_request = 0.0
        self._min_interval = 1.2  # polite rate-limit

    # ──────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────
    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _safe_get(self, url: str, timeout: int = 20,
                  extra_headers: dict = None) -> Optional[str]:
        """HTTP GET with rate-limit, returns text or None."""
        self._rate_limit()
        try:
            headers = dict(self.session.headers)
            if extra_headers:
                headers.update(extra_headers)
            resp = self.session.get(url, timeout=timeout, headers=headers)
            if resp.status_code == 200:
                return resp.text
            logger.warning("HTTP %d from %s", resp.status_code, url)
        except Exception as exc:
            logger.error("Request failed for %s: %s", url, exc)
        return None

    # ──────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────
    def get_all_predictions(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Fetch AI predictions from all available sources.
        Returns flat list of prediction dicts, each tagged with 'source'.
        """
        all_preds: List[Dict] = []
        scrapers = [
            ("ai-goalie.com",          self._scrape_aigoalie),
            ("betswithbots.com",       self._scrape_betswithbots),
            ("soccertips.ai",          self._scrape_soccertips),
            ("footballpredictions.ai", self._scrape_footballpredictions),
        ]
        for src, fn in scrapers:
            try:
                preds = fn(target_date) or []
                for p in preds:
                    p["source"] = src
                all_preds.extend(preds)
                logger.info("[OK] %s: %d predictions", src, len(preds))
            except Exception as exc:
                logger.error("[ERR] %s: %s", src, exc)
        logger.info("Total: %d predictions from %d sources",
                     len(all_preds), len(scrapers))
        return all_preds

    def get_consensus_predictions(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Combine predictions from multiple AI sources into consensus.
        Groups by match, averages probabilities, votes on BTTS / O/U.
        """
        all_preds = self.get_all_predictions(target_date)
        if not all_preds:
            return []

        groups: Dict[str, Dict] = {}
        for pred in all_preds:
            key = self._match_key(pred.get("home_team", ""),
                                  pred.get("away_team", ""))
            if not key or key == "_vs_":
                continue
            if key not in groups:
                groups[key] = {
                    "home_team": pred.get("home_team", ""),
                    "away_team": pred.get("away_team", ""),
                    "league":   pred.get("league", ""),
                    "kickoff_time": pred.get("kickoff_time", ""),
                    "predictions": [],
                }
            groups[key]["predictions"].append(pred)

        consensus: List[Dict] = []
        for key, grp in groups.items():
            preds = grp["predictions"]
            n = len(preds)

            h_vals = [p["home_win_pct"] for p in preds if p.get("home_win_pct")]
            d_vals = [p["draw_pct"]     for p in preds if p.get("draw_pct")]
            a_vals = [p["away_win_pct"] for p in preds if p.get("away_win_pct")]

            avg_h = sum(h_vals) / len(h_vals) if h_vals else None
            avg_d = sum(d_vals) / len(d_vals) if d_vals else None
            avg_a = sum(a_vals) / len(a_vals) if a_vals else None

            winner = confidence = None
            if avg_h is not None and avg_a is not None:
                ad = avg_d or 0
                if avg_h >= ad and avg_h >= avg_a:
                    winner, confidence = "1", avg_h
                elif avg_a >= ad and avg_a >= avg_h:
                    winner, confidence = "2", avg_a
                else:
                    winner, confidence = "X", ad

            # BTTS voting
            btts_v = [p["btts"] for p in preds
                      if p.get("btts") and p["btts"].lower() in ("yes", "no")]
            btts_yes = sum(1 for v in btts_v if v.lower() == "yes")
            btts_con = ("Yes" if btts_yes > len(btts_v) / 2 else "No") if btts_v else None

            # Over/Under 2.5 voting
            ou_v = [p["over_under_25"] for p in preds
                    if p.get("over_under_25") and p["over_under_25"].lower() in ("over", "under")]
            ou_o = sum(1 for v in ou_v if v.lower() == "over")
            ou_con = ("Over" if ou_o > len(ou_v) / 2 else "Under") if ou_v else None

            sources = list({p.get("source", "") for p in preds})

            consensus.append({
                "home_team":  grp["home_team"],
                "away_team":  grp["away_team"],
                "league":     grp["league"],
                "kickoff_time": grp["kickoff_time"],
                "num_sources": n,
                "sources":     sources,
                "avg_home_win_pct":  round(avg_h, 1) if avg_h else None,
                "avg_draw_pct":      round(avg_d, 1) if avg_d else None,
                "avg_away_win_pct":  round(avg_a, 1) if avg_a else None,
                "consensus_winner":     winner,
                "consensus_confidence": round(confidence, 1) if confidence else None,
                "btts_consensus":       btts_con,
                "over_under_consensus": ou_con,
                "individual_predictions": preds,
            })

        consensus.sort(key=lambda x: x["num_sources"], reverse=True)
        return consensus

    # ──────────────────────────────────────────────
    #  Match-key helpers
    # ──────────────────────────────────────────────
    def _match_key(self, home: str, away: str) -> str:
        h, a = self._norm(home), self._norm(away)
        return f"{h}_vs_{a}" if h and a else ""

    def _norm(self, name: str) -> str:
        """Normalize team name for cross-source matching."""
        if not name:
            return ""
        n = name.strip().lower()
        # Strip common suffixes
        for sfx in (" fc", " sc", " cf", " bc", " fk", " sk", " afc",
                     " calcio", " sad", " sfc", " ssd", " kv", " sc braga",
                     " bk", " if", " ff", " gf", " boldklub"):
            if n.endswith(sfx):
                n = n[: -len(sfx)].strip()
        # Strip common prefixes
        for pfx in ("fc ", "sc ", "fk ", "sk ", "ac ", "as ", "us ",
                     "ca ", "cd ", "rcd ", "tsg ", "1899 "):
            if n.startswith(pfx):
                n = n[len(pfx):].strip()
        aliases = {
            "as roma": "roma", "ac milan": "milan",
            "fc barcelona": "barcelona", "fc porto": "porto",
            "sporting cp": "sporting", "atalanta bc": "atalanta",
            "atalanta bergamo": "atalanta", "us cremonese": "cremonese",
            "cagliari calcio": "cagliari",
            "rcd espanyol barcelona": "espanyol", "rcd espanyol": "espanyol",
            "villarreal cf": "villarreal",
            "middlesbrough fc": "middlesbrough",
            "sheffield utd": "sheffield united", "sheff utd": "sheffield united",
            "clermont foot 63": "clermont", "clermont foot": "clermont",
            "amiens sc": "amiens",
            "fenerbahce sk": "fenerbahce", "kasimpasa sk": "kasimpasa",
            "genclerbirligi sk": "genclerbirligi",
            "vejle boldklub": "vejle", "fc fredericia": "fredericia",
            "odense boldklub": "odense", "aarhus gf": "aarhus", "agf": "aarhus",
            "fc famalicao": "famalicao", "cd mirandes": "mirandes",
            "avs futebol": "avs", "avs futebol sad": "avs",
            "racing santander": "racing santander",
            "tsg 1899 hoffenheim": "hoffenheim", "tsg hoffenheim": "hoffenheim",
            "1899 hoffenheim": "hoffenheim",
            "rb leipzig": "leipzig", "rasenballsport leipzig": "leipzig",
            "bayern munich": "bayern", "bayern münchen": "bayern",
            "borussia dortmund": "dortmund", "bor. dortmund": "dortmund",
            "borussia mönchengladbach": "gladbach", "bor. m'gladbach": "gladbach",
        }
        n = (n.replace("ç", "c").replace("ã", "a").replace("é", "e")
              .replace("ü", "u").replace("ş", "s").replace("ı", "i")
              .replace("ö", "o").replace("ë", "e").replace("ñ", "n"))
        if n in aliases:
            n = aliases[n]
        n = re.sub(r"[^\w\s]", "", n)
        return re.sub(r"\s+", " ", n).strip()

    # ══════════════════════════════════════════════
    #  1. AI-GOALIE.COM
    # ══════════════════════════════════════════════
    def _scrape_aigoalie(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Parse HTML table rows.
        Each row has: alt="TeamA logo", alt="TeamB logo", and NN% confidence.
        Odds appear as "1.47 vs -1.92".
        """
        if target_date:
            url = f"https://ai-goalie.com/{target_date}.html"
        else:
            today = date.today().strftime("%d.%m.%Y")
            url = f"https://ai-goalie.com/{today}.html"

        html = self._safe_get(url)
        if not html:
            return []

        predictions: List[Dict] = []
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)

        for row in rows:
            try:
                # Must contain a percentage
                pct_m = re.search(r"(\d{1,3})%", row)
                if not pct_m:
                    continue
                win_pct = int(pct_m.group(1))
                if win_pct < 1 or win_pct > 100:
                    continue

                # Team names from "alt" attributes on logo images
                alts = re.findall(r'alt="([^"]+)"', row)
                team_alts = [a.replace(" logo", "").replace(" Logo", "")
                             for a in alts
                             if "logo" in a.lower() and "information" not in a.lower()]
                if len(team_alts) < 2:
                    continue
                home, away = team_alts[0].strip(), team_alts[1].strip()

                # Kickoff time
                stripped = _strip_html(row)
                time_m = re.search(r"\b(\d{1,2}:\d{2})\b", stripped)

                # Odds pair: "1.47 vs -1.92" or "1.47 vs 1.92"
                odds_m = re.search(r"(-?\d+\.\d+)\s+vs\s+(-?\d+\.\d+)", stripped)
                odds_home = float(odds_m.group(1)) if odds_m else None
                odds_away = float(odds_m.group(2)) if odds_m else None

                # Determine who the AI picked (the team whose odds appear first
                # if positive, or the team with the higher percentage)
                # AI-Goalie shows one percentage for the "selected" team.
                # We assume the first team (home) is selected if odds_home > 0
                predictions.append({
                    "home_team":    home,
                    "away_team":    away,
                    "kickoff_time": time_m.group(1) if time_m else "",
                    "home_win_pct": win_pct,
                    "away_win_pct": 100 - win_pct,
                    "draw_pct":     None,
                    "odds_home":    abs(odds_home) if odds_home else None,
                    "odds_away":    abs(odds_away) if odds_away else None,
                })
            except Exception:
                continue

        return predictions

    # ══════════════════════════════════════════════
    #  2. BETSWITHBOTS.COM
    # ══════════════════════════════════════════════
    def _scrape_betswithbots(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Match links: /result/ID/LEAGUE/Home/Away
        Nearby text contains H% D% A%, predicted score, best odds.
        """
        url = "https://www.betswithbots.com/"
        html = self._safe_get(url)
        if not html:
            return []

        predictions: List[Dict] = []
        links = re.findall(r'href="/result/(\d+)/(\d+)/([^"]+)/([^"]+)"', html)

        for match_id, league_id, home_enc, away_enc in links:
            try:
                home = requests.utils.unquote(home_enc).strip()
                away = requests.utils.unquote(away_enc).strip()

                marker = f"/result/{match_id}/{league_id}/{home_enc}/{away_enc}"
                pos = html.find(marker)
                if pos < 0:
                    continue

                ctx = _strip_html(html[pos: pos + 500])

                nums = re.findall(r"\b(\d{1,3})\b", ctx)
                pcts: List[int] = []
                for n in nums:
                    v = int(n)
                    if 1 <= v <= 99:
                        pcts.append(v)
                    if len(pcts) == 3:
                        break

                if len(pcts) != 3 or not (80 <= sum(pcts) <= 120):
                    continue

                pred_w = re.search(r"\b([1X2])\b", ctx)
                odds_m = re.search(r"(\d\.\d{2})", ctx)
                score  = re.search(r"(\d)-(\d)", ctx)

                predictions.append({
                    "home_team":        home,
                    "away_team":        away,
                    "home_win_pct":     pcts[0],
                    "draw_pct":         pcts[1],
                    "away_win_pct":     pcts[2],
                    "predicted_winner": pred_w.group(1) if pred_w else None,
                    "odds_best":        float(odds_m.group(1)) if odds_m else None,
                    "predicted_score":  f"{score.group(1)}-{score.group(2)}" if score else None,
                })
            except Exception:
                continue

        return predictions

    # ══════════════════════════════════════════════
    #  3. SOCCERTIPS.AI
    # ══════════════════════════════════════════════
    def _scrape_soccertips(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        win-draw-win tips page → list of match links with title="View match: A vs B"
        and tip value (1, X, 2, 1X, X2).
        Individual match pages have detailed percentages (H/D/A) + BTTS + O/U.
        We scrape the tips list (fast) and a few top match pages (detailed).
        """
        tips_url = "https://soccertips.ai/football-betting-tips/win-draw-win-tips/"
        html = self._safe_get(tips_url, extra_headers={"Referer": "https://www.google.com/"})
        if not html:
            return []

        predictions: List[Dict] = []

        # Extract match titles & links
        # Pattern: title="View match: TeamA vs TeamB"
        matches = re.findall(
            r'title="View match:\s*([^"]+?)\s+vs\s+([^"]+?)"',
            html, re.I,
        )
        # Extract tip values nearby
        tip_values = re.findall(
            r'class="prediction-card__tip-value"[^>]*>\s*([^<]*?)\s*<',
            html,
        )

        # Pair matches with tips (they appear in order)
        seen = set()
        for i, (home, away) in enumerate(matches):
            home, away = home.strip(), away.strip()
            key = f"{home}|{away}"
            if key in seen:
                continue
            seen.add(key)

            tip = tip_values[i].strip() if i < len(tip_values) else ""
            tip_code = None
            h_pct = d_pct = a_pct = None

            if tip in ("1", "1X"):
                tip_code = "1"
            elif tip in ("2", "X2"):
                tip_code = "2"
            elif tip == "X":
                tip_code = "X"

            predictions.append({
                "home_team":        home,
                "away_team":        away,
                "tip":              tip,
                "predicted_winner": tip_code,
                "home_win_pct":     h_pct,
                "draw_pct":         d_pct,
                "away_win_pct":     a_pct,
            })

        # Enrich top-N matches with detailed percentages from match pages
        max_detail = min(20, len(predictions))
        match_links = re.findall(
            r'href="(https://soccertips\.ai/match/[^"]+)"', html
        )
        unique_links = list(dict.fromkeys(match_links))

        for idx in range(min(max_detail, len(unique_links))):
            url = unique_links[idx]
            try:
                mhtml = self._safe_get(url, timeout=12,
                                       extra_headers={"Referer": tips_url})
                if not mhtml:
                    continue
                mtext = _strip_html(mhtml)
                mtext = unescape(mtext)

                # Find H/D/A percentages: "TeamA 26%  Draw 26%  TeamB 46%"
                pct_m = re.search(
                    r'(\w[\w\s\.\-\']+?)\s+(\d{1,3})%\s+'
                    r'Draw\s+(\d{1,3})%\s+'
                    r'(\w[\w\s\.\-\']+?)\s+(\d{1,3})%',
                    mtext,
                )
                if pct_m and idx < len(predictions):
                    predictions[idx]["home_win_pct"] = int(pct_m.group(2))
                    predictions[idx]["draw_pct"]     = int(pct_m.group(3))
                    predictions[idx]["away_win_pct"] = int(pct_m.group(5))

                # BTTS
                btts_m = re.search(r'BTTS\s+(Yes|No)', mtext, re.I)
                if btts_m and idx < len(predictions):
                    predictions[idx]["btts"] = btts_m.group(1).capitalize()

                # Over/Under 2.5
                ou_m = re.search(r'Over 2\.5 Goals.*?(\d{1,3})%', mtext)
                if ou_m and idx < len(predictions):
                    ou_pct = int(ou_m.group(1))
                    predictions[idx]["over_under_25"] = "Over" if ou_pct >= 50 else "Under"

            except Exception as exc:
                logger.debug("SoccerTips match page error: %s", exc)

        return predictions

    # ══════════════════════════════════════════════
    #  4. FOOTBALLPREDICTIONS.AI
    # ══════════════════════════════════════════════
    def _scrape_footballpredictions(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Uses multiple pages:
        - /football-predictions/1x2-predictions/ → tip 1/X/2
        - /football-predictions/btts-predictions/ → Yes/No
        - /football-predictions/over-under-predictions/ → Over/Under 2.5
        Each page has <div class="footgame"> blocks with match-team divs,
        match-tip-show, and embedded JSON-LD SportsEvent schema.
        """
        predictions: List[Dict] = []
        match_data: Dict[str, Dict] = {}  # keyed by "home|away"

        pages = [
            ("https://footballpredictions.ai/football-predictions/1x2-predictions/", "1x2"),
            ("https://footballpredictions.ai/football-predictions/btts-predictions/", "btts"),
            ("https://footballpredictions.ai/football-predictions/over-under-predictions/", "ou"),
        ]

        for page_url, page_type in pages:
            html = self._safe_get(page_url)
            if not html:
                continue

            # Split on footgame divs
            game_blocks = re.split(r'<div class="footgame[^"]*">', html)

            for block in game_blocks[1:]:  # skip before first
                try:
                    # Extract team names from match-team divs
                    teams = re.findall(
                        r'<div class="match-team">\s*([^<]+?)\s*</div>', block
                    )
                    if len(teams) < 2:
                        continue
                    home = teams[0].strip()
                    away = teams[1].strip()
                    key = f"{home}|{away}"

                    # Extract tip from match-tip-show
                    tip_m = re.search(
                        r'<div class="match-tip-show">\s*([^<]+?)\s*</div>', block
                    )
                    tip = tip_m.group(1).strip() if tip_m else ""

                    # Extract time
                    time_m = re.search(
                        r'<span class="match-date__time"[^>]*>\s*(\d{1,2}:\d{2})\s*</span>',
                        block,
                    )
                    kickoff = time_m.group(1) if time_m else ""

                    # Extract league from JSON-LD if available
                    schema_m = re.search(
                        r'"location":\s*\{[^}]*"name":\s*"([^"]+)"', block
                    )
                    league = schema_m.group(1) if schema_m else ""

                    if key not in match_data:
                        match_data[key] = {
                            "home_team":    home,
                            "away_team":    away,
                            "kickoff_time": kickoff,
                            "league":       league,
                        }

                    entry = match_data[key]

                    if page_type == "1x2":
                        entry["tip_1x2"] = tip
                        # Assign approximate pcts from tip
                        if tip == "1":
                            entry.update(home_win_pct=60, draw_pct=22, away_win_pct=18, predicted_winner="1")
                        elif tip == "2":
                            entry.update(home_win_pct=18, draw_pct=22, away_win_pct=60, predicted_winner="2")
                        elif tip == "X":
                            entry.update(home_win_pct=28, draw_pct=44, away_win_pct=28, predicted_winner="X")
                        elif tip == "1X":
                            entry.update(home_win_pct=42, draw_pct=35, away_win_pct=23, predicted_winner="1")
                        elif tip == "X2":
                            entry.update(home_win_pct=23, draw_pct=35, away_win_pct=42, predicted_winner="2")
                    elif page_type == "btts":
                        entry["btts"] = tip  # "Yes" or "No"
                    elif page_type == "ou":
                        entry["over_under_25"] = tip  # "Over" or "Under"

                except Exception:
                    continue

        # Build final list
        for key, data in match_data.items():
            predictions.append({
                "home_team":        data.get("home_team", ""),
                "away_team":        data.get("away_team", ""),
                "kickoff_time":     data.get("kickoff_time", ""),
                "league":           data.get("league", ""),
                "home_win_pct":     data.get("home_win_pct"),
                "draw_pct":         data.get("draw_pct"),
                "away_win_pct":     data.get("away_win_pct"),
                "predicted_winner": data.get("predicted_winner"),
                "tip":              data.get("tip_1x2", ""),
                "btts":             data.get("btts"),
                "over_under_25":    data.get("over_under_25"),
            })

        return predictions

    # ──────────────────────────────────────────────
    #  UTILITY
    # ──────────────────────────────────────────────
    def get_available_sources(self) -> List[str]:
        return self.SOURCES.copy()

    def get_source_status(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        urls = {
            "ai-goalie.com":          "https://ai-goalie.com/",
            "betswithbots.com":       "https://www.betswithbots.com/",
            "soccertips.ai":          "https://soccertips.ai/",
            "footballpredictions.ai": "https://footballpredictions.ai/football-predictions/",
        }
        for name, url in urls.items():
            try:
                resp = self.session.head(url, timeout=5)
                status[name] = resp.status_code in (200, 301, 302)
            except Exception:
                status[name] = False
        return status
