#!/usr/bin/env python3
"""
FlashScore Scraper - Henter live scores, resultater og odds fra FlashScore.dk
"""
import urllib.request
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class FlashScoreScraper:
    BASE_URL = "https://d.flashscore.dk/x/feed"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://www.flashscore.dk/",
        "Accept": "text/plain, */*; q=0.01",
        "X-Fsign": "SW9D1eZo",
    }

    def _fetch(self, url: str) -> Optional[str]:
        try:
            req = urllib.request.Request(url, headers=self.HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  ⚠ FlashScore fetch error: {e}")
            return None

    def fetch_todays_matches(self) -> List[Dict]:
        url = f"{self.BASE_URL}/f_1_0_2_da_1"
        raw = self._fetch(url)
        if not raw or len(raw) < 50:
            return []
        return self._parse_feed(raw)

    def fetch_matches_by_date(self, date: datetime) -> List[Dict]:
        date_str = date.strftime("%Y%m%d")
        url = f"{self.BASE_URL}/f_1_0_2_da_1_{date_str}"
        raw = self._fetch(url)
        if not raw or len(raw) < 50:
            return []
        return self._parse_feed(raw)

    def _parse_feed(self, raw: str) -> List[Dict]:
        matches = []
        lines = raw.split("~")
        current_league = ""
        current_country = ""

        for line in lines:
            parts = line.split("¬")
            data = {}
            for part in parts:
                if "÷" in part:
                    key, _, value = part.partition("÷")
                    data[key] = value

            if "ZA" in data:
                current_league = data.get("ZA", "")
                current_country = data.get("ZB", "")

            if "AA" in data and data.get("AE") and data.get("AF"):
                status_code = data.get("AB", "1")
                status_map = {
                    "1": "SCHEDULED", "2": "LIVE", "3": "FINISHED",
                    "6": "LIVE", "8": "FINISHED", "9": "FINISHED",
                    "4": "POSTPONED", "5": "CANCELLED",
                }
                home_score = self._safe_int(data.get("AG"))
                away_score = self._safe_int(data.get("AH"))

                matches.append({
                    "flashscore_id": data.get("AA", ""),
                    "homeTeam": data.get("AE", ""),
                    "awayTeam": data.get("AF", ""),
                    "homeScore": home_score,
                    "awayScore": away_score,
                    "status": status_map.get(status_code, "SCHEDULED"),
                    "league": current_league,
                    "country": current_country,
                    "minute": data.get("BJ", ""),
                    "htHomeScore": self._safe_int(data.get("BA")),
                    "htAwayScore": self._safe_int(data.get("BB")),
                    "source": "flashscore",
                })

        return matches

    def fetch_odds(self, match_id: str) -> Optional[Dict]:
        url = f"{self.BASE_URL}/do_1_{match_id}_1"
        raw = self._fetch(url)
        if not raw:
            return None
        return self._parse_odds(raw)

    def _parse_odds(self, raw: str) -> Dict:
        odds = {"bookmakers": [], "average": {}, "highest": {}}
        parts = raw.split("¬")
        current_bm = ""
        current_odds = {}

        for part in parts:
            if "÷" in part:
                key, _, value = part.partition("÷")
                if key == "OD":
                    if current_bm and current_odds:
                        odds["bookmakers"].append({"name": current_bm, "odds": current_odds.copy()})
                    current_bm = value
                    current_odds = {}
                elif key == "OE":
                    current_odds["home"] = self._safe_float(value)
                elif key == "OF":
                    current_odds["draw"] = self._safe_float(value)
                elif key == "OG":
                    current_odds["away"] = self._safe_float(value)

        if current_bm and current_odds:
            odds["bookmakers"].append({"name": current_bm, "odds": current_odds.copy()})

        if odds["bookmakers"]:
            for key in ["home", "draw", "away"]:
                vals = [b["odds"].get(key, 0) for b in odds["bookmakers"] if b["odds"].get(key)]
                if vals:
                    odds["average"][key] = round(sum(vals) / len(vals), 2)
                    odds["highest"][key] = round(max(vals), 2)

        return odds

    def _safe_int(self, val) -> Optional[int]:
        try:
            return int(val) if val else None
        except (ValueError, TypeError):
            return None

    def _safe_float(self, val) -> Optional[float]:
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None


def fetch_flashscore_data() -> Dict:
    scraper = FlashScoreScraper()
    print("  📡 Fetching FlashScore matches...")
    today = scraper.fetch_todays_matches()
    print(f"  ✅ {len(today)} today matches")

    tomorrow = scraper.fetch_matches_by_date(datetime.now() + timedelta(days=1))
    print(f"  ✅ {len(tomorrow)} tomorrow matches")

    all_matches = today + tomorrow

    # Fetch odds for top matches
    for match in all_matches[:10]:
        if match.get("flashscore_id"):
            odds = scraper.fetch_odds(match["flashscore_id"])
            if odds and odds.get("bookmakers"):
                match["flashscore_odds"] = odds

    return {"matches": all_matches, "today": today, "tomorrow": tomorrow}


if __name__ == "__main__":
    print("🏟 FlashScore Scraper")
    data = fetch_flashscore_data()
    print(f"Total: {len(data['matches'])} matches")
    for m in data["matches"][:10]:
        print(f"  {m['homeTeam']} vs {m['awayTeam']} [{m['status']}] - {m['league']}")
