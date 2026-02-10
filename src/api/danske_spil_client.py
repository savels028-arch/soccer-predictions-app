"""
Danske Spil Oddset Scraper - Henter odds fra Danske Spil via Kambi API.

Danske Spil bruger Kambi som odds-provider.  Kambi's offentlige offering-API
leverer JSON med alle tilgængelige events og bet-offers (1X2, O/U, BTTS, m.m.)

INGEN API-nøgle nødvendig – data hentes fra den offentlige Kambi CDN.

Struktur:
  Base: https://eu-offering-api.kambicdn.com/offering/v2018/dkdli/
  Endpoints:
    listView/football.json                        – alle fodboldkampe
    listView/football/england/premier_league.json – specifik liga
    betoffer/event/{eventId}.json                  – odds for enkelt kamp
"""
import requests
import logging
import time
import re
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class DanskeSpilClient:
    """
    Henter aktuelle odds fra Danske Spil (Oddset) via Kambi's offentlige API.
    Matcher holdnavne med appens predictions for at vise, hvilke spil der
    kan placeres hos Danske Spil.
    """

    BASE_URL = "https://eu-offering-api.kambicdn.com/offering/v2018"

    # Operator codes – prøv Danske Spil først, fallback til andre DK-operatører
    OPERATORS = ["dkdli", "ubdk"]

    # Kambi league-paths for de ligaer vi supporterer
    LEAGUE_PATHS = {
        "PL":  "football/england/premier_league",
        "PD":  "football/spain/la_liga",
        "BL1": "football/germany/bundesliga",
        "SA":  "football/italy/serie_a",
        "FL1": "football/france/ligue_1",
        "CL":  "football/champions_league",
        "EL":  "football/europa_league",
        "DED": "football/netherlands/eredivisie",
        "PPL": "football/portugal/primeira_liga",
        "BSA": "football/brazil/serie_a",
    }

    # Alternativ: hent ALLE fodboldkampe i ét kald
    ALL_FOOTBALL_PATH = "football"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://danskespil.dk/",
            "Origin": "https://danskespil.dk",
        })
        self._last_request = 0.0
        self._min_interval = 2.5     # pæn rate-limit – Kambi er streng
        self._retry_after = 0.0      # ekstra pause efter 429
        self._active_operator = None  # caches den operator der virker

    # ──────────────────────────────────────────────
    #  Rate limiting & HTTP
    # ──────────────────────────────────────────────
    def _rate_limit(self):
        now = time.time()
        # Respektér 429 retry-after
        if now < self._retry_after:
            wait = self._retry_after - now
            logger.debug("Venter %.1fs (429 retry-after)", wait)
            time.sleep(wait)

        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _get_json(self, path: str, params: dict = None) -> Optional[Dict]:
        """GET JSON fra Kambi API med rate-limit, operator-fallback og retry."""
        default_params = {"lang": "da_DK", "market": "DK", "useCombined": "true"}
        if params:
            default_params.update(params)

        # Bestem hvilke operators vi prøver
        if self._active_operator:
            operators_to_try = [self._active_operator]
        else:
            operators_to_try = list(self.OPERATORS)

        for operator in operators_to_try:
            url = f"{self.BASE_URL}/{operator}/listView/{path}.json"

            for attempt in range(2):
                self._rate_limit()
                try:
                    resp = self.session.get(url, params=default_params, timeout=15)
                    if resp.status_code == 200:
                        self._active_operator = operator
                        if operator != self.OPERATORS[0]:
                            logger.info("Bruger fallback-operator '%s' (Danske Spil rate-limited)", operator)
                        return resp.json()
                    elif resp.status_code == 429:
                        retry = min(int(resp.headers.get("Retry-After", 5)), 10)
                        logger.warning("Kambi 429 for '%s' – prøver næste operator", operator)
                        time.sleep(retry)
                        break  # prøv næste operator
                    else:
                        logger.warning("Kambi HTTP %d for %s (%s)", resp.status_code, path, operator)
                        break
                except Exception as exc:
                    logger.error("Kambi request fejl for %s (%s): %s", path, operator, exc)
                    time.sleep(3)
                    break

        return None

    # ──────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────
    def get_all_football_odds(self) -> List[Dict]:
        """
        Hent alle tilgængelige fodboldkampe med 1X2-odds fra Danske Spil.
        Returnerer liste af normaliserede kamp-dicts.
        """
        data = self._get_json(self.ALL_FOOTBALL_PATH)
        if not data:
            # Fallback: hent liga for liga
            return self._fetch_by_league()
        return self._parse_events(data)

    def get_league_odds(self, league_code: str) -> List[Dict]:
        """Hent odds for en specifik liga."""
        path = self.LEAGUE_PATHS.get(league_code)
        if not path:
            logger.warning("Ukendt liga-kode for Danske Spil: %s", league_code)
            return []
        data = self._get_json(path)
        if not data:
            return []
        events = self._parse_events(data)
        for e in events:
            e["league_code"] = league_code
        return events

    def get_todays_odds(self) -> List[Dict]:
        """Hent alle kampe der spilles i dag med odds."""
        all_events = self.get_all_football_odds()
        today = date.today().isoformat()
        return [e for e in all_events if e.get("match_date", "").startswith(today)]

    def get_upcoming_odds(self, days: int = 3) -> List[Dict]:
        """Hent kampe de næste N dage med odds."""
        all_events = self.get_all_football_odds()
        cutoff = (datetime.now() + timedelta(days=days)).isoformat()
        now = datetime.now().isoformat()
        return [
            e for e in all_events
            if now <= e.get("match_date", "") <= cutoff
        ]

    # ──────────────────────────────────────────────
    #  MATCHING – krydsreferér med appens predictions
    # ──────────────────────────────────────────────
    def match_predictions_with_odds(
        self,
        predictions: List[Dict],
        danske_spil_events: List[Dict] = None,
    ) -> List[Dict]:
        """
        Match appens predictions med Danske Spil odds.

        Args:
            predictions: Liste af prediction-dicts fra PredictionEngine
                         (skal have 'home_team' og 'away_team')
            danske_spil_events: Valgfri – allerede hentede DS-events.
                                Hvis None, hentes de automatisk.

        Returns:
            Liste af dicts med prediction + DS odds info.
            Hver dict har feltet 'danske_spil': {...} hvis kampen
            er tilgængelig hos Danske Spil, ellers None.
        """
        if danske_spil_events is None:
            danske_spil_events = self.get_all_football_odds()

        # Byg lookup-index over Danske Spil events (normaliserede navne)
        ds_index: Dict[str, Dict] = {}
        for ev in danske_spil_events:
            key = self._match_key(ev.get("home_team", ""), ev.get("away_team", ""))
            if key:
                ds_index[key] = ev

        results = []
        for pred in predictions:
            home = pred.get("home_team", "") or pred.get("home_team_name", "")
            away = pred.get("away_team", "") or pred.get("away_team_name", "")
            key = self._match_key(home, away)

            # Prøv direkte match
            ds_match = ds_index.get(key)

            # Prøv fuzzy match hvis direkte match fejler
            if not ds_match and key:
                ds_match = self._fuzzy_find(key, ds_index)

            result = dict(pred)
            if ds_match:
                result["danske_spil"] = {
                    "available": True,
                    "event_id": ds_match.get("event_id"),
                    "home_odds": ds_match.get("home_odds"),
                    "draw_odds": ds_match.get("draw_odds"),
                    "away_odds": ds_match.get("away_odds"),
                    "over_25_odds": ds_match.get("over_25_odds"),
                    "under_25_odds": ds_match.get("under_25_odds"),
                    "btts_yes_odds": ds_match.get("btts_yes_odds"),
                    "btts_no_odds": ds_match.get("btts_no_odds"),
                    "ds_home_team": ds_match.get("home_team"),
                    "ds_away_team": ds_match.get("away_team"),
                    "league": ds_match.get("league"),
                    "kickoff": ds_match.get("match_date"),
                    "deeplink": ds_match.get("deeplink"),
                }
            else:
                result["danske_spil"] = {"available": False}
            results.append(result)

        available = sum(1 for r in results if r["danske_spil"]["available"])
        logger.info(
            "Danske Spil matching: %d/%d predictions tilgængelige",
            available, len(results),
        )
        return results

    # ──────────────────────────────────────────────
    #  PARSING
    # ──────────────────────────────────────────────
    def _parse_events(self, data: Dict) -> List[Dict]:
        """Parse Kambi API response til normaliserede event-dicts."""
        events = []
        raw_events = data.get("events", [])

        for item in raw_events:
            try:
                event_data = item.get("event", {})
                bet_offers = item.get("betOffers", [])

                if not event_data:
                    continue

                # Event info
                event_id = event_data.get("id")
                name = event_data.get("name", "")           # "Arsenal FC - Chelsea FC"
                english_name = event_data.get("englishName", name)
                start = event_data.get("start", "")          # ISO datetime
                state = event_data.get("state", "")          # NOT_STARTED, STARTED, FINISHED
                group = event_data.get("group", "")          # liga-navn

                # Parse holdnavne fra "Home - Away"
                home_team, away_team = self._parse_team_names(name)
                if not home_team or not away_team:
                    home_team, away_team = self._parse_team_names(english_name)
                if not home_team:
                    continue

                # Parse liga fra path
                paths = event_data.get("path", [])
                league_name = ""
                league_code = ""
                if paths:
                    for p in paths:
                        pname = p.get("name", "")
                        if pname and pname != "Fodbold" and pname != "Football":
                            league_name = pname
                    # Prøv at mappe til vores liga-kode
                    league_code = self._league_name_to_code(league_name)

                # Parse 1X2 odds fra betOffers
                home_odds = draw_odds = away_odds = None
                over_25_odds = under_25_odds = None
                btts_yes_odds = btts_no_odds = None

                for offer in bet_offers:
                    criterion = offer.get("criterion", {})
                    offer_label = criterion.get("label", "").lower()
                    bet_type = offer.get("betOfferType", {})
                    bet_type_name = bet_type.get("name", "").lower()
                    bet_type_id = bet_type.get("id", 0)

                    # 1X2 (Resultat / Match / Fuldtid / Kamp) — also match betOfferType id 2
                    if offer_label in ("resultat", "match", "fuld tid", "fuldtid", "kamp") or bet_type_id == 2:
                        outcomes = offer.get("outcomes", [])
                        for oc in outcomes:
                            label = oc.get("label", "").strip()
                            # Kambi odds er i milliodds (1850 = 1.85)
                            odds_val = oc.get("odds")
                            if odds_val:
                                odds_decimal = odds_val / 1000.0
                            else:
                                odds_decimal = None

                            if label == "1":
                                home_odds = odds_decimal
                            elif label == "X":
                                draw_odds = odds_decimal
                            elif label == "2":
                                away_odds = odds_decimal

                    # Over/Under 2.5
                    elif "over/under" in offer_label or "mål" in offer_label:
                        for oc in offer.get("outcomes", []):
                            label = oc.get("label", "").lower()
                            odds_val = oc.get("odds")
                            odds_decimal = odds_val / 1000.0 if odds_val else None
                            if "over" in label:
                                over_25_odds = odds_decimal
                            elif "under" in label:
                                under_25_odds = odds_decimal

                    # BTTS (Begge hold scorer)
                    elif "begge" in offer_label or "btts" in offer_label:
                        for oc in offer.get("outcomes", []):
                            label = oc.get("label", "").lower()
                            odds_val = oc.get("odds")
                            odds_decimal = odds_val / 1000.0 if odds_val else None
                            if label in ("ja", "yes"):
                                btts_yes_odds = odds_decimal
                            elif label in ("nej", "no"):
                                btts_no_odds = odds_decimal

                # Byg deeplink til Danske Spil
                deeplink = None
                if event_id:
                    slug = english_name.lower().replace(" ", "-").replace(".", "")
                    deeplink = f"https://danskespil.dk/oddset/event/{event_id}/{slug}"

                event_dict = {
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_date": start,
                    "state": state,
                    "league": league_name,
                    "league_code": league_code,
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "over_25_odds": over_25_odds,
                    "under_25_odds": under_25_odds,
                    "btts_yes_odds": btts_yes_odds,
                    "btts_no_odds": btts_no_odds,
                    "deeplink": deeplink,
                    "source": "danske_spil",
                }
                events.append(event_dict)

            except Exception as exc:
                logger.debug("Parse error for event: %s", exc)
                continue

        logger.info("Parsede %d events fra Kambi/Danske Spil", len(events))
        return events

    def _fetch_by_league(self) -> List[Dict]:
        """Fallback: hent liga for liga i stedet for alle på én gang."""
        all_events = []
        for code, path in self.LEAGUE_PATHS.items():
            events = self.get_league_odds(code)
            all_events.extend(events)
        return all_events

    # ──────────────────────────────────────────────
    #  TEAM NAME HELPERS
    # ──────────────────────────────────────────────
    @staticmethod
    def _parse_team_names(name: str):
        """Parse 'Home - Away' eller 'Home vs Away' til (home, away)."""
        if not name:
            return "", ""
        # Kambi bruger " - " som separator
        for sep in (" - ", " vs ", " v "):
            if sep in name:
                parts = name.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        return "", ""

    def _match_key(self, home: str, away: str) -> str:
        """Normaliseret nøgle til matching."""
        h, a = self._norm(home), self._norm(away)
        return f"{h}_vs_{a}" if h and a else ""

    def _norm(self, name: str) -> str:
        """Normalisér holdnavn for cross-source matching."""
        if not name:
            return ""
        n = name.strip().lower()

        # Strip common suffixes
        for sfx in (" fc", " sc", " cf", " bc", " fk", " sk", " afc",
                     " calcio", " sad", " sfc", " ssd", " kv",
                     " bk", " if", " ff", " gf", " boldklub",
                     " 1846", " 1848", " 1909"):
            if n.endswith(sfx):
                n = n[: -len(sfx)].strip()

        # Strip common prefixes
        for pfx in ("fc ", "sc ", "fk ", "sk ", "ac ", "as ", "us ",
                     "ca ", "cd ", "rcd ", "tsg ", "1899 ", "1. "):
            if n.startswith(pfx):
                n = n[len(pfx):].strip()

        # Known aliases (Kambi-navne → normaliserede)
        aliases = {
            "as roma": "roma", "ac milan": "milan",
            "fc barcelona": "barcelona", "fc porto": "porto",
            "sporting cp": "sporting", "atalanta bc": "atalanta",
            "atalanta bergamo": "atalanta",
            "villarreal cf": "villarreal",
            "rb leipzig": "leipzig", "rasenballsport leipzig": "leipzig",
            "bayern munich": "bayern", "bayern münchen": "bayern",
            "fc bayern münchen": "bayern", "fc bayern munich": "bayern",
            "bayern munchen": "bayern",
            "borussia dortmund": "dortmund", "bor. dortmund": "dortmund",
            "borussia mönchengladbach": "gladbach", "bor. m'gladbach": "gladbach",
            "borussia monchengladbach": "gladbach",
            "tsg 1899 hoffenheim": "hoffenheim", "tsg hoffenheim": "hoffenheim",
            "1899 hoffenheim": "hoffenheim",
            "paris saint-germain": "psg", "paris sg": "psg", "paris saint germain": "psg",
            "atletico madrid": "atletico", "atlético de madrid": "atletico",
            "atlético madrid": "atletico", "atletico de madrid": "atletico",
            "inter milan": "inter", "inter milano": "inter", "internazionale": "inter",
            "ssc napoli": "napoli", "ssc neapel": "napoli",
            "manchester united": "man united", "manchester city": "man city",
            "tottenham hotspur": "tottenham", "wolverhampton wanderers": "wolves",
            "wolverhampton": "wolves",
            "brighton & hove albion": "brighton", "brighton and hove albion": "brighton",
            "sheffield utd": "sheffield united", "sheff utd": "sheffield united",
            "newcastle united": "newcastle",
            "west ham united": "west ham",
            "nottingham forest": "nott forest",
            "real sociedad de fútbol": "real sociedad",
            "real betis balompié": "real betis",
        }

        if n in aliases:
            n = aliases[n]

        # Transliterate accented chars
        n = (n.replace("ç", "c").replace("ã", "a").replace("é", "e")
              .replace("ü", "u").replace("ş", "s").replace("ı", "i")
              .replace("ö", "o").replace("ë", "e").replace("ñ", "n")
              .replace("á", "a").replace("í", "i").replace("ó", "o")
              .replace("ú", "u").replace("ý", "y"))

        # Check aliases again after transliteration
        if n in aliases:
            n = aliases[n]

        n = re.sub(r"[^\w\s]", "", n)
        return re.sub(r"\s+", " ", n).strip()

    def _fuzzy_find(self, key: str, index: Dict[str, Dict]) -> Optional[Dict]:
        """Fuzzy-match en prediction-key mod Danske Spil-index."""
        parts = key.split("_vs_")
        if len(parts) != 2:
            return None
        pred_home, pred_away = parts

        best_match = None
        best_score = 0

        for ds_key, ds_event in index.items():
            ds_parts = ds_key.split("_vs_")
            if len(ds_parts) != 2:
                continue
            ds_home, ds_away = ds_parts

            score = 0
            # Check home team
            if pred_home == ds_home:
                score += 2
            elif pred_home in ds_home or ds_home in pred_home:
                score += 1
            elif self._common_words(pred_home, ds_home):
                score += 1

            # Check away team
            if pred_away == ds_away:
                score += 2
            elif pred_away in ds_away or ds_away in pred_away:
                score += 1
            elif self._common_words(pred_away, ds_away):
                score += 1

            if score > best_score and score >= 2:
                best_score = score
                best_match = ds_event

        return best_match

    @staticmethod
    def _common_words(a: str, b: str) -> bool:
        """Tjek om to hold deler et 'signifikant' ord (>3 chars)."""
        words_a = {w for w in a.split() if len(w) > 3}
        words_b = {w for w in b.split() if len(w) > 3}
        return bool(words_a & words_b)

    def _league_name_to_code(self, league_name: str) -> str:
        """Map Kambi liga-navn til vores liga-kode."""
        if not league_name:
            return ""
        ln = league_name.lower()
        mapping = {
            "premier league": "PL",
            "la liga": "PD",
            "bundesliga": "BL1",
            "serie a": "SA",
            "ligue 1": "FL1",
            "champions league": "CL",
            "europa league": "EL",
            "eredivisie": "DED",
            "primeira liga": "PPL",
            "série a": "BSA",
            "superligaen": "DK_SL",
            "1st division": "DK_1D",
        }
        for key, code in mapping.items():
            if key in ln:
                return code
        return ""

    # ──────────────────────────────────────────────
    #  UTILITY
    # ──────────────────────────────────────────────
    def get_status(self) -> Dict[str, Any]:
        """Tjek om Kambi API'et er tilgængeligt."""
        operator = self._active_operator or self.OPERATORS[0]
        try:
            self._rate_limit()
            resp = self.session.head(
                f"{self.BASE_URL}/{operator}/listView/football.json",
                params={"lang": "da_DK", "market": "DK"},
                timeout=8,
            )
            return {
                "available": resp.status_code == 200,
                "status_code": resp.status_code,
                "source": f"Kambi / Danske Spil ({operator})",
            }
        except Exception as exc:
            return {"available": False, "error": str(exc), "source": "Kambi / Danske Spil"}
