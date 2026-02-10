"""
Soccer Predictions Pro - Flask Web Application
Replaces tkinter GUI with a browser-based UI.
"""
import logging
import os
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, render_template

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def create_flask_app(db_manager, data_aggregator, prediction_engine):
    """Create and configure Flask application with all API routes."""

    app = Flask(__name__,
                template_folder=str(Path(__file__).parent / "templates"))

    # ─── State ───
    matches_cache = []
    predictions_cache = {}
    ai_predictions_cache = []
    cache_lock = threading.Lock()

    # ═══════════════════════════════════════════
    # PAGES
    # ═══════════════════════════════════════════
    @app.route("/")
    def index():
        return render_template("index.html")

    # ═══════════════════════════════════════════
    # API: CONFIG
    # ═══════════════════════════════════════════
    @app.route("/api/config")
    def api_config():
        from config.settings import LEAGUES
        leagues_out = {}
        for code, info in LEAGUES.items():
            leagues_out[code] = {
                "name": info.get("name", code),
                "emoji": info.get("emoji", "🏟️"),
                "country": info.get("country", ""),
            }
        return jsonify({"leagues": leagues_out})

    # ═══════════════════════════════════════════
    # API: STATUS (sidebar info)
    # ═══════════════════════════════════════════
    @app.route("/api/status")
    def api_status():
        try:
            return jsonify({
                "match_count": db_manager.get_match_count(),
                "prediction_count": db_manager.get_prediction_count(),
                "is_trained": prediction_engine.is_trained,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: MATCHES (Dashboard)
    # ═══════════════════════════════════════════
    @app.route("/api/matches")
    def api_matches():
        nonlocal matches_cache
        try:
            force = request.args.get("force", "0") == "1"
            matches = data_aggregator.fetch_todays_matches(force_refresh=force)
            with cache_lock:
                matches_cache = matches
            return jsonify({"matches": _serialize_matches(matches)})
        except Exception as e:
            logger.error(f"matches error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/upcoming")
    def api_upcoming():
        nonlocal matches_cache
        try:
            matches = data_aggregator.fetch_upcoming_matches(days=7)
            with cache_lock:
                matches_cache.extend(matches)
            return jsonify({"matches": _serialize_matches(matches)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: LIVE
    # ═══════════════════════════════════════════
    @app.route("/api/live")
    def api_live():
        try:
            matches = data_aggregator.fetch_live_matches()
            return jsonify({"matches": _serialize_matches(matches)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: PREDICTIONS (ML models)
    # ═══════════════════════════════════════════
    @app.route("/api/predictions", methods=["POST"])
    def api_predictions():
        nonlocal predictions_cache
        try:
            with cache_lock:
                mc = list(matches_cache) if matches_cache else []

            if not mc:
                mc = data_aggregator.fetch_todays_matches()
                with cache_lock:
                    matches_cache.extend(mc)

            scheduled = [m for m in mc
                         if m.get("status") in ("SCHEDULED", "TIMED", "NOT_STARTED")]
            if not scheduled:
                scheduled = mc

            preds = prediction_engine.predict_all_matches(scheduled)
            with cache_lock:
                predictions_cache = preds

            # Serialize
            out = {}
            for match_key, model_preds in preds.items():
                out[match_key] = [_serialize_pred(p) for p in model_preds]
            return jsonify({"predictions": out})
        except Exception as e:
            logger.error(f"predictions error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: AI SITES
    # ═══════════════════════════════════════════
    @app.route("/api/ai_predictions")
    def api_ai_predictions():
        nonlocal ai_predictions_cache
        try:
            consensus = data_aggregator.fetch_ai_predictions(force_refresh=True)
            with cache_lock:
                ai_predictions_cache = consensus
            return jsonify({"consensus": consensus})
        except Exception as e:
            # If caching failed but we got consensus data in memory, still return it
            with cache_lock:
                if ai_predictions_cache:
                    logger.warning(f"ai_predictions cache error (using in-memory): {e}")
                    return jsonify({"consensus": ai_predictions_cache})
            logger.error(f"ai_predictions error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: CONSENSUS + DANSKE SPIL
    # ═══════════════════════════════════════════
    @app.route("/api/consensus_danske_spil")
    def api_consensus_danske_spil():
        try:
            with cache_lock:
                mc = list(matches_cache) if matches_cache else None
            result = data_aggregator.build_consensus_with_danske_spil(
                prediction_engine=prediction_engine,
                matches=mc,
                force_refresh=True,
            )
            return jsonify(result)
        except Exception as e:
            logger.error(f"consensus_danske_spil error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: COMPARISON
    # ═══════════════════════════════════════════
    @app.route("/api/comparison")
    def api_comparison():
        try:
            perfs = prediction_engine.get_model_comparison()
            return jsonify({"performances": perfs})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: SUGGESTIONS
    # ═══════════════════════════════════════════
    @app.route("/api/suggestions")
    def api_suggestions():
        try:
            with cache_lock:
                preds = dict(predictions_cache)

            if not preds:
                # Generate first
                mc = list(matches_cache) if matches_cache else data_aggregator.fetch_todays_matches()
                scheduled = [m for m in mc
                             if m.get("status") in ("SCHEDULED", "TIMED", "NOT_STARTED")]
                if not scheduled:
                    scheduled = mc
                preds = prediction_engine.predict_all_matches(scheduled)
                with cache_lock:
                    predictions_cache.update(preds)

            suggestions = []
            for match_key, model_preds in preds.items():
                for p in model_preds:
                    if p.get("model_name") == "ensemble" and p.get("value_rating", 0) > 0.05:
                        suggestions.append({**_serialize_pred(p), "match_key": match_key})

            suggestions.sort(key=lambda x: x.get("value_rating", 0), reverse=True)
            return jsonify({"suggestions": suggestions})
        except Exception as e:
            logger.error(f"suggestions error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: HISTORY
    # ═══════════════════════════════════════════
    _last_history_update = {"time": None, "saved": 0}

    def _update_history_results():
        """Scan today's finished matches and save prediction results to DB."""
        nonlocal ai_predictions_cache
        saved = 0
        try:
            matches = data_aggregator.fetch_todays_matches(force_refresh=False)
            # Also try to get AI predictions (use cache, don't force)
            ai_preds = []
            try:
                ai_preds = data_aggregator.fetch_ai_predictions(force_refresh=False)
            except Exception:
                pass
            with cache_lock:
                if ai_preds:
                    ai_predictions_cache = ai_preds
                local_ai = list(ai_predictions_cache)

            # Build AI index
            ai_idx = {}
            for ai in local_ai:
                h = (ai.get("home_team") or "").lower().strip()
                a = (ai.get("away_team") or "").lower().strip()
                if h and a:
                    ai_idx[f"{h}|{a}"] = ai
                    h_short = h.split()[0] if h else ""
                    a_short = a.split()[0] if a else ""
                    if h_short and a_short:
                        ai_idx[f"{h_short}|{a_short}"] = ai

            for m in matches:
                if m.get("status") != "FINISHED":
                    continue
                hs = m.get("home_score")
                aws = m.get("away_score")
                if hs is None or aws is None:
                    continue
                if hs > aws:
                    actual = "HOME_WIN"
                elif hs < aws:
                    actual = "AWAY_WIN"
                else:
                    actual = "DRAW"

                home = m.get("home_team_name", "")
                away = m.get("away_team_name", "")
                pred_info = None

                # Source 1: DB predictions
                try:
                    preds = db_manager.get_predictions_by_teams(home, away)
                    if preds:
                        ens = next((p for p in preds if p["model_name"] == "ensemble"), None)
                        best = ens or preds[0]
                        predicted = best.get("predicted_outcome")
                        if predicted:
                            pred_info = {
                                "predicted_outcome": predicted,
                                "confidence": best.get("confidence", 0),
                                "source": best.get("model_name", "ML"),
                                "home_win_prob": best.get("home_win_prob"),
                                "draw_prob": best.get("draw_prob"),
                                "away_win_prob": best.get("away_win_prob"),
                            }
                except Exception:
                    pass

                # Source 2: AI consensus
                if not pred_info and ai_idx:
                    h_low = home.lower().strip()
                    a_low = away.lower().strip()
                    ai_match = ai_idx.get(f"{h_low}|{a_low}")
                    if not ai_match:
                        h_short = h_low.split()[0] if h_low else ""
                        a_short = a_low.split()[0] if a_low else ""
                        if h_short and a_short:
                            ai_match = ai_idx.get(f"{h_short}|{a_short}")
                    if not ai_match:
                        for k, v in ai_idx.items():
                            kh, ka = k.split("|", 1)
                            if (kh in h_low or h_low in kh) and (ka in a_low or a_low in ka):
                                ai_match = v
                                break
                    if ai_match:
                        winner = ai_match.get("consensus_winner")
                        if winner:
                            predicted = {"1": "HOME_WIN", "X": "DRAW", "2": "AWAY_WIN"}.get(winner, winner)
                            conf = ai_match.get("consensus_confidence")
                            conf_val = (conf / 100.0) if conf and conf > 1 else conf
                            pred_info = {
                                "predicted_outcome": predicted,
                                "confidence": conf_val,
                                "source": f"AI Sites ({ai_match.get('num_sources', '?')} kilder)",
                                "home_win_prob": (ai_match.get("avg_home_win_pct") or 0) / 100.0 if (ai_match.get("avg_home_win_pct") or 0) > 1 else ai_match.get("avg_home_win_pct"),
                                "draw_prob": (ai_match.get("avg_draw_pct") or 0) / 100.0 if (ai_match.get("avg_draw_pct") or 0) > 1 else ai_match.get("avg_draw_pct"),
                                "away_win_prob": (ai_match.get("avg_away_win_pct") or 0) / 100.0 if (ai_match.get("avg_away_win_pct") or 0) > 1 else ai_match.get("avg_away_win_pct"),
                            }

                # Source 3: ML predictions cache
                if not pred_info:
                    with cache_lock:
                        ml_preds = predictions_cache.get(f"{home} vs {away}")
                    if ml_preds:
                        ens = next((p for p in ml_preds if p.get("model_name") == "ensemble"), None)
                        best = ens or ml_preds[0]
                        predicted = best.get("predicted_outcome")
                        if predicted:
                            pred_info = {
                                "predicted_outcome": predicted,
                                "confidence": best.get("confidence", 0),
                                "source": best.get("model_name", "ML"),
                            }

                if pred_info:
                    is_correct = pred_info["predicted_outcome"] == actual
                    was_new = db_manager.save_prediction_result({
                        "match_date": m.get("match_date", ""),
                        "home_team": home,
                        "away_team": away,
                        "league_code": m.get("league_code", ""),
                        "home_score": hs,
                        "away_score": aws,
                        "actual_outcome": actual,
                        "predicted_outcome": pred_info["predicted_outcome"],
                        "confidence": pred_info.get("confidence", 0),
                        "source": pred_info.get("source", "Unknown"),
                        "is_correct": is_correct,
                        "home_win_prob": pred_info.get("home_win_prob"),
                        "draw_prob": pred_info.get("draw_prob"),
                        "away_win_prob": pred_info.get("away_win_prob"),
                    })
                    if was_new:
                        saved += 1

            import datetime
            _last_history_update["time"] = datetime.datetime.now().strftime("%H:%M:%S")
            _last_history_update["saved"] = saved
            logger.info(f"History auto-update: {saved} new results saved")
        except Exception as e:
            logger.error(f"History auto-update error: {e}", exc_info=True)
        return saved

    @app.route("/api/history/update", methods=["POST"])
    def api_history_update():
        """Manually trigger a history results update."""
        try:
            saved = _update_history_results()
            summary = db_manager.get_prediction_results_summary()
            results = db_manager.get_all_prediction_results()
            return jsonify({
                "summary": summary,
                "results": results,
                "new_saved": saved,
                "last_update": _last_history_update["time"],
            })
        except Exception as e:
            logger.error(f"history update error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/history")
    def api_history():
        try:
            summary = db_manager.get_prediction_results_summary()
            results = db_manager.get_all_prediction_results()
            return jsonify({
                "summary": summary,
                "results": results,
                "last_update": _last_history_update["time"],
            })
        except Exception as e:
            logger.error(f"history error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Background thread: auto-update history every 10 minutes
    def _history_bg_worker():
        import time as _time
        _time.sleep(30)  # Wait for app startup
        while True:
            try:
                _update_history_results()
            except Exception as e:
                logger.error(f"History bg worker error: {e}")
            _time.sleep(600)  # Every 10 minutes

    _hist_thread = threading.Thread(target=_history_bg_worker, daemon=True)
    _hist_thread.start()

    # ═══════════════════════════════════════════
    # API: TRAINING
    # ═══════════════════════════════════════════
    @app.route("/api/train", methods=["POST"])
    def api_train():
        try:
            if prediction_engine.is_training:
                return jsonify({"error": "Træning er allerede i gang..."}), 409

            leagues = request.json.get("leagues") if request.json else None
            if not leagues:
                leagues = ["PL", "PD", "BL1", "SA", "FL1"]

            results = prediction_engine.train_models(leagues)
            return jsonify({"results": results})
        except Exception as e:
            logger.error(f"train error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # API: SETTINGS
    # ═══════════════════════════════════════════
    @app.route("/api/save_keys", methods=["POST"])
    def api_save_keys():
        try:
            body = request.json or {}
            fd_key = body.get("fd_key", "").strip()
            af_key = body.get("af_key", "").strip()
            if fd_key:
                os.environ["FOOTBALL_DATA_API_KEY"] = fd_key
            if af_key:
                os.environ["API_FOOTBALL_KEY"] = af_key
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/clear_cache", methods=["POST"])
    def api_clear_cache():
        try:
            db_manager.clear_expired_cache()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ═══════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════
    def _serialize_matches(matches, enrich_finished=True):
        """Ensure matches are JSON-serializable. For FINISHED matches, attach prediction results."""
        # Build AI prediction lookup index
        ai_index = {}
        if enrich_finished:
            with cache_lock:
                for ai in ai_predictions_cache:
                    h = (ai.get("home_team") or "").lower().strip()
                    a = (ai.get("away_team") or "").lower().strip()
                    if h and a:
                        ai_index[f"{h}|{a}"] = ai
                        # Also store short name keys for fuzzy matching
                        h_short = h.split()[0] if h else ""
                        a_short = a.split()[0] if a else ""
                        if h_short and a_short:
                            ai_index[f"{h_short}|{a_short}"] = ai

        out = []
        for m in matches:
            item = dict(m)
            # Ensure extra_data is a dict
            extra = item.get("extra_data")
            if extra and not isinstance(extra, dict):
                try:
                    import json
                    item["extra_data"] = json.loads(extra) if isinstance(extra, str) else {}
                except Exception:
                    item["extra_data"] = {}

            # Enrich finished matches with prediction results
            if enrich_finished and item.get("status") == "FINISHED":
                hs = item.get("home_score")
                aws = item.get("away_score")
                if hs is not None and aws is not None:
                    # Determine actual outcome
                    if hs > aws:
                        actual = "HOME_WIN"
                    elif hs < aws:
                        actual = "AWAY_WIN"
                    else:
                        actual = "DRAW"
                    item["actual_outcome"] = actual

                    home = item.get("home_team_name", "")
                    away = item.get("away_team_name", "")
                    found_prediction = False

                    # 1. Look up predictions from DB
                    try:
                        preds = db_manager.get_predictions_by_teams(home, away)
                        if preds:
                            ens = next((p for p in preds if p["model_name"] == "ensemble"), None)
                            best = ens or preds[0]
                            predicted = best.get("predicted_outcome")
                            item["prediction"] = {
                                "model": best.get("model_name", "?"),
                                "predicted_outcome": predicted,
                                "confidence": best.get("confidence", 0),
                                "is_correct": predicted == actual if predicted else None,
                                "home_win_prob": best.get("home_win_prob"),
                                "draw_prob": best.get("draw_prob"),
                                "away_win_prob": best.get("away_win_prob"),
                            }
                            item["prediction_models"] = [
                                {
                                    "model": p["model_name"],
                                    "predicted_outcome": p.get("predicted_outcome"),
                                    "confidence": p.get("confidence", 0),
                                    "is_correct": p.get("predicted_outcome") == actual if p.get("predicted_outcome") else None,
                                }
                                for p in preds
                            ]
                            found_prediction = True
                    except Exception as e:
                        logger.debug(f"DB prediction lookup failed for {home} vs {away}: {e}")

                    # 2. Look up from AI consensus predictions (in-memory cache)
                    if not found_prediction and ai_index:
                        h_low = home.lower().strip()
                        a_low = away.lower().strip()
                        ai_match = ai_index.get(f"{h_low}|{a_low}")
                        # Fuzzy: try short name
                        if not ai_match:
                            h_short = h_low.split()[0] if h_low else ""
                            a_short = a_low.split()[0] if a_low else ""
                            if h_short and a_short:
                                ai_match = ai_index.get(f"{h_short}|{a_short}")
                        # Fuzzy: substring
                        if not ai_match:
                            for k, v in ai_index.items():
                                kh, ka = k.split("|", 1)
                                if (kh in h_low or h_low in kh) and (ka in a_low or a_low in ka):
                                    ai_match = v
                                    break
                        if ai_match:
                            winner = ai_match.get("consensus_winner")
                            if winner:
                                predicted = {"1": "HOME_WIN", "X": "DRAW", "2": "AWAY_WIN"}.get(winner, winner)
                                conf = ai_match.get("consensus_confidence")
                                conf_val = (conf / 100.0) if conf and conf > 1 else conf
                                item["prediction"] = {
                                    "model": f"AI Sites ({ai_match.get('num_sources', '?')} kilder)",
                                    "predicted_outcome": predicted,
                                    "confidence": conf_val,
                                    "is_correct": predicted == actual,
                                    "home_win_prob": (ai_match.get("avg_home_win_pct") or 0) / 100.0 if (ai_match.get("avg_home_win_pct") or 0) > 1 else ai_match.get("avg_home_win_pct"),
                                    "draw_prob": (ai_match.get("avg_draw_pct") or 0) / 100.0 if (ai_match.get("avg_draw_pct") or 0) > 1 else ai_match.get("avg_draw_pct"),
                                    "away_win_prob": (ai_match.get("avg_away_win_pct") or 0) / 100.0 if (ai_match.get("avg_away_win_pct") or 0) > 1 else ai_match.get("avg_away_win_pct"),
                                }
                                found_prediction = True

                    # 3. Look up from in-memory ML predictions cache
                    if not found_prediction:
                        with cache_lock:
                            ml_preds = predictions_cache.get(f"{home} vs {away}")
                        if ml_preds:
                            ens = next((p for p in ml_preds if p.get("model_name") == "ensemble"), None)
                            best = ens or ml_preds[0]
                            predicted = best.get("predicted_outcome")
                            item["prediction"] = {
                                "model": best.get("model_name", "?"),
                                "predicted_outcome": predicted,
                                "confidence": best.get("confidence", 0),
                                "is_correct": predicted == actual if predicted else None,
                            }
                            item["prediction_models"] = [
                                {
                                    "model": p.get("model_name", "?"),
                                    "predicted_outcome": p.get("predicted_outcome"),
                                    "confidence": p.get("confidence", 0),
                                    "is_correct": p.get("predicted_outcome") == actual if p.get("predicted_outcome") else None,
                                }
                                for p in ml_preds
                            ]

                    # 4. Auto-save prediction result to DB for history
                    if item.get("prediction") and item["prediction"].get("predicted_outcome"):
                        pred = item["prediction"]
                        try:
                            db_manager.save_prediction_result({
                                "match_date": item.get("match_date", ""),
                                "home_team": home,
                                "away_team": away,
                                "league_code": item.get("league_code", ""),
                                "home_score": hs,
                                "away_score": aws,
                                "actual_outcome": actual,
                                "predicted_outcome": pred["predicted_outcome"],
                                "confidence": pred.get("confidence", 0),
                                "source": pred.get("model", "Unknown"),
                                "is_correct": pred.get("is_correct", False),
                                "home_win_prob": pred.get("home_win_prob"),
                                "draw_prob": pred.get("draw_prob"),
                                "away_win_prob": pred.get("away_win_prob"),
                            })
                        except Exception as e:
                            logger.debug(f"Auto-save prediction result failed: {e}")

            out.append(item)
        return out

    def _serialize_pred(p):
        """Serialize a prediction dict for JSON."""
        safe = {}
        for k, v in p.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, dict):
                safe[k] = v
            elif isinstance(v, (list, tuple)):
                safe[k] = list(v)
            else:
                safe[k] = str(v)
        return safe

    return app
