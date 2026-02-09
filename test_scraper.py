#!/usr/bin/env python3
"""Test all scrapers."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from src.api.prediction_scraper import PredictionScraper

def main():
    scraper = PredictionScraper()
    t0 = time.time()
    preds = scraper.get_all_predictions()
    elapsed = time.time() - t0
    
    print(f"\n{'='*70}")
    print(f"  SCRAPING COMPLETE - {len(preds)} predictions in {elapsed:.1f}s")
    print(f"{'='*70}")
    
    # Group by source
    by_source = {}
    for p in preds:
        src = p.get('source', '?')
        by_source.setdefault(src, []).append(p)
    
    for src, items in sorted(by_source.items()):
        print(f"\n{'─'*60}")
        print(f"  {src}: {len(items)} predictions")
        print(f"{'─'*60}")
        for p in items[:5]:
            home = p.get('home_team', '?')
            away = p.get('away_team', '?')
            h = p.get('home_win_pct', '?')
            d = p.get('draw_pct', '?')
            a = p.get('away_win_pct', '?')
            tip = p.get('predicted_winner') or p.get('tip', '')
            btts = p.get('btts', '')
            ou = p.get('over_under_25', '')
            extras = []
            if tip:
                extras.append(f"tip={tip}")
            if btts:
                extras.append(f"btts={btts}")
            if ou:
                extras.append(f"o/u={ou}")
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            print(f"  {home:25s} vs {away:25s}  H:{h}% D:{d}% A:{a}%{extra_str}")
        if len(items) > 5:
            print(f"  ... and {len(items)-5} more")

    # Now test consensus
    print(f"\n{'='*70}")
    print("  CONSENSUS PREDICTIONS")
    print(f"{'='*70}")
    consensus = scraper.get_consensus_predictions()
    multi_source = [c for c in consensus if c['num_sources'] >= 2]
    print(f"\n  Matches with 2+ sources: {len(multi_source)}")
    for c in multi_source[:10]:
        src_names = ', '.join(c['sources'])
        print(f"\n  {c['home_team']} vs {c['away_team']} ({c['num_sources']} sources)")
        print(f"    Avg: H:{c['avg_home_win_pct']}% D:{c['avg_draw_pct']}% A:{c['avg_away_win_pct']}%")
        print(f"    Winner: {c['consensus_winner']} ({c['consensus_confidence']}%)")
        if c.get('btts_consensus'):
            print(f"    BTTS: {c['btts_consensus']}")
        if c.get('over_under_consensus'):
            print(f"    O/U 2.5: {c['over_under_consensus']}")
        print(f"    Sources: {src_names}")

if __name__ == '__main__':
    main()
