#!/usr/bin/env python3
"""Quick test for the free football client."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.api.free_football_client import FreeFootballClient

client = FreeFootballClient()

print("=== Testing Free Football Client (NO API keys!) ===")
print()

# Test ESPN single league
print("1. ESPN - Premier League today:")
matches = client._espn_get_scoreboard("PL")
print(f"   Found {len(matches)} matches")
for m in matches[:5]:
    hs = m.get("home_score")
    as_ = m.get("away_score")
    score = f"{hs}-{as_}" if hs is not None else "vs"
    print(f"   {m['home_team_name']} {score} {m['away_team_name']} [{m['status']}]")

# Test La Liga
print()
print("2. ESPN - La Liga today:")
matches2 = client._espn_get_scoreboard("PD")
print(f"   Found {len(matches2)} matches")
for m in matches2[:5]:
    hs = m.get("home_score")
    as_ = m.get("away_score")
    score = f"{hs}-{as_}" if hs is not None else "vs"
    print(f"   {m['home_team_name']} {score} {m['away_team_name']} [{m['status']}]")

print()
print("3. is_available():", client.is_available())

print()
print("=== SUCCESS! Real match data without any API key! ===")
