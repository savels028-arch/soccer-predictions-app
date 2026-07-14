# International / World Cup data bundle

This directory contains the separate national-team model used for World Cup
fixtures. It is deliberately isolated from all domestic club features.

Files:

- `results_1990_plus.csv.gz`: compact normalized scored matches from 1990 onward
- `international_elo_v1.json`: ratings, frozen parameters and validation evidence
- `manifest.json`: immutable source URL, commit, license and SHA-256 checksums
- `validation_report.md`: selection split, untouched holdout and limitations

The source is Mart Jürisoo's men's full-international results dataset, released
under CC0 and [mirrored by OpenFootball](https://github.com/openfootball/internationals).
The exact upstream commit and file checksum are pinned in `manifest.json`; the
raw repository is not vendored.

Rebuild (network is needed only when `--source` is omitted):

```bash
venv/bin/python scripts/build_international_model.py
```

Or rebuild from an already downloaded pinned file:

```bash
venv/bin/python scripts/build_international_model.py --source /path/to/results.csv
```

The runtime loader verifies the normalized snapshot checksum, artifact checksum,
source provenance, schema and every holdout gate. Any mismatch, unknown team,
insufficient history or fixture at/before the training cutoff fails closed.

This is forecast-only. The source has no historical pre-match odds, so these
results do not establish betting ROI or value. World Cup forecasts retain an
`ABSTAIN` decision status and cannot enter coupons or P&L until a separate,
odds-aware betting strategy passes the existing promotion gates.
