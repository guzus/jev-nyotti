# BitMEX one/five-minute history for action imitation

`fetch_action_market.py` acquires public XBTUSD candles from the official BitMEX
`GET /api/v1/trade/bucketed` endpoint. It uses no API key, proxy, alternate market,
paid resource, interpolation, or modified OHLC prices.

```sh
python3 training/fetch_action_market.py --output-dir .runtime/action-market
python3 training/fetch_action_market.py --bin-size 1m --output-dir .runtime/action-market-1m
```

The default interval includes every five-minute bucket end from
`2018-03-01T00:00:00.000Z` through `2022-01-01T00:00:00.000Z`. Repeating exactly
the command resumes. `--bin-size 1m` selects one-minute buckets and names outputs
accordingly; never share an output directory between cadences or run simultaneous
downloaders against the same public rate limit. The default 5m range has 403,777
bars (404 requests); 1m has 2,018,881 bars (2,019 requests). At 2.2-second pacing,
the latter requires at least 74 minutes plus response latency and retries.
`--max-pages-run 1` performs one new page as a smoke test;
`--start` and `--end` can select a smaller interval within the default period.
One output directory belongs to one fixed interval. A file lock prevents two
downloaders from writing it concurrently.

To train on an early contiguous period while the same paced stream continues,
add `--export-prefix 2018-06-01T00:00:00Z`. Once available, the downloader writes
an immutable, independently hashed 132,481-bar 1m snapshot under
`prefixes/2018-06-01T000000Z/`. The snapshot includes precisely its stated end,
even when a fetched page extends later; request provenance records included
rows separately. Repeated `--export-prefix` flags support additional snapshots.
The full requested range and its checkpoint remain unchanged. Existing exported
files are hash-checked on restart rather than silently replaced.

Each response is stored atomically under `pages/` with its exact response SHA-256,
request URL, retrieval time, and rate-limit headers. `checkpoint.json` records
completed pages and the next candle end. Resume revalidates hashes, symbols,
ordering, and exact one/five-minute continuity rather than trusting that cursor. A page
persisted immediately before a crash can be recovered even if the checkpoint
write did not finish. Any rejected response is retained separately for diagnosis.

Only a fully contiguous, validated range produces `bitmex-xbtusd-5m.jsonl` and
`bitmex-xbtusd-5m.metadata.json`. The metadata contains the final file hash,
request provenance, range, count, and validation statistics. A missing or invalid
candle stops acquisition; it is never replaced with an invented candle.

Important input semantics match `fetch_market.py`:

- Timestamps mark **bucket end**, not its start. Conservative training features
  should use bucket ends strictly before the action cutoff.
- `open` equals the preceding bucket's close and can be outside the current
  bucket's `low`–`high`. It remains unchanged. `close` must be within that range.
- Prices are USD/XBT. `volume` is raw XBTUSD inverse-contract count (USD1 face
  value), not BTC base volume; `homeNotional` is a different field.
- `partial=false` requests completed candles. Requested history ends in 2022.

The downloader spaces successful requests by at least 2.2 seconds to remain
below the documented unauthenticated limit of 30/minute. It respects
`retry-after` and `x-ratelimit-reset`, retries transient failures at most five
times per page, and stops rather than waiting beyond a five-minute retry delay.
Nonretryable HTTP errors (including 403) stop immediately. Requests time out
after 45 seconds; the page count is bounded by the interval size plus ten.

```sh
PYTHONPATH=training python3 -m unittest discover -s training/tests -p 'test_action_market.py' -v
```

Primary documentation, checked 2026-09-23:

- [Bucket-end and previous-close semantics](https://docs.bitmex.com/api-explorer/get-trade-bucketed)
- [REST filtering and rate limits](https://www.bitmex.com/app/restAPI)
- [XBTUSD contract](https://www.bitmex.com/app/contract/XBTUSD)
- [Instrument field definitions](https://docs.bitmex.com/api-explorer/get-instruments)
