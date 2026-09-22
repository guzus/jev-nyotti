"""Download the fixed pilot's official BitMEX XBTUSD hourly history.

No credentials, alternate providers, proxies, synthetic filling, or paid resources.
Timestamps identify bucket END, and open is the previous bucket's close.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request

ENDPOINT = "https://www.bitmex.com/api/v1/trade/bucketed"
START = "2018-03-01T00:00:00.000Z"
END = "2022-01-01T00:00:00.000Z"
DATA_NAME = "bitmex-xbtusd-1h.jsonl"
METADATA_NAME = "bitmex-xbtusd-1h.metadata.json"


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("BitMEX timestamp must have explicit UTC timezone")
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("BitMEX timestamp must be an hourly boundary")
    return parsed


def iso(value):
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_row(row, expected):
    if row.get("symbol") != "XBTUSD" or timestamp(row["timestamp"]) != expected:
        raise ValueError("missing, duplicate, unordered, or incorrect-symbol candle")
    for key in ("open", "high", "low", "close"):
        value = row[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid OHLC field: {key}")
    if not row["low"] <= row["close"] <= row["high"]:
        raise ValueError("close outside low/high range")
    # Do not require open inside high/low: BitMEX explicitly uses prior close.
    volume = row["volume"]
    if type(volume) not in (int, float) or not math.isfinite(volume) or volume < 0:
        raise ValueError("invalid raw contract volume")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("unexpected redirect from the official BitMEX endpoint")


def fetch():
    rows, requests_log = [], []
    expected = timestamp(START)
    end = timestamp(END)
    # Explicitly avoid machine-configured proxies; contact the official host only.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    while expected <= end:
        if len(requests_log) >= 40:
            raise ValueError("unexpected pagination length; stopped before further requests")
        parameters = {"symbol": "XBTUSD", "binSize": "1h", "partial": "false",
                      "count": 1000, "reverse": "false", "startTime": iso(expected), "endTime": END}
        url = ENDPOINT + "?" + urllib.parse.urlencode(parameters)
        with opener.open(url, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP status: {response.status}")
            body = response.read()
            batch = json.loads(body)
        if not isinstance(batch, list) or not 1 <= len(batch) <= 1000:
            raise ValueError("empty or malformed BitMEX page before range completion")
        requests_log.append({"url": url, "status": 200, "rows": len(batch),
                             "response_sha256": hashlib.sha256(body).hexdigest(),
                             "retrieved_at": iso(datetime.now(timezone.utc))})
        for row in batch:
            if expected > end:
                raise ValueError("unexpected candle beyond requested end")
            validate_row(row, expected)
            rows.append(row)
            expected += timedelta(hours=1)
        print(json.dumps({"pages": len(requests_log), "rows": len(rows),
                          "last": rows[-1]["timestamp"]}), flush=True)
        if expected <= end:
            time.sleep(.25)
    expected_count = int((end - timestamp(START)).total_seconds() / 3600) + 1
    if len(rows) != expected_count:
        raise ValueError("incomplete hourly range")
    metadata = {
        "endpoint": ENDPOINT,
        "documentation": "https://docs.bitmex.com/api-explorer/get-trade-bucketed",
        "contract_documentation": "https://www.bitmex.com/app/contract/XBTUSD",
        "volume_documentation": "https://docs.bitmex.com/api-explorer/get-instruments",
        "retrieved_at": iso(datetime.now(timezone.utc)),
        "symbol": "XBTUSD", "binSize": "1h", "partial": False,
        "range_inclusive": [START, END],
        "timestamp_semantics": "END of period (UTC); use only timestamp <= decision cutoff, or < for strict boundary availability.",
        "open_semantics": "Open equals previous bucket close and may lie outside current bucket high/low.",
        "volume_units": "contracts (XBTUSD inverse contract: USD1 face value); not BTC base volume",
        "ohlc_price_units": "USD per XBT",
        "count": len(rows), "expected_inclusive_hour_count": expected_count,
        "duplicate_count": 0, "gaps": [],
        "open_outside_high_low_count": sum(not r["low"] <= r["open"] <= r["high"] for r in rows),
        "close_outside_high_low_count": 0,
        "previous_close_open_mismatch_count": sum(a["close"] != b["open"] for a, b in zip(rows, rows[1:])),
        "requests": requests_log,
    }
    content = ''.join(json.dumps(row, separators=(",", ":")) + '\n' for row in rows).encode()
    metadata["sha256"] = hashlib.sha256(content).hexdigest()
    return content, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Local directory for the two new cache files; existing outputs are never overwritten")
    args = parser.parse_args(argv)
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / DATA_NAME, directory / METADATA_NAME)
    if any(path.exists() or path.is_symlink() for path in paths):
        parser.error("output already exists; choose a fresh directory")
    content, metadata = fetch()
    # Exclusive creation repeats the overwrite protection after network calls.
    with paths[0].open("xb") as handle:
        handle.write(content)
    with paths[1].open("x") as handle:
        handle.write(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"data_path": str(paths[0]), "metadata_path": str(paths[1]),
                      "count": metadata["count"], "gaps": [], "sha256": metadata["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
