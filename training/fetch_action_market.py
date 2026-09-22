"""Resumable official BitMEX XBTUSD one/five-minute candles for action imitation.

Public, unauthenticated requests only. Never interpolates gaps or changes prices.
Each candle timestamp is its END; volume is raw inverse-contract volume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://www.bitmex.com/api/v1/trade/bucketed"
START = "2018-03-01T00:00:00.000Z"
END = "2022-01-01T00:00:00.000Z"
STEP = timedelta(minutes=5)
PACE_SECONDS = 2.2  # Official unauthenticated limit: 30 requests per minute.
MAX_ATTEMPTS = 5
MAX_RETRY_SECONDS = 300
MAX_PAGE_BYTES = 8 * 1024 * 1024
DATA_NAME = "bitmex-xbtusd-5m.jsonl"
METADATA_NAME = "bitmex-xbtusd-5m.metadata.json"


def iso(value):
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def step_for(bin_size):
    if bin_size not in ("1m", "5m"):
        raise ValueError("bin size must be 1m or 5m")
    return timedelta(minutes=int(bin_size[:-1]))


def timestamp(value, bin_size="5m"):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must have an explicit UTC timezone")
    if parsed.minute % int(step_for(bin_size).total_seconds() / 60) or parsed.second or parsed.microsecond:
        raise ValueError(f"timestamp must be a {bin_size} boundary")
    return parsed


def validate_row(row, expected, bin_size="5m"):
    if row.get("symbol") != "XBTUSD" or timestamp(row["timestamp"], bin_size) != expected:
        raise ValueError(f"gap/duplicate/order/symbol error: expected XBTUSD {iso(expected)}, got {row.get('symbol')} {row.get('timestamp')}")
    for field in ("open", "high", "low", "close"):
        value = row[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid OHLC {field} at {row['timestamp']}; no synthetic repair")
    if not row["low"] <= row["close"] <= row["high"]:
        raise ValueError("close outside high/low")
    # BitMEX open is the previous close, even outside current high/low.
    volume = row["volume"]
    if type(volume) not in (int, float) or not math.isfinite(volume) or volume < 0:
        raise ValueError("invalid raw contract volume")


def page_url(expected, end, bin_size="5m"):
    return ENDPOINT + "?" + urllib.parse.urlencode({
        "symbol": "XBTUSD", "binSize": bin_size, "partial": "false", "count": 1000,
        "reverse": "false", "startTime": iso(expected), "endTime": iso(end),
    })


def retry_delay(headers, attempt, now=None):
    now = time.time() if now is None else now
    delay = max(PACE_SECONDS, 2 ** attempt)
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - now)
    if headers.get("x-ratelimit-reset"):
        delay = max(delay, float(headers["x-ratelimit-reset"]) - now)
    if not math.isfinite(delay) or delay > MAX_RETRY_SECONDS:
        raise RuntimeError("server requested a long pause; stop and resume later")
    return delay


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("unexpected redirect away from the official BitMEX endpoint")


def fetch_page(url, opener, sleep=time.sleep):
    for attempt in range(MAX_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "jev-nyotti-research/1.0"})
            with opener.open(request, timeout=45) as response:
                if response.status != 200:
                    raise ValueError(f"unexpected HTTP status {response.status}")
                body = response.read(MAX_PAGE_BYTES + 1)
                if len(body) > MAX_PAGE_BYTES:
                    raise ValueError("oversized response")
                headers = {key.lower(): value for key, value in response.headers.items()}
            pause = PACE_SECONDS
            if int(headers.get("x-ratelimit-remaining", "30")) <= 1:
                pause = retry_delay(headers, 0)
            return {
                "url": url, "status": 200, "retrieved_at": iso(datetime.now(timezone.utc)),
                "response_sha256": hashlib.sha256(body).hexdigest(), "body": body.decode("utf-8"),
                "rate_limit_headers": {k: v for k, v in headers.items() if k.startswith("x-ratelimit")},
                "attempts": attempt + 1,
            }, pause
        except urllib.error.HTTPError as error:
            if error.code != 429 and not 500 <= error.code <= 599:
                raise
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            sleep(retry_delay({k.lower(): v for k, v in error.headers.items()}, attempt))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt + 1 == MAX_ATTEMPTS:
                raise
            sleep(max(PACE_SECONDS, 2 ** attempt))
    raise AssertionError("unreachable")


def atomic_json(path, data):
    if path.is_symlink():
        raise ValueError("refusing symlink cache file")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_page(envelope, expected, end, bin_size="5m"):
    step = step_for(bin_size)
    if envelope["url"] != page_url(expected, end, bin_size) or envelope["status"] != 200:
        raise ValueError("checkpoint request does not match requested range")
    if hashlib.sha256(envelope["body"].encode()).hexdigest() != envelope["response_sha256"]:
        raise ValueError("checkpoint page response hash mismatch")
    rows = json.loads(envelope["body"])
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("empty or malformed page before completion")
    for row in rows:
        if expected > end:
            raise ValueError("unexpected candle beyond requested end")
        validate_row(row, expected, bin_size)
        expected += step
    return rows, expected


def export_cache_range(directory, identity, requests, end, destination):
    """Revalidate cached responses and publish an exact immutable range snapshot."""
    bin_size = identity["binSize"]
    first, final = map(lambda value: timestamp(value, bin_size), identity["range_inclusive"])
    step = step_for(bin_size)
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / f"bitmex-xbtusd-{bin_size}.jsonl"
    metadata_path = destination / f"bitmex-xbtusd-{bin_size}.metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["range_inclusive"] != [iso(first), iso(end)] or metadata["binSize"] != bin_size:
            raise ValueError("existing snapshot range does not match")
        if not data_path.exists() or hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("existing snapshot data hash mismatch")
        return metadata
    temporary = data_path.with_suffix(".jsonl.tmp")
    digest, count, outside, mismatches, last_close = hashlib.sha256(), 0, 0, 0, None
    expected, included_requests = first, []
    with temporary.open("wb") as handle:
        for index, request in enumerate(requests):
            if expected > end:
                break
            envelope = json.loads((directory / "pages" / f"{index:06}.json").read_text())
            if envelope["response_sha256"] != request["response_sha256"]:
                raise ValueError("snapshot page hash mismatch")
            rows, next_expected = validate_page(envelope, expected, final, bin_size)
            included = 0
            for row in rows:
                if timestamp(row["timestamp"], bin_size) > end:
                    break
                outside += not row["low"] <= row["open"] <= row["high"]
                mismatches += last_close is not None and last_close != row["open"]
                last_close = row["close"]
                content = (json.dumps(row, separators=(",", ":")) + "\n").encode()
                handle.write(content)
                digest.update(content)
                included += 1
                count += 1
            included_requests.append(request | {"included_rows": included})
            expected = next_expected
        handle.flush()
        os.fsync(handle.fileno())
    expected_count = int((end - first) / step) + 1
    if count != expected_count:
        raise ValueError("snapshot does not contain the complete requested range")
    temporary.replace(data_path)
    metadata = identity | {
        "range_inclusive": [iso(first), iso(end)], "source_request_range_inclusive": identity["range_inclusive"],
        "documentation": "https://docs.bitmex.com/api-explorer/get-trade-bucketed",
        "rate_limit_documentation": "https://www.bitmex.com/app/restAPI#Request-Rate-Limits",
        "contract_documentation": "https://www.bitmex.com/app/contract/XBTUSD",
        "volume_documentation": "https://docs.bitmex.com/api-explorer/get-instruments",
        "retrieved_at": iso(datetime.now(timezone.utc)),
        "timestamp_semantics": "END of period (UTC); only timestamps < cutoff are used for conservative availability.",
        "open_semantics": "Previous bucket close; may lie outside current high/low. Preserved without repair.",
        "volume_units": "contracts (XBTUSD inverse contract: USD1 face value); not BTC base volume",
        "ohlc_price_units": "USD per XBT", "count": count, "expected_inclusive_candle_count": expected_count,
        "duplicate_count": 0, "gaps": [], "open_outside_high_low_count": outside,
        "close_outside_high_low_count": 0, "previous_close_open_mismatch_count": mismatches,
        "sha256": digest.hexdigest(), "requests": included_requests,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def download(directory, start=START, end=END, max_pages_run=None, fetcher=None, sleep=time.sleep, bin_size="5m", prefix_ends=None):
    first, final = timestamp(start, bin_size), timestamp(end, bin_size)
    if not timestamp(START) <= first <= final <= timestamp(END):
        raise ValueError("range must be within the authorized 2018-03-01 through 2022-01-01 historical interval")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prefixes = sorted(set(timestamp(value, bin_size) for value in (prefix_ends or [])))
        if any(not first <= value <= final for value in prefixes):
            raise ValueError("prefix end must lie within the requested range")
        return _download_locked(directory, first, final, max_pages_run, fetcher, sleep, bin_size, prefixes)


def _download_locked(directory, first, final, max_pages_run, fetcher, sleep, bin_size, prefixes):
    step = step_for(bin_size)
    data_name = f"bitmex-xbtusd-{bin_size}.jsonl"
    pages_dir = directory / "pages"
    pages_dir.mkdir(exist_ok=True)
    identity = {"endpoint": ENDPOINT, "symbol": "XBTUSD", "binSize": bin_size, "partial": False,
                "range_inclusive": [iso(first), iso(final)]}
    checkpoint_path = directory / "checkpoint.json"
    prior = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    if prior and prior["identity"] != identity:
        raise ValueError("resume range differs from checkpoint; choose a new output directory")
    expected, count, open_outside, mismatches, last_close = first, 0, 0, 0, None
    requests = []
    expected_count = int((final - first) / step) + 1
    existing = sorted(pages_dir.glob("*.json"))
    if len(existing) > math.ceil(expected_count / 1000) + 10:
        raise ValueError("unexpected checkpoint page count")
    if prior and len(existing) < len(prior["requests"]):
        raise ValueError("checkpoint page disappeared")

    def absorb(envelope, index):
        nonlocal expected, count, open_outside, mismatches, last_close
        rows, next_expected = validate_page(envelope, expected, final, bin_size)
        if prior and index < len(prior["requests"]):
            if envelope["response_sha256"] != prior["requests"][index]["response_sha256"]:
                raise ValueError("saved page differs from committed checkpoint hash")
        for row in rows:
            open_outside += not row["low"] <= row["open"] <= row["high"]
            mismatches += last_close is not None and last_close != row["open"]
            last_close = row["close"]
        count += len(rows)
        expected = next_expected
        requests.append({key: value for key, value in envelope.items() if key != "body"} | {"rows": len(rows)})

    for index, path in enumerate(existing):
        if path.name != f"{index:06}.json" or path.is_symlink():
            raise ValueError("missing/nonsequential/symlink checkpoint page")
        absorb(json.loads(path.read_text()), index)

    def progress():
        result = {"identity": identity, "complete": expected > final, "count": count,
                  "expected_count": expected_count, "next_timestamp": iso(expected),
                  "last_timestamp": iso(expected - step) if count else None, "requests": requests,
                  "updated_at": iso(datetime.now(timezone.utc))}
        atomic_json(checkpoint_path, result)
        return result

    def publish_prefixes():
        for prefix in prefixes:
            if expected <= prefix:
                continue
            destination = directory / "prefixes" / prefix.strftime("%Y-%m-%dT%H%M%SZ")
            existed = (destination / f"bitmex-xbtusd-{bin_size}.metadata.json").exists()
            metadata = export_cache_range(directory, identity, requests, prefix, destination)
            if not existed:
                print(json.dumps({"prefix_ready": str(destination), "count": metadata["count"], "sha256": metadata["sha256"]}), flush=True)
        # Each immutable snapshot is validated on restart and written once per run.
        prefixes[:] = [prefix for prefix in prefixes if expected <= prefix]

    publish_prefixes()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    fetcher = fetcher or (lambda url: fetch_page(url, opener, sleep))
    fetched = 0
    while expected <= final:
        if max_pages_run is not None and fetched >= max_pages_run:
            return progress()
        if len(requests) >= math.ceil(expected_count / 1000) + 10:
            raise ValueError("unexpected pagination length; stopped before further requests")
        envelope, pause = fetcher(page_url(expected, final, bin_size))
        try:
            # Validate before making this response a reusable checkpoint.
            validate_page(envelope, expected, final, bin_size)
        except (ValueError, KeyError, TypeError) as error:
            atomic_json(directory / "rejected-response.json", envelope | {"validation_error": str(error)})
            raise
        page_path = pages_dir / f"{len(requests):06}.json"
        atomic_json(page_path, envelope)
        absorb(envelope, len(requests))
        fetched += 1
        status = progress()
        publish_prefixes()
        print(json.dumps({key: status[key] for key in ("complete", "count", "expected_count", "last_timestamp")}), flush=True)
        if expected <= final:
            sleep(pause)
    if count != expected_count:
        raise ValueError("incomplete candle range")
    metadata = export_cache_range(directory, identity, requests, final, directory)
    return progress() | {"data_path": str(directory / data_name), "sha256": metadata["sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-size", choices=("1m", "5m"), default="5m")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--max-pages-run", type=int, help="Stop cleanly after N new pages; rerun to resume")
    parser.add_argument("--export-prefix", action="append", default=[], help="Publish an immutable contiguous prefix once this inclusive end is cached; repeatable")
    args = parser.parse_args(argv)
    if args.max_pages_run is not None and args.max_pages_run < 1:
        parser.error("--max-pages-run must be positive")
    result = download(args.output_dir, args.start, args.end, args.max_pages_run, bin_size=args.bin_size, prefix_ends=args.export_prefix)
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}), flush=True)


if __name__ == "__main__":
    main()
