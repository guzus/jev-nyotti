import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from datetime import timedelta

from fetch_action_market import (
    DATA_NAME, END, MAX_ATTEMPTS, METADATA_NAME, PACE_SECONDS, START, STEP,
    download, fetch_page, iso, page_url, retry_delay, timestamp, validate_page, validate_row,
)


def candle(end, **updates):
    return dict(timestamp=iso(end), symbol="XBTUSD", open=99, high=102, low=100,
                close=101, volume=1000) | updates


def envelope(rows, first, last):
    body = json.dumps(rows)
    return dict(url=page_url(first, last), status=200, body=body,
                response_sha256=hashlib.sha256(body.encode()).hexdigest(),
                retrieved_at="2026-09-23T00:00:00.000Z", attempts=1)


class ActionMarketTests(unittest.TestCase):
    def test_prefix_export_is_exact_contiguous_and_immutable_while_download_is_partial(self):
        first = timestamp(START)
        last = first + 3 * STEP
        prefix = first + STEP
        with tempfile.TemporaryDirectory() as tmp:
            def first_page(url):
                return envelope([candle(first), candle(prefix, open=101), candle(prefix + STEP, open=101)], first, last), 0
            result = download(tmp, START, iso(last), max_pages_run=1, fetcher=first_page,
                              sleep=lambda _: None, prefix_ends=[iso(prefix)])
            self.assertFalse(result["complete"])
            prefix_dir = Path(tmp) / "prefixes/2018-03-01T000500Z"
            data = (prefix_dir / DATA_NAME).read_bytes()
            metadata = json.loads((prefix_dir / METADATA_NAME).read_text())
            self.assertEqual(metadata["count"], 2)
            self.assertEqual(metadata["range_inclusive"], [START, iso(prefix)])
            self.assertEqual(metadata["requests"][0]["included_rows"], 2)
            self.assertEqual(len(data.splitlines()), 2)
            self.assertFalse((Path(tmp) / DATA_NAME).exists())
            download(tmp, START, iso(last), fetcher=lambda _: (envelope([candle(last, open=101)], last, last), 0),
                     prefix_ends=[iso(prefix)], sleep=lambda _: None)
            self.assertEqual((prefix_dir / DATA_NAME).read_bytes(), data)
            self.assertEqual(len((Path(tmp) / DATA_NAME).read_bytes().splitlines()), 4)

    def test_one_minute_uses_distinct_request_identity_and_output_name(self):
        first = timestamp(START)
        last = first + timedelta(minutes=1)
        rows = [candle(first), candle(last, open=101)]
        page = envelope(rows, first, last)
        page["url"] = page_url(first, last, "1m")
        with tempfile.TemporaryDirectory() as tmp:
            result = download(tmp, START, iso(last), bin_size="1m", fetcher=lambda _: (page, 0))
            self.assertEqual(result["count"], 2)
            self.assertTrue((Path(tmp) / "bitmex-xbtusd-1m.jsonl").exists())
            self.assertEqual(result["identity"]["binSize"], "1m")
            self.assertFalse((Path(tmp) / DATA_NAME).exists())
        with self.assertRaisesRegex(ValueError, "5m boundary"):
            timestamp(iso(last), "5m")

    def test_five_minute_end_timestamp_and_raw_previous_close_open(self):
        first = timestamp(START)
        validate_row(candle(first), first)  # open 99 outside [100,102] is real semantics.
        for bad in ("2018-03-01T00:01:00Z", "2018-03-01T00:00:01Z", "2018-03-01T00:00:00"):
            with self.assertRaises(ValueError):
                timestamp(bad)
        for updates in (dict(close=103), dict(volume=-1), dict(open=None), dict(volume=True)):
            with self.assertRaises(ValueError):
                validate_row(candle(first, **updates), first)

    def test_gap_duplicate_and_tampered_response_rejected(self):
        first = timestamp(START)
        end = first + 2 * STEP
        for rows in ([candle(first), candle(end)], [candle(first), candle(first)]):
            with self.assertRaisesRegex(ValueError, "gap/duplicate"):
                validate_page(envelope(rows, first, end), first, end)
        page = envelope([candle(first)], first, end)
        page["body"] = page["body"].replace('1000', '1001')
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_page(page, first, end)

    def test_resume_restores_cursor_from_verified_pages_and_publishes_only_when_complete(self):
        first = timestamp(START)
        last = first + 2 * STEP
        with tempfile.TemporaryDirectory() as tmp:
            first_fetch = lambda url: (envelope([candle(first)], first, last), 0)
            partial = download(tmp, START, iso(last), max_pages_run=1, fetcher=first_fetch, sleep=lambda _: None)
            self.assertFalse(partial["complete"])
            self.assertFalse((Path(tmp) / DATA_NAME).exists())
            calls = []

            def rest(url):
                calls.append(url)
                return envelope([candle(first + STEP, open=101), candle(last, open=101)], first + STEP, last), 0

            result = download(tmp, START, iso(last), fetcher=rest, sleep=lambda _: None)
            self.assertEqual(calls, [page_url(first + STEP, last)])
            self.assertTrue(result["complete"])
            self.assertEqual(result["count"], 3)
            data = (Path(tmp) / DATA_NAME).read_bytes()
            metadata = json.loads((Path(tmp) / METADATA_NAME).read_text())
            self.assertEqual(metadata["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(metadata["gaps"], [])
            self.assertEqual(metadata["open_outside_high_low_count"], 1)
            self.assertEqual(metadata["previous_close_open_mismatch_count"], 0)
            download(tmp, START, iso(last), fetcher=lambda _: self.fail("complete cache made network request"))
            with self.assertRaisesRegex(ValueError, "range differs"):
                download(tmp, START, END)

    def test_changed_committed_page_hash_and_missing_page_rejected(self):
        first = timestamp(START)
        last = first + STEP
        with tempfile.TemporaryDirectory() as tmp:
            download(tmp, START, iso(last), max_pages_run=1,
                     fetcher=lambda _: (envelope([candle(first)], first, last), 0), sleep=lambda _: None)
            path = Path(tmp) / "pages/000000.json"
            path.write_text(json.dumps(envelope([candle(first, volume=999)], first, last)))
            with self.assertRaisesRegex(ValueError, "committed checkpoint hash"):
                download(tmp, START, iso(last))
            path.unlink()
            with self.assertRaisesRegex(ValueError, "disappeared"):
                download(tmp, START, iso(last))

    def test_crash_between_page_write_and_checkpoint_can_resume(self):
        first = timestamp(START)
        with tempfile.TemporaryDirectory() as tmp:
            pages = Path(tmp) / "pages"
            pages.mkdir()
            (pages / "000000.json").write_text(json.dumps(envelope([candle(first)], first, first)))
            result = download(tmp, START, START, fetcher=lambda _: self.fail("should recover saved response"))
            self.assertTrue(result["complete"])

    def test_invalid_response_is_retained_for_audit_but_never_committed(self):
        first = timestamp(START)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "gap/duplicate"):
                download(tmp, START, iso(first + STEP),
                         fetcher=lambda _: (envelope([candle(first + STEP)], first, first + STEP), 0))
            self.assertTrue((Path(tmp) / "rejected-response.json").exists())
            self.assertEqual(list((Path(tmp) / "pages").glob("*.json")), [])
            self.assertFalse((Path(tmp) / DATA_NAME).exists())

    def test_rate_limit_headers_and_bounded_retries(self):
        self.assertEqual(retry_delay({"retry-after": "20", "x-ratelimit-reset": "125"}, 0, now=100), 25)
        with self.assertRaisesRegex(RuntimeError, "long pause"):
            retry_delay({"retry-after": "301"}, 0)

        class Errors:
            def __init__(self, code):
                self.code, self.calls = code, 0

            def open(self, request, timeout):
                self.calls += 1
                raise urllib.error.HTTPError(request.full_url, self.code, "test", {"Retry-After": "3"}, io.BytesIO())

        errors, sleeps = Errors(429), []
        with self.assertRaises(urllib.error.HTTPError):
            fetch_page(page_url(timestamp(START), timestamp(END)), errors, sleep=sleeps.append)
        self.assertEqual(errors.calls, MAX_ATTEMPTS)
        self.assertEqual(len(sleeps), MAX_ATTEMPTS - 1)
        self.assertTrue(all(delay >= PACE_SECONDS for delay in sleeps))
        errors = Errors(403)
        with self.assertRaises(urllib.error.HTTPError):
            fetch_page(page_url(timestamp(START), timestamp(END)), errors, sleep=lambda _: self.fail("403 retried"))
        self.assertEqual(errors.calls, 1)


if __name__ == "__main__":
    unittest.main()
