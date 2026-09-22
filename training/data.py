"""Deterministic synthetic fixtures; these are NOT trader records or trade advice."""
from __future__ import annotations

import random

from jev_inference.prompt import format_prompt, messages_for_job
from jev_inference.schemas import Job

RUBRIC = (
    "Synthetic plumbing exercise only, not a trading strategy. "
    "Select long when features.changePct is greater than 1, short when it is "
    "less than -1, and hold otherwise. Ignore all other features. "
    "No real trader history or future outcomes are supplied."
)
ACTIONS = ("long", "short", "hold")


def synthetic_examples(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for i in range(count):
        action = ACTIONS[i % 3]
        change = round(rng.uniform(1.2, 4.0) if action == "long" else
                       rng.uniform(-4.0, -1.2) if action == "short" else
                       rng.uniform(-0.8, 0.8), 3)
        price = rng.uniform(100, 90000)
        candles = []
        for j in range(24):
            close = price * (1 + rng.uniform(-0.004, 0.004))
            candles.append({"time": 1700000000 + j * 900, "open": round(price, 2),
                            "high": round(max(price, close) * 1.001, 2),
                            "low": round(min(price, close) * 0.999, 2),
                            "close": round(close, 2), "volume": round(rng.uniform(2, 200), 2)})
            price = close
        options = list(ACTIONS)
        rng.shuffle(options)
        job = Job(
            state={"provenance": "SYNTHETIC_REHEARSAL_NOT_TRADER_DATA",
                   "symbol": rng.choice(["BTCUSD", "ETHUSD", "SOLUSD"]),
                   "interval_minutes": 15, "position": "flat",
                   "features": {"lastClose": round(price, 2), "changePct": change,
                                "rsi14": round(rng.uniform(10, 90), 2),
                                "volatilityPct": round(rng.uniform(0.1, 2), 2),
                                "volumeRatio": round(rng.uniform(0.2, 3), 2)},
                   "recent_closed_candles": candles,
                   "missing": ["order_book", "funding", "news", "trader_history"]},
            instructions=RUBRIC,
            options=[{"name": name, "description": f"Synthetic {name} class."} for name in options],
        )
        examples.append({"job": job, "target_index": options.index(action)})
    return examples


def encode_example(tokenizer, labels, example: dict, max_length: int) -> dict:
    prompt = format_prompt(tokenizer, messages_for_job(example["job"], labels))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target = labels[example["target_index"]]
    input_ids = tokenizer.encode(prompt + target.text, add_special_tokens=False)
    if input_ids != prompt_ids + [target.token_id]:
        raise ValueError("label continuation does not match the serving tokenizer")
    if len(input_ids) > max_length:
        raise ValueError(f"example length {len(input_ids)} exceeds {max_length}; never truncate labels")
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + [target.token_id],
            "target_index": example["target_index"]}
