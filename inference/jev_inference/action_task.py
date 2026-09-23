"""ACTION_V1 shared prompt contract. Pure stdlib: the only prompt builder for dataset,
training, replay and the live /action endpoint (the Node server sends raw candles).

Decision every 15 minutes at a UTC boundary. Input is 96 closed 15-minute candles
ending exactly at the cutoff plus the carried paper/teacher position state. Every
market quantity is unit-free (returns and relative volume) so BitMEX contract
volume and Kraken base volume are comparable. `server/paper.ts` ports only
apply_action and is parity-tested against this file.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

TASK = 'ACTION_V1'  # interface/protocol family (API, replay, paper rule)
# Prompt revision 2 (ACTION_V2.md): no self-referential time fields; neutral '<ASSET>/USD' market.
PROMPT_REVISION = 2
INTERVAL_MINUTES = 15
STEP = INTERVAL_MINUTES * 60
LOOKBACK = 96
SHOWN = 24

FLAT_OPTIONS = ('hold', 'open_long', 'open_short')
POSITION_OPTIONS = ('hold', 'add', 'reduce', 'close')
ALL_ACTIONS = ('hold', 'open_long', 'open_short', 'add', 'reduce', 'close')

INSTRUCTIONS = (
    "Imitate the historical BitMEX XBTUSD trader's first execution episode in the next 15 minutes, "
    "given only this closed-candle snapshot and the current position state. "
    "Choose hold when the trader would not execute. Market values are unit-free returns and relative volume. "
    "This is experimental behavior imitation, not a recommendation or an order. "
    "Do not infer missing balances, leverage, news, order book or future prices."
)

DESCRIPTIONS = {
    'hold': 'No execution in the next 15 minutes.',
    'open_long': 'Buy from zero exposure, opening a long position.',
    'open_short': 'Sell from zero exposure, opening a short position.',
    'add': 'Execute in the direction of the current position, increasing it.',
    'reduce': 'Execute against the current position, decreasing it without fully closing.',
    'close': 'Execute against the current position until it is fully closed (a reversal also counts as close).',
}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def options_for(side: str) -> tuple[str, ...]:
    if side == 'flat':
        return FLAT_OPTIONS
    if side in ('long', 'short'):
        return POSITION_OPTIONS
    raise ValueError('side must be flat, long or short')


def r(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        raise ValueError('nonfinite feature')
    out = round(value, digits)
    return 0.0 if out == 0 else out  # normalize -0.0 so JS and Python serialize identically


def validate_candles(candles: list[dict], cutoff: int) -> None:
    """`time` is candle OPEN in UTC seconds; the last candle must close exactly at cutoff."""
    if len(candles) != LOOKBACK:
        raise ValueError(f'requires {LOOKBACK} closed candles')
    if cutoff % STEP:
        raise ValueError('cutoff must be a 15-minute UTC boundary')
    for i, c in enumerate(candles):
        if c['time'] != cutoff - (LOOKBACK - i) * STEP:
            raise ValueError('candles must be contiguous and end at cutoff')
        values = [c[k] for k in ('open', 'high', 'low', 'close', 'volume')]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError('nonfinite candle')
        if c['low'] <= 0 or c['high'] < c['low'] or not c['low'] <= c['close'] <= c['high'] or c['volume'] < 0:
            raise ValueError('invalid OHLCV')


def market_view(candles: list[dict]) -> dict:
    closes = [c['close'] for c in candles]
    rets = [b / a - 1 for a, b in zip(closes, closes[1:])]
    mean_ret = sum(rets) / len(rets)
    vol = math.sqrt(sum((x - mean_ret) ** 2 for x in rets) / len(rets))
    deltas = [b - a for a, b in zip(closes[-15:], closes[-14:])]
    gain = sum(max(d, 0) for d in deltas) / 14
    loss = sum(max(-d, 0) for d in deltas) / 14
    rsi = (50.0 if gain == 0 else 100.0) if loss == 0 else 100 - 100 / (1 + gain / loss)
    mean_volume = sum(c['volume'] for c in candles) / len(candles)
    last = closes[-1]

    def rel_volume(v: float) -> float:
        return r(v / mean_volume, 3) if mean_volume > 0 else 0.0

    features = dict(
        return_1h_pct=r(100 * (last / closes[-5] - 1)),
        return_4h_pct=r(100 * (last / closes[-17] - 1)),
        return_24h_pct=r(100 * (last / candles[0]['open'] - 1)),
        rsi14=r(rsi, 2),
        volatility_pct=r(100 * vol),
        from_24h_high_pct=r(100 * (last / max(c['high'] for c in candles) - 1)),
        from_24h_low_pct=r(100 * (last / min(c['low'] for c in candles) - 1)),
        last_volume_rel=rel_volume(candles[-1]['volume']),
    )
    recent = []
    for c in candles[-SHOWN:]:
        recent.append(dict(
            ret_pct=r(100 * (c['close'] / c['open'] - 1)),
            high_pct=r(100 * (c['high'] / c['open'] - 1)),
            low_pct=r(100 * (c['low'] / c['open'] - 1)),
            vol_rel=rel_volume(c['volume']),
        ))
    return dict(features=features, recent=recent)


def position_view(position: dict, cutoff: int, mark: float) -> dict:
    """position: {side, entry_price|None, opened_at|None, last_trade_at|None} (UTC seconds).

    Revision 2 shows only side and unrealized return. Time since the holder's own last execution
    and position age are deliberately omitted: under teacher forcing they encode the trader's own
    bursts, and in closed loop the model's quiet stretch fed back into itself (ACTION_V1 failure).
    """
    side = position['side']
    options_for(side)
    view = dict(side=side)
    if side != 'flat':
        entry = position['entry_price']
        if not entry or entry <= 0:
            raise ValueError('open position requires entry_price')
        signed = mark / entry - 1 if side == 'long' else entry / mark - 1
        view.update(unrealized_return_pct=r(100 * signed))
    return view


def market_label(asset: str) -> str:
    """Neutral market string used identically in training, replay and live serving."""
    if not asset.isalnum() or not asset.isupper() or len(asset) > 10:
        raise ValueError('asset must be an uppercase ticker')
    return f'{asset}/USD'


def build_job(*, candles: list[dict], cutoff: int, position: dict, market: str, order: list[str] | None = None) -> dict:
    """Return the scoring Job as a plain dict. `order` permutes options (training only)."""
    validate_candles(candles, cutoff)
    names = list(order) if order is not None else list(options_for(position['side']))
    if sorted(names) != sorted(options_for(position['side'])):
        raise ValueError('options do not match position side')
    view = market_view(candles)
    state = dict(task=TASK, prompt_revision=PROMPT_REVISION, market=market, interval_minutes=INTERVAL_MINUTES, data_cutoff=iso(cutoff),
                 position=position_view(position, cutoff, candles[-1]['close']),
                 features=view['features'], recent_closed_candles=view['recent'],
                 missing=['equity', 'leverage', 'order_book', 'news', 'future_prices'])
    return dict(state=state, instructions=INSTRUCTIONS,
                options=[dict(name=n, description=DESCRIPTIONS[n]) for n in names])


# Paper execution rule for replay/live (disclosed; teacher sizes are not imitated in V1).
MAX_UNITS = 3


def apply_action(position: dict, action: str, price: float, t: int) -> dict:
    """Unit-sized paper inventory. open=1 unit, add=+1 (max 3), reduce=halve, close=flat."""
    if action not in options_for(position['side']):
        raise ValueError('action not allowed for position side')
    p = dict(position)
    if action == 'hold':
        return p
    p['last_trade_at'] = t
    if action in ('open_long', 'open_short'):
        p.update(side='long' if action == 'open_long' else 'short', units=1.0, entry_price=price, opened_at=t)
    elif action == 'add':
        units = p['units']
        added = min(1.0, MAX_UNITS - units)
        if added > 0:
            p['entry_price'] = (units + added) / (units / p['entry_price'] + added / price)
            p['units'] = units + added
    elif action == 'reduce':
        p['units'] = p['units'] / 2
    elif action == 'close':
        p.update(side='flat', units=0.0, entry_price=None, opened_at=None)
    return p


def flat_position() -> dict:
    return dict(side='flat', units=0.0, entry_price=None, opened_at=None, last_trade_at=None)
