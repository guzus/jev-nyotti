"""Causal XBTUSD inverse inventory accounting. No account equity is invented.

Contract quantity is signed USD1 face-value contracts; entry price is harmonic.
Funding/fees are separate BTC cash movements and never change contract inventory.
"""
from dataclasses import dataclass
import math


@dataclass
class Inventory:
    quantity: int = 0
    entry_price: float | None = None
    opened_at: float | None = None
    last_trade_at: float | None = None
    realized_xbt: float = 0.0

    def apply(self, delta: int, price: float, timestamp: float) -> str:
        if type(delta) is not int or delta == 0 or not math.isfinite(price) or price <= 0 or not math.isfinite(timestamp):
            raise ValueError('invalid fill')
        if self.last_trade_at is not None and timestamp < self.last_trade_at:
            raise ValueError('fills must be chronological')
        prior = self.quantity
        next_quantity = prior + delta
        if prior == 0:
            self.entry_price = price
            self.opened_at = timestamp
            action = 'open_long' if delta > 0 else 'open_short'
        elif prior * delta > 0:
            assert self.entry_price is not None
            self.entry_price = (abs(prior)+abs(delta))/(abs(prior)/self.entry_price+abs(delta)/price)
            action = 'add_long' if prior > 0 else 'add_short'
        else:
            assert self.entry_price is not None
            closed = min(abs(prior), abs(delta))
            self.realized_xbt += (1 if prior > 0 else -1)*closed*(1/self.entry_price-1/price)
            if next_quantity == 0:
                self.entry_price = None
                self.opened_at = None
                action = 'close_long' if prior > 0 else 'close_short'
            elif prior*next_quantity < 0:
                self.entry_price = price
                self.opened_at = timestamp
                action = 'reverse_to_long' if next_quantity > 0 else 'reverse_to_short'
            else:
                action = 'reduce_long' if prior > 0 else 'reduce_short'
        self.quantity = next_quantity
        self.last_trade_at = timestamp
        return action

    def snapshot(self, mark: float, cutoff: float) -> dict:
        if not math.isfinite(mark) or mark <= 0 or not math.isfinite(cutoff):
            raise ValueError('invalid mark or cutoff')
        if self.last_trade_at is not None and self.last_trade_at >= cutoff:
            raise ValueError('snapshot must exclude fills at or after cutoff')
        unrealized = self.quantity*(1/self.entry_price-1/mark) if self.entry_price else 0.0
        return dict(quantity_contracts=self.quantity,
                    side='long' if self.quantity>0 else 'short' if self.quantity<0 else 'flat',
                    entry_price=self.entry_price,
                    position_age_seconds=cutoff-self.opened_at if self.opened_at is not None else 0,
                    time_since_trade_seconds=cutoff-self.last_trade_at if self.last_trade_at is not None else None,
                    unrealized_xbt=unrealized,
                    unrealized_return_on_entry_value=unrealized/(abs(self.quantity)/self.entry_price) if self.entry_price else 0.0)


def classify_delta(prior: int, delta: int) -> str:
    """Direction/intent only; no inferred fills or prices."""
    if type(prior) is not int or type(delta) is not int:
        raise ValueError('integral quantities required')
    if not delta:return 'hold'
    if not prior:return 'open_long' if delta>0 else 'open_short'
    if prior*delta>0:return 'add_long' if prior>0 else 'add_short'
    after=prior+delta
    if not after:return 'close_long' if prior>0 else 'close_short'
    if prior*after<0:return 'reverse_to_long' if after>0 else 'reverse_to_short'
    return 'reduce_long' if prior>0 else 'reduce_short'
