"""Pure historical input preparation; no network or GPU imports."""
import hashlib
import json
import math
from datetime import datetime, timezone

STEP = 14400
ACTIONS = ['long', 'short', 'flat']
INSTRUCTIONS = "Imitate the historical trader's position side at the end of the next hour using only this closed-candle snapshot and the assumed position side immediately before the cutoff. Long and short describe hypothetical signed exposure; flat means zero exposure. This is experimental behavior prediction, not price direction, a recommendation or an order. Training used hourly BitMEX XBTUSD; these spot markets and other candle intervals are out of distribution. Do not infer missing balances, leverage, news or future executions."

def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace('+00:00', 'Z')

def epoch(t):
    return int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp())

def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def features(candles):
    closes = [c['close'] for c in candles]
    changes = [b/a-1 for a,b in zip(closes,closes[1:])]
    mean = sum(changes)/len(changes)
    deltas = [b-a for a,b in zip(closes[-15:],closes[-14:])]
    gain = sum(max(d,0) for d in deltas)/14
    loss = sum(max(-d,0) for d in deltas)/14
    volume = sum(c['volume'] for c in candles[-21:-1])/20
    return dict(lastClose=closes[-1],changePct=(closes[-1]/closes[0]-1)*100,rsi14=(50 if gain == 0 else 100) if loss == 0 else 100-100/(1+gain/loss),volatilityPct=math.sqrt(sum((c-mean)**2 for c in changes)/len(changes))*100,volumeRatio=candles[-1]['volume']/volume if volume else None)

def index_candles(candles):
    result = {}
    for c in candles:
        t = c['time']
        if not isinstance(t,int) or t % 3600 or t in result:
            raise ValueError('duplicate or unaligned candle')
        if not all(isinstance(c[k],(int,float)) and math.isfinite(c[k]) for k in ('open','high','low','close','volume')):
            raise ValueError('nonfinite candle')
        if not (0 < c['low'] <= min(c['open'],c['close']) <= max(c['open'],c['close']) <= c['high']) or c['volume'] < 0:
            raise ValueError('invalid OHLCV')
        result[t] = c
    return result

def window(index, cutoff):
    try:
        prior = [index[t] for t in range(cutoff-96*3600,cutoff,3600)]
        execution = [index[t] for t in range(cutoff,cutoff+STEP,3600)]
    except KeyError as e:
        raise ValueError(f'missing required hour {e.args[0]}') from e
    aggregate = dict(time=cutoff,open=execution[0]['open'],high=max(c['high'] for c in execution),low=min(c['low'] for c in execution),close=execution[-1]['close'],volume=sum(c['volume'] for c in execution))
    return prior, aggregate

def job(symbol, source, cutoff, previous, prior):
    from .schemas import Job, Option
    return Job(state=dict(symbol=symbol,market=source+'; research transfer from BitMEX XBTUSD training',interval_minutes=60,data_cutoff=iso(cutoff),position_side_before_cutoff=previous,position_assumption='Hypothetical previous model prediction carried forward, initialized flat at replay start.',features=features(prior),volume_unit='base asset',recent_closed_candles=prior[-24:],missing=['equity','leverage','order_book','news','future_executions','actual_trader_position']),instructions=INSTRUCTIONS,options=[Option(name=a,description=d) for a,d in zip(ACTIONS,['Positive signed position after the next hour.','Negative signed position after the next hour.','Zero position after the next hour.'])])
