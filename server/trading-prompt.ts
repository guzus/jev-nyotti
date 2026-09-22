import type { Config } from './config.js';
import type { SystemOneRequest } from './contracts.js';
import type { Market } from './market.js';

export function tradingPrompt(config: Config, market: Market): SystemOneRequest {
  if (config.trainingStatus === 'fine_tuned') {
    return {
      model: config.modelId,
      state: {
        symbol: market.symbol, market: 'Kraken spot USD; research transfer from BitMEX XBTUSD training',
        interval_minutes: market.interval, data_cutoff: market.asOf,
        position_side_before_cutoff: 'flat', position_assumption: 'Hypothetical zero exposure, not a real trader account.',
        features: market.features, volume_unit: 'base asset',
        recent_closed_candles: market.candles.slice(-24),
        missing: ['equity', 'leverage', 'order_book', 'news', 'future_executions', 'actual_trader_position'],
      },
      questions: { direction: { type: 'choice',
        instructions: "Imitate the historical trader's position side at the end of the next hour using only this closed-candle snapshot and the assumed position side immediately before the cutoff. Long and short describe hypothetical signed exposure; flat means zero exposure. This is experimental behavior prediction, not price direction, a recommendation or an order. Training used hourly BitMEX XBTUSD; these spot markets and other candle intervals are out of distribution. Do not infer missing balances, leverage, news or future executions.",
        criteria: { long: 'Positive signed position after the next hour.', short: 'Negative signed position after the next hour.', flat: 'Zero position after the next hour.' },
      } },
    };
  }
  return {
    model: config.modelId,
    state: { symbol: market.symbol, market: 'Kraken spot USD, not futures', interval_minutes: market.interval,
      data_cutoff: market.asOf, position: 'flat', features: market.features,
      feature_definitions: { changePct: 'change across 96 closed candles', rsi14: 'simple mean gain/loss over 14 intervals', volatilityPct: 'population standard deviation of close returns in percent', volumeRatio: 'last closed candle volume / preceding 20-candle mean' },
      recent_closed_candles: market.candles.slice(-24), missing: ['order_book', 'funding', 'news', 'trader_history'] },
    questions: { direction: { type: 'choice', instructions: 'Given only the supplied closed-candle market snapshot, classify a hypothetical research stance for the next candle. No orders will be executed. Prefer hold if there is no clear directional evidence; a short is a directional research stance, not a spot short order. Do not infer missing news or trader history.',
      criteria: { long: 'Upward directional stance supported by the snapshot.', short: 'Downward directional stance supported by the snapshot.', hold: 'Insufficient directional evidence; remain flat.' } } },
  };
}
