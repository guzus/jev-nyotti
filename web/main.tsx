import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowDownLeft, ArrowRight, ArrowUpRight, Clock3, Copy, LoaderCircle,
  Minus, RefreshCw, Share2, Sparkles,
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import './styles.css';

type SymbolCode = 'BTCUSD' | 'ETHUSD' | 'SOLUSD';
type Interval = 15 | 60 | 240;
type Action = 'long' | 'short' | 'hold';
type Status = {
  model: string; trainingStatus: 'base'; providerConfigured: boolean;
  apiAuthRequired: boolean; inferenceMode: 'live' | 'unconfigured';
};
type Candle = { time: number; open: number; high: number; low: number; close: number; volume: number };
type Market = {
  symbol: SymbolCode; interval: Interval; source: string; asOf: string; fetchedAt: string;
  candles: Candle[];
  features: { lastClose: number; changePct: number; rsi14: number | null; volatilityPct: number | null; volumeRatio: number | null };
};
type Decision = {
  id: string; symbol: SymbolCode; interval: Interval; action: Action; summary: string;
  model: string; trainingStatus: 'base'; generatedAt: string; marketAsOf: string;
  latencyMs: number; cached: boolean;
  scores?: Record<Action, number>; scoreType: 'model_relative_likelihood' | 'not_available';
};
const ASSETS: { symbol: SymbolCode; code: string; name: string; icon: string }[] = [
  { symbol: 'BTCUSD', code: 'BTC', name: 'Bitcoin', icon: '₿' },
  { symbol: 'ETHUSD', code: 'ETH', name: 'Ethereum', icon: 'Ξ' },
  { symbol: 'SOLUSD', code: 'SOL', name: 'Solana', icon: '◎' },
];
const ACTIONS = {
  long: { name: '롱 관점', label: 'LONG', icon: ArrowUpRight },
  short: { name: '숏 관점', label: 'SHORT', icon: ArrowDownLeft },
  hold: { name: '관망', label: 'HOLD', icon: Minus },
};
const currency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const number = (value: number, digits = 2) => new Intl.NumberFormat('ko-KR', { maximumFractionDigits: digits }).format(value);
const clock = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '시각 정보 없음' : new Intl.DateTimeFormat('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }).format(date);
};
const intervalLabel = (interval: Interval) => interval === 15 ? '15분' : interval === 60 ? '1시간' : '4시간';

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error?.message || `요청을 처리하지 못했어요. (${response.status})`);
  if (!body) throw new Error('서버 응답을 읽지 못했어요. 다시 시도해 주세요.');
  return body as T;
}

function PriceChart({ candles, interval }: { candles: Candle[]; interval: Interval }) {
  return <div className="price-chart" role="img" aria-label={`${candles.length}개 ${intervalLabel(interval)} 봉의 종가 추이`}>
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={candles} margin={{ top: 20, right: 8, left: 8, bottom: 0 }}>
        <defs><linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#c4ef8d" stopOpacity={0.2} /><stop offset="100%" stopColor="#c4ef8d" stopOpacity={0} /></linearGradient></defs>
        <CartesianGrid vertical={false} stroke="#29362f" strokeDasharray="3 6" />
        <XAxis dataKey="time" axisLine={false} tickLine={false} minTickGap={54} tick={{ fill: '#9ba99f', fontSize: 11 }} tickMargin={17} tickFormatter={(value: number) => new Intl.DateTimeFormat('ko-KR', interval === 15 ? { hour: '2-digit', minute: '2-digit', hour12: false } : { month: 'numeric', day: 'numeric' }).format(value * 1000)} />
        <YAxis orientation="right" domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{ fill: '#9ba99f', fontSize: 11 }} tickMargin={10} width={76} tickFormatter={(value: number) => number(value, value < 100 ? 2 : 0)} />
        <Tooltip content={({ active, payload }) => {
          const candle = payload?.[0]?.payload as Candle | undefined;
          if (!active || !candle) return null;
          return <div className="chart-tooltip"><span>{clock(new Date(candle.time * 1000).toISOString())}</span><strong>{currency(candle.close)}</strong><small>종가 · 거래량 {number(candle.volume)}</small></div>;
        }} cursor={{ stroke: '#849d88', strokeDasharray: '4 4' }} />
        <Area type="linear" dataKey="close" stroke="#c4ef8d" strokeWidth={2} fill="url(#price-fill)" isAnimationActive={false} activeDot={{ r: 4, stroke: '#17231d', strokeWidth: 3, fill: '#d2fb9e' }} />
      </AreaChart>
    </ResponsiveContainer>
  </div>;
}

function App() {
  const [symbol, setSymbol] = useState<SymbolCode>('BTCUSD');
  const [interval, setInterval] = useState<Interval>(15);
  const [status, setStatus] = useState<Status | null>(null);
  const [statusError, setStatusError] = useState('');
  const [market, setMarket] = useState<Market | null>(null);
  const [marketError, setMarketError] = useState('');
  const [marketLoading, setMarketLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [decisionError, setDecisionError] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [sharedLoading, setSharedLoading] = useState(false);
  const [copyState, setCopyState] = useState('');
  const analyzeAbort = useRef<AbortController | null>(null);
  const sharedAbort = useRef<AbortController | null>(null);
  const asset = ASSETS.find((item) => item.symbol === symbol)!;

  useEffect(() => {
    const controller = new AbortController();
    setStatusError('');
    request<Status>('/api/status', { signal: controller.signal }).then((data) => {
      if (!controller.signal.aborted) setStatus(data);
    }).catch((error) => {
      if (!controller.signal.aborted) { setStatus(null); setStatusError(error.message); }
    });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    const controller = new AbortController();
    setMarketLoading(true); setMarketError(''); setMarket(null);
    request<Market>(`/api/market?symbol=${symbol}&interval=${interval}`, { signal: controller.signal }).then((data) => {
      if (!controller.signal.aborted) setMarket(data);
    }).catch((error) => {
      if (!controller.signal.aborted) setMarketError(error.message);
    }).finally(() => { if (!controller.signal.aborted) setMarketLoading(false); });
    return () => controller.abort();
  }, [symbol, interval, refresh]);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('decision');
    if (!id) return;
    const controller = new AbortController(); sharedAbort.current = controller;
    setSharedLoading(true);
    request<Decision>(`/api/decisions/${encodeURIComponent(id)}`, { signal: controller.signal }).then((data) => {
      if (controller.signal.aborted) return;
      setSymbol(data.symbol); setInterval(data.interval); setDecision(data);
    }).catch((error) => { if (!controller.signal.aborted) setDecisionError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setSharedLoading(false); });
    return () => controller.abort();
  }, []);

  useEffect(() => () => analyzeAbort.current?.abort(), []);

  function changeMarket(nextSymbol: SymbolCode, nextInterval: Interval) {
    if (nextSymbol === symbol && nextInterval === interval) return;
    analyzeAbort.current?.abort(); sharedAbort.current?.abort();
    setAnalyzing(false); setSharedLoading(false); setDecision(null); setDecisionError(''); setCopyState('');
    setSymbol(nextSymbol); setInterval(nextInterval);
    const url = new URL(window.location.href); url.searchParams.delete('decision'); window.history.replaceState(null, '', url);
  }

  async function analyze() {
    analyzeAbort.current?.abort();
    const controller = new AbortController(); analyzeAbort.current = controller;
    setAnalyzing(true); setDecisionError(''); setCopyState('');
    try {
      const data = await request<Decision>('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol, interval }), signal: controller.signal,
      });
      if (!controller.signal.aborted) {
        setDecision(data);
        const url = new URL(window.location.href); url.searchParams.set('decision', data.id); window.history.replaceState(null, '', url);
      }
    } catch (error) { if (!controller.signal.aborted) setDecisionError(error instanceof Error ? error.message : '분석을 완료하지 못했어요.'); }
    finally { if (!controller.signal.aborted) setAnalyzing(false); }
  }

  async function share(asJson = false) {
    if (!decision) return;
    try {
      const url = new URL(window.location.origin); url.searchParams.set('decision', decision.id);
      await navigator.clipboard.writeText(asJson ? JSON.stringify(decision, null, 2) : url.toString());
      setCopyState(asJson ? 'JSON 복사 완료' : '공유 링크 복사 완료');
    } catch { setCopyState('복사하지 못했어요. 브라우저 주소를 직접 복사해 주세요.'); }
  }

  const action = decision ? ACTIONS[decision.action] : null;
  const ActionIcon = action?.icon || Minus;
  const rawScores = decision?.scores;
  const scoresValid = rawScores && (['long', 'short', 'hold'] as const).every((key) => Number.isFinite(rawScores[key]) && rawScores[key] >= 0);
  const scoreTotal = scoresValid ? rawScores.long + rawScores.short + rawScores.hold : 0;
  const busy = analyzing || sharedLoading;
  const canAnalyze = status?.providerConfigured && !!market?.candles.length && !marketLoading && !busy;

  return <>
    <header className="site-header">
      <a className="brand" href="/" aria-label="Jev Trading Lab 홈"><span className="brand-mark">j<span>.</span></span><span>jev<span className="brand-divider">/</span><span className="brand-sub">trading lab</span></span></a>
      <div className="header-model"><span>Qwen3.5 <strong>4B</strong></span><span className="base-label">기본 모델</span></div>
    </header>

    <main>
      <h1 className="sr-only">Jev 시장 분석</h1>
      <section className="terminal" aria-label="시장 분석">
        <div className="terminal-toolbar">
          <div className="asset-tabs" role="group" aria-label="거래 종목 선택">{ASSETS.map((item) => <button key={item.symbol} aria-pressed={symbol === item.symbol} className={symbol === item.symbol ? 'selected' : ''} onClick={() => changeMarket(item.symbol, interval)}><span className={`coin coin-${item.code.toLowerCase()}`}>{item.icon}</span>{item.code}<span className="quote-currency"> / USD</span></button>)}</div>
          <div className="feed-state"><span className={`status-dot ${market ? 'online' : ''}`} />{market ? market.source : marketLoading ? '연결 중' : '연결 확인 필요'}</div>
        </div>

        <div className="terminal-body">
          <section className="market-panel" aria-label={`${asset.name} 시장 차트`}>
            <div className="market-heading"><div><div className="asset-name">{asset.name}<span>{asset.code} / USD</span></div><div className="price-row"><strong className={marketLoading ? 'price-skeleton' : ''}>{market ? currency(market.features.lastClose) : marketLoading ? '\u00A0' : '—'}</strong>{market && <span className={`price-change ${market.features.changePct >= 0 ? 'positive' : 'negative'}`}>{market.features.changePct >= 0 ? '+' : ''}{number(market.features.changePct)}%<small>구간 등락</small></span>}</div></div>
              <button className="icon-button refresh-button" aria-label="시장 데이터와 연결 상태 새로고침" disabled={marketLoading || busy} onClick={() => setRefresh((count) => count + 1)}><RefreshCw size={17} className={marketLoading ? 'spin' : ''} /></button>
            </div>
            <div className="chart-toolbar"><span>PRICE <span className="chart-label-detail">/ 종가</span></span><div className="interval-tabs" role="group" aria-label="차트 시간 간격">{([15, 60, 240] as Interval[]).map((item) => <button key={item} onClick={() => changeMarket(symbol, item)} aria-pressed={item === interval} className={item === interval ? 'selected' : ''}>{intervalLabel(item)}</button>)}</div></div>
            {marketLoading ? <div className="chart-placeholder" role="status"><div className="chart-loader"><LoaderCircle size={21} className="spin" /><span>시세 불러오는 중</span></div></div> : marketError ? <div className="chart-placeholder chart-error" role="alert"><h3>차트를 불러오지 못했어요.</h3><p>{marketError}</p><button className="secondary-button" onClick={() => setRefresh((count) => count + 1)}><RefreshCw size={14} /> 다시 불러오기</button></div> : market?.candles.length ? <PriceChart candles={market.candles} interval={interval} /> : <div className="chart-placeholder"><p>표시할 시장 데이터가 없어요.</p></div>}
            <div className="chart-caption"><span>{market ? `${market.candles.length}개 봉 · ${intervalLabel(interval)} 간격` : '종가 기준 · USD'}</span><span>{market ? `${clock(market.asOf)} 기준` : '시장 데이터 수신 대기'}</span></div>
            <div className="market-stats">
              <div><span>RSI <small>14</small></span><strong>{market?.features.rsi14 != null ? number(market.features.rsi14, 1) : '—'}</strong></div>
              <div><span title="봉 수익률의 표준편차">변동성</span><strong>{market?.features.volatilityPct != null ? `${number(market.features.volatilityPct)}%` : '—'}</strong></div>
              <div><span title="평균 대비 거래량">거래량 비율</span><strong>{market?.features.volumeRatio != null ? `${number(market.features.volumeRatio)}×` : '—'}</strong></div>
            </div>
          </section>

          <aside className="decision-panel" aria-label="AI 시장 판단" aria-busy={busy}>
            <div className="decision-heading"><h2>AI 판단</h2><span>{asset.code} · {intervalLabel(interval)}</span></div>
            <div className="decision-content" aria-live="polite">
              {busy ? <div className="decision-empty"><LoaderCircle size={22} className="spin" /><p>{sharedLoading ? '공유 결과 불러오는 중' : '분석 중 · 첫 요청은 1분 이상 걸릴 수 있어요.'}</p></div> : decision && action ? <div className={`decision-result result-${decision.action}`}>
                <div className="result-direction"><span className="direction-symbol"><ActionIcon size={34} strokeWidth={1.5} /></span><div><span>{action.label}</span><h3>{action.name}</h3></div></div>
                {scoresValid && scoreTotal > 0 && decision.scoreType === 'model_relative_likelihood' ? <div className="score-chart"><div className="score-heading">행동별 상대 점수</div>{(['long', 'short', 'hold'] as const).map((key) => <div className={`score-row score-${key}`} key={key}><span>{ACTIONS[key].label}</span><div className="score-track"><div style={{ width: `${rawScores[key] / scoreTotal * 100}%` }} /></div><strong>{rawScores[key].toFixed(3)}</strong></div>)}<p>상대 점수 · 수익 확률 아님</p></div> : <div className="score-unavailable">상대 점수 없음</div>}
                <div className="result-metadata"><span><Clock3 size={12} /> {number(decision.latencyMs / 1000, 1)}초{decision.cached ? ' · 저장된 결과' : ''}</span><span>{clock(decision.generatedAt)}</span></div>
                <div className="result-data-date">시장 기준 · {clock(decision.marketAsOf)}</div>
                <div className="share-actions"><button className="secondary-button" onClick={() => share()}><Share2 size={14} /> 공유</button><button className="icon-button" aria-label="판단 데이터 JSON 복사" onClick={() => share(true)}><Copy size={16} /></button></div>
              </div> : <div className="decision-empty"><div className="stance-options" aria-hidden="true"><span>LONG</span><span>SHORT</span><span>HOLD</span></div><p>지금 시장에 대한 모델의 선택은?</p></div>}
            </div>
            {decisionError && <div className="inline-error" role="alert">{decisionError}</div>}
            {copyState && <p className="copy-feedback" role="status">{copyState}</p>}
            <div className="decision-action"><button className="analyze-button" disabled={!canAnalyze} onClick={analyze}>{analyzing ? <LoaderCircle size={18} className="spin" /> : <Sparkles size={18} />}<span>{analyzing ? '분석 중' : decision ? '다시 분석' : '분석하기'}</span>{!analyzing && <ArrowRight size={18} />}</button>
              {(statusError || !status?.providerConfigured) && <p className="model-state" role="status">{statusError ? '모델 연결 확인 실패 · 새로고침해 주세요.' : !status ? '모델 연결 확인 중' : '모델 서버 연결 전'}</p>}
            </div>
          </aside>
        </div>
      </section>
    </main>
    <footer><span>워뇨띠 거래내역 미학습</span><span>연구용 · 주문 실행 없음</span></footer>
  </>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
