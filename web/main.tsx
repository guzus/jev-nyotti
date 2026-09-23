import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowDownLeft, ArrowUpRight, Clock3, Copy, Link2, LoaderCircle,
  Minus, RefreshCw, Github,
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import './styles.css';
import { Performance } from './performance.js';
import { startAnalytics } from './analytics.js';
import { ActionDecisionView, PaperOverview, actionModelLabel, type ActionDecision } from './action.js';
import { PolicyValidation } from './action-performance.js';
import { policyInfo } from './policy-info.js';

type SymbolCode = import('../server/contracts.js').TradeRequest['symbol'];
type Interval = 15 | 60 | 240;
type Action = 'long' | 'short' | 'hold' | 'flat';
type Status = {
  model: string; trainingStatus: 'base' | 'fine_tuned' | 'action_v1'; providerConfigured: boolean; task?: 'ACTION_V1' | null;
  action?: { model: string; revision: string } | null;
  scheduledAnalysisEnabled?: boolean; apiAuthRequired: boolean; inferenceMode: 'live' | 'unconfigured'; gaMeasurementId?: string | null;
};
type Candle = { time: number; open: number; high: number; low: number; close: number; volume: number };
type Market = {
  symbol: SymbolCode; interval: Interval; source: string; asOf: string; fetchedAt: string;
  cachedDecision: AnyDecision | null;
  cache: { refreshIntervalMs: number; checkedAt: string | null; nextRefreshAt: string | null; refreshing: boolean; error: string | null };
  candles: Candle[];
  features: { lastClose: number; changePct: number; rsi14: number | null; volatilityPct: number | null; volumeRatio: number | null };
};
type Decision = {
  id: string; symbol: SymbolCode; interval: Interval; action: Action; summary: string;
  model: string; trainingStatus: 'base' | 'fine_tuned'; generatedAt: string; marketAsOf: string;
  latencyMs: number; cached: boolean;
  semantics?: 'next_hour_position_side';
  scores?: Partial<Record<Action, number>>; scoreType: 'model_relative_likelihood' | 'not_available';
};
type AnyDecision = Decision | ActionDecision;
const isAction = (d: AnyDecision | null): d is ActionDecision => !!d && 'task' in d && d.task === 'ACTION_V1';
type Asset = { symbol: SymbolCode; code: string; name: string; icon: string };
// Dated CoinGecko global-volume snapshot; see docs/volume-ranking.json.
const ASSETS: Asset[] = [
  { symbol: 'BTCUSD', code: 'BTC', name: 'Bitcoin', icon: '₿' },
  { symbol: 'ETHUSD', code: 'ETH', name: 'Ethereum', icon: 'Ξ' },
  { symbol: 'XRPUSD', code: 'XRP', name: 'XRP', icon: 'X' },
  { symbol: 'SOLUSD', code: 'SOL', name: 'Solana', icon: '◎' },
  { symbol: 'DOGEUSD', code: 'DOGE', name: 'Dogecoin', icon: 'Ð' },
  { symbol: 'BNBUSD', code: 'BNB', name: 'BNB', icon: 'B' },
  { symbol: 'SUIUSD', code: 'SUI', name: 'Sui', icon: 'S' },
  { symbol: 'NEARUSD', code: 'NEAR', name: 'NEAR Protocol', icon: 'N' },
  { symbol: 'PEPEUSD', code: 'PEPE', name: 'Pepe', icon: 'P' },
  { symbol: 'ZECUSD', code: 'ZEC', name: 'Zcash', icon: 'Z' },
];
// Preserve the identity and chart for previously published share links.
const LEGACY_ASSETS: Asset[] = [
  { symbol: 'ADAUSD', code: 'ADA', name: 'Cardano', icon: 'A' },
  { symbol: 'AVAXUSD', code: 'AVAX', name: 'Avalanche', icon: 'A' },
  { symbol: 'LINKUSD', code: 'LINK', name: 'Chainlink', icon: 'L' },
  { symbol: 'DOTUSD', code: 'DOT', name: 'Polkadot', icon: '●' },
  { symbol: 'LTCUSD', code: 'LTC', name: 'Litecoin', icon: 'Ł' },
];
const ACTIONS = {
  long: { name: '롱 관점', label: 'LONG', icon: ArrowUpRight },
  short: { name: '숏 관점', label: 'SHORT', icon: ArrowDownLeft },
  hold: { name: '관망', label: 'HOLD', icon: Minus },
  flat: { name: '무포지션', label: 'FLAT', icon: Minus },
};
// Korean market convention: rising = red, falling = blue. Mirrors the CSS custom properties.
const CHART = { up: '#bd3425', down: '#2058c7', grid: '#ebe6dc', tick: '#736c60', cursor: '#b8b1a3' };
const priceDigits = (value: number) => value < 0.0001 ? 10 : value < 0.01 ? 8 : value < 1 ? 6 : value < 10 ? 4 : 2;
const currency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: priceDigits(value) }).format(value);
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

const BrandMark = () => <svg className="brand-mark" viewBox="0 0 64 64" aria-hidden="true" focusable="false">
  <rect width="64" height="64" rx="18" fill="currentColor" />
  <path d="M39 27v15a8 8 0 0 1-8 8h-9" fill="none" stroke="#f4f1ea" strokeWidth="7" strokeLinecap="round" />
  <circle cx="39" cy="15" r="5.5" fill="#e0432f" />
</svg>;

function PriceChart({ candles, interval, trend }: { candles: Candle[]; interval: Interval; trend: 'up' | 'down' }) {
  const color = CHART[trend];
  return <div className="price-chart" role="img" aria-label={`${candles.length}개 ${intervalLabel(interval)} 봉의 종가 추이`}>
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={candles} margin={{ top: 16, right: 0, left: 0, bottom: 0 }}>
        <defs><linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.18} /><stop offset="100%" stopColor={color} stopOpacity={0} /></linearGradient></defs>
        <CartesianGrid vertical={false} stroke={CHART.grid} />
        <XAxis dataKey="time" axisLine={false} tickLine={false} minTickGap={56} tick={{ fill: CHART.tick, fontSize: 11 }} tickMargin={12} tickFormatter={(value: number) => new Intl.DateTimeFormat('ko-KR', interval === 15 ? { hour: '2-digit', minute: '2-digit', hour12: false } : { month: 'numeric', day: 'numeric' }).format(value * 1000)} />
        <YAxis orientation="right" domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{ fill: CHART.tick, fontSize: 11 }} tickMargin={8} width={candles.at(-1)!.close < 0.0001 ? 88 : 64} tickFormatter={(value: number) => number(value, value < 100 ? priceDigits(value) : 0)} />
        <Tooltip content={({ active, payload }) => {
          const candle = payload?.[0]?.payload as Candle | undefined;
          if (!active || !candle) return null;
          return <div className="chart-tooltip"><span>{clock(new Date(candle.time * 1000).toISOString())}</span><strong>{currency(candle.close)}</strong><small>종가 · 거래량 {number(candle.volume)}</small></div>;
        }} cursor={{ stroke: CHART.cursor, strokeDasharray: '3 5' }} />
        <Area type="linear" dataKey="close" stroke={color} strokeWidth={1.75} fill="url(#price-fill)" isAnimationActive={false} activeDot={{ r: 4.5, stroke: '#ffffff', strokeWidth: 2.5, fill: color }} />
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
  const [decision, setDecision] = useState<AnyDecision | null>(null);
  const [decisionError, setDecisionError] = useState('');
  const [shared, setShared] = useState(() => new URLSearchParams(window.location.search).has('decision'));
  const [sharedLoading, setSharedLoading] = useState(false);
  const [copyState, setCopyState] = useState('');
  const sharedAbort = useRef<AbortController | null>(null);
  const asset = [...ASSETS, ...LEGACY_ASSETS].find((item) => item.symbol === symbol)!;

  useEffect(() => {
    const controller = new AbortController();
    setStatusError('');
    request<Status>('/api/status', { signal: controller.signal }).then((data) => {
      if (!controller.signal.aborted) { setStatus(data); startAnalytics(data.gaMeasurementId); }
    }).catch((error) => {
      if (!controller.signal.aborted) { setStatus(null); setStatusError(error.message); }
    });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    const controller = new AbortController();
    let polling: ReturnType<typeof window.setTimeout>;
    async function load(first = false) {
      if (first) setMarketLoading(true);
      try {
        const data = await request<Market>(`/api/market?symbol=${symbol}&interval=${interval}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setMarket(data); setMarketError('');
        // A shared URL is an immutable historical result, even as the chart updates.
        if (!shared) setDecision(data.cachedDecision);
      } catch (error) {
        if (!controller.signal.aborted) setMarketError(error instanceof Error ? error.message : '시세를 불러오지 못했어요.');
      } finally {
        if (!controller.signal.aborted) {
          setMarketLoading(false);
          polling = window.setTimeout(() => { void load(); }, 30000);
        }
      }
    }
    void load(true);
    return () => { controller.abort(); window.clearTimeout(polling); };
  }, [symbol, interval, refresh, shared]);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('decision');
    if (!id) return;
    const controller = new AbortController(); sharedAbort.current = controller;
    setSharedLoading(true);
    request<AnyDecision>(`/api/decisions/${encodeURIComponent(id)}`, { signal: controller.signal }).then((data) => {
      if (controller.signal.aborted) return;
      setMarket(null); setSymbol(data.symbol); setInterval(data.interval); setDecision(data);
    }).catch((error) => { if (!controller.signal.aborted) setDecisionError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setSharedLoading(false); });
    return () => controller.abort();
  }, []);


  function changeMarket(nextSymbol: SymbolCode, nextInterval: Interval) {
    if (nextSymbol === symbol && nextInterval === interval) return;
    sharedAbort.current?.abort();
    setShared(false); setSharedLoading(false); setMarket(null); setDecision(null); setDecisionError(''); setCopyState('');
    setSymbol(nextSymbol); setInterval(nextInterval);
    const url = new URL(window.location.href); url.searchParams.delete('decision'); window.history.replaceState(null, '', url);
  }

  function showLatest() {
    sharedAbort.current?.abort(); setSharedLoading(false); setShared(false); setDecisionError('');
    setDecision(market?.symbol === symbol && market?.interval === interval ? market.cachedDecision : null);
    const url = new URL(window.location.href); url.searchParams.delete('decision'); window.history.replaceState(null, '', url);
  }

  async function share(asJson = false) {
    if (!decision) return;
    try {
      const url = new URL(window.location.origin); url.searchParams.set('decision', decision.id);
      await navigator.clipboard.writeText(asJson ? JSON.stringify(decision, null, 2) : url.toString());
      setCopyState(asJson ? 'JSON 복사 완료' : '공유 링크 복사 완료');
    } catch { setCopyState('복사하지 못했어요. 브라우저 주소를 직접 복사해 주세요.'); }
  }

  const actionMode = status?.task === 'ACTION_V1';
  // ACTION_V1 identity comes from the served /action response, never a hardcoded model name.
  const actionModel = actionMode && decision && isAction(decision) ? actionModelLabel(decision) : null;
  const servedPolicy = actionMode ? policyInfo(status?.action?.revision) : null;
  const legacy = decision && !isAction(decision) ? decision : null;
  const action = legacy ? ACTIONS[legacy.action] : null;
  const ActionIcon = action?.icon || Minus;
  const rawScores = legacy?.scores;
  const scoreKeys: Action[] = ['long', 'short', rawScores?.flat != null ? 'flat' : 'hold'];
  const scoresValid = rawScores && scoreKeys.every((key) => typeof rawScores[key] === 'number' && Number.isFinite(rawScores[key]) && rawScores[key]! >= 0);
  const scoreTotal = scoresValid ? scoreKeys.reduce((total, key) => total + rawScores[key]!, 0) : 0;
  const busy = sharedLoading;
  const trend: 'up' | 'down' = market && market.features.changePct < 0 ? 'down' : 'up';
  const renderCacheStatus = () => <div className="cache-status" role="status">
    {shared ? <><span>공유된 시점의 판단</span><button className="secondary-button" onClick={showLatest}>최신 저장 결과 보기</button></> : <>
      <span><Clock3 size={14} aria-hidden="true" />{status?.scheduledAnalysisEnabled === false ? '자동 갱신 일시 중지' : actionMode ? (interval === 15 ? '전체 종목 · 15분 봉 마감마다 판단' : '1시간·4시간은 시장 보기 전용') : '전체 종목 · 10분마다 자동 갱신'}</span>
      <small>{market?.cache.error ? `갱신 지연 · ${market.cache.error}` : market?.cache.refreshing ? '새 판단을 갱신하고 있어요.' : market?.cache.checkedAt ? `최근 확인 ${clock(market.cache.checkedAt)}` : status?.scheduledAnalysisEnabled === false ? '저장된 결과만 표시합니다.' : '첫 자동 분석을 준비하고 있어요.'}</small>
      {market?.cache.error && decision && <small>이전 저장 결과를 표시하고 있어요.</small>}
    </>}
    {(statusError || status?.providerConfigured === false) && <small>{statusError ? '모델 연결 확인 실패' : '모델 서버 연결 전'}</small>}
  </div>;

  return <>
    <header className="site-header">
      <a className="brand" href="/" aria-label="jev뇨띠 홈"><BrandMark /><span className="brand-name">jev뇨띠</span></a>
      <div className="model-chip" title={actionMode ? '페이퍼 포지션을 이어가는 15분 행동 모방 실험' : status?.trainingStatus === 'fine_tuned' ? '거래 기록으로 학습한 실험용 LoRA' : 'Qwen3.5-4B 모델'}><span className="model-chip-name">{actionMode ? actionModel ?? 'ACTION_V1 모델' : 'Qwen3.5-4B'}</span><span className="model-chip-base">{actionMode ? `${servedPolicy?.version ?? 'ACTION'} · 15분` : status?.trainingStatus === 'fine_tuned' ? 'LoRA 학습' : status ? '기본 모델' : '연결 확인 중'}</span></div>
    </header>

    <main>
      <h1 className="sr-only">jev뇨띠 시장 분석</h1>
      <p className="ranking-note"><a href="https://www.coingecko.com/en/highlights/high-volume" target="_blank" rel="noopener noreferrer">CoinGecko 글로벌 24h 거래대금 TOP 10</a><span>스테이블 제외 · <time dateTime="2026-09-22T08:38:10Z">2026. 9. 22. 17:38 KST</time> 기준</span></p>
      <div className="market-bar">
        <div className="segmented asset-tabs" role="group" aria-label="거래 종목 선택">{ASSETS.map((item) => <button key={item.symbol} type="button" aria-pressed={symbol === item.symbol} onClick={() => changeMarket(item.symbol, interval)}><span className={`coin coin-${item.code.toLowerCase()}`} aria-hidden="true">{item.icon}</span>{item.code}</button>)}</div>
        <div className="segmented interval-tabs" role="group" aria-label="차트 시간 간격">{([15, 60, 240] as Interval[]).map((item) => <button key={item} type="button" aria-pressed={item === interval} onClick={() => changeMarket(symbol, item)}>{intervalLabel(item)}</button>)}</div>
      </div>

      <div className="workspace">
        <section className="card market-card" aria-label={`${asset.name} 시장 차트`}>
          <div className="market-heading">
            <div className="asset-identity">
              <span className={`coin coin-${asset.code.toLowerCase()} coin-large`} aria-hidden="true">{asset.icon}</span>
              <div>
                <div className="asset-name">{asset.name}</div>
                <div className="asset-code">{asset.code} / USD<span className="feed-state"><span className={`status-dot ${market ? 'online' : ''}`} aria-hidden="true" />{market ? market.source : marketLoading ? '연결 중' : '연결 확인 필요'}</span></div>
              </div>
            </div>
            <button type="button" className="icon-button" aria-label="시장 데이터와 연결 상태 새로고침" disabled={marketLoading || busy} onClick={() => setRefresh((count) => count + 1)}><RefreshCw size={17} className={marketLoading ? 'spin' : ''} /></button>
          </div>
          <div className="price-row">
            <strong className={marketLoading ? 'price-skeleton' : ''}>{market ? currency(market.features.lastClose) : marketLoading ? ' ' : '—'}</strong>
            {market && <span className="price-delta"><span className={`price-change ${trend}`}>{market.features.changePct >= 0 ? '+' : ''}{number(market.features.changePct)}%</span><small>구간 등락</small></span>}
          </div>

          {marketLoading && !market ? <div className="chart-placeholder" role="status"><div className="chart-loader"><LoaderCircle size={20} className="spin" /><span>시세 불러오는 중</span></div></div> : marketError && !market ? <div className="chart-placeholder chart-error" role="alert"><h3>차트를 불러오지 못했어요.</h3><p>{marketError}</p><button type="button" className="secondary-button" onClick={() => setRefresh((count) => count + 1)}><RefreshCw size={14} /> 다시 불러오기</button></div> : market?.candles.length ? <PriceChart candles={market.candles} interval={interval} trend={trend} /> : <div className="chart-placeholder"><p>표시할 시장 데이터가 없어요.</p></div>}
          {marketError && market && <p className="inline-error" role="status">시세 갱신 지연 · 마지막 수신 데이터를 표시합니다.</p>}
          <div className="chart-caption"><span>{market ? `${market.candles.length}개 봉 · ${intervalLabel(interval)} 간격 · 종가` : '종가 기준 · USD'}</span><span>{market ? `${clock(market.asOf)} 기준` : '시장 데이터 수신 대기'}</span></div>

          <dl className="market-stats">
            <div><dt>RSI <small>14</small></dt><dd>{market?.features.rsi14 != null ? number(market.features.rsi14, 1) : '—'}</dd></div>
            <div><dt title="봉 수익률의 표준편차">변동성</dt><dd>{market?.features.volatilityPct != null ? `${number(market.features.volatilityPct)}%` : '—'}</dd></div>
            <div><dt title="평균 대비 거래량">거래량 비율</dt><dd>{market?.features.volumeRatio != null ? `${number(market.features.volumeRatio)}×` : '—'}</dd></div>
          </dl>
        </section>

        <aside className="card decision-card" aria-label="jev뇨띠 시장 판단" aria-busy={busy}>
          <div className="decision-heading"><h2>jev뇨띠 판단</h2><span>{asset.code} · {intervalLabel(interval)}</span></div>
          <div className="decision-content" aria-live="polite">
            {busy ? <div className="decision-empty"><LoaderCircle size={22} className="spin" /><p>공유 결과 불러오는 중</p></div> : isAction(decision) ? <><ActionDecisionView decision={decision} /><div className="share-actions"><button type="button" className="secondary-button" onClick={() => share()}><Link2 size={15} /> 공유 링크</button><button type="button" className="secondary-button" aria-label="판단 데이터 JSON 복사" onClick={() => share(true)}><Copy size={15} /> JSON</button></div></> : actionMode && interval !== 15 && !decision ? <div className="decision-empty"><p>ACTION_V1은 15분 봉 마감에서만 판단합니다.<br />1시간·4시간 차트는 시장 보기 전용이에요.</p><button type="button" className="secondary-button" onClick={() => changeMarket(symbol, 15)}>15분 판단 보기</button></div> : legacy && action ? <div className={`decision-result result-${legacy.action}`}>
              <div className="result-direction"><span className="direction-symbol"><ActionIcon size={30} strokeWidth={2} /></span><div><span className="direction-label">{action.label}</span><h3>{action.name}</h3></div></div>
              {scoresValid && scoreTotal > 0 && legacy.scoreType === 'model_relative_likelihood' ? <div className="score-chart"><div className="score-heading"><span>{legacy.semantics === 'next_hour_position_side' ? '1시간 후 포지션 상대 점수' : '행동별 상대 점수'}</span><span>수익 확률 아님</span></div>{scoreKeys.map((key) => <div className={`score-row score-${key}${key === legacy.action ? ' score-chosen' : ''}`} key={key}><span>{ACTIONS[key].label}</span><div className="score-track"><div style={{ width: `${rawScores[key]! / scoreTotal * 100}%` }} /></div><strong>{rawScores[key]!.toFixed(3)}</strong></div>)}</div> : <div className="score-unavailable">상대 점수 없음</div>}
              {legacy.semantics === 'next_hour_position_side' && <p className="experiment-note">이전 포지션 없음 가정 · BTC 1시간 학습을 다른 시장에도 적용한 실험입니다. 수익성 미검증.</p>}
              <div className="result-metadata">
                {shared && <span>{legacy.trainingStatus === 'fine_tuned' ? 'LoRA 학습 모델' : '기본 모델'}의 저장 결과</span>}
                <span><Clock3 size={12} aria-hidden="true" /> {legacy.cached ? '저장된 분석' : '새 분석'} · {clock(legacy.generatedAt)}</span>
                <span>시장 기준 {clock(legacy.marketAsOf)} · 최초 분석 {number(legacy.latencyMs / 1000, 1)}초</span>
              </div>
              <div className="share-actions"><button type="button" className="secondary-button" onClick={() => share()}><Link2 size={15} /> 공유 링크</button><button type="button" className="secondary-button" aria-label="판단 데이터 JSON 복사" onClick={() => share(true)}><Copy size={15} /> JSON</button></div>
            </div> : <div className="decision-empty"><div className="stance-options" aria-hidden="true"><span className="stance-long">LONG</span><span className="stance-short">SHORT</span><span className="stance-hold">HOLD</span></div><p>아직 저장된 판단이 없어요.<br />자동 분석이 끝나면 여기에 표시됩니다.</p></div>}
          </div>
          {decisionError && <div className="inline-error" role="alert">{decisionError}</div>}
          {copyState && <p className="copy-feedback" role="status">{copyState}</p>}
          {renderCacheStatus()}
        </aside>
      </div>
      {actionMode && <PaperOverview refresh={refresh} onSelect={(next) => changeMarket(next as SymbolCode, 15)} />}
      {actionMode && <PolicyValidation revision={status?.action?.revision ?? null} />}
      <Performance />
    </main>
    <footer>
      <span>{actionMode ? `${servedPolicy ? `${servedPolicy.version} · ${servedPolicy.model}` : '행동 모방'} · 교육용 · 주문 실행 없음` : status?.trainingStatus === 'fine_tuned' ? '거래 기록으로 학습한 Qwen3.5-4B · 교육용' : status ? 'Qwen3.5-4B 기본 모델 서빙 중' : '모델 연결 확인 중'}</span>
      <div className="footer-resources">
        <nav className="project-links" aria-label="프로젝트 링크">
          <a href="https://github.com/guzus/jev-nyotti" target="_blank" rel="noopener noreferrer" aria-label="GitHub 소스 코드 (새 탭)"><Github className="project-logo" size={17} aria-hidden="true" />GitHub<ArrowUpRight size={13} aria-hidden="true" /></a>
          <a href={actionMode ? servedPolicy?.hfUrl ?? 'https://huggingface.co/guzus/jev-nyotti-action' : 'https://huggingface.co/guzus/jev-nyotti'} target="_blank" rel="noopener noreferrer" aria-label={actionMode ? 'Hugging Face 서빙 모델 (새 탭)' : 'Hugging Face 학습 어댑터 (새 탭)'}><img className="project-logo" src="/huggingface.svg" width="19" height="18" alt="" />Hugging Face<ArrowUpRight size={13} aria-hidden="true" /></a>
          <a href="https://x.com/uncanny_guzus/status/2102341117232693582" target="_blank" rel="noopener noreferrer" aria-label="Release tweet (opens in a new tab)">Release tweet<ArrowUpRight size={13} aria-hidden="true" /></a>
        </nav>
        <span>DYOR NFA</span>
      </div>
    </footer>
  </>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
