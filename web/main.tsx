import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowDownLeft, ArrowRight, ArrowUpRight, Check, ChevronRight, Clock3,
  Code2, Copy, ExternalLink, Layers3, LoaderCircle, Minus, RefreshCw,
  Share2, Sparkles, X,
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
  long: { name: '롱 관점', label: 'LONG', icon: ArrowUpRight, description: '상승 방향에 무게를 두고 있어요.' },
  short: { name: '숏 관점', label: 'SHORT', icon: ArrowDownLeft, description: '하락 방향에 무게를 두고 있어요.' },
  hold: { name: '관망', label: 'HOLD', icon: Minus, description: '지금은 지켜보는 쪽에 무게를 두고 있어요.' },
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

function ApiDrawer({ onClose, configured }: { onClose: () => void; configured: boolean }) {
  const dialog = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [copyError, setCopyError] = useState(false);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.current?.querySelector<HTMLButtonElement>('button')?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
      if (event.key === 'Tab') {
        const items = dialog.current?.querySelectorAll<HTMLElement>('button, a[href], input, [tabindex="0"]');
        if (!items?.length) return;
        const first = items[0]; const last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    }
    document.addEventListener('keydown', keydown);
    return () => { document.body.style.overflow = originalOverflow; document.removeEventListener('keydown', keydown); previous?.focus(); };
  }, [onClose]);
  async function copy(key: string, text: string) {
    try { await navigator.clipboard.writeText(text); setCopied(key); setCopyError(false); }
    catch { setCopyError(true); }
  }
  const origin = window.location.origin;
  return <div className="drawer-backdrop" onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="api-drawer" ref={dialog} role="dialog" aria-modal="true" aria-labelledby="api-title">
      <div className="drawer-top"><span className="eyebrow">BUILD WITH JEV</span><button className="icon-button" onClick={onClose} aria-label="API 연결 닫기"><X size={21} /></button></div>
      <Code2 size={34} strokeWidth={1.4} className="drawer-symbol" />
      <h2 id="api-title">판단을 연결하세요.</h2>
      <p className="drawer-description">같은 Qwen3.5-4B 모델을 Jev와 기존 도구에서 호출하는 연결 주소입니다.</p>
      {!configured && <p className="drawer-connection-note">현재 모델 서버 연결 전입니다. 연결이 완료되면 추론 요청을 사용할 수 있어요.</p>}
      <div className="endpoint">
        <h3>TypeSafe / Jev</h3><p>System One 호환 엔드포인트</p>
        <div className="endpoint-value"><code>{origin}/v1/systemone</code><button className="icon-button" aria-label="TypeSafe 주소 복사" onClick={() => copy('jev', `${origin}/v1/systemone`)}>{copied === 'jev' ? <Check size={17} /> : <Copy size={17} />}</button></div>
      </div>
      <div className="integration-note"><Layers3 size={19} /><div><strong>서버에서 안전하게 연결</strong><p>Bearer API 키를 서버 환경 변수로 설정하세요. 프론트엔드 코드에는 넣지 마세요.</p></div></div>
      <div className="drawer-facts"><span>모델</span><code>Qwen/Qwen3.5-4B</code><span>호출 방식</span><code>POST /v1/systemone</code></div>
      <a className="text-link" href="https://docs.typesafe.ai/introduction" target="_blank" rel="noreferrer">TypeSafe 문서 보기 <ExternalLink size={14} /></a>
      <p className="copy-feedback" aria-live="polite">{copyError ? '복사하지 못했어요. 주소를 선택해 직접 복사해 주세요.' : copied ? '주소를 복사했어요.' : ''}</p>
    </div>
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
  const [apiOpen, setApiOpen] = useState(false);
  const [copyState, setCopyState] = useState('');
  const analyzeAbort = useRef<AbortController | null>(null);
  const sharedAbort = useRef<AbortController | null>(null);
  const asset = ASSETS.find((item) => item.symbol === symbol)!;
  const closeApi = useCallback(() => setApiOpen(false), []);

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
      setCopyState(asJson ? '판단 데이터를 복사했어요.' : '이 판단의 공유 링크를 복사했어요.');
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
      <nav aria-label="주 메뉴"><a href="#about" className="model-nav">모델 소개</a><button className="api-nav" onClick={() => setApiOpen(true)}><Code2 size={16} /> API 연결 <ArrowUpRight size={14} /></button></nav>
    </header>

    <main>
      <section className="intro" aria-labelledby="page-title">
        <div><div className="intro-kicker"><span className="small-rule" /> A DIFFERENT READ ON THE MARKET</div><h1 id="page-title">같은 시장,<br className="mobile-break" /> 다른 판단.</h1><p>시장의 움직임을 보고, AI의 관점과 비교해 보세요.</p></div>
        <div className="model-stamp"><span className="stamp-icon"><Layers3 size={20} strokeWidth={1.5} /></span><div><span className="eyebrow">POWERED BY</span><strong>Qwen3.5 <span>4B</span></strong><span className="base-label">BASE MODEL · 추가 학습 전</span></div></div>
      </section>

      <section className="terminal" aria-label="시장 분석">
        <div className="terminal-toolbar">
          <div className="asset-tabs" role="group" aria-label="거래 종목 선택">{ASSETS.map((item) => <button key={item.symbol} aria-pressed={symbol === item.symbol} className={symbol === item.symbol ? 'selected' : ''} onClick={() => changeMarket(item.symbol, interval)}><span className={`coin coin-${item.code.toLowerCase()}`}>{item.icon}</span>{item.code}<span className="quote-currency"> / USD</span></button>)}</div>
          <div className="feed-state"><span className={`status-dot ${market ? 'online' : ''}`} />{market ? `${market.source} 시장 데이터` : marketLoading ? '시장 연결 중' : '시장 연결 확인 필요'}</div>
        </div>

        <div className="terminal-body">
          <section className="market-panel" aria-label={`${asset.name} 시장 차트`}>
            <div className="market-heading"><div><div className="asset-name">{asset.name}<span>{asset.code} / USD</span></div><div className="price-row"><strong className={marketLoading ? 'price-skeleton' : ''}>{market ? currency(market.features.lastClose) : marketLoading ? '\u00A0' : '—'}</strong>{market && <span className={`price-change ${market.features.changePct >= 0 ? 'positive' : 'negative'}`}>{market.features.changePct >= 0 ? '+' : ''}{number(market.features.changePct)}%<small>구간 등락</small></span>}</div></div>
              <button className="icon-button refresh-button" aria-label="시장 데이터와 연결 상태 새로고침" disabled={marketLoading || busy} onClick={() => setRefresh((count) => count + 1)}><RefreshCw size={17} className={marketLoading ? 'spin' : ''} /></button>
            </div>
            <div className="chart-toolbar"><span>PRICE <span className="chart-label-detail">/ 종가</span></span><div className="interval-tabs" role="group" aria-label="차트 시간 간격">{([15, 60, 240] as Interval[]).map((item) => <button key={item} onClick={() => changeMarket(symbol, item)} aria-pressed={item === interval} className={item === interval ? 'selected' : ''}>{intervalLabel(item)}</button>)}</div></div>
            {marketLoading ? <div className="chart-placeholder" role="status"><div className="chart-loader"><LoaderCircle size={21} className="spin" /><span>실제 시장 데이터를 불러오고 있어요</span></div></div> : marketError ? <div className="chart-placeholder chart-error" role="alert"><span className="eyebrow">DATA UNAVAILABLE</span><h3>차트를 불러오지 못했어요.</h3><p>{marketError}</p><button className="secondary-button" onClick={() => setRefresh((count) => count + 1)}><RefreshCw size={14} /> 다시 불러오기</button></div> : market?.candles.length ? <PriceChart candles={market.candles} interval={interval} /> : <div className="chart-placeholder"><p>표시할 시장 데이터가 없어요.</p></div>}
            <div className="chart-caption"><span>{market ? `${market.candles.length}개 봉 · ${intervalLabel(interval)} 간격` : '종가 기준 · USD'}</span><span>{market ? `${clock(market.asOf)} 기준` : '시장 데이터 수신 대기'}</span></div>
            <div className="market-stats">
              <div><span>RSI <small>14</small></span><strong>{market?.features.rsi14 != null ? number(market.features.rsi14, 1) : '—'}</strong><small>상대강도지수</small></div>
              <div><span>변동성</span><strong>{market?.features.volatilityPct != null ? `${number(market.features.volatilityPct)}%` : '—'}</strong><small>봉 수익률 기준</small></div>
              <div><span>거래량 비율</span><strong>{market?.features.volumeRatio != null ? `${number(market.features.volumeRatio)}×` : '—'}</strong><small>평균 대비 거래량</small></div>
            </div>
          </section>

          <aside className="decision-panel" aria-label="AI 시장 판단" aria-busy={busy}>
            <div className="decision-heading"><span className="eyebrow">THE OTHER PERSPECTIVE</span><span className="model-pill">4B</span></div>
            <h2>AI의 시선</h2><p className="decision-intro">{asset.code}의 흐름을 읽고<br />지금의 관점을 제안해요.</p>
            <div className="decision-content" aria-live="polite">
              {busy ? <div className="decision-empty thinking"><div className="thinking-orbit"><Sparkles size={27} strokeWidth={1.3} /></div><h3>{sharedLoading ? '공유된 판단을 불러오는 중' : '시장의 맥락을 읽는 중'}</h3><p>{sharedLoading ? '저장된 분석 결과를 확인하고 있어요.' : '가격, 거래량, 최근 흐름을 함께 살펴봐요.'}</p></div> : decision && action ? <div className={`decision-result result-${decision.action}`}>
                <div className="result-direction"><span className="direction-symbol"><ActionIcon size={34} strokeWidth={1.5} /></span><div><span>{action.label}</span><h3>{action.name}</h3></div></div>
                <p className="result-summary">{decision.summary || action.description}</p>
                {scoresValid && scoreTotal > 0 && decision.scoreType === 'model_relative_likelihood' ? <div className="score-chart"><div className="score-heading">행동별 상대 점수</div>{(['long', 'short', 'hold'] as const).map((key) => <div className={`score-row score-${key}`} key={key}><span>{ACTIONS[key].label}</span><div className="score-track"><div style={{ width: `${rawScores[key] / scoreTotal * 100}%` }} /></div><strong>{rawScores[key].toFixed(3)}</strong></div>)}<p>모델의 상대 점수이며, 보정된 성공 확률이 아니에요.</p></div> : <div className="score-unavailable">이 응답은 행동별 점수를 제공하지 않아요.</div>}
                <div className="result-metadata"><span><Clock3 size={12} /> {number(decision.latencyMs / 1000, 1)}초{decision.cached ? ' · 저장된 결과' : ''}</span><span>{clock(decision.generatedAt)}</span></div>
                <div className="result-data-date">판단에 사용한 데이터 · {clock(decision.marketAsOf)}</div>
                <div className="share-actions"><button className="secondary-button" onClick={() => share()}><Share2 size={14} /> 판단 공유</button><button className="icon-button" aria-label="판단 데이터 JSON 복사" onClick={() => share(true)}><Copy size={16} /></button></div>
              </div> : <div className="decision-empty"><div className="empty-visual" aria-hidden="true"><span /><span /><span /><Sparkles size={24} strokeWidth={1.4} /></div><h3>당신의 관점은 어떤가요?</h3><p>롱, 숏, 아니면 관망.<br />AI의 판단과 나란히 놓아보세요.</p></div>}
            </div>
            {decisionError && <div className="inline-error" role="alert">{decisionError}</div>}
            {copyState && <p className="copy-feedback" role="status">{copyState}</p>}
            <div className="decision-action"><button className="analyze-button" disabled={!canAnalyze} onClick={analyze}>{analyzing ? <LoaderCircle size={18} className="spin" /> : <Sparkles size={18} />}<span>{analyzing ? '판단을 생성하고 있어요' : decision ? '다시 판단하기' : 'AI 판단 보기'}</span>{!analyzing && <ArrowRight size={18} />}</button>
              <p className="model-state">{statusError ? '모델 연결 상태를 확인하지 못했어요. 새로고침해 주세요.' : !status ? '모델 연결 확인 중' : !status.providerConfigured ? '모델 서버 연결 전 · 현재는 시장 데이터만 볼 수 있어요.' : 'Qwen3.5-4B · 공개 기본 모델'}</p>
            </div>
          </aside>
        </div>
        <div className="terminal-foot"><span><span className="foot-dot" /> MARKET DATA × LANGUAGE MODEL</span><span>매매 실행 없이, 판단만 살펴보는 실험</span></div>
      </section>

      <section id="about" className="about-section" aria-labelledby="about-title"><div className="about-index">01 <span>/ ABOUT THE LAB</span></div><div className="about-copy"><h2 id="about-title">판단의 차이를 탐구하는 공간.</h2><p>Jev Trading Lab은 실제 시장 데이터를 Qwen3.5-4B에 전달하고, 모델이 읽어낸 관점을 보여줍니다. 현재는 추가 학습을 하지 않은 기본 모델입니다. 워뇨띠 매매기록으로 학습한 모델이 아닙니다.</p></div><button className="about-api" onClick={() => setApiOpen(true)}><span>내 도구에서도<br />같은 모델 사용하기</span><span className="round-arrow"><ArrowUpRight size={22} /></span></button></section>
    </main>
    <footer><span>jev / trading lab <span className="footer-year">© {new Date().getFullYear()}</span></span><p>연구·체험용 분석입니다. 실제 주문은 실행하지 않으며, 수익을 보장하지 않습니다.</p><a href="https://docs.typesafe.ai/introduction" target="_blank" rel="noreferrer">TypeSafe <ChevronRight size={13} /></a></footer>
    {apiOpen && <ApiDrawer onClose={closeApi} configured={!!status?.providerConfigured} />}
  </>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
