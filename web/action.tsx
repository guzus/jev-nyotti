import { useEffect, useState } from 'react';
import { ArrowDownLeft, ArrowUpRight, Minus, Plus, Scissors, X } from 'lucide-react';

export type ActionDecision = import('../server/contracts.js').ActionDecision;
type ActionName = ActionDecision['action'];
type Side = ActionDecision['paper']['side'];
type PaperRow = {
  symbol: string; transfer: ActionDecision['transfer']; updatedCutoff: string | null;
  position: { side: Side; units: number; entryPrice: number | null };
  markPrice: number | null; unitReturnPct: number; unrealizedPct: number; realizedPct: number; feesPct: number; trades: number;
  lastAction: { cutoff: string; action: ActionName | null } | null;
};

export const ACTION_LABELS: Record<ActionName, string> = {
  hold: '관망', open_long: '롱 진입', open_short: '숏 진입', add: '추가', reduce: '축소', close: '청산',
};
const ACTION_ICONS = { hold: Minus, open_long: ArrowUpRight, open_short: ArrowDownLeft, add: Plus, reduce: Scissors, close: X };
const SIDE_LABELS: Record<Side, string> = { flat: '무포지션', long: '롱', short: '숏' };
export const ACTION_DISCLAIMER = '교육용 · 주문 실행 없음 · BTC 외 코인은 검증되지 않은 전이';
const pct = (n: number, digits = 2) => `${n > 0 ? '+' : ''}${n.toFixed(digits)}%`;
const tone = (n: number) => n > 0 ? 'up' : n < 0 ? 'down' : '';
const time = (iso: string) => new Intl.DateTimeFormat('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(iso));
const price = (n: number) => new Intl.NumberFormat('en-US', { maximumFractionDigits: n < 1 ? 8 : 2 }).format(n);
const units = (n: number) => `${Number(n.toFixed(3))}단위`;
const position = (side: Side, n: number) => side === 'flat' ? SIDE_LABELS.flat : `${SIDE_LABELS[side]} ${units(n)}`;
const actionTone = (a: ActionName) => a === 'open_long' ? 'long' : a === 'open_short' ? 'short' : a === 'hold' ? 'hold' : 'flat';

/** Latest ACTION_V1 decision for one coin: executed paper action, probabilities, position and log. */
export function ActionDecisionView({ decision }: { decision: ActionDecision }) {
  const Icon = ACTION_ICONS[decision.action];
  const p = decision.paper;
  const log = decision.actionLog ?? [];
  const top = Math.max(...decision.options.map((o) => o.probability));
  return <div className={`decision-result result-${actionTone(decision.action)}`}>
    <div className="result-direction"><span className="direction-symbol"><Icon size={30} strokeWidth={2} /></span><div><span className="direction-label">{decision.action.toUpperCase()}</span><h3>{ACTION_LABELS[decision.action]}</h3></div></div>
    <p className={`transfer-badge ${decision.transfer}`}>{decision.transfer === 'in_distribution' ? 'BTC 15분 · 학습 분포' : '검증되지 않은 전이 · BTC로만 학습'}</p>
    <div className="score-chart"><div className="score-heading"><span>행동별 모델 확률 · 관망 마진 {decision.holdMargin}</span><span>수익 확률 아님</span></div>
      {decision.options.map((o) => <div className={`score-row action-score score-${actionTone(o.name)}${o.name === decision.action ? ' score-chosen' : ''}`} key={o.name}><span>{ACTION_LABELS[o.name]}</span><div className="score-track"><div style={{ width: `${o.probability * 100}%` }} /></div><strong>{(o.probability * 100).toFixed(1)}%</strong></div>)}
      {decision.options.find((o) => o.name === decision.action)!.probability < top && <p className="experiment-note">관망에 마진을 뺀 뒤 가장 높은 행동을 선택하므로 최고 확률과 다를 수 있습니다.</p>}
    </div>
    <dl className="paper-grid">
      <div><dt>페이퍼 포지션</dt><dd>{position(p.side, p.units)}</dd></div>
      <div><dt>평균 진입가</dt><dd>{p.entryPrice === null ? '—' : price(p.entryPrice)}</dd></div>
      <div><dt>평가 손익</dt><dd className={tone(p.unrealizedPct)}>{p.side === 'flat' ? '—' : pct(p.unrealizedPct)}</dd></div>
      <div><dt>실현 손익 · 수수료 후</dt><dd className={tone(p.realizedPct)}>{pct(p.realizedPct)}</dd></div>
    </dl>
    <p className="paper-note">1단위 명목 대비 % · 봉 마감 종가 {price(p.markPrice)} 체결 가정 · 단위당 수수료 0.075% · 누적 수수료 {p.feesPct.toFixed(3)}% · 거래 {p.trades}회{decision.missedCutoffs > 0 ? ` · 직전 ${decision.missedCutoffs}개 판단 누락(소급 없음)` : ''}</p>
    {log.length > 0 && <details className="action-log"><summary>행동 기록 · 최근 {log.length}건</summary><ol>{log.map((row) => <li key={`${row.cutoff}-${row.kind}`}>
      <time>{time(row.cutoff)}</time>
      {row.kind === 'gap' ? <span className="log-gap">{row.missedCutoffs}개 판단 누락 · 소급 없음</span> : <><strong>{ACTION_LABELS[row.action!]}</strong><span>{position(row.sideBefore, row.unitsBefore)} → {position(row.sideAfter, row.unitsAfter)}</span>{row.realizedPct !== 0 && <span className={tone(row.realizedPct)}>{pct(row.realizedPct, 3)}</span>}</>}
    </li>)}</ol></details>}
    <p className="experiment-note">{ACTION_DISCLAIMER}</p>
  </div>;
}

/** All scheduled coins' ACTION_V1 paper positions from GET /api/paper. */
export function PaperOverview({ refresh, onSelect }: { refresh: number; onSelect: (symbol: string) => void }) {
  const [rows, setRows] = useState<PaperRow[] | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const load = () => fetch('/api/paper', { signal: controller.signal }).then((r) => { if (!r.ok) throw Error(); return r.json(); })
      .then((data: { symbols: PaperRow[] }) => { setRows(data.symbols); setError(false); }).catch((e) => { if (e.name !== 'AbortError') setError(true); });
    void load();
    const timer = window.setInterval(load, 60000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [refresh]);
  return <section className="card paper-overview" aria-label="코인별 페이퍼 포지션">
    <div className="performance-heading"><div><span className="performance-eyebrow">ACTION_V1 · PAPER</span><h2>코인별 페이퍼 포지션</h2><p>15분 봉 마감마다 판단 · 1단위 명목 대비 % · {ACTION_DISCLAIMER}</p></div></div>
    {error ? <p className="inline-error" role="alert">페이퍼 포지션을 불러오지 못했어요.</p> : !rows ? <p className="paper-note" role="status">페이퍼 포지션 확인 중</p> : <div className="paper-table" role="table">
      <div role="row" className="paper-table-head"><span role="columnheader">코인</span><span role="columnheader">포지션</span><span role="columnheader">최근 행동</span><span role="columnheader">평가</span><span role="columnheader">실현</span></div>
      {rows.map((row) => <button type="button" role="row" key={row.symbol} onClick={() => onSelect(row.symbol)}>
        <span role="cell">{row.symbol.replace(/USD$/, '')}{row.transfer === 'untested_transfer' && <small> 전이</small>}</span>
        <span role="cell">{position(row.position.side, row.position.units)}</span>
        <span role="cell">{row.lastAction?.action ? ACTION_LABELS[row.lastAction.action] : '—'}</span>
        <span role="cell" className={tone(row.unrealizedPct)}>{row.position.side === 'flat' ? '—' : pct(row.unrealizedPct)}</span>
        <span role="cell" className={tone(row.realizedPct)}>{pct(row.realizedPct)}</span>
      </button>)}
    </div>}
  </section>;
}
