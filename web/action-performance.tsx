import { useState } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ACTION_DISCLAIMER, ACTION_LABELS } from './action.js';

type Report = ReturnType<typeof import('../server/pnl.js').simulateActionReplay>;
type Row = { symbol: string; marketAsOf: string; action: keyof typeof ACTION_LABELS; sideAfter: 'flat' | 'long' | 'short'; unitsAfter: number; price: number };
export type ActionReplayPayload = { task: 'ACTION_V1'; status: 'ready'; model: string; revision: string; holdMargin: number; generatedAt: string; source: string; history: Row[]; report: Report };
const pct = (n: number, digits = 2) => `${n > 0 ? '+' : ''}${n.toFixed(digits)}%`;
const side = { flat: '무포지션', long: '롱', short: '숏' };

/** Performance card for an imported ACTION_V1 closed-loop replay (unit-based paper PnL). */
export function ActionPerformance({ data }: { data: ActionReplayPayload }) {
  const [symbol, setSymbol] = useState('전체'), [page, setPage] = useState(0);
  const r = data.report, first = r.curve[0], last = r.curve.at(-1)!;
  const history = data.history.filter((row) => symbol === '전체' || row.symbol === symbol);
  const trades = r.perSymbol.reduce((a, row) => a + row.trades, 0);
  return <section className="card performance" aria-label="ACTION_V1 페이퍼 재현 PnL">
    <div className="performance-heading"><div><span className="performance-eyebrow">ACTION_V1 REPLAY</span><h2>jev뇨띠 행동 재현 PnL</h2><p>모델 자신의 페이퍼 포지션을 이어간 15분 폐루프 재현 · 1단위 명목 대비 %</p></div><span className="performance-badge">사후 재현</span></div>
    <div className="performance-metrics">
      <div><span>평균 손익 · 수수료 후</span><strong>{pct(r.pnlPct)}</strong></div>
      <div><span>최대 낙폭 · %p</span><strong>{r.maxDrawdownPts.toFixed(2)}</strong></div>
      <div><span>1단위 매수 후 보유</span><strong>{pct(r.buyHoldPct)}</strong></div>
      <div><span>체결 · 종목 평균 누적 수수료</span><strong>{trades.toLocaleString()}회 · {r.feesPct.toFixed(2)}%</strong></div>
    </div>
    <div className="performance-chart" role="img" aria-label="ACTION_V1 페이퍼 손익 및 매수 후 보유 곡선"><ResponsiveContainer width="100%" height="100%"><LineChart data={r.curve}><CartesianGrid vertical={false} stroke="#e8e2d8" /><XAxis dataKey="time" tickFormatter={(v) => String(v).slice(5, 10)} minTickGap={65} /><YAxis tickFormatter={(v) => `${Number(v).toFixed(1)}%`} width={60} /><Tooltip formatter={(v) => pct(Number(v))} labelFormatter={(v) => String(v).slice(0, 16).replace('T', ' ')} /><Line name="jev뇨띠" dataKey="pnlPct" stroke="#a95546" dot={false} isAnimationActive={false} /><Line name="매수 후 보유" dataKey="buyHoldPct" stroke="#99958c" strokeDasharray="4 4" dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div>
    <p className="performance-range">실선 jev뇨띠 · 점선 1단위 매수 후 보유 · UTC · {first.time.slice(0, 16).replace('T', ' ')} — {last.time.slice(0, 16).replace('T', ' ')} · <span title={data.source}>{data.source}</span> · <span title={data.revision}>{data.revision.split('@').at(-1)?.slice(0, 7)}</span> · 관망 마진 {data.holdMargin}</p>
    <div className="performance-symbols">{r.perSymbol.map((row) => <div key={row.symbol}><span>{row.symbol.replace(/USDT?$/, '')}</span><strong>{pct(row.pnlPct)}</strong><span>진입 {row.opens} · 청산 {row.closes} · 보유 {row.timeInPositionPct.toFixed(0)}%</span></div>)}</div>
    {!!data.history.length && <details className="performance-rules"><summary>실행된 행동 기록 · {data.history.length.toLocaleString()}건 (관망 제외)</summary>
      <div className="replay-history-controls"><label>종목 <select value={symbol} onChange={(e) => { setSymbol(e.target.value); setPage(0); }}><option>전체</option>{r.perSymbol.map((row) => <option key={row.symbol}>{row.symbol}</option>)}</select></label><span>최신순 · UTC · 행동 → 이후 포지션</span></div>
      <div className="replay-history-rows">{history.slice(page * 20, (page + 1) * 20).map((row) => <div key={row.symbol + row.marketAsOf}><time>{row.marketAsOf.slice(0, 16).replace('T', ' ')}</time><span>{row.symbol.replace(/USDT?$/, '')}</span><span><strong>{ACTION_LABELS[row.action]}</strong> → {side[row.sideAfter]}{row.unitsAfter ? ` ${Number(row.unitsAfter.toFixed(3))}단위` : ''}</span></div>)}</div>
      <div className="replay-history-controls"><button className="secondary-button" disabled={!page} onClick={() => setPage((v) => v - 1)}>이전</button><span>{page + 1} / {Math.max(1, Math.ceil(history.length / 20))}</span><button className="secondary-button" disabled={(page + 1) * 20 >= history.length} onClick={() => setPage((v) => v + 1)}>다음</button></div>
    </details>}
    <details className="performance-rules"><summary>페이퍼 재현 계산 기준</summary>
      <p>매 15분 봉 마감마다 한 번 판단합니다. 무포지션에서 롱 진입·숏 진입은 1단위, 추가는 +1단위(최대 3단위, 조화평균 진입가), 축소는 보유 단위의 절반, 청산은 전량입니다.</p>
      <p>체결은 판단 시점 봉의 종가로 가정하고, 거래 단위마다 수수료 {r.assumptions.feeBpsPerUnit / 100}%를 뺍니다. 손익은 1단위 명목 대비 %이며, 전체 수치는 종목별 손익의 단순 평균입니다. 낙폭은 %p입니다.</p>
      <p>펀딩비·슬리피지·마지막 미청산 포지션의 청산 비용은 미반영입니다. 교사 거래자의 수량은 모방하지 않습니다. 실제 운용 수익이 아닙니다. {ACTION_DISCLAIMER}.</p>
    </details>
  </section>;
}
