/**
 * Validation facts for served numeric policies, keyed by the pinned artifact SHA-256 that the
 * /action revision reports (numeric-<kind>:sha256:<sha>). A revision that is not listed here gets
 * no version label and no validation claims, so the UI never attributes one model's results to
 * another. Source of every number: training/ACTION_V*.md and ACTION_V*_RESULTS.json.
 */
export type WindowResult = { label: string; period: string; netPct: number; holdPct: number; feesPct?: number; confirmatory?: boolean };
export type PolicyInfo = {
  version: string;
  model: string;
  gate: string;
  caveat: string;
  rules: string;
  windows: WindowResult[];
  contractUrl: string;
  hfUrl: string;
};

const POLICIES: Record<string, PolicyInfo> = {
  '95531fcda384c095101051bf689947f46177a3a97e47806d3d16bb25f157d284': {
    version: 'V7',
    model: 'GBM 모방 모델 + 비용 규칙',
    gate: '사전 등록 게이트 통과 (2022 하락장, 처음 보는 구간)',
    caveat: '상승장에서는 단순 보유보다 못했습니다 · 변동성이 낮으면 거의 거래하지 않습니다',
    rules: '24시간 추세와 같은 방향으로만 진입하고, 진입 후 최소 32봉(8시간) 보유합니다. 청산 후 8봉(2시간)은 다시 진입하지 않으며, 추가·축소 없이 1단위 진입·전량 청산만 합니다.',
    windows: [
      { label: '검증 (게이트)', period: '2022-04 ~ 12 · 약 9개월', netPct: 19.0, holdPct: -70.2, feesPct: 15.0, confirmatory: true },
      { label: '고정 벤치마크', period: '2025-01 ~ 2026-06 · 약 18개월', netPct: -15.9, holdPct: -48.9, feesPct: 18.1 },
      { label: '탐색 (표본 내)', period: '2023 · 1년', netPct: 44.7, holdPct: 311.1, feesPct: 10.9 },
      { label: '선택', period: '2024 · 1년', netPct: -1.7, holdPct: 114.3, feesPct: 12.8 },
    ],
    contractUrl: 'https://github.com/guzus/jev-nyotti/blob/main/training/ACTION_V7.md',
    hfUrl: 'https://huggingface.co/guzus/jev-nyotti-action',
  },
};

export function policyInfo(revision: string | null | undefined): PolicyInfo | null {
  const sha = revision?.match(/sha256:([0-9a-f]{64})/)?.[1];
  return sha ? POLICIES[sha] ?? null : null;
}

export const PNL_UNITS = '구간 누적 · 1단위 명목 대비 % · 복리 아님 · 연 수익률 아님';
