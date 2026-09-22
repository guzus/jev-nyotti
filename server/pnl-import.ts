import { readFileSync, mkdirSync, writeFileSync, renameSync, unlinkSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { simulatePortfolio } from './pnl.js';

const timestamp = z.string().datetime({ offset: true });
const side = z.enum(['long', 'short', 'flat']);
export const pnlImportSchema = z.object({
  model: z.string().trim().min(1).max(300), revision: z.string().trim().min(1).max(500),
  generatedAt: timestamp, source: z.string().trim().min(1).max(300),
  priorPolicy: z.literal('previous_prediction'),
  initialCapital: z.number().positive().optional(),
  feeBps: z.number().nonnegative().lt(10000).optional(), slippageBps: z.number().nonnegative().lt(10000).optional(),
  series: z.array(z.object({
    symbol: z.string().trim().min(1).max(40),
    decisions: z.array(z.object({ marketAsOf: timestamp, action: side, previousAction: side })).min(1),
    candles: z.array(z.object({ time: z.number().int().positive(), open: z.number().positive(), high: z.number().positive(), low: z.number().positive(), close: z.number().positive(), volume: z.number().nonnegative() })).min(1),
  })).min(1).max(100),
}).strict();

export function makePnlReport(input: unknown) {
  const parsed = pnlImportSchema.parse(input);
  const report = simulatePortfolio(parsed);
  const latestCutoff = Math.max(...parsed.series.map(s => Date.parse(s.decisions.at(-1)!.marketAsOf)));
  if (Date.parse(parsed.generatedAt) < latestCutoff + 3600000) throw new Error('generatedAt precedes the final completed execution candle');
  return { status: 'ready' as const, model: parsed.model, revision: parsed.revision, generatedAt: parsed.generatedAt, source: parsed.source, priorPolicy: parsed.priorPolicy, report };
}

export function importPnlReport(inputFile: string, outputDirectory: string) {
  const result = makePnlReport(JSON.parse(readFileSync(inputFile, 'utf8')));
  mkdirSync(outputDirectory, { recursive: true });
  const path = join(outputDirectory, 'pnl-report.json');
  const temporary = join(outputDirectory, `.pnl-report-${randomUUID()}.tmp`);
  try {
    writeFileSync(temporary, JSON.stringify(result), { mode: 0o600, flag: 'wx' });
    renameSync(temporary, path);
  } catch (error) {
    try { unlinkSync(temporary); } catch { /* Already renamed or never created. */ }
    throw error;
  }
  return { path, result };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    if (process.argv.length !== 4) throw new Error('Usage: node dist/server/pnl-import.js INPUT.json OUTPUT_DIRECTORY');
    const { path, result } = importPnlReport(process.argv[2], process.argv[3]);
    console.log(`Saved ${result.report.perSymbol.length} symbols and ${result.report.curve.length - 1} hourly points to ${path}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : 'PnL import failed');
    process.exitCode = 1;
  }
}
