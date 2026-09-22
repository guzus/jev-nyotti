import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startAnalytics } from '../web/analytics.js';

test('analytics is production-only, strips URL parameters and initializes once across polling', () => {
  const scripts: { id: string; src: string; async: boolean }[] = [];
  const page = { location: { hostname: 'localhost', origin: 'http://localhost:3000', pathname: '/', search: '?decision=private-id' }, dataLayer: [] as IArguments[] };
  Object.defineProperty(globalThis, 'window', { value: page, configurable: true });
  Object.defineProperty(globalThis, 'document', { value: {
    getElementById: (id: string) => scripts.find(script => script.id === id),
    createElement: () => ({}), head: { appendChild: (script: typeof scripts[number]) => scripts.push(script) },
  }, configurable: true });
  try {
    startAnalytics('G-LQLNM6WNZK'); assert.equal(scripts.length, 0);
    page.location = { ...page.location, hostname: 'jt.memtherscan.xyz', origin: 'https://jt.memtherscan.xyz' };
    startAnalytics('not-a-tag'); assert.equal(scripts.length, 0);
    startAnalytics('G-LQLNM6WNZK'); startAnalytics('G-LQLNM6WNZK');
    assert.equal(scripts.length, 1); assert.equal(page.dataLayer.length, 2);
    assert.equal(scripts[0].src, 'https://www.googletagmanager.com/gtag/js?id=G-LQLNM6WNZK');
    const config = Array.from(page.dataLayer[1]);
    assert.equal(config[0], 'config'); assert.equal(config[1], 'G-LQLNM6WNZK');
    assert.equal(config[2].page_location, 'https://jt.memtherscan.xyz/');
    assert.equal(config[2].allow_google_signals, false);
  } finally {
    Reflect.deleteProperty(globalThis, 'window'); Reflect.deleteProperty(globalThis, 'document');
  }
});
