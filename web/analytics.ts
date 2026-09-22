/** Load GA once, from public runtime configuration rather than build-time secrets. */
export function startAnalytics(measurementId?: string | null) {
  if (!measurementId || !/^G-[A-Z0-9]+$/.test(measurementId)) return;
  if (!['jt.memtherscan.xyz', 'jevtrade.up.railway.app'].includes(window.location.hostname)) return;
  if (document.getElementById('google-analytics')) return;
  const analyticsWindow = window as typeof window & { dataLayer?: unknown[]; gtag?: (...args: unknown[]) => void };
  analyticsWindow.dataLayer ??= [];
  analyticsWindow.gtag = function () { analyticsWindow.dataLayer!.push(arguments); };
  analyticsWindow.gtag('js', new Date());
  analyticsWindow.gtag('config', measurementId, {
    // Shared decision IDs and arbitrary URL parameters are unnecessary for traffic analytics.
    page_location: window.location.origin + window.location.pathname,
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  });
  const script = document.createElement('script');
  script.id = 'google-analytics';
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
  document.head.appendChild(script);
}
