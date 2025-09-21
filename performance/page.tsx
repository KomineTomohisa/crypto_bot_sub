// app/performance/page.tsx
export const revalidate = 60; // ISR 60秒ごと再取得

type Symbols = Record<string, { trades: number; win_rate?: number; avg_pnl_pct?: number }>;
type Metrics = {
  period_start: string;
  period_end: string;
  total_trades: number;
  win_rate?: number;
  avg_pnl_pct?: number;
  symbols: Symbols;
};

async function getMetrics(): Promise<Metrics> {
  const base = process.env.NEXT_PUBLIC_API_BASE!;
  const res = await fetch(`${base}/public/metrics`, { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`Failed to load metrics: ${res.status}`);
  return res.json();
}

const pct = (v?: number) => (typeof v === "number" ? (v * 100).toFixed(2) + "%" : "—");

export default async function Page() {
  let m: Metrics | null = null;
  try {
    m = await getMetrics();
  } catch (e) {
    return (
      <main className="p-6 max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Performance</h1>
        <div className="text-red-600">読み込みに失敗しました。APIの疎通と CORS を確認してください。</div>
      </main>
    );
  }

  return (
    <main className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Performance (Last Window)</h1>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="rounded-2xl shadow p-4">
          <div className="text-sm text-gray-500">Win Rate</div>
          <div className="text-2xl font-semibold">{pct(m?.win_rate)}</div>
        </div>
        <div className="rounded-2xl shadow p-4">
          <div className="text-sm text-gray-500">Avg PnL %</div>
          <div className="text-2xl font-semibold">{pct(m?.avg_pnl_pct)}</div>
        </div>
        <div className="rounded-2xl shadow p-4">
          <div className="text-sm text-gray-500">Total Trades</div>
          <div className="text-2xl font-semibold">{m?.total_trades ?? "—"}</div>
        </div>
      </div>

      <h2 className="text-xl font-semibold mt-8 mb-2">By Symbol</h2>
      <div className="overflow-x-auto rounded-2xl shadow">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left">Symbol</th>
              <th className="px-4 py-2 text-right">Trades</th>
              <th className="px-4 py-2 text-right">Win Rate</th>
              <th className="px-4 py-2 text-right">Avg PnL %</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(m?.symbols || {}).map(([sym, s]) => (
              <tr key={sym} className="border-t">
                <td className="px-4 py-2">{sym}</td>
                <td className="px-4 py-2 text-right">{s.trades}</td>
                <td className="px-4 py-2 text-right">{pct(s.win_rate)}</td>
                <td className="px-4 py-2 text-right">{pct(s.avg_pnl_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-gray-500 mt-4">
        Period: {new Date(m!.period_start).toLocaleString()} — {new Date(m!.period_end).toLocaleString()}
      </p>
    </main>
  );
}
