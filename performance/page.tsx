export const revalidate = 0;

type Daily = { date: string; total_trades: number; win_rate?: number | null; avg_pnl_pct?: number | null };

function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

async function fetchDaily(days = 30): Promise<Daily[]> {
  const base = apiBase();
  const res = await fetch(`${base}/public/performance/daily?days=${days}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

import ChartClient from "./ChartClient";

export default async function Page() {
  try {
    const days = 30;
    const data = await fetchDaily(days);

    return (
      <main className="p-6 max-w-6xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">Performance (Last {days} days)</h1>

        <section className="rounded-2xl border p-4 shadow">
          <h2 className="font-semibold mb-2">Win Rate & Avg PnL (%)</h2>
          <ChartClient data={data} />
        </section>

        <section className="rounded-2xl border p-4 shadow">
          <h2 className="font-semibold mb-2">Daily Table</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left">Date</th>
                  <th className="px-3 py-2 text-right">Trades</th>
                  <th className="px-3 py-2 text-right">Win Rate</th>
                  <th className="px-3 py-2 text-right">Avg PnL</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r, i) => (
                  <tr key={i} className="border-t">
                    <td className="px-3 py-2">{r.date}</td>
                    <td className="px-3 py-2 text-right">{r.total_trades}</td>
                    <td className="px-3 py-2 text-right">{r.win_rate != null ? `${(r.win_rate*100).toFixed(1)}%` : "—"}</td>
                    <td className="px-3 py-2 text-right">{r.avg_pnl_pct != null ? `${(r.avg_pnl_pct*100).toFixed(2)}%` : "—"}</td>
                  </tr>
                ))}
                {data.length === 0 && (
                  <tr><td className="px-3 py-6 text-center text-gray-500" colSpan={4}>データがありません</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Performance</h1>
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
      </main>
    );
  }
}
