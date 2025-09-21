export const revalidate = 30;
type Sig = { symbol: string; side: string; price?: number; generated_at?: string; strength_score?: number };

function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

async function getSignals(): Promise<Sig[]> {
  const base = apiBase();
  const res = await fetch(`${base}/public/signals?limit=50`, { next: { revalidate: 30 } });
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

const fmt = (v: any) => (v ?? "—");
const dt = (s?: string) => (s ? new Date(s).toLocaleString() : "—");

export default async function Page() {
  let rows: Sig[] = [];
  try { rows = await getSignals(); } catch (e) {
    return <main className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Signals</h1>
      <div className="text-red-600">読み込みに失敗しました。</div>
    </main>;
  }

  return (
    <main className="p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Signals (Latest)</h1>
      <div className="overflow-x-auto rounded-2xl shadow">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left">Time</th>
              <th className="px-4 py-2 text-left">Symbol</th>
              <th className="px-4 py-2 text-left">Side</th>
              <th className="px-4 py-2 text-right">Price</th>
              <th className="px-4 py-2 text-right">Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t">
                <td className="px-4 py-2">{dt(r.generated_at)}</td>
                <td className="px-4 py-2">{fmt(r.symbol)}</td>
                <td className="px-4 py-2">{fmt(r.side)}</td>
                <td className="px-4 py-2 text-right">{fmt(r.price)}</td>
                <td className="px-4 py-2 text-right">{fmt(r.strength_score)}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td className="px-4 py-6 text-center text-gray-500" colSpan={5}>データがありません</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}