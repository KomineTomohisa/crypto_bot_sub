export const revalidate = 0;

type Sig = {
  symbol: string;
  side: string;
  price?: number;
  generated_at?: string;
  strength_score?: number;
};
type Meta = { total: number; page: number; limit: number; hasNext: boolean };

function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

function buildQS(q: Record<string, string | number | undefined>) {
  const params = new URLSearchParams();
  Object.entries(q).forEach(([k, v]) => {
    if (v !== undefined && v !== "") params.set(k, String(v));
  });
  return params.toString();
}

async function fetchSignals(q: { symbol?: string; since?: string; page: number; limit: number }) {
  const base = apiBase();
  const qs = buildQS(q);
  const res = await fetch(`${base}/public/signals?${qs}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  const items: Sig[] = await res.json();
  const h = res.headers;
  const meta: Meta = {
    total: Number(h.get("x-total-count") ?? items.length),
    page: Number(h.get("x-page") ?? q.page),
    limit: Number(h.get("x-limit") ?? q.limit),
    hasNext: (h.get("x-has-next") ?? "false") === "true",
  };
  return { items, meta };
}

const fmt = (v: string | number | null | undefined) => (v ?? "—");
const dt = (s?: string) => (s ? new Date(s).toLocaleString() : "—");

// ★ ここがポイント：searchParams を Promise で受けて await する
export default async function Page({
  searchParams,
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = searchParams ? await searchParams : undefined;

  const symbol =
    typeof sp?.symbol === "string" ? sp.symbol :
    Array.isArray(sp?.symbol) ? sp?.symbol[0] : undefined;

  const since =
    typeof sp?.since === "string" ? sp.since :
    Array.isArray(sp?.since) ? sp?.since[0] : undefined;

  const page = Number(typeof sp?.page === "string" ? sp.page :
              Array.isArray(sp?.page) ? sp.page[0] : 1);
  const limit = Number(typeof sp?.limit === "string" ? sp.limit :
               Array.isArray(sp?.limit) ? sp.limit[0] : 50);

  const { items, meta } = await fetchSignals({ symbol, since, page, limit });

  const prevQS = buildQS({ symbol, since, page: Math.max(1, meta.page - 1), limit: meta.limit });
  const nextQS = buildQS({ symbol, since, page: meta.page + 1, limit: meta.limit });

  return (
    <main className="p-6 max-w-6xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Signals</h1>

      <form method="get" className="grid grid-cols-1 sm:grid-cols-5 gap-3 items-end">
        <div>
          <label className="block text-sm text-gray-600">Symbol</label>
          <input
            name="symbol"
            defaultValue={symbol ?? ""}
            placeholder="例: sol_jpy / BTCUSDT"
            className="w-full border rounded-xl px-3 py-2"
          />
        </div>
        <div className="sm:col-span-2">
          <label className="block text-sm text-gray-600">Since (ISO / datetime-local可)</label>
          <input
            name="since"
            defaultValue={since ?? ""}
            placeholder="2025-09-01T00:00:00Z or 2025-09-01T00:00"
            className="w-full border rounded-xl px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600">Limit</label>
          <input
            type="number"
            name="limit"
            min={1}
            max={200}
            defaultValue={limit || 50}
            className="w-full border rounded-xl px-3 py-2"
          />
        </div>
        <div>
          <button className="w-full rounded-2xl shadow px-4 py-2">Apply</button>
        </div>
      </form>

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
            {items.map((r, i) => (
              <tr key={i} className="border-t">
                <td className="px-4 py-2">{dt(r.generated_at!)}</td>
                <td className="px-4 py-2">{fmt(r.symbol)}</td>
                <td className="px-4 py-2">{fmt(r.side)}</td>
                <td className="px-4 py-2 text-right">{fmt(r.price)}</td>
                <td className="px-4 py-2 text-right">{fmt(r.strength_score)}</td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr>
                <td className="px-4 py-6 text-center text-gray-500" colSpan={5}>
                  データがありません
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          Total: {meta.total} / Page: {meta.page}
        </div>
        <div className="space-x-2">
          <a
            href={`?${prevQS}`}
            className={`rounded-xl px-3 py-2 border ${meta.page > 1 ? "" : "pointer-events-none opacity-40"}`}
          >
            ← Prev
          </a>
          <a
            href={`?${nextQS}`}
            className={`rounded-xl px-3 py-2 border ${meta.hasNext ? "" : "pointer-events-none opacity-40"}`}
          >
            Next →
          </a>
        </div>
      </div>
    </main>
  );
}
