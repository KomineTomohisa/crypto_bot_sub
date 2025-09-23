// /app/signals/page.tsx
export const revalidate = 0;

type Sig = {
  signal_id?: number | string;
  symbol: string;
  side: string;                 // BUY / SELL / EXIT-LONG 等
  price?: number;
  generated_at?: string;        // ISO8601
  strength_score?: number;
  reason?: string;
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

async function fetchSymbols(daysForList = 90): Promise<string[]> {
  const base = apiBase();
  const url = new URL(`${base}/public/symbols`);
  url.searchParams.set("days", String(daysForList));
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
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
const num = (v?: number, d = 2) => (v == null ? "—" : v.toFixed(d));

/** クイック期間（since ISO） */
function QuickSince({ basePath, symbol, since }: { basePath: string; symbol?: string; since?: string }) {
  const mk = (days: number) => {
    const s = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    const p = new URLSearchParams({ since: s });
    if (symbol) p.set("symbol", symbol);
    return `${basePath}?${p.toString()}`;
  };
  const baseBtn =
    "px-3 py-1.5 rounded-xl border text-sm hover:bg-gray-50 dark:hover:bg-gray-800";
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="期間のクイック切替">
      {[7, 30, 90].map((d) => (
        <a key={d} href={mk(d)} className={`${baseBtn} border-gray-300 dark:border-gray-700`}>
          過去{d}日
        </a>
      ))}
      {since && (
        <span className="text-xs text-gray-500 ml-1">
          since: {new Date(since).toLocaleString()}
        </span>
      )}
    </div>
  );
}

// ★ Next.js 15想定：searchParams を Promise で受けられるように
export default async function Page({
  searchParams,
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = searchParams ? await searchParams : undefined;

  const symbol =
    typeof sp?.symbol === "string" ? sp.symbol :
    Array.isArray(sp?.symbol) ? sp?.symbol[0] : undefined;

  // デフォルトは「過去30日」
  const defaultSince = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
  const since =
    typeof sp?.since === "string" ? sp.since :
    Array.isArray(sp?.since) ? sp?.since[0] : defaultSince;

  const page = Number(typeof sp?.page === "string" ? sp.page :
              Array.isArray(sp?.page) ? sp.page[0] : 1);
  const limit = Number(typeof sp?.limit === "string" ? sp.limit :
               Array.isArray(sp?.limit) ? sp.limit[0] : 50);

  try {
    const [symbols, { items, meta }] = await Promise.all([
      fetchSymbols(90),
      fetchSignals({ symbol, since, page, limit }),
    ]);

    const prevQS = buildQS({ symbol, since, page: Math.max(1, meta.page - 1), limit: meta.limit });
    const nextQS = buildQS({ symbol, since, page: meta.page + 1, limit: meta.limit });

    // CSV（最大件数は用途に合わせて調整可）
    const csvUrl = `/api/public/export/signals.csv?${buildQS({
      symbol,
      since,
      limit: 1000,
    })}`;

    return (
      <main className="p-6 md:p-8 max-w-6xl mx-auto space-y-8">
        {/* ヘッダ */}
        <header className="space-y-2">
          <h1 className="text-2xl md:text-3xl font-bold">Signals（シグナル一覧）</h1>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            期間・シンボルで絞り込み、ページング可能。CSVダウンロードにも対応。
          </p>
        </header>

        {/* フィルタ行：クイック期間 + フォーム */}
        <section
          aria-labelledby="filters-heading"
          className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm space-y-3"
        >
          <h2 id="filters-heading" className="sr-only">フィルタ</h2>

          <div className="flex items-center justify-between flex-wrap gap-3">
            <QuickSince basePath="/signals" symbol={symbol} since={since} />
            <div className="text-xs text-gray-500">
              時刻は <b>ローカル表示</b>（保存はUTC）。大小文字非区別（<code>lower(symbol)</code>）。
            </div>
          </div>

          <form method="get" className="grid grid-cols-1 sm:grid-cols-6 gap-3 items-end">
            <div className="sm:col-span-2">
              <label className="block text-sm text-gray-600 dark:text-gray-300">Symbol（任意）</label>
              <select
                name="symbol"
                defaultValue={symbol ?? ""}
                className="w-full border rounded-xl px-3 py-2 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700"
              >
                <option value="">（全体）</option>
                {symbols.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div className="sm:col-span-3">
              <label className="block text-sm text-gray-600 dark:text-gray-300">
                Since（ISO / datetime-local 可）
              </label>
              <input
                name="since"
                defaultValue={since}
                placeholder="2025-09-01T00:00:00Z または 2025-09-01T00:00"
                className="w-full border rounded-xl px-3 py-2 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700"
              />
            </div>

            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300">Limit</label>
              <input
                type="number"
                name="limit"
                min={1}
                max={200}
                defaultValue={limit || 50}
                className="w-full border rounded-xl px-3 py-2 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700"
              />
            </div>

            <div className="flex gap-2 sm:col-span-6 sm:justify-end">
              <button className="rounded-2xl shadow px-4 py-2 bg-gray-900 text-white dark:bg-white dark:text-gray-900">
                Apply
              </button>
              <a
                className="rounded-2xl border px-4 py-2 border-gray-300 dark:border-gray-700 text-center"
                href={`/signals?${buildQS({ since: defaultSince })}`}
                title="フィルタをクリア（過去30日）"
              >
                Reset
              </a>
            </div>
          </form>
        </section>

        {/* CSV エクスポート */}
        <section
          aria-labelledby="csv-heading"
          className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm space-y-3"
        >
          <h2 id="csv-heading" className="font-semibold">CSV エクスポート</h2>
          <div className="flex flex-wrap gap-3">
            <a
              className="px-4 py-2 border rounded-xl border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
              href={csvUrl}
              download
              target="_blank"
              rel="noopener"
            >
              シグナルCSV（最大1000件）
            </a>
          </div>
          <p className="text-xs text-gray-500">※ ブラウザが自動でダウンロードを開始します。</p>
        </section>

        {/* テーブル */}
        <section
          aria-labelledby="table-heading"
          className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm"
        >
          <h2 id="table-heading" className="font-semibold mb-2">一覧</h2>
          <div className="overflow-x-auto rounded-2xl">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
                <tr>
                  <th className="px-4 py-2 text-left">Time</th>
                  <th className="px-4 py-2 text-left">Symbol</th>
                  <th className="px-4 py-2 text-left">Side</th>
                  <th className="px-4 py-2 text-right">Price</th>
                  <th className="px-4 py-2 text-right">Score</th>
                  <th className="px-4 py-2 text-left">Reason</th>
                </tr>
              </thead>
              <tbody>
                {items.map((r, i) => (
                  <tr key={String(r.signal_id ?? `${r.symbol}-${r.generated_at}-${i}`)} className="border-t border-gray-100 dark:border-gray-800">
                    <td className="px-4 py-2">{dt(r.generated_at!)}</td>
                    <td className="px-4 py-2">{fmt(r.symbol)}</td>
                    <td className="px-4 py-2">{fmt(r.side)}</td>
                    <td className="px-4 py-2 text-right">{fmt(r.price)}</td>
                    <td className="px-4 py-2 text-right">{num(r.strength_score, 2)}</td>
                    <td className="px-4 py-2 max-w-[480px]">
                      <div className="truncate" title={r.reason ?? ""}>{r.reason ?? "—"}</div>
                    </td>
                  </tr>
                ))}
                {items.length === 0 && (
                  <tr>
                    <td className="px-4 py-6 text-center text-gray-500" colSpan={6}>
                      データがありません（期間やシンボルを変更してお試しください）
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* ページング */}
          <div className="flex items-center justify-between mt-3">
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Total: {meta.total} / Page: {meta.page} / Limit: {meta.limit}
            </div>
            <div className="space-x-2">
              <a
                href={`?${prevQS}`}
                className={`rounded-xl px-3 py-2 border ${meta.page > 1 ? "border-gray-300 dark:border-gray-700" : "pointer-events-none opacity-40 border-gray-200 dark:border-gray-800"}`}
              >
                ← Prev
              </a>
              <a
                href={`?${nextQS}`}
                className={`rounded-xl px-3 py-2 border ${meta.hasNext ? "border-gray-300 dark:border-gray-700" : "pointer-events-none opacity-40 border-gray-200 dark:border-gray-800"}`}
              >
                Next →
              </a>
            </div>
          </div>
        </section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Signals</h1>
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
        <p className="text-sm text-gray-600 mt-2">
          ネットワークまたはAPIの一時的な不調の可能性があります。時間をおいて再度お試しください。
        </p>
      </main>
    );
  }
}
