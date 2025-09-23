// app/transparency/page.tsx
export const revalidate = 0;

type Daily = {
  date: string;
  total_trades: number;
  win_rate?: number | null;
  avg_pnl_pct?: number | null;
};

function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

// ダウンロードリンクは同一オリジン固定で安全に
const publicBase = "/api";

async function fetchDaily(days = 30, symbol?: string): Promise<Daily[]> {
  const base = apiBase();
  const url = new URL(
    symbol
      ? `${base}/public/performance/daily/by-symbol`
      : `${base}/public/performance/daily`
  );
  url.searchParams.set("days", String(days));
  if (symbol) url.searchParams.set("symbol", symbol);

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed: ${res.status}`);
  return res.json();
}

async function fetchSymbols(daysForList = 90): Promise<string[]> {
  const base = apiBase();
  const url = new URL(`${base}/public/symbols`);
  url.searchParams.set("days", String(daysForList));
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

import ChartClient from "../performance/ChartClient";

function pct(v?: number | null, digits = 1) {
  return v != null ? `${(v * 100).toFixed(digits)}%` : "—";
}

export default async function Page({
  searchParams,
}: {
  // Next.js 15 対応：Promise を許容
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = searchParams ? await searchParams : undefined;
  const days = Number(
    typeof sp?.days === "string" ? sp.days :
    Array.isArray(sp?.days) ? sp.days[0] : 30
  );
  const symbol =
    typeof sp?.symbol === "string"
      ? sp.symbol
      : Array.isArray(sp?.symbol)
      ? sp.symbol[0]
      : undefined;

  try {
    const [data, symbols] = await Promise.all([
      fetchDaily(days, symbol),
      fetchSymbols(90),
    ]);

    // 直近days日の since を計算（signals CSV用）
    const sinceIso = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
    const csvDailyUrl = symbol
      ? `${publicBase}/public/export/performance/daily_by_symbol.csv?symbol=${encodeURIComponent(
          symbol
        )}&days=${days}`
      : `${publicBase}/public/export/performance/daily.csv?days=${days}`;
    const csvSignalsUrl = `${publicBase}/public/export/signals.csv?since=${encodeURIComponent(
      sinceIso
    )}&limit=1000${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`;

    return (
      <main className="p-6 max-w-6xl mx-auto space-y-8">
        <header className="space-y-2">
          <h1 className="text-2xl font-bold">
            Transparency（直近{days}日{symbol ? ` / ${symbol}` : ""}）
          </h1>
          <p className="text-sm text-gray-600">
            直近期間の結果と算出ロジックを公開します。
          </p>
        </header>

        {/* フィルタ（GET提出） */}
        <form method="get" className="grid grid-cols-1 sm:grid-cols-6 gap-3 items-end">
          <div>
            <label className="block text-sm text-gray-600">Days</label>
            <input
              name="days"
              type="number"
              min={7}
              max={365}
              defaultValue={days}
              className="w-full border rounded-xl px-3 py-2"
            />
          </div>
          <div className="sm:col-span-3">
            <label className="block text-sm text-gray-600">Symbol（任意）</label>
            <select
              name="symbol"
              defaultValue={symbol ?? ""}
              className="w-full border rounded-xl px-3 py-2"
            >
              <option value="">（全体）</option>
              {symbols.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <button className="w-full rounded-2xl shadow px-4 py-2">Apply</button>
            <a
              className="w-full text-center rounded-2xl border px-4 py-2"
              href="/transparency?days=30"
              title="フィルタをクリア"
            >
              Reset
            </a>
          </div>
        </form>

        {/* ダウンロード */}
        <section className="rounded-2xl border p-4 shadow space-y-3">
          <h2 className="font-semibold">CSV エクスポート</h2>
          <div className="flex flex-wrap gap-3">
            <a
              className="px-4 py-2 border rounded-xl"
              href={csvDailyUrl}
              download
              target="_blank"
              rel="noopener"
            >
              {symbol ? `日次指標CSV（${symbol} / ${days}日）` : `日次指標CSV（${days}日）`}
            </a>
            <a
              className="px-4 py-2 border rounded-xl"
              href={csvSignalsUrl}
              download
              target="_blank"
              rel="noopener"
            >
              シグナルCSV（{days}日{symbol ? ` / ${symbol}` : ""}）
            </a>
          </div>
          <p className="text-xs text-gray-500">※ ブラウザがダウンロードを開始します。</p>
        </section>

        {/* グラフ */}
        <section className="rounded-2xl border p-4 shadow space-y-2">
          <h2 className="font-semibold">30日推移（Win Rate / Avg PnL%）</h2>
          <ChartClient data={data} />
        </section>

        {/* テーブル */}
        <section className="rounded-2xl border p-4 shadow">
          <h2 className="font-semibold mb-2">日次テーブル</h2>
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
                    <td className="px-3 py-2 text-right">{pct(r.win_rate, 1)}</td>
                    <td className="px-3 py-2 text-right">{pct(r.avg_pnl_pct, 2)}</td>
                  </tr>
                ))}
                {data.length === 0 && (
                  <tr>
                    <td className="px-3 py-6 text-center text-gray-500" colSpan={4}>
                      データがありません
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* 算出ロジック */}
        <section className="rounded-2xl border p-4 shadow space-y-2">
          <h2 className="font-semibold">算出ロジック（概要）</h2>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>タイムゾーンは <b>JST（UTC+9）</b> を基準に日次を区切ります。</li>
            <li>日次指標は、該当日のトレード（<code>trades</code>）を対象に算出します。</li>
            <li><b>Win</b> の定義：<code>pnl &gt; 0</code> のトレード数。</li>
            <li><b>Win Rate</b>： <code>wins / total_trades</code>。</li>
            <li><b>Avg PnL%</b>： その日の <code>pnl_pct</code> の単純平均。</li>
            <li>シンボル比較は大小文字を区別しません（<code>lower(symbol)</code>）。</li>
            <li>該当日のトレードが無い場合、Win Rate / Avg PnL% は <code>null</code>（表示上は「—」）。</li>
          </ul>
          <p className="text-xs text-gray-500">
            ※ 具体実装はサーバ側の <code>compute_metrics()</code> に準拠しています。
          </p>
        </section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Transparency</h1>
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
      </main>
    );
  }
}
