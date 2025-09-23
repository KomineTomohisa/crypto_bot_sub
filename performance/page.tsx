// /app/performance/page.tsx
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

/** 全体 or シンボル別の日次系列を取得 */
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

/** シンボル候補（UI用） */
async function fetchSymbols(daysForList = 90): Promise<string[]> {
  const base = apiBase();
  const url = new URL(`${base}/public/symbols`);
  url.searchParams.set("days", String(daysForList));
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

import ChartClient from "./ChartClient";

function pct(v?: number | null, digits = 1) {
  return v != null ? `${(v * 100).toFixed(digits)}%` : "—";
}

/** クイック期間切替（7/30/90日） */
function QuickRanges({ days, symbol }: { days: number; symbol?: string }) {
  const makeHref = (d: number) => {
    const p = new URLSearchParams({ days: String(d) });
    if (symbol) p.set("symbol", symbol);
    return `/performance?${p.toString()}`;
  };
  const baseBtn =
    "px-3 py-1.5 rounded-xl border text-sm hover:bg-gray-50 dark:hover:bg-gray-800";
  const active =
    "bg-gray-900 text-white border-gray-900 dark:bg-white dark:text-gray-900 dark:border-white";
  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="期間のクイック切替">
      {[7, 30, 90].map((d) => (
        <a
          key={d}
          href={makeHref(d)}
          className={`${baseBtn} ${days === d ? active : "border-gray-300 dark:border-gray-700"}`}
          aria-current={days === d ? "page" : undefined}
        >
          {d}日
        </a>
      ))}
    </div>
  );
}

export default async function Page({
  searchParams,
}: {
  // Next 15 互換：Promise も許容
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
    // 初期描画に必要なものをSSRで取得
    const [data, symbols] = await Promise.all([
      fetchDaily(days, symbol),
      fetchSymbols(90),
    ]);

    // CSVリンク（全体 or シンボル別）
    const csvDailyUrl = symbol
      ? `/api/public/export/performance/daily_by_symbol.csv?symbol=${encodeURIComponent(
          symbol
        )}&days=${days}`
      : `/api/public/export/performance/daily.csv?days=${days}`;

    return (
      <main className="p-6 md:p-8 max-w-6xl mx-auto space-y-8">
        {/* ヘッダ */}
        <header className="space-y-2">
          <h1 className="text-2xl md:text-3xl font-bold">Performance（日次集計）</h1>
          <p className="text-sm text-gray-600 dark:text-gray-300">
            直近{days}日の Win Rate / Avg PnL% / Trades を公開。シンボル切替・CSVダウンロードに対応。
          </p>
        </header>

        {/* フィルタ欄：クイックレンジ＋フォーム */}
        <section
          aria-labelledby="filters-heading"
          className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm space-y-3"
        >
          <h2 id="filters-heading" className="sr-only">フィルタ</h2>

          <div className="flex items-center justify-between flex-wrap gap-3">
            <QuickRanges days={days} symbol={symbol} />
            <div className="text-xs text-gray-500">
              JST基準の<strong>日次</strong>で集計（大小文字非区別: <code>lower(symbol)</code>）
            </div>
          </div>

          <form method="get" className="grid grid-cols-1 sm:grid-cols-6 gap-3 items-end">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300">Days</label>
              <input
                name="days"
                type="number"
                min={7}
                max={365}
                defaultValue={days}
                className="w-full border rounded-xl px-3 py-2 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700"
              />
            </div>
            <div className="sm:col-span-3">
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
            <div className="flex gap-2 sm:col-span-2">
              <button className="w-full rounded-2xl shadow px-4 py-2 bg-gray-900 text-white dark:bg-white dark:text-gray-900">
                Apply
              </button>
              <a
                className="w-full text-center rounded-2xl border px-4 py-2 border-gray-300 dark:border-gray-700"
                href="/performance?days=30"
                title="フィルタをクリア"
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
              href={csvDailyUrl}
              download
              target="_blank"
              rel="noopener"
            >
              {symbol ? `日次指標CSV（${symbol} / ${days}日）` : `日次指標CSV（${days}日）`}
            </a>
          </div>
          <p className="text-xs text-gray-500">※ ブラウザが自動でダウンロードを開始します。</p>
        </section>

        {/* グラフ */}
        <section
          aria-labelledby="chart-heading"
          className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm space-y-2"
        >
          <h2 id="chart-heading" className="font-semibold">日次推移（Win Rate / Avg PnL%）</h2>
          {Array.isArray(data) && data.length > 0 ? (
            <ChartClient data={data} />
          ) : (
            <div className="h-32 grid place-items-center text-sm text-gray-500">
              表示できるデータがありません（期間・シンボルを変更してお試しください）
            </div>
          )}
        </section>

        {/* 日次テーブル */}
        <section
          aria-labelledby="table-heading"
          className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm"
        >
          <h2 id="table-heading" className="font-semibold mb-2">日次テーブル</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
                <tr>
                  <th scope="col" className="px-3 py-2 text-left">Date（JST）</th>
                  <th scope="col" className="px-3 py-2 text-right">Trades</th>
                  <th scope="col" className="px-3 py-2 text-right">Win Rate</th>
                  <th scope="col" className="px-3 py-2 text-right">Avg PnL</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r, i) => (
                  <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
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
          <p className="text-xs text-gray-500 mt-2">※ Win/Avg PnL% は当日トレードに基づく単純集計です。</p>
        </section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Performance</h1>
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
        <p className="text-sm text-gray-600 mt-2">
          ネットワークまたはAPIの一時的な不調の可能性があります。時間をおいて再度お試しください。
        </p>
      </main>
    );
  }
}
