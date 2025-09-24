// /app/performance/page.tsx
export const revalidate = 0;

import Link from "next/link";
import {
  PageHeader,
  Section,
  FilterBar,
  CsvButtons,
} from "@/components/ui";
import { Notes } from "@/components/ui/Notes";

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
        <Link
          key={d}
          href={makeHref(d)}
          className={`${baseBtn} ${days === d ? active : "border-gray-300 dark:border-gray-700"}`}
          aria-current={days === d ? "page" : undefined}
        >
          {d}日
        </Link>
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
        <PageHeader
          title="Performance（日次集計）"
          description={<>直近{days}日の Win Rate / Avg PnL% / Trades を公開。シンボル切替・CSVダウンロードに対応。</>}
        />

        {/* フィルタ欄：2段構成（1段目: クイック、2段目: 入力群 を箱いっぱいで） */}
        <FilterBar
          left={
            <div className="space-y-3">
              {/* --- 1段目：クイックレンジ（左寄せ） --- */}
              <div className="flex gap-2">
                <QuickRanges days={days} symbol={symbol} />
              </div>

              {/* --- 2段目：Days / Symbol / Apply / Reset を横一列で展開 --- */}
              <form method="get" className="grid grid-cols-1 md:grid-cols-12 gap-3 items-end">
                {/* Days（2カラム） */}
                <div className="md:col-span-2">
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

                {/* Symbol（6カラム） */}
                <div className="md:col-span-6">
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

                {/* Apply（2カラム） */}
                <div className="md:col-span-2 flex">
                  <button className="w-full rounded-2xl shadow px-4 py-2 bg-gray-900 text-white dark:bg-white dark:text-gray-900">
                    Apply
                  </button>
                </div>

                {/* Reset（2カラム） */}
                <div className="md:col-span-2 flex">
                  <Link
                    className="w-full text-center rounded-2xl border px-4 py-2 border-gray-300 dark:border-gray-700"
                    href="/performance?days=30"
                    title="フィルタをクリア"
                  >
                    Reset
                  </Link>
                </div>
              </form>
            </div>
          }
          right={
            <Notes
              items={[
                { label: <>JST基準の<b>日次</b>集計</>, tooltip: "日付の区切りは JST (UTC+9)" },
                { label: <>大小文字非区別: <code>lower(symbol)</code></> },
              ]}
            />
          }
        />

        {/* グラフ */}
         <Section title="日次推移（Win Rate / Avg PnL%）">
          {Array.isArray(data) && data.length > 0 ? (
            <ChartClient data={data} />
          ) : (
            <div className="h-32 grid place-items-center text-sm text-gray-500">
              表示できるデータがありません（期間・シンボルを変更してお試しください）
            </div>
          )}
        </Section>

        {/* 日次テーブル */}
        <Section
          title="日次テーブル"
          headerRight={
            <CsvButtons
              links={[
                {
                  href: csvDailyUrl,
                  label: symbol ? `CSV（${symbol} / ${days}日）` : `CSV（${days}日）`,
                  download: true,
                  target: "_blank",
                  rel: "noopener",
                },
              ]}
            />
          }
        >
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
        </Section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <PageHeader title="Performance" />
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
        <p className="text-sm text-gray-600 mt-2">
          ネットワークまたはAPIの一時的な不調の可能性があります。時間をおいて再度お試しください。
        </p>
      </main>
    );
  }
}
