export const revalidate = 0;

import Link from "next/link";
import {
  PageHeader,
  Section,
  FilterBar,
  CsvButtons,
} from "@/components/ui";
import { Notes } from "@/components/ui/Notes";
import { QuickFilters } from "@/components/ui/QuickFilters";
import { FilterCard } from "@/components/ui/FilterCard";
import ChartClient from "../performance/ChartClient";

/* ===== 型 ===== */
type Daily = {
  date: string;
  total_trades: number;
  win_rate?: number | null;
  avg_pnl_pct?: number | null;
};

type SignalRecord = {
  signal_id?: number | string;
  symbol: string;
  side: string;        // "BUY" | "SELL" | "EXIT-LONG" | "LONG-ENTRY" | ...
  reason?: string;
  price?: number;
  created_at?: string | number; // API起源によっては数値UNIXもあり得る
};

type TimeLike = string | number | Date | null | undefined;
type SignalTimeFields = {
  created_at?: TimeLike;
  generated_at?: TimeLike;
  time?: TimeLike;
  timestamp?: TimeLike;
  createdAt?: TimeLike;
  generatedAt?: TimeLike;
};

/* ===== ヘルパー（ESLint: any 不使用） ===== */
function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

function normSide(side?: string) {
  return (side ?? "").trim().toUpperCase();
}

function sideLabel(side?: string) {
  const s = normSide(side);
  if (s.includes("EXIT")) return "EXIT";
  if (s.includes("BUY") || s.includes("LONG"))  return "BUY";
  if (s.includes("SELL") || s.includes("SHORT")) return "SELL";
  return s || "—";
}

function posLabel(side?: string) {
  const s = normSide(side);
  if (s.includes("LONG"))  return "ロング";
  if (s.includes("SHORT")) return "ショート";
  return "—";
}

// BUY=緑 / SELL=赤 / EXIT=グレー
function sidePillClass(side?: string) {
  const s = normSide(side);
  if (s.includes("EXIT")) return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
  if (s.includes("BUY") || s.includes("LONG"))
    return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300";
  if (s.includes("SELL") || s.includes("SHORT"))
    return "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300";
  return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300";
}

function parseDateFlexible(v: TimeLike): Date | null {
  if (v == null) return null;

  if (v instanceof Date) return isNaN(v.getTime()) ? null : v;

  if (typeof v === "number") {
    const ms = v > 1e12 ? v : v * 1000; // 秒/ミリ秒両対応
    const d = new Date(ms);
    return isNaN(d.getTime()) ? null : d;
  }

  if (typeof v === "string") {
    const trimmed = v.trim();
    if (!trimmed) return null;

    if (/^\d+$/.test(trimmed)) {
      const num = Number(trimmed);
      const ms = num > 1e12 ? num : num * 1000;
      const d = new Date(ms);
      return isNaN(d.getTime()) ? null : d;
    }

    const d = new Date(trimmed);
    return isNaN(d.getTime()) ? null : d;
  }

  return null;
}

function fmtSignalTimeJST(sig: SignalTimeFields): string {
  const cand: TimeLike =
    sig.created_at ??
    sig.generated_at ??
    sig.time ??
    sig.timestamp ??
    sig.createdAt ??
    sig.generatedAt;

  const d = parseDateFlexible(cand);
  if (!d) return "Invalid Date";
  return d.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" });
}

function pct(v?: number | null, digits = 1) {
  return v != null ? `${(v * 100).toFixed(digits)}%` : "—";
}

/* ===== データ取得 ===== */
const publicBase = "/api";

/** 日次系列（全体 or シンボル別） */
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

/** シンボル候補 */
async function fetchSymbols(daysForList = 90): Promise<string[]> {
  const base = apiBase();
  const url = new URL(`${base}/public/symbols`);
  url.searchParams.set("days", String(daysForList));
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

/** 最新シグナル（直近days日から10件） */
async function fetchLatestSignals(days = 30, symbol?: string): Promise<SignalRecord[]> {
  const base = apiBase();
  const url = new URL(`${base}/public/signals`);
  url.searchParams.set("limit", "10");
  const sinceIso = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();
  url.searchParams.set("since", sinceIso);
  if (symbol) url.searchParams.set("symbol", symbol);

  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

/* ===== ページ本体 ===== */
export default async function Page({
  searchParams,
}: {
  // Next.js 15 の型ずれ吸収：Promise 許容
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = searchParams ? await searchParams : undefined;
  const daysRaw =
    typeof sp?.days === "string" ? sp.days :
    Array.isArray(sp?.days) ? sp?.days[0] : "30";
  const daysNum = Number(daysRaw);
  const days = Number.isFinite(daysNum)
    ? Math.min(365, Math.max(7, daysNum))
    : 30;
  const symbol =
    typeof sp?.symbol === "string"
      ? sp.symbol
      : Array.isArray(sp?.symbol)
      ? sp?.symbol[0]
      : undefined;

  try {
    // 初期描画に必要なものをSSRでまとめて取得
    const [data, initialSymbols, latestSignals] = await Promise.all([
      fetchDaily(days, symbol),
      fetchSymbols(90),
      fetchLatestSignals(days, symbol),
    ]);

    // CSVリンク（sinceはdays起点）
    const csvDailyUrl = symbol
      ? `${publicBase}/public/export/performance/daily_by_symbol.csv?symbol=${encodeURIComponent(
          symbol
        )}&days=${days}`
      : `${publicBase}/public/export/performance/daily.csv?days=${days}`;

    return (
      <main className="p-6 md:p-8 max-w-3xl mx-auto space-y-8">
        {/* ヘッダ */}
        <PageHeader
          title="透明性（パフォーマンス＆シグナル）"
          description={<>直近{days}日の結果と主要指標、最新シグナルを公開します。シンボル切替・CSVダウンロードに対応。</>}
        />

        {/* フィルタ行：クイックレンジ＋フォーム（カード折りたたみ） */}
        <FilterBar
          left={
            <div className="space-y-3">
              {/* --- 1段目：クイックレンジ --- */}
              <div className="flex gap-2">
                <QuickFilters mode="range" basePath="/transparency" symbol={symbol} activeDays={days} />
              </div>

              {/* --- 2段目：Days / Symbol / Apply / Reset --- */}
              <FilterCard title="検索フィルタ" defaultOpen={false}>
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
                      {initialSymbols.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>

                  {/* Apply（2カラム） */}
                  <div className="md:col-span-2 flex">
                    <button
                      className="w-full h-10 rounded-2xl shadow px-4 whitespace-nowrap text-sm
                                bg-gray-900 text-white dark:bg-white dark:text-gray-900"
                      title="フィルタを適用"
                    >
                      <span className="md:inline hidden">Apply</span>
                      <span className="md:hidden inline">OK</span>
                    </button>
                  </div>

                  {/* Reset（2カラム） */}
                  <div className="md:col-span-2 flex">
                    <Link
                      className="w-full h-10 text-center rounded-2xl border px-4 py-2 whitespace-nowrap text-sm
                                border-gray-300 dark:border-gray-700 grid place-items-center"
                      href="/transparency?days=30"
                      title="フィルタをクリア"
                    >
                      <span className="md:inline hidden">Reset</span>
                      <span className="md:hidden inline">CLR</span>
                    </Link>
                  </div>
                </form>
              </FilterCard>
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
              <div className="text-center space-y-2">
                <div>表示できるデータがありません。</div>
                <div className="flex gap-2 justify-center">
                  <Link
                    className="rounded-2xl border px-3 py-1.5 border-gray-300 dark:border-gray-700"
                    href={`/transparency?days=${Math.max(30, Math.min(90, days))}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`}
                  >
                    期間を90日に広げる
                  </Link>
                  {symbol && (
                    <Link
                      className="rounded-2xl border px-3 py-1.5 border-gray-300 dark:border-gray-700"
                      href={`/transparency?days=${days}`}
                    >
                      シンボル解除
                    </Link>
                  )}
                </div>
              </div>
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
                  <th scope="col" className="px-3 py-2 text-right">
                    <abbr title="当日の全トレード件数">Trades</abbr>
                  </th>
                  <th scope="col" className="px-3 py-2 text-right">
                    <abbr title="Winの定義: pnl > 0。Win Rate = wins / total_trades">Win Rate</abbr>
                  </th>
                  <th scope="col" className="px-3 py-2 text-right">
                    <abbr title="当日の pnl_pct の単純平均（JST日単位）">Avg PnL</abbr>
                  </th>
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
                      <div className="space-y-2">
                        <div>データがありません。</div>
                        <div className="flex gap-2 justify-center">
                          <Link
                            className="rounded-2xl border px-3 py-1.5 border-gray-300 dark:border-gray-700"
                            href={`/transparency?days=${Math.max(30, Math.min(90, days))}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`}
                          >
                            期間を90日に広げる
                          </Link>
                          {symbol && (
                            <Link
                              className="rounded-2xl border px-3 py-1.5 border-gray-300 dark:border-gray-700"
                              href={`/transparency?days=${days}`}
                            >
                              シンボル解除
                            </Link>
                          )}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-2">※ Win/Avg PnL% は当日トレードに基づく単純集計です。</p>
        </Section>

        {/* 最新シグナル10件（カードUI） */}
        <Section
          title={<>最新シグナル（10件）{symbol ? ` / ${symbol}` : ""}</>}
          subtitle={<>※ 直近 {days} 日のデータから抽出しています。</>}
        >
          {latestSignals.length === 0 ? (
            <div className="text-sm text-gray-500">該当するシグナルが見つかりませんでした。</div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {latestSignals.map((s, i) => (
                <article
                  key={String(s.signal_id ?? `${s.symbol}-${s.created_at}-${i}`)}
                  className="group rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-shadow p-4"
                  aria-label={`${s.symbol} の ${s.side} シグナル`}
                >
                  {/* 1段目：シグナル（BUY/SELL/EXIT）＋ ロング/ショート */}
                  <div className="flex items-center justify-between">
                    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${sidePillClass(s.side)}`}>
                      {sideLabel(s.side)}
                    </span>
                    <span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs border border-gray-200 dark:border-gray-700">
                      {posLabel(s.side)}
                    </span>
                  </div>

                  {/* 2段目：価格 */}
                  <div className="mt-3 text-base font-semibold">
                    {typeof s.price === "number" ? `price: ${s.price}` : "price: —"}
                  </div>

                  {/* 3段目：時刻（JST、柔軟パース） */}
                  <div className="mt-2 text-xs text-gray-500">
                    {fmtSignalTimeJST(s)}
                  </div>
                </article>
              ))}
            </div>
          )}

          {/* Tailwindのパージ対策：色クラスのsafelist（未使用警告なしの直接配置） */}
          <div className="hidden">
            <span className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300" />
            <span className="bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300" />
            <span className="bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300" />
          </div>
        </Section>

        {/* 算出ロジック（概要） */}
        <Section title="算出ロジック（概要）">
          <ul className="list-disc pl-5 space-y-1 text-sm text-gray-700 dark:text-gray-200">
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
        </Section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <PageHeader title="Transparency" />
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
        <p className="text-sm text-gray-600 mt-2">
          ネットワークまたはAPIの一時的な不調の可能性があります。時間をおいて再度お試しください。
        </p>
      </main>
    );
  }
}
