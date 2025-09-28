export const revalidate = 0;

import Link from "next/link";
import { PageHeader, Section } from "@/components/ui";
import ChartClient from "@/app/performance/ChartClient";

/* =========================
   Types
   ========================= */
type TimeLike = string | number | Date | null | undefined;

type Kpis = {
  win_rate?: number | null;     // 0.0-1.0
  avg_pnl_pct?: number | null;  // 0.0123 (=1.23%)
  total_trades?: number | null;
  recent_signal_count?: number | null; // 未使用だがKPI APIに残っている前提
};

type Daily = {
  date: string;                 // JST日（YYYY-MM-DD）
  win_rate?: number | null;
  avg_pnl_pct?: number | null;
  total_trades: number;
};

type Position = {
  position_id?: string | number;
  symbol: string;               // 例: btc_jpy
  side: "LONG" | "SHORT";       // 保有方向
  size: number;                 // ロット
  entry_price: number;
  entry_time?: TimeLike;        // 取得日時
  unrealized_pnl_pct?: number | null; // 含み損益（%）0.0123形式を想定
};

/* =========================
   Helpers (no-any)
   ========================= */
function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

function normSide(side?: string) {
  return (side ?? "").trim().toUpperCase();
}

function pillClassForSide(side?: string) {
  const s = normSide(side);
  if (s.includes("SHORT"))
    return "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300";
  if (s.includes("LONG"))
    return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300";
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
    const t = v.trim();
    if (!t) return null;
    if (/^\d+$/.test(t)) {
      const num = Number(t);
      const ms = num > 1e12 ? num : num * 1000;
      const d = new Date(ms);
      return isNaN(d.getTime()) ? null : d;
    }
    const d = new Date(t);
    return isNaN(d.getTime()) ? null : d;
  }
  return null;
}

function fmtJST(v: TimeLike): string {
  const d = parseDateFlexible(v);
  return d ? d.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo" }) : "Invalid Date";
}

function pct(v?: number | null, digits = 1) {
  return v != null ? `${(v * 100).toFixed(digits)}%` : "—";
}

/* =========================
   Data fetchers (SSR)
   ========================= */
const DAYS = 7; // ★ 7日固定

async function fetchKpis(symbol?: string): Promise<Kpis> {
  const base = apiBase();
  const url = new URL(`${base}/public/metrics`);
  url.searchParams.set("days", String(DAYS));
  if (symbol) url.searchParams.set("symbol", symbol);
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) {
    return { win_rate: null, avg_pnl_pct: null, total_trades: null, recent_signal_count: null };
  }
  return res.json();
}

async function fetchDaily(symbol?: string): Promise<Daily[]> {
  const base = apiBase();
  const url = new URL(
    symbol
      ? `${base}/public/performance/daily/by-symbol`
      : `${base}/public/performance/daily`
  );
  url.searchParams.set("days", String(DAYS));
  if (symbol) url.searchParams.set("symbol", symbol);
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

/** 現在の保有ポジション一覧（未実装なら空配列でもOK） */
async function fetchOpenPositions(symbol?: string): Promise<Position[]> {
  const base = apiBase();
  const url = new URL(`${base}/public/positions/open`);
  if (symbol) url.searchParams.set("symbol", symbol);
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

/* =========================
   Page (SSR)
   ========================= */
export default async function Page({
  searchParams,
}: {
  // Next.js 15 対応（Promiseも許容）だが今回は7日固定なので symbol 以外無視
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = searchParams ? await searchParams : undefined;

  const symbol =
    typeof sp?.symbol === "string"
      ? sp.symbol
      : Array.isArray(sp?.symbol)
      ? sp?.symbol[0]
      : undefined;

  const [kpis, daily, positions] = await Promise.all([
    fetchKpis(symbol),
    fetchDaily(symbol),
    fetchOpenPositions(symbol),
  ]);

  return (
    <main className="p-6 md:p-8 max-w-6xl mx-auto space-y-8">
      <PageHeader
        title="Dashboard"
        description={<>過去<b>{DAYS}日</b>のKPI、簡易チャート、現在の保有ポジションを表示します。</>}
      />

      {/* 1. KPI（7日固定） */}
      <Section title={`KPI（直近 ${DAYS} 日）`}>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <KpiCard label="Win Rate" value={pct(kpis.win_rate, 1)} hint="直近期間の勝率" />
          <KpiCard label="Avg PnL%" value={kpis.avg_pnl_pct != null ? `${(kpis.avg_pnl_pct * 100).toFixed(2)}%` : "—"} hint="日次平均（単純）" />
          <KpiCard label="Total Trades" value={kpis.total_trades ?? "—"} hint="直近期間の総トレード数" />
          <KpiCard label="Positions (Open)" value={positions.length} hint="現在の保有数" />
        </div>
      </Section>

      {/* 2. パフォーマンス（簡易チャート、7日固定） */}
      <Section
        title={`パフォーマンス（直近 ${DAYS} 日）`}
        headerRight={
          <Link className="text-sm underline" href={`/performance?days=${DAYS}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`}>
            詳細へ →
          </Link>
        }
      >
        {daily.length > 0 ? (
          <ChartClient data={daily} />
        ) : (
          <div className="h-32 grid place-items-center text-sm text-gray-500">データがありません。</div>
        )}
      </Section>

      {/* 3. 現在の保有ポジション（最新シグナルの代替） */}
      <Section title="現在の保有ポジション">
        {positions.length === 0 ? (
          <div className="text-sm text-gray-500">保有中のポジションはありません。</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {positions.map((p) => (
              <article
                key={String(p.position_id ?? `${p.symbol}-${p.entry_time ?? ""}-${p.side}`)}
                className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm"
              >
                {/* 1段目：方向ピル＋シンボル */}
                <div className="flex items-center justify-between">
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${pillClassForSide(p.side)}`}>
                    {normSide(p.side).includes("SHORT") ? "SHORT" : "LONG"}
                  </span>
                  <span className="inline-flex items-center rounded-full px-2 py-0.5 text-xs border border-gray-200 dark:border-gray-700">
                    {p.symbol}
                  </span>
                </div>

                {/* 2段目：サイズ・建値 */}
                <div className="mt-3 text-sm">
                  <div>size: <b>{p.size}</b></div>
                  <div>entry: <b>{p.entry_price}</b></div>
                </div>

                {/* 3段目：エントリー時刻・含み損益 */}
                <div className="mt-2 text-xs text-gray-500">Entry Time: {fmtJST(p.entry_time)}</div>
                <div className="mt-1 text-base font-semibold">
                  Unrealized PnL%: {p.unrealized_pnl_pct != null ? `${(p.unrealized_pnl_pct * 100).toFixed(2)}%` : "—"}
                </div>
              </article>
            ))}
          </div>
        )}
      </Section>

      {/* 4. クイックアクセス（7日固定リンク） */}
      <Section title="クイックアクセス">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <ShortcutCard href={`/transparency?days=${DAYS}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`} title="Transparency" desc="公開指標・CSV・最新実績" />
          <ShortcutCard href={`/performance?days=${DAYS}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`} title="Performance" desc="推移と比較の詳細分析" />
          <ShortcutCard href={`/signals?since=${encodeURIComponent(new Date(Date.now()-DAYS*24*60*60*1000).toISOString())}${symbol ? `&symbol=${encodeURIComponent(symbol)}` : ""}`} title="Signals" desc="最新シグナル一覧（参考）" />
          <ShortcutCard href="/learn" title="Learn" desc="指標や戦略の解説" />
        </div>
      </Section>

      {/* Tailwind safelist（色） */}
      <div className="hidden">
        <span className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300" />
        <span className="bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300" />
        <span className="bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300" />
      </div>
    </main>
  );
}

/* =========================
   Local UI mini components
   ========================= */
function KpiCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number | null;
  hint?: string;
}) {
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="mt-1 text-2xl font-bold">{value ?? "—"}</div>
      {hint && <div className="mt-1 text-xs text-gray-500">{hint}</div>}
    </div>
  );
}

function ShortcutCard({ href, title, desc }: { href: string; title: string; desc: string }) {
  return (
    <Link
      href={href}
      className="block rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm hover:shadow-md transition-shadow"
    >
      <div className="font-semibold">{title}</div>
      <div className="mt-1 text-sm text-gray-600 dark:text-gray-300">{desc}</div>
      <div className="mt-2 text-xs underline">開く →</div>
    </Link>
  );
}
