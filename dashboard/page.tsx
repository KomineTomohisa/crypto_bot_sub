export const revalidate = 0;

import Link from "next/link";
import { PageHeader, Section } from "@/components/ui";
import ChartClient from "@/app/performance/ChartClient";

/* =========================
   Types
   ========================= */
type TimeLike = string | number | Date | null | undefined;

type Daily = {
  date: string;                 // JST日（YYYY-MM-DD）
  win_rate?: number | null;
  avg_pnl_pct?: number | null;
  total_trades: number;
};

type Position = {
  position_id: number | string;
  symbol: string;
  side: 'LONG' | 'SHORT' | string;
  size: number | null;
  entry_price: number | null;
  entry_time: string | null;

  // ↓ API /public/positions/open_with_price の追加フィールド
  current_price?: number | null;
  unrealized_pnl_pct?: number | null;
  price_ts?: string | null;
};

type KPI = {
  trade_count: number;
  win_count: number;
  win_rate: number | null;
  total_pnl: number | null;
  avg_pnl_pct: number | null;
  avg_holding_hours: number | null;
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

// ▼ ここを置換
function fmtPctSmart(v?: number | null, digits = 1) {
  if (v == null) return "—";
  const abs = Math.abs(v);
  // 1以下なら比率(0.0123=1.23%)、1超なら既に%値(1.23=1.23%)
  const val = abs <= 1 ? v * 100 : v;
  return `${val.toFixed(digits)}%`;
}

// 価格鮮度（分）
const STALE_TTL_MIN = 10;
function isStale(ts?: string | null, ttlMin = STALE_TTL_MIN) {
  if (!ts) return true;
  const ageMin = (Date.now() - new Date(ts).getTime()) / 60000;
  return ageMin > ttlMin;
}

/* =========================
   Data fetchers (SSR)
   ========================= */
const DAYS = 7; // ★ 7日固定

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
  const url = new URL(`${base}/public/positions/open_with_price`);
  if (symbol) url.searchParams.set("symbol", symbol);
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

async function fetchKPI(symbol?: string): Promise<KPI | null> {
  const base = apiBase();
  const url = new URL(`${base}/public/kpi/7days`);
  if (symbol) url.searchParams.set("symbol", symbol);
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return null;
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

  const [kpi, daily, positions] = await Promise.all([
    fetchKPI(symbol),
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
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <KpiCard title="Trades (7d)" value={kpi?.trade_count ?? "—"} />
          <KpiCard title="Win Rate (7d)" value={fmtPctSmart(kpi?.win_rate)} />
          <KpiCard title="Total PnL (7d)" value={kpi?.total_pnl != null ? kpi.total_pnl.toFixed(2) : "—"} />
          <KpiCard title="Avg PnL% (7d)" value={fmtPctSmart(kpi?.avg_pnl_pct)} />
          <KpiCard title="Avg Holding Hours (7d)" value={kpi?.avg_holding_hours != null ? kpi.avg_holding_hours.toFixed(1) : "—"} />
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
                  <div>entry: <b>{p.entry_price?.toLocaleString?.('ja-JP') ?? p.entry_price ?? '—'}</b></div>
                </div>

                {/* 3段目：エントリー時刻 */}
                <div className="mt-2 text-xs text-gray-500">
                  Entry Time: {fmtJST(p.entry_time)}
                </div>

                {/* 価格 + 鮮度バッジ */}
                <div className="mt-1 text-xs">
                  Price: {p.current_price != null ? p.current_price.toLocaleString('ja-JP') : '—'}
                  {isStale(p.price_ts) && (
                    <span className="ml-2 rounded px-1.5 py-0.5 text-[10px] bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300">
                      stale
                    </span>
                  )}
                </div>

                {/* 含み損益％（色分け） */}
                <div className="mt-1 text-base font-semibold">
                  <span
                    className={
                      "text-sm " +
                      (p.unrealized_pnl_pct == null
                        ? "text-gray-500"
                        : p.unrealized_pnl_pct > 0
                          ? "text-emerald-600 dark:text-emerald-400"
                          : p.unrealized_pnl_pct < 0
                            ? "text-rose-600 dark:text-rose-400"
                            : "text-gray-500")
                    }
                  >
                    PnL%: {p.unrealized_pnl_pct == null ? '—' : `${p.unrealized_pnl_pct.toFixed(2)}%`}
                  </span>
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
  title,
  value,
  hint,
}: {
  title: string;
  value: string | number | null;
  hint?: string;
}) {
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm">
      <div className="text-xs text-gray-500">{title}</div>
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
