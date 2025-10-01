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
  win_rate?: number | null;     // 0.42 など（%換算は表示側）
  avg_pnl_pct?: number | null;  // その日の平均損益率（%換算は表示側）
  total_trades: number;

  // 追加（あれば使う）：日次の平均保有時間（時間）
  avg_holding_hours?: number | null;

  // 追加（任意）：その日の総合PnL%（あるなら使う。無ければ avg_pnl_pct を近似使用）
  total_pnl_pct?: number | null;
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
  price_ts?: string | number | null;
};

type KPI = {
  trade_count: number;
  win_count: number;
  win_rate: number | null;
  total_pnl: number | null;
  avg_pnl_pct: number | null;
  avg_holding_hours: number | null;
};

type PositionWithPriceTS = Position & { price_ts?: string | number | null };

/* =========================
   Helpers (no-any)
   ========================= */
function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
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

// ▼ ここを置換
function fmtPctSmart(v?: number | null, digits = 1) {
  if (v == null) return "—";
  const abs = Math.abs(v);
  // 1以下なら比率(0.0123=1.23%)、1超なら既に%値(1.23=1.23%)
  const val = abs <= 1 ? v * 100 : v;
  return `${val.toFixed(digits)}%`;
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

function fmtJST(v: TimeLike, time: boolean = true): string {
  const d = parseDateFlexible(v);
  if (!d) return "";
  const optsDate: Intl.DateTimeFormatOptions = {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  };
  const optsTime: Intl.DateTimeFormatOptions = time
    ? { hour: "2-digit", minute: "2-digit", hour12: false }
    : {};
  // 例: "2025/09/29 11:35" → "/" を "-" に
  const s = new Intl.DateTimeFormat("ja-JP", { ...optsDate, ...optsTime }).format(d);
  return s.replace(/\//g, "-");
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
    <main className="p-6 md:p-8 max-w-4xl mx-auto space-y-8">
      <PageHeader
        title="Dashboard"
        description={<>過去<b>{DAYS}日</b>のKPI、簡易チャート、現在の保有ポジションを表示します。</>}
      />
      
      {/* 1. KPI（7日固定） */}
      <Section
        title="主要指標（直近7日間）"
        headerRight={
          (() => {
            const latest = daily.at(-1)?.date;
            // KPIの鮮度：当日JSTと一致していなければ stale
            const jstNow = new Date().toLocaleDateString("ja-JP", { timeZone: "Asia/Tokyo" }).replaceAll("/", "-");
            const stale = latest && latest !== jstNow;
            return (
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span>最終更新: {latest ?? "—"} JST</span>
                {stale && (
                  <span className="ml-1 rounded px-1.5 py-0.5 bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300">
                    stale
                  </span>
                )}
              </div>
            );
          })()
        }
      >
        {(() => {
          // ---------- 共通ユーティリティ ----------
          const getLastTwo = (arr: Array<number | null | undefined>): [number | null, number | null] => {
            const vals = arr.filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
            const n = vals.length;
            return n >= 2 ? [vals[n - 2], vals[n - 1]] : [null, null];
          };
          // 0.42（比率）は%に直し、52 はそのまま扱う。差分も%で返す。
          const toPctDelta = (a?: number | null, b?: number | null): number | null => {
            if (a == null || b == null) return null;
            const conv = (x: number) => Math.abs(x) <= 1 ? x * 100 : x; // 0.42 -> 42
            return Number((conv(b) - conv(a)).toFixed(2));
          };

          // 7日分 日次系列
          const seriesTrades   = daily.map(d => d.total_trades ?? 0);
          const seriesWinRate  = daily.map(d => d.win_rate ?? null);
          const seriesAvgPnl   = daily.map(d => d.avg_pnl_pct ?? null);
          const seriesHoldH    = daily.map(d => d.avg_holding_hours ?? null);

          // 前日比（直近2点）
          const [tPrev,  tLast ] = getLastTwo(daily.map(d => d.total_trades));
          const [wrPrev, wrLast] = getLastTwo(seriesWinRate);
          const [apPrev, apLast] = getLastTwo(seriesAvgPnl);
          const [hhPrev, hhLast] = getLastTwo(seriesHoldH);

          const deltaTrades   = (tPrev != null && tLast != null) ? (tLast - tPrev) : null;           // 件
          const deltaWinRate  = toPctDelta(wrPrev, wrLast);                                           // %
          const deltaAvgPnl   = toPctDelta(apPrev, apLast);                                           // %
          const deltaHoldHour = (hhPrev != null && hhLast != null) ? Number((hhLast - hhPrev).toFixed(1)) : null; // h

          // ---- Max Drawdown / Sharpe / Sortino （7日）----
          // 7日分の日次リターン（%）を作る：total_pnl_pct があれば優先、無ければ avg_pnl_pct を使用
          const dayReturnsPct = daily.map(d => {
            const v = (d.total_pnl_pct ?? d.avg_pnl_pct);
            if (v == null || Number.isNaN(Number(v))) return 0;
            // 0.42 -> 42,  2.5 -> 2.5
            return Math.abs(Number(v)) <= 1 ? Number(v) * 100 : Number(v);
          });

          // 累積曲線（%ベースの擬似エクイティ）を作成して MaxDD を算出
          const equity = (() => {
            let acc = 0;
            return dayReturnsPct.map(r => (acc += r));
          })();

          const maxDrawdownPct = (() => {
            let peak = -Infinity, maxDD = 0;
            for (const x of equity) {
              peak = Math.max(peak, x);
              const dd = x - peak; // マイナス値
              maxDD = Math.min(maxDD, dd);
            }
            return Number(maxDD.toFixed(2)); // 例: -12.34（%）
          })();

          const sharpeSortino = (() => {
            const xs = dayReturnsPct;                 // 日次%（例: 1.2, -0.8, ...）
            const n  = xs.length || 1;
            const mean = xs.reduce((a, b) => a + b, 0) / n;
            const std  = Math.sqrt(xs.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / n) || 0;

            const downside = xs.filter(v => v < 0);
            const dMean = downside.length ? downside.reduce((a, b) => a + b, 0) / downside.length : 0;
            const dStd  = downside.length
              ? Math.sqrt(downside.reduce((s, v) => s + Math.pow(v - dMean, 2), 0) / downside.length)
              : 0;

            // 年率化（参考）：日次→年次換算。営業日換算なら√252でもOK
            const ann = Math.sqrt(365);
            const sharpe  = std  ? (mean / std) * ann : 0;
            const sortino = dStd ? (mean / dStd) * ann : 0;

            return { sharpe: Number(sharpe.toFixed(2)), sortino: Number(sortino.toFixed(2)) };
          })();

          return (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* 既存KPI（%表記に統一） */}
              <KpiCardPro
                title="取引回数"
                value={kpi?.trade_count}
                valueRender={(v) => (v ?? "—")}
                series={seriesTrades.map(v => v ?? 0)}
                delta={deltaTrades}
                deltaUnit=""
                hint="7日間の合計トレード数"
              />
              <KpiCardPro
                title="勝率"
                value={kpi?.win_rate}
                valueRender={(v) => fmtPctSmart(v as number | null | undefined)}  // 0.42 -> 42.0%
                series={seriesWinRate.map(v => (v ?? 0))}
                delta={deltaWinRate}
                deltaUnit="%"
                hint="全取引に占める勝ちトレードの割合"
              />
              <KpiCardPro
                title="総損益"
                value={kpi?.total_pnl}
                valueRender={(v) => (v != null ? Number(v).toFixed(2) : "—")}
                series={seriesAvgPnl.map(v => (v ?? 0))}
                delta={deltaAvgPnl}
                deltaUnit="%"
                hint="7日間の合計損益額"
              />
              <KpiCardPro
                title="平均損益率"
                value={kpi?.avg_pnl_pct}
                valueRender={(v) => fmtPctSmart(v as number | null | undefined)}
                series={seriesAvgPnl.map(v => (v ?? 0))}
                delta={deltaAvgPnl}
                deltaUnit="%"
                hint="1トレードあたりの平均損益率"
              />
              <KpiCardPro
                title="平均保有時間"
                value={kpi?.avg_holding_hours}
                valueRender={(v) => (v != null ? Number(v).toFixed(1) : "—")}
                series={seriesHoldH.map(v => (v ?? 0))}
                delta={deltaHoldHour}
                deltaUnit="h"
                hint="1ポジションの平均保有時間"
              />

              {/* 追加KPI：MaxDD / Sharpe / Sortino（7日） */}
              <KpiCardPro
                title="最大ドローダウン"
                value={maxDrawdownPct}
                valueRender={(v) => (v != null ? `${Number(v).toFixed(2)}%` : "—")}
                series={equity}                            // 疑似エクイティ曲線
                delta={null}
                deltaUnit=""
                hint="7日間での累積最大下落率"
              />
              <KpiCardPro
                title="シャープレシオ"
                value={sharpeSortino.sharpe}
                valueRender={(v) => (v != null ? Number(v).toFixed(2) : "—")}
                series={dayReturnsPct}
                delta={null}
                deltaUnit=""
                hint="リスクあたりの収益性（年率換算）。1以上なら安定的に良好"
              />
              <KpiCardPro
                title="ソルティノレシオ"
                value={sharpeSortino.sortino}
                valueRender={(v) => (v != null ? Number(v).toFixed(2) : "—")}
                series={dayReturnsPct}
                delta={null}
                deltaUnit=""
                hint="下方リスク限定のリスク調整リターン（年率換算）。1以上なら望ましい水準"
              />
            </div>
          );
        })()}
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
      <Section
        title="現在の保有ポジション"
        headerRight={
          (() => {
            // positions の price_ts の最大（最新）を取得して時刻表示
            const toDateMs = (v: string | number | null | undefined): number | null => {
              const d = parseDateFlexible(v);
              return d ? d.getTime() : null;
            };
            const times = positions
              .map(p => toDateMs(p.price_ts))
              .filter((t): t is number => typeof t === "number");
            const latestMs = times.length ? Math.max(...times) : null;

            const latestLabel = latestMs ? fmtJST(latestMs) : "—";
            // しきい値：5分より古ければ stale
            const stale = latestMs ? (Date.now() - latestMs) > 5 * 60 * 1000 : false;

            return (
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span>更新: {latestLabel}</span>
                {stale && (
                  <span className="ml-1 rounded px-1.5 py-0.5 bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300">
                    stale
                  </span>
                )}
              </div>
            );
          })()
        }
      >
        {positions.length === 0 ? (
          <div className="text-sm text-gray-500">保有中のポジションはありません。</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
            {positions.map((p, i) => (
              <PositionCard key={`${p.symbol}-${i}`} p={p} />
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
function Sparkline({
  values,
  className,
  maxHeight = 36,
}: {
  values: number[];
  className?: string;
  maxHeight?: number;
}) {
  const w = 120;
  const h = maxHeight;
  const n = Math.max(values.length, 1);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const range = max - min || 1;

  const points = values.map((v, i) => {
    const x = (i / (n - 1 || 1)) * (w - 2) + 1;
    const y = h - ((v - min) / range) * (h - 2) - 1;
    return `${x},${y}`;
  });

  return (
    <svg
      className={className}
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      aria-hidden="true"
    >
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        className="text-gray-400 dark:text-gray-500"
      />
    </svg>
  );
}

function DeltaBadge({
  delta,
  unit = "",
}: {
  delta: number | null;
  unit?: string;
}) {
  if (delta == null || Number.isNaN(delta)) {
    return <span className="text-xs text-gray-400">—</span>;
  }
  const sign = delta === 0 ? "" : delta > 0 ? "+" : "";
  const positive = delta > 0;
  const negative = delta < 0;
  return (
    <span
      className={
        "inline-flex items-center text-xs font-medium " +
        (positive
          ? "text-emerald-600 dark:text-emerald-400"
          : negative
            ? "text-rose-600 dark:text-rose-400"
            : "text-gray-500")
      }
      aria-label={`delta ${delta}${unit}`}
      title={`前後比較の差分：${delta}${unit}`}
    >
      {positive ? "▲" : negative ? "▼" : "■"} {sign}
      {Math.abs(delta).toFixed(Math.abs(delta) < 1 ? 2 : 1)}
      {unit}
    </span>
  );
}

function KpiCardPro<T>({
  title,
  value,
  valueRender,
  series,
  delta,
  deltaUnit,
  hint,
}: {
  title: string;
  value: T | null | undefined;
  valueRender: (v: T | null | undefined) => string | number;
  series: number[];
  delta: number | null;
  deltaUnit: string;
  hint?: string;
}) {
  const rendered = valueRender(value);
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs text-gray-500">{title}</div>
          <div className="mt-1 text-2xl font-bold tabular-nums">{rendered}</div>
        </div>
        <DeltaBadge delta={delta} unit={deltaUnit} />
      </div>

      <div className="mt-3">
        {series.length > 0 ? (
          <Sparkline values={series} />
        ) : (
          <div className="h-9 rounded bg-gray-100 dark:bg-gray-800 animate-pulse" />
        )}
      </div>

      {hint && <div className="mt-2 text-xs text-gray-500">{hint}</div>}
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

function isStaleTS(ts?: string | number | null, minutes = 5): boolean {
  if (!ts) return false;
  const t = typeof ts === "number" ? ts : Date.parse(ts);
  if (Number.isNaN(t)) return false;
  const now = Date.now();
  return now - t > minutes * 60 * 1000;
}

function Pill({ children, tone = "default" }: { children: React.ReactNode; tone?: "green" | "red" | "default" }) {
  const map: Record<string, string> = {
    green: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    red:   "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
    default: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };
  return (
    <span className={`inline-flex items-center rounded px-2 py-0.5 text-[11px] font-medium ${map[tone]}`}>
      {children}
    </span>
  );
}

/** PnL% を -10%〜+10% に収めてバー表示（外れ値は端でクリップ） */
/** PnL% を可視化するメーター。デフォルトは ±10% スケール */
function PnlMeter({ pct, range = 10 }: { pct: number | null | undefined; range?: number }) {
  if (pct == null || Number.isNaN(Number(pct))) {
    return (
      <div className="mt-2">
        <div className="h-2 rounded bg-gray-100 dark:bg-gray-800" />
        <div className="mt-1 flex justify-between text-[10px] text-gray-500">
          <span>-{range}%</span><span>0%</span><span>+{range}%</span>
        </div>
      </div>
    );
  }

  const v = Number(pct);
  const clipped = Math.abs(v) > range;
  const clamped = Math.max(-range, Math.min(range, v));
  const pos = ((clamped + range) / (2 * range)) * 100; // 0..100

  return (
    <div className="mt-2" aria-label={`PnL ${v.toFixed(2)}% （基準 ±${range}%）`}>
      <div className="relative h-3 rounded bg-gray-100 dark:bg-gray-800 overflow-hidden">
        {/* 中央線(0%) */}
        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-300/70 dark:bg-gray-700/70 -translate-x-1/2" />
        {/* フィル（0% から現在値方向へ伸ばす） */}
        <div
          className={`absolute top-0 bottom-0 ${clamped >= 0 ? "bg-emerald-500/80" : "bg-rose-500/80"}`}
          style={{
            left: clamped >= 0 ? "50%" : `${pos}%`,
            right: clamped >= 0 ? `${100 - pos}%` : "50%",
          }}
        />
        {/* クリップ表示（端に到達した合図） */}
        {clipped && (
          <div
            className="absolute top-1/2 -translate-y-1/2 text-[10px] px-1 py-0.5 rounded bg-black/50 text-white"
            style={{ left: clamped >= 0 ? "calc(100% - 34px)" : "6px" }}
            title="表示レンジ外（clipped）"
          >
            clipped
          </div>
        )}
      </div>

      {/* 目盛りと数値バッジ */}
      <div className="mt-1 flex items-center justify-between text-[10px] text-gray-500">
        <span>-{range}%</span>
        <span>0%</span>
        <span>+{range}%</span>
      </div>

      <div className="mt-1 text-right">
        <span
          className={[
            "inline-block text-[11px] px-1.5 py-0.5 rounded",
            clamped >= 0
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
              : "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
          ].join(" ")}
        >
          {v >= 0 ? "+" : ""}
          {v.toFixed(2)}%
        </span>
      </div>
    </div>
  );
}

function StatRow({
  label,
  value,
  sub,
  emphasize = false,
  valueClassName = "",
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  emphasize?: boolean;
  valueClassName?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="text-xs text-gray-500">{label}</div>
      <div
        className={[
          "tabular-nums break-words leading-tight",
          emphasize ? "text-xl font-bold" : "text-base font-semibold",
          valueClassName,
        ].join(" ")}
      >
        {value}
      </div>
      {sub && <div className="text-xs text-gray-500">{sub}</div>}
    </div>
  );
}

function PositionCard({ p }: { p: PositionWithPriceTS }) {
  const symbol = (p?.symbol ?? "").toString().toUpperCase();
  const side   = (p?.side ?? "").toString().toUpperCase() as "LONG" | "SHORT" | string;
  const size   = p?.size ?? null;
  const entryPrice = p?.entry_price ?? null;
  const entryTime  = p?.entry_time ?? null;
  const curPrice   = p?.current_price ?? null;
  const pnlPct     = typeof p?.unrealized_pnl_pct === "number" ? p.unrealized_pnl_pct : null;
  const priceTS    = p?.price_ts ?? null;
  const stale      = isStaleTS(priceTS, 5);

  // 金額PnL（size × 価格差）
  const pnlAbs: number | null = (() => {
    if (size == null || entryPrice == null || curPrice == null) return null;
    const diff = side === "SHORT" ? (entryPrice - curPrice) : (curPrice - entryPrice);
    return diff * size;
  })();

  // 保有時間（h）
  const holdH: string | null = (() => {
    if (!entryTime) return null;
    const d = new Date(entryTime);
    if (Number.isNaN(d.getTime())) return null;
    const h = (Date.now() - d.getTime()) / 3600000;
    return h >= 0 ? h.toFixed(1) : null;
  })();

  const sideTone: "green" | "red" = side === "LONG" ? "green" : "red";
  const pnlTone =
    pnlPct == null ? "default"
    : pnlPct >= 0   ? "green"
    :                 "red";

  return (
    <article className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm">
      {/* ヘッダ */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="text-sm font-semibold truncate">{symbol}</div>
          <Pill tone={sideTone}>{side}</Pill>
          {stale && <Pill>stale</Pill>}
        </div>
        {p?.position_id != null && (
          <div className="text-[11px] text-gray-500 shrink-0">ID: {String(p.position_id)}</div>
        )}
      </div>

      {/* 本体（すべて縦並び、行間ゆったり） */}
      <div className="mt-4 space-y-4">
        {/* 1) 現在価格 */}
        <StatRow
          label="現在価格"
          value={
            <span className="whitespace-nowrap">
              {curPrice != null ? `￥${curPrice.toLocaleString("ja-JP")}` : "—"}
            </span>
          }
          emphasize
        />

        {/* 2) 含み損益（額＋%） */}
        <StatRow
          label="含み損益"
          value={
            <span className={[
              pnlTone === "green" ? "text-emerald-600" : pnlTone === "red" ? "text-rose-600" : "",
              "whitespace-nowrap",
            ].join(" ")}>
              {pnlAbs == null ? "—" : pnlAbs.toLocaleString("ja-JP")}
              <span className="ml-1 text-xs text-gray-500">
                {pnlPct == null ? "" : `(${pnlPct.toFixed(2)}%)`}
              </span>
            </span>
          }
          sub={<PnlMeter pct={typeof pnlPct === "number" ? pnlPct : null} range={10} />}
          emphasize
        />

        {/* 3) サイズ / エントリー */}
        <StatRow
          label="サイズ / エントリー"
          value={
            entryPrice != null && size != null ? (
              <span className="break-words">
                <span className="whitespace-nowrap">{size}</span>
                <span className="mx-1">@</span>
                <span className="whitespace-nowrap">{Number(entryPrice).toLocaleString("ja-JP")}</span>
              </span>
            ) : "—"
          }
        />

        {/* 補足行（必要なら出す） */}
        <div className="text-sm text-gray-600 dark:text-gray-300">
          保有時間: <span className="font-semibold">{holdH ?? "—"} 時間</span>
        </div>
      </div>

      {/* 詳細リンク */}
      {p?.symbol && (
        <div className="mt-3 text-right">
          <Link
            href={`/transparency?days=7&symbol=${encodeURIComponent(p.symbol)}`}
            className="text-xs underline"
            prefetch={false}
          >
            この銘柄の詳細 →
          </Link>
        </div>
      )}
    </article>
  );
}