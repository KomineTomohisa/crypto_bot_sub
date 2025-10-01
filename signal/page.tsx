// /app/signals/page.tsx
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
import { TablePagination } from "@/components/ui/TablePagination";
import SinceField from "./SinceField";
import ExpandableText from "./ExpandableText";
import { FilterCard } from "@/components/ui/FilterCard";

type Sig = {
  signal_id?: number | string;
  symbol: string;
  side: string;
  price?: number;
  generated_at?: string;
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

// Next.js 15想定：searchParams を Promise で受けられるように
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
      <main className="p-6 md:p-8 max-w-3xl mx-auto space-y-8">
        {/* ヘッダ */}
        <PageHeader
          title="Signals（シグナル一覧）"
          description={<>期間・シンボルで絞り込み、ページング可能。CSVダウンロードにも対応。</>}
        />

        {/* フィルタ行：すべて1列（横スクロール対応） */}
        <FilterBar
          left={
            <div className="space-y-3">
              {/* --- 1段目: クイックレンジ --- */}
              <div className="flex gap-2">
                <QuickFilters mode="since" basePath="/signals" symbol={symbol} />
                {since && (
                  <span className="text-xs text-gray-500 ml-1 whitespace-nowrap">
                    since: {new Date(since).toLocaleString()}
                  </span>
                )}
              </div>

              {/* --- 2段目: Symbol / Since / Limit / Apply / Reset --- */}
              <FilterCard title="検索フィルタ" defaultOpen={false}>
                <form method="get" className="grid grid-cols-1 md:grid-cols-12 gap-3 md:items-end">
                  {/* Symbol（3/12） */}
                  <div className="md:col-span-3">
                    <label className="block text-sm text-gray-600 dark:text-gray-300">Symbol（任意）</label>
                    <select
                      name="symbol"
                      defaultValue={symbol ?? ""}
                      className="h-10 w-full border rounded-xl px-3 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700"
                    >
                      <option value="">（全体）</option>
                      {symbols.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>

                  {/* Since（5/12） */}
                  <div className="md:col-span-5">
                    <label className="block text-sm text-gray-600 dark:text-gray-300">
                      Since（日付/時刻）
                      <abbr title="入力はローカル時刻、送信時はISO(UTC)に変換されます" className="ml-1 cursor-help">ⓘ</abbr>
                    </label>
                    <SinceField
                      defaultValueISO={since || defaultSince}
                      withLabel={false}
                      inputClassName="h-10"
                    />
                  </div>

                  {/* Limit＆ボタン（4/12） */}
                  <div className="md:col-span-4">
                    <div className="grid grid-cols-12 gap-3 md:items-end">
                      {/* Limit（4/12） */}
                      <div className="col-span-6 sm:col-span-4">
                        <label className="block text-sm text-gray-600 dark:text-gray-300">Limit</label>
                        <input
                          type="number"
                          name="limit"
                          min={1}
                          max={200}
                          defaultValue={limit || 50}
                          className="h-10 w-full border rounded-xl px-3 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700"
                        />
                      </div>

                      {/* Apply（4/12） */}
                      <div className="col-span-6 sm:col-span-4 flex">
                        <button
                          className="h-10 w-full rounded-2xl shadow px-3 whitespace-nowrap text-sm shrink-0
                                    bg-gray-900 text-white dark:bg-white dark:text-gray-900"
                          title="フィルタを適用"
                        >
                          <span className="md:inline hidden">Apply</span>
                          <span className="md:hidden inline">OK</span>
                        </button>
                      </div>

                      {/* Reset（4/12） */}
                      <div className="col-span-12 sm:col-span-4 flex">
                        <Link
                          className="h-10 w-full text-center rounded-2xl border px-3 py-2 whitespace-nowrap text-sm shrink-0
                                    border-gray-300 dark:border-gray-700 grid place-items-center"
                          href={`/signals?${buildQS({ since: defaultSince })}`}
                          title="フィルタをリセット（過去30日）"
                        >
                          <span className="md:inline hidden">Reset</span>
                          <span className="md:hidden inline">CLR</span>
                        </Link>
                      </div>
                    </div>
                  </div>
                </form>
              </FilterCard>

            </div>
          }
          right={
            <Notes
              items={[
                { label: <>時刻は<b>ローカル表示</b></>, tooltip: "保存はUTC、表示は端末のタイムゾーン" },
                { label: <>大小文字非区別: <code>lower(symbol)</code></> },
              ]}
            />
          }
        />

        {/* テーブル */}
        <Section
          title="一覧"
          headerRight={
            <CsvButtons
              links={[
                {
                  href: csvUrl,
                  label: "CSV（最大1000件）",
                  download: true,
                  target: "_blank",
                  rel: "noopener",
                },
              ]}
            />
          }
        >
          {/* ページング（上） */}
          <TablePagination
            prevHref={`?${prevQS}`}
            nextHref={`?${nextQS}`}
            canPrev={meta.page > 1}
            canNext={meta.hasNext}
          />

          {/* テーブル本体 */}
          <div className="overflow-x-auto rounded-2xl mt-3">
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
                  <tr
                    key={String(r.signal_id ?? `${r.symbol}-${r.generated_at}-${i}`)}
                    className="border-t border-gray-100 dark:border-gray-800"
                  >
                    <td className="px-4 py-2">{dt(r.generated_at!)}</td>
                    <td className="px-4 py-2">{fmt(r.symbol)}</td>
                    <td className="px-4 py-2">{fmt(r.side)}</td>
                    <td className="px-4 py-2 text-right">{fmt(r.price)}</td>
                    <td className="px-4 py-2 text-right">{num(r.strength_score, 2)}</td>
                    <td className="px-4 py-2 max-w-[480px]">
                      <ExpandableText text={r.reason} maxChars={140} />
                    </td>
                  </tr>
                ))}
                {items.length === 0 && (
                  <tr>
                    <td className="px-4 py-6 text-center text-gray-500" colSpan={6}>
                      <div className="space-y-2">
                        <div>データがありません。</div>
                        <div className="flex gap-2 justify-center">
                          <Link
                            className="rounded-2xl border px-3 py-1.5 border-gray-300 dark:border-gray-700"
                            href={`/signals?${buildQS({ symbol, since: new Date(Date.now() - 90*24*60*60*1000).toISOString(), page: 1, limit: meta.limit })}`}
                          >
                            期間を90日に広げる
                          </Link>
                          {symbol && (
                            <Link
                              className="rounded-2xl border px-3 py-1.5 border-gray-300 dark:border-gray-700"
                              href={`/signals?${buildQS({ since: defaultSince, page: 1, limit: meta.limit })}`}
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

          {/* ページング（下） */}
          <TablePagination
            prevHref={`?${prevQS}`}
            nextHref={`?${nextQS}`}
            canPrev={meta.page > 1}
            canNext={meta.hasNext}
          />
        </Section>
      </main>
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return (
      <main className="p-6 max-w-6xl mx-auto">
        <PageHeader title="Signals" />
        <div className="text-red-600">読み込みに失敗しました。{msg}</div>
        <p className="text-sm text-gray-600 mt-2">
          ネットワークまたはAPIの一時的な不調の可能性があります。時間をおいて再度お試しください。
        </p>
      </main>
    );
  }
}
