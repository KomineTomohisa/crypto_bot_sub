"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

type Position = {
  position_id: number | string;
  symbol: string;
  side: "LONG" | "SHORT" | string;
  size: number | null;
  entry_price: number | null;
  entry_time: string | null;
  current_price?: number | null;
  unrealized_pnl_pct?: number | null;
  price_ts?: string | number | null;
};

type Candle15m = {
  time: string; // ISO文字列
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number | null;
};

type RawCandle = {
  time: string;
  open: number | string;
  high: number | string;
  low: number | string;
  close: number | string;
  volume?: number | string | null;
};

type Props = {
  positions: Position[];
};

type ChartPoint = Candle15m & {
  timeLabel: string;
  isEntry: boolean;
  entrySide?: string;
  entryPrice: number | null;
};

type EntryStarDotProps = {
  cx?: number;
  cy?: number;
  payload?: ChartPoint;
};

// エントリー価格の位置に★を描画
const EntryStarDot = ({ cx, cy, payload }: EntryStarDotProps) => {
  if (!payload) return null;
  if (!payload.isEntry || payload.entryPrice == null) return null;
  if (cx == null || cy == null) return null;

  const isLong = payload.entrySide === "LONG";

  return (
    <text
      x={cx}
      y={cy}
      dy={-8}
      textAnchor="middle"
      fontSize={16}
      fontWeight="bold"
      fill={isLong ? "#22c55e" : "#ef4444"} // LONG=緑 / SHORT=赤
    >
      ★
    </text>
  );
};

// ---- 15分足取得（クライアント側 fetch） ----
async function fetchCandles15minClient(
  symbol: string,
  days: number = 7,
): Promise<Candle15m[]> {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (!base) {
    console.warn("NEXT_PUBLIC_API_BASE が設定されていません");
    return [];
  }

  const url = new URL(`${base}/public/candles_15min`);
  url.searchParams.set("symbol", symbol);
  url.searchParams.set("days", String(days));

  const res = await fetch(url.toString());
  if (!res.ok) {
    console.warn("candles_15min fetch failed", res.status);
    return [];
  }

  const raw: unknown = await res.json();

  const toNum = (v: unknown): number =>
    typeof v === "number" ? v : v == null ? NaN : Number(v);

  if (!Array.isArray(raw)) {
    return [];
  }

  return (raw as RawCandle[]).map((r) => ({
    time: String(r.time),
    open: toNum(r.open),
    high: toNum(r.high),
    low: toNum(r.low),
    close: toNum(r.close),
    volume:
      r.volume == null || Number.isNaN(Number(r.volume))
        ? null
        : Number(r.volume),
  }));
}

// 含み損益（額）を計算（円換算）
function computeUnrealizedPnlAmount(p: Position): number | null {
  if (p.entry_price == null || p.current_price == null || p.size == null) {
    return null;
  }

  const entry = p.entry_price;
  const current = p.current_price;
  const size = p.size;
  const side = (p.side ?? "").toUpperCase();

  let diff: number;
  if (side === "LONG") {
    diff = current - entry;
  } else if (side === "SHORT") {
    diff = entry - current;
  } else {
    diff = current - entry;
  }
  return diff * size;
}

// ---- メインコンポーネント ----
export default function PositionsTabsClient({ positions }: Props) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [candles, setCandles] = useState<Candle15m[]>([]);
  const [loading, setLoading] = useState(false);

  const active = positions[activeIndex];
  const activeSymbol = active?.symbol ?? null;

  useEffect(() => {
    if (!activeSymbol) return;

    let cancelled = false;
    setLoading(true);

    fetchCandles15minClient(activeSymbol, 7)
      .then((data) => {
        if (cancelled) return;
        setCandles(data);
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeSymbol]);

  // チャート用データ生成（エントリー足を1本だけ特定し、その足に entryPrice を入れる）
  const chartData: ChartPoint[] = useMemo(() => {
    if (candles.length === 0) return [];

    const entryTs = active?.entry_time
      ? new Date(active.entry_time).getTime()
      : null;

    let entryIndex = -1;
    let minDiff = Number.POSITIVE_INFINITY;

    if (entryTs != null) {
      candles.forEach((c, idx) => {
        const candleTime = new Date(c.time).getTime();
        const diff = Math.abs(entryTs - candleTime);
        if (diff < minDiff) {
          minDiff = diff;
          entryIndex = idx;
        }
      });

      // 一番近い足でも15分以上離れていたら「該当なし」
      if (minDiff > 15 * 60 * 1000) {
        entryIndex = -1;
      }
    }

    return candles.map((c, idx) => {
      const isEntry = idx === entryIndex;
      const entryPriceForChart =
        isEntry && active?.entry_price != null ? active.entry_price : null;

      return {
        ...c,
        timeLabel: new Date(c.time).toLocaleString("ja-JP", {
          timeZone: "Asia/Tokyo",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        }),
        isEntry,
        entrySide: active?.side?.toUpperCase(),
        entryPrice: entryPriceForChart,
      };
    });
  }, [candles, active]);

  if (positions.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        保有中のポジションはありません。
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* タブバー */}
      <div className="flex flex-wrap gap-2">
        {positions.map((p, i) => {
          const sym = (p.symbol ?? "").toUpperCase();
          const isActive = i === activeIndex;
          return (
            <button
              key={`${p.position_id}-${i}`}
              type="button"
              onClick={() => setActiveIndex(i)}
              className={[
                "px-3 py-1.5 rounded-full text-xs font-medium border transition",
                "max-w-[140px] truncate",
                isActive
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800",
              ].join(" ")}
            >
              {sym}
            </button>
          );
        })}
      </div>

      {/* 選択中ポジションの情報 + チャート */}
      {active && (
        <div className="grid grid-cols-1 md:grid-cols-10 gap-4 items-start">
          {/* 左：ポジション概要（幅3） */}
          <div className="md:col-span-3 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold truncate">
                {(active.symbol ?? "").toUpperCase()}
              </div>
              <span
                className={[
                  "inline-flex items-center rounded px-2 py-0.5 text-[11px] font-medium",
                  active.side?.toUpperCase() === "LONG"
                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                    : "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
                ].join(" ")}
              >
                {active.side?.toUpperCase()}
              </span>
            </div>

            {(() => {
              const amount = computeUnrealizedPnlAmount(active);
              const pct = active.unrealized_pnl_pct;
              const isPositive =
                typeof amount === "number"
                  ? amount >= 0
                  : typeof pct === "number"
                  ? pct >= 0
                  : null;
              const colorClass =
                isPositive == null
                  ? ""
                  : isPositive
                  ? "text-emerald-600"
                  : "text-rose-600";

              return (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">サイズ</span>
                    <span className="font-semibold">
                      {active.size != null ? active.size : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">エントリー価格</span>
                    <span className="font-semibold">
                      {active.entry_price != null
                        ? active.entry_price.toLocaleString("ja-JP")
                        : "—"}
                    </span>
                  </div>
                  <div className="flex justify_between">
                    <span className="text-gray-500">現在価格</span>
                    <span className="font-semibold">
                      {active.current_price != null
                        ? active.current_price.toLocaleString("ja-JP")
                        : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">含み損益（額）</span>
                    <span className={["font-semibold", colorClass].join(" ")}>
                      {typeof amount === "number"
                        ? `${amount >= 0 ? "+" : ""}${amount.toLocaleString(
                            "ja-JP",
                            { maximumFractionDigits: 0 },
                          )} 円`
                        : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">含み損益（%）</span>
                    <span className={["font-semibold", colorClass].join(" ")}>
                      {typeof pct === "number"
                        ? `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`
                        : "—"}
                    </span>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* 右：15分足チャート（幅7） */}
          <div className="md:col-span-7 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm h-[420px]">
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-semibold text-gray-700 dark:text-gray-200">
                価格推移（15分足, 直近7日）
              </div>
            </div>

            {loading ? (
              <div className="h-full grid place-items-center text-xs text-gray-500">
                読み込み中…
              </div>
            ) : chartData.length === 0 ? (
              <div className="h-full grid.place-items-center text-xs text-gray-500">
                チャート用のローソク足データがありません。
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timeLabel"
                    minTickGap={16}
                    tick={{ fontSize: 10 }}
                  />
                  <YAxis
                    tick={{ fontSize: 10 }}
                    width={60}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: 11,
                    }}
                    formatter={(value: number | string, name: string) => {
                      if (name === "close") {
                        return [
                          Number(value).toLocaleString("ja-JP"),
                          "終値",
                        ];
                      }
                      return [String(value), name];
                    }}
                    labelFormatter={(label: string) => `時刻: ${label}`}
                  />

                  {/* 終値のライン（dotなし） */}
                  <Line
                    type="monotone"
                    dataKey="close"
                    dot={false}
                    strokeWidth={1.5}
                  />

                  {/* エントリー価格に★を描画するための「見えないライン」 */}
                  <Line
                    type="monotone"
                    dataKey="entryPrice"
                    stroke="none"
                    dot={<EntryStarDot />} // entryPrice の位置に★
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
