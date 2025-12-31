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
  ReferenceArea,
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

type SrZone = {
  zone_id: string;
  zone_type: string;
  price_center: number;
  price_low: number;
  price_high: number;
  strength: number;
};

type RawCandle = {
  time: string;
  open: number | string;
  high: number | string;
  low: number | string;
  close: number | string;
  volume?: number | string | null;
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

// エントリー価格の位置に★を描画する dot コンポーネント
const EntryStarDot = ({ cx, cy, payload }: EntryStarDotProps) => {
  if (!payload) return null;
  if (!payload.isEntry || payload.entryPrice == null) return null;
  if (cx == null || cy == null) return null;

  const isLong = (payload.entrySide ?? "").toUpperCase() === "LONG";

  return (
    <text
      x={cx}
      y={cy}
      textAnchor="middle"
      fontSize={14}
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
    console.warn("candles_15min fetch failed", res.status, await res.text());
    return [];
  }

  const raw = (await res.json()) as unknown;
  if (!Array.isArray(raw)) {
    console.warn("candles_15min response is not array");
    return [];
  }

  const toNum = (v: number | string): number => {
    const n = typeof v === "number" ? v : Number(v);
    return Number.isNaN(n) ? 0 : n;
  };

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

// ---- SRゾーン取得（クライアント側 fetch） ----
async function fetchSrZonesClient(
  symbol: string,
  timeframe: string = "1hour",
  lookbackDays: number = 30,
): Promise<SrZone[]> {
  const base = process.env.NEXT_PUBLIC_API_BASE;
  if (!base) {
    console.warn("NEXT_PUBLIC_API_BASE が設定されていません");
    return [];
  }

  const url = new URL(`${base}/public/support_resistance_zones`);
  url.searchParams.set("symbol", symbol);
  url.searchParams.set("timeframe", timeframe);
  url.searchParams.set("lookback_days", String(lookbackDays));

  const res = await fetch(url.toString());
  if (!res.ok) {
    console.warn(
      "support_resistance_zones fetch failed",
      res.status,
      await res.text(),
    );
    return [];
  }

  const raw = (await res.json()) as unknown;
  if (!Array.isArray(raw)) return [];

  // any を使わず Record<string, unknown> で扱う
  const rows = raw as Array<Record<string, unknown>>;

  return rows.map((z) => ({
    zone_id: String(z["zone_id"]),
    zone_type: String(z["zone_type"] ?? ""),
    price_center: Number(z["price_center"]),
    price_low: Number(z["price_low"]),
    price_high: Number(z["price_high"]),
    strength:
      z["strength"] != null ? Number(z["strength"] as number | string) : 0,
  }));
}

// 含み損益（額）を計算（円換算）
function computeUnrealizedPnlAmount(p: Position): number | null {
  if (p.entry_price == null || p.current_price == null || p.size == null) {
    return null;
  }
  if (Number.isNaN(p.entry_price) || Number.isNaN(p.current_price)) {
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

type Props = {
  positions: Position[];
};

// ---- メインコンポーネント ----
export default function PositionsTabsClient({ positions }: Props) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [candles, setCandles] = useState<Candle15m[]>([]);
  const [srZones, setSrZones] = useState<SrZone[]>([]);
  const [loading, setLoading] = useState(false);

  const active = positions[activeIndex];
  const activeSymbol = active?.symbol ?? null;

  useEffect(() => {
    if (!activeSymbol) return;

    let cancelled = false;
    setLoading(true);

    Promise.all([
      fetchCandles15minClient(activeSymbol, 7),
      fetchSrZonesClient(activeSymbol, "1hour", 30),
    ])
      .then(([candlesData, srZonesData]) => {
        if (cancelled) return;
        setCandles(candlesData);
        setSrZones(srZonesData);
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
    }

    const sideUpper = (active?.side ?? "").toUpperCase();

    return candles.map((c, idx) => {
      const isEntry = idx === entryIndex;
      const entryPrice =
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
        entrySide: sideUpper,
        entryPrice,
      };
    });
  }, [candles, active?.entry_price, active?.entry_time, active?.side]);

  const pnlInfo = useMemo(() => {
    const amount = active ? computeUnrealizedPnlAmount(active) : null;
    const pct = active?.unrealized_pnl_pct ?? null;

    let colorClass = "text-gray-700";
    if (typeof amount === "number" && typeof pct === "number") {
      if (amount > 0 && pct > 0) colorClass = "text-emerald-500";
      else if (amount < 0 && pct < 0) colorClass = "text-red-500";
    }

    return { amount, pct, colorClass };
  }, [active]);

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
                "px-3 py-1.5 rounded-full border text-xs md:text-sm transition-colors",
                isActive
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50",
              ].join(" ")}
            >
              {sym}
            </button>
          );
        })}
      </div>

      {/* 詳細エリア */}
      {active && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* 左：ポジション情報 */}
          <div className="lg:col-span-1 space-y-3 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <h3 className="text-sm font-semibold text-gray-800 mb-1">
              ポジション詳細
            </h3>
            <div className="text-xs space-y-1.5">
              <div className="flex justify-between">
                <span className="text-gray-500">シンボル</span>
                <span className="font-semibold">
                  {(active.symbol ?? "").toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">サイド</span>
                <span className="font-semibold">
                  {(active.side ?? "").toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">サイズ</span>
                <span className="font-semibold">
                  {active.size != null
                    ? active.size.toLocaleString("ja-JP")
                    : "—"}
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
              <div className="flex justify-between">
                <span className="text-gray-500">現在価格</span>
                <span className="font-semibold">
                  {active.current_price != null
                    ? active.current_price.toLocaleString("ja-JP")
                    : "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">含み損益（額）</span>
                <span className={[pnlInfo.colorClass, "font-semibold"].join(" ")}>
                  {typeof pnlInfo.amount === "number"
                    ? `${pnlInfo.amount >= 0 ? "+" : ""}${pnlInfo.amount.toLocaleString(
                        "ja-JP",
                        { maximumFractionDigits: 0 },
                      )} 円`
                    : "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">含み損益（%）</span>
                <span className={[pnlInfo.colorClass, "font-semibold"].join(" ")}>
                  {typeof pnlInfo.pct === "number"
                    ? `${pnlInfo.pct >= 0 ? "+" : ""}${pnlInfo.pct.toFixed(2)}%`
                    : "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">エントリー日時</span>
                <span className="font-semibold">
                  {active.entry_time
                    ? new Date(active.entry_time).toLocaleString("ja-JP", {
                        timeZone: "Asia/Tokyo",
                      })
                    : "—"}
                </span>
              </div>
            </div>
          </div>

          {/* 右：価格推移チャート */}
          <div className="lg:col-span-2 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-800">
                価格推移（15分足 / 直近7日）
              </h3>
              {loading && (
                <span className="text-xs text-gray-500">読み込み中...</span>
              )}
            </div>

            <div className="h-72">
              {chartData.length === 0 ? (
                <div className="flex h-full items-center justify-center text-xs text-gray-400">
                  データがありません
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
                      domain={[
                        (dataMin: number) => dataMin * 0.995,
                        (dataMax: number) => dataMax * 1.005,
                      ]}
                      tickFormatter={(value: number) =>
                        value % 1 === 0
                          ? value.toLocaleString("ja-JP")              // 整数 → そのまま区切りつき
                          : value.toLocaleString("ja-JP", {            // 小数 → 最大2桁
                              minimumFractionDigits: 0,
                              maximumFractionDigits: 2,
                            })
                      }
                    />
                    <Tooltip
                      contentStyle={{
                        fontSize: 11,
                      }}
                      formatter={(value, name) => {
                        if (name === "close") {
                          const v = typeof value === "number" ? value : Number(value);
                          if (value == null || Number.isNaN(v)) return ["-", "終値"];
                          return [v.toLocaleString("ja-JP"), "終値"];
                        }

                        if (value == null) return ["-", String(name)];
                        return [String(value), String(name)];
                      }}
                      labelFormatter={(label: string) => `時刻: ${label}`}
                    />

                    {/* SRゾーン帯を表示 */}
                    {srZones.map((z) => {
                      const isSupport =
                        (z.zone_type ?? "").toLowerCase() === "support";
                      return (
                        <ReferenceArea
                          key={z.zone_id}
                          y1={z.price_low}
                          y2={z.price_high}
                          stroke={isSupport ? "#22c55e" : "#ef4444"}
                          strokeOpacity={0.8}
                          fill={isSupport ? "#22c55e" : "#ef4444"}
                          fillOpacity={0.06}
                        />
                      );
                    })}

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
        </div>
      )}
    </div>
  );
}
