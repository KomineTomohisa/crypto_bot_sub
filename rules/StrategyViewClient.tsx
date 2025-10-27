"use client";

import { useMemo, useState } from "react";
import type { Rule } from "./types";

/* =========================
   型・ユーティリティ
   ========================= */

type Side = "buy" | "sell";
type Segment = {
  start: number; // 0..1
  end: number;   // 0..1 (start <= end)
  disabled: boolean; // action === "disable" なら true
  raw: Rule;
};

type Marker = {
  pos: number;   // 0..1
  disabled: boolean;
  label: string;
  raw: Rule;
};

type IndicatorBucket = {
  score_col: string;
  // timeframeごとにバーを並べる
  byTimeframe: Record<string, { segments: Segment[]; markers: Marker[]; extraNotes: string[] }>;
  totals: { active: number; openEnded: number; count: number };
};

type SideGroup = {
  side: Side;
  indicators: IndicatorBucket[];
};

type SymbolGroup = {
  symbol: string;
  sides: SideGroup[];
};

const INDICATOR_ORDER = [
  "bb_score_short",
  "bb_score_long",
  "rsi_score_short",
  "rsi_score_long",
  "adx_score_short",
  "adx_score_long",
  "atr_score_short",
  "atr_score_long",
  "ma_score_short",
  "ma_score_long",
];

function clamp01(x: number): number {
  if (Number.isNaN(x)) return x;
  return Math.max(0, Math.min(1, x));
}
function toNum(v: string | number | null): number | null {
  if (v == null) return null;
  if (typeof v === "number") return v;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function buildSegmentsAndMarkers(rule: Rule): { segs: Segment[]; marks: Marker[]; notes: string[] } {
  const disabled = rule.action === "disable";
  const segs: Segment[] = [];
  const marks: Marker[] = [];
  const notes: string[] = [];

  const v1n = toNum(rule.v1);
  const v2n = toNum(rule.v2);

  switch (rule.op) {
    case "between": {
      if (v1n == null || v2n == null) { notes.push("between値不足"); break; }
      const a = clamp01(Math.min(v1n, v2n));
      const b = clamp01(Math.max(v1n, v2n));
      if (!Number.isFinite(a) || !Number.isFinite(b)) { notes.push("between数値不正"); break; }
      segs.push({ start: a, end: b, disabled, raw: rule });
      break;
    }
    case "<":
    case "<=": {
      if (v1n == null) { notes.push(`${rule.op} 値不足`); break; }
      const x = clamp01(v1n);
      segs.push({ start: 0, end: x, disabled, raw: rule });
      break;
    }
    case ">":
    case ">=": {
      if (v1n == null) { notes.push(`${rule.op} 値不足`); break; }
      const x = clamp01(v1n);
      segs.push({ start: x, end: 1, disabled, raw: rule });
      break;
    }
    case "==":
    case "!=": {
      if (v1n == null) { notes.push(`${rule.op} 値不足`); break; }
      const x = clamp01(v1n);
      marks.push({ pos: x, disabled, label: rule.op === "==" ? "=" : "≠", raw: rule });
      break;
    }
    case "is_null": {
      notes.push("値なし条件");
      break;
    }
    case "is_not_null": {
      notes.push("値あり条件");
      break;
    }
    default: {
      notes.push(`未対応op: ${rule.op}`);
      break;
    }
  }

  return { segs, marks, notes };
}

function sideColor(side: Side): string {
  // 無効以外のときの色。buy=緑, sell=赤
  return side === "buy" ? "bg-emerald-500" : "bg-rose-500";
}

function pillToneForSide(side: Side): "green" | "red" {
  return side === "buy" ? "green" : "red";
}

/* =========================
   UIパーツ
   ========================= */

function Pill({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "green" | "red" | "blue" | "default";
}) {
  const map: Record<string, string> = {
    green:
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    red: "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
    blue: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
    default:
      "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };
  return (
    <span
      className={`inline-flex items-center rounded px-2 py-0.5 text-[11px] font-medium ${map[tone]}`}
    >
      {children}
    </span>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-6 rounded bg-emerald-500" />
        <span>buy 有効領域</span>
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-6 rounded bg-rose-500" />
        <span>sell 有効領域</span>
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-6 rounded bg-gray-400" />
        <span>無効（disable）</span>
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-0.5 rounded bg-gray-700" />
        <span>一致/不一致マーカー（= / ≠）</span>
      </span>
      <span className="inline-flex items-center gap-1">
        <span className="inline-block h-2 w-10 rounded bg-gray-200 border border-gray-300" />
        <span>0–1 バー（下に 0 / 1 目盛）</span>
      </span>
    </div>
  );
}

/* =========================
   本体
   ========================= */

export default function StrategyViewClient({ data }: { data: Rule[] }) {
  // symbol > side(buy/sell) > indicator(score_col) の3階層へ整形
  const groups = useMemo<SymbolGroup[]>(() => {
    // symbolごと
    const bySymbol = new Map<string, Rule[]>();
    for (const r of data) {
      const k = r.symbol ?? "(unknown)";
      if (!bySymbol.has(k)) bySymbol.set(k, []);
      bySymbol.get(k)!.push(r);
    }

    const result: SymbolGroup[] = [];
    for (const [symbol, rules] of bySymbol.entries()) {
      // sideごと
      const sides: Side[] = ["buy", "sell"];
      const sideGroups: SideGroup[] = [];
      for (const side of sides) {
        const sideRules = rules.filter(
          (r) => (r.target_side ?? "").toLowerCase() === side
        );

        // indicatorごと
        const byIndicator = new Map<string, Rule[]>();
        for (const r of sideRules) {
          const k = r.score_col ?? "(unknown)";
          if (!byIndicator.has(k)) byIndicator.set(k, []);
          byIndicator.get(k)!.push(r);
        }

        const indicators: IndicatorBucket[] = [];
        for (const [score_col, indRules] of byIndicator.entries()) {
          // timeframeごとのバーに分ける
          const byTF = new Map<string, { segments: Segment[]; markers: Marker[]; extraNotes: string[] }>();
          for (const r of indRules) {
            const tf = r.timeframe ?? "";
            if (!byTF.has(tf)) byTF.set(tf, { segments: [], markers: [], extraNotes: [] });
            const built = buildSegmentsAndMarkers(r);
            byTF.get(tf)!.segments.push(...built.segs);
            byTF.get(tf)!.markers.push(...built.marks);
            byTF.get(tf)!.extraNotes.push(...built.notes);
          }

          // totals
          const totals = {
            active: indRules.filter((r) => r.active).length,
            openEnded: indRules.filter((r) => r.valid_to == null).length,
            count: indRules.length,
          };

          // TFキー順：15m, 1h, 4h などを自然順に（数字 → 文字）
          const entries = Array.from(byTF.entries()).sort((a, b) =>
            a[0].localeCompare(b[0], "en", { numeric: true })
          );
          const byTimeframe: IndicatorBucket["byTimeframe"] = {};
          for (const [tf, v] of entries) byTimeframe[tf] = v;

          indicators.push({ score_col, byTimeframe, totals });
        }

        // インジケータの表示順を定義に従ってソート
        indicators.sort((a, b) => {
          const ai = INDICATOR_ORDER.indexOf(a.score_col);
          const bi = INDICATOR_ORDER.indexOf(b.score_col);
          if (ai !== -1 && bi !== -1) return ai - bi;
          if (ai !== -1) return -1;
          if (bi !== -1) return 1;
          return a.score_col.localeCompare(b.score_col);
        });

        sideGroups.push({ side, indicators });
      }

      // 通貨ごと：ルール件数が多いほうを先に表示（視認性）
      sideGroups.sort((a, b) => {
        const ca = a.indicators.reduce((acc, ib) => acc + ib.totals.count, 0);
        const cb = b.indicators.reduce((acc, ib) => acc + ib.totals.count, 0);
        return cb - ca;
      });

      result.push({ symbol, sides: sideGroups });
    }

    // 通貨自体も件数の多い順
    result.sort((a, b) => {
      const ca = a.sides.reduce(
        (acc, sg) => acc + sg.indicators.reduce((acc2, ib) => acc2 + ib.totals.count, 0),
        0
      );
      const cb = b.sides.reduce(
        (acc, sg) => acc + sg.indicators.reduce((acc2, ib) => acc2 + ib.totals.count, 0),
        0
      );
      return cb - ca;
    });

    return result;
  }, [data]);

  /* 折りたたみ状態（通貨ごと） */
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const toggleSymbol = (symbol: string) =>
    setCollapsed((prev) => ({ ...prev, [symbol]: !prev[symbol] }));

  return (
    <div className="space-y-6">
      {/* 凡例 */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-800 p-3 bg-white dark:bg-gray-900">
        <Legend />
      </div>

      {groups.map((g) => {
        const isCollapsed = !!collapsed[g.symbol];
        return (
          <div
            key={g.symbol}
            className="rounded-2xl border border-gray-200 dark:border-gray-800 p-4 bg-white dark:bg-gray-900"
          >
            {/* 親：通貨（ヘッダ + 折りたたみトグル） */}
            <div className="flex items-center justify-between">
              <div className="text-lg font-semibold">{g.symbol}</div>
              {/* ざっくりKPI */}
              <div className="flex items-center gap-2 text-xs">
                <Pill>
                  総{" "}
                  {g.sides.reduce(
                    (acc, sg) =>
                      acc + sg.indicators.reduce((a, ib) => a + ib.totals.count, 0),
                    0
                  )}
                </Pill>
                <Pill tone="green">
                  Active{" "}
                  {g.sides.reduce(
                    (acc, sg) => acc + sg.indicators.reduce((a, ib) => a + ib.totals.active, 0),
                    0
                  )}
                </Pill>
                <Pill tone="blue">
                  OpenEnd{" "}
                  {g.sides.reduce(
                    (acc, sg) =>
                      acc + sg.indicators.reduce((a, ib) => a + ib.totals.openEnded, 0),
                    0
                  )}
                </Pill>
                <button
                  onClick={() => toggleSymbol(g.symbol)}
                  className="ml-2 inline-flex items-center gap-1 rounded-lg border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs hover:bg-gray-50 dark:hover:bg-gray-800"
                  aria-expanded={!isCollapsed}
                  aria-controls={`symbol-content-${g.symbol}`}
                  title={isCollapsed ? "展開" : "折りたたみ"}
                >
                  {isCollapsed ? "展開" : "折りたたみ"}
                </button>
              </div>
            </div>

            {/* 折りたたみ対象 */}
            {!isCollapsed && (
              <div id={`symbol-content-${g.symbol}`} className="mt-3 space-y-4">
                {/* 子：サイド（buy/sell） */}
                {g.sides.map((sg) => (
                  <div
                    key={`${g.symbol}__${sg.side}`}
                    className="rounded-xl border border-gray-100 dark:border-gray-800 p-3 bg-gray-50 dark:bg-gray-950"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-sm font-medium flex items-center gap-2">
                        <Pill tone={pillToneForSide(sg.side)}>{sg.side}</Pill>
                        <span className="text-xs text-gray-500">
                          ルール {sg.indicators.reduce((a, ib) => a + ib.totals.count, 0)}
                        </span>
                      </div>
                    </div>

                    {/* 指標ボックス群（縦並び） */}
                    <div className="space-y-2">
                      {sg.indicators.map((ib) => (
                        <div
                          key={`${g.symbol}__${sg.side}__${ib.score_col}`}
                          className="rounded-lg border border-gray-200 dark:border-gray-800 p-3 bg-white dark:bg-gray-900"
                        >
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-medium">{ib.score_col}</div>
                            <div className="flex items-center gap-2 text-xs">
                              <Pill>TF {Object.keys(ib.byTimeframe).length}</Pill>
                              <Pill tone="green">Active {ib.totals.active}</Pill>
                              <Pill tone="blue">OpenEnd {ib.totals.openEnded}</Pill>
                            </div>
                          </div>

                          {/* timeframeごとのバー */}
                          <div className="mt-2 space-y-3">
                            {Object.entries(ib.byTimeframe).map(([tf, v]) => (
                              <div key={`${ib.score_col}__${tf}`}>
                                <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                                  <span>
                                    TF: <b>{tf}</b>
                                  </span>
                                  <span>
                                    seg {v.segments.length} / mark {v.markers.length}
                                  </span>
                                </div>

                                {/* 0..1 バー */}
                                <div className="relative w-full h-3 rounded bg-gray-200 dark:bg-gray-800 overflow-hidden">
                                  {/* Segment: 有効領域（side色） or 無効（グレー） */}
                                  {v.segments.map((s, idx) => {
                                    const left = `${s.start * 100}%`;
                                    const width = `${Math.max(0, s.end - s.start) * 100}%`;
                                    const color = s.disabled ? "bg-gray-400" : sideColor(sg.side);
                                    const title = `${ib.score_col} ${s.raw.op} ${s.raw.v1 ?? ""}${
                                      s.raw.op === "between" ? ` ~ ${s.raw.v2 ?? ""}` : ""
                                    } ${s.disabled ? "(無効)" : ""}`;
                                    return (
                                      <div
                                        key={idx}
                                        className={`absolute top-0 h-3 ${color}`}
                                        style={{ left, width }}
                                        title={title}
                                      />
                                    );
                                  })}

                                  {/* Marker: = / ≠ などの位置 */}
                                  {v.markers.map((m, idx) => {
                                    const left = `${m.pos * 100}%`;
                                    const color = m.disabled ? "bg-gray-600" : "bg-gray-900";
                                    const title = `${ib.score_col} ${m.label} ${m.raw.v1 ?? ""} ${m.disabled ? "(無効)" : ""}`;
                                    return (
                                      <div
                                        key={`mk-${idx}`}
                                        className={`absolute top-0 h-3 w-0.5 ${color}`}
                                        style={{ left }}
                                        title={title}
                                      />
                                    );
                                  })}
                                </div>

                                {/* ▼ 0 と 1 の目盛（はっきり表示） */}
                                <div className="mt-1 text-[11px] text-gray-600 dark:text-gray-300 flex items-center justify-between">
                                  <span className="font-semibold">0</span>
                                  <span className="font-semibold">1</span>
                                </div>

                                {/* 追加メモ（未対応op/NULL系） */}
                                {v.extraNotes.length > 0 && (
                                  <div className="mt-1 text-[11px] text-gray-500">
                                    {Array.from(new Set(v.extraNotes)).join(" / ")}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}

                      {sg.indicators.length === 0 && (
                        <div className="rounded-lg border border-dashed border-gray-300 dark:border-gray-700 p-6 text-center text-sm text-gray-500">
                          {sg.side} 側の指標はありません。
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {groups.length === 0 && (
        <div className="rounded-xl border border-dashed border-gray-300 dark:border-gray-700 p-8 text-center text-sm text-gray-500">
          条件に一致するルールがありません。上のフィルタを変更して再検索してください。
        </div>
      )}
    </div>
  );
}
