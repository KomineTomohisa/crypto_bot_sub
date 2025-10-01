// components/ui/QuickFilters.tsx
import React from "react";
import Link from "next/link";

/**
 * QuickFilters
 * - mode="range"  : days を [7,30,90] などで切替（/performance, /transparency）
 * - mode="since"  : since(ISO) を [7,30,90] 日前で切替（/signals）
 * - basePath      : 遷移先のパス（例："/performance"）
 * - symbol        : 任意で ?symbol= を付加
 * - activeDays    : 現在選択中の日数（range時のアクティブ表示に使用）
 * - daysOptions   : ボタン候補（既定 [7,30,90]）
 */
export function QuickFilters({
  mode,
  basePath,
  symbol,
  activeDays,
  daysOptions = [7, 30, 90],
}: {
  mode: "range" | "since";
  basePath: string;
  symbol?: string;
  activeDays?: number;
  daysOptions?: number[];
}) {
  const baseBtn =
    "px-3 py-1.5 rounded-xl border text-sm hover:bg-gray-50 dark:hover:bg-gray-800";
  const activeCls =
    "bg-gray-900 text-white border-gray-900 dark:bg-white dark:text-gray-900 dark:border-white";

  const mkHref = (d: number) => {
    const p = new URLSearchParams();
    if (mode === "range") {
      p.set("days", String(d));
    } else {
      const since = new Date(Date.now() - d * 24 * 60 * 60 * 1000).toISOString();
      p.set("since", since);
    }
    if (symbol) p.set("symbol", symbol);
    return `${basePath}?${p.toString()}`;
  };

  return (
    <div className="flex flex-wrap gap-2" role="group" aria-label="クイック切替">
      {daysOptions.map((d) => (
        <Link
          key={d}
          href={mkHref(d)}
          className={`${baseBtn} ${mode === "range" && activeDays === d ? activeCls : "border-gray-300 dark:border-gray-700"}`}
          aria-current={mode === "range" && activeDays === d ? "page" : undefined}
        >
          {mode === "range" ? `${d}日` : `過去${d}日`}
        </Link>
      ))}
    </div>
  );
}

