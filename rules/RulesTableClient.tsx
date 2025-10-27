"use client";

import { useMemo, useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import type { Rule } from "./types";

/* ============== ユーティリティ（JST整形） ============== */
function parseDateFlexible(v: string | number | Date | null | undefined): Date | null {
  if (v == null) return null;
  if (v instanceof Date) return isNaN(v.getTime()) ? null : v;
  if (typeof v === "number") {
    const ms = v > 1e12 ? v : v * 1000;
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

function fmtJST(v: string | number | Date | null | undefined, withTime = true): string {
  const d = parseDateFlexible(v);
  if (!d) return "";
  const optsDate: Intl.DateTimeFormatOptions = {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  };
  const optsTime: Intl.DateTimeFormatOptions = withTime
    ? { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }
    : {};
  const s = new Intl.DateTimeFormat("ja-JP", { ...optsDate, ...optsTime }).format(d);
  return s.replace(/\//g, "-");
}

/* ============== UI ============== */
function Pill({ children, tone = "default" }: { children: React.ReactNode; tone?: "green" | "red" | "default" }) {
  const map: Record<string, string> = {
    green: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    red: "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
    default: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };
  return <span className={`inline-flex items-center rounded px-2 py-0.5 text-[11px] font-medium ${map[tone]}`}>{children}</span>;
}

function RowMenuSkeleton() {
  return (
    <div className="inline-flex items-center gap-1 text-xs text-gray-500">
      <span className="px-2 py-1 rounded border border-gray-200 dark:border-gray-800">編集</span>
      <span className="px-2 py-1 rounded border border-gray-200 dark:border-gray-800">複製</span>
      <span className="px-2 py-1 rounded border border-gray-200 dark:border-gray-800">無効化</span>
    </div>
  );
}

/* ============== 本体 ============== */
export default function RulesTableClient({ initialData }: { initialData: Rule[] }) {
  const sp = useSearchParams();
  const [sortKey, setSortKey] = useState<keyof Rule>("priority");
  const [sortAsc, setSortAsc] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const pageSize = 20;

  const filteredByQ = useMemo(() => {
    const q = (sp.get("q") ?? "").toLowerCase();
    if (!q) return initialData;
    return initialData.filter((r) =>
      [
        r.symbol,
        r.timeframe,
        r.score_col,
        r.op,
        r.v1?.toString() ?? "",
        r.v2?.toString() ?? "",
        r.target_side,
        r.action,
        String(r.priority),
        r.version ?? "",
        r.user_id ?? "",
        r.strategy_id ?? "",
        r.notes ?? "",
      ]
        .join("\t")
        .toLowerCase()
        .includes(q)
    );
  }, [initialData, sp]);

  const sorted = useMemo(() => {
    const arr = [...filteredByQ];
    arr.sort((a, b) => {
      const av = a[sortKey] as unknown;
      const bv = b[sortKey] as unknown;

      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;

      if (typeof av === "number" && typeof bv === "number") return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return arr;
  }, [filteredByQ, sortKey, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  useEffect(() => {
    setPage(1);
  }, [sorted.length]);

  const pageData = sorted.slice((page - 1) * pageSize, page * pageSize);

  const header = (key: keyof Rule, label: string, align: "left" | "right" | "center" = "left") => (
    <th
      className={`px-3 py-2 text-${align} cursor-pointer select-none`}
      onClick={() => {
        if (sortKey === key) setSortAsc(!sortAsc);
        else {
          setSortKey(key);
          setSortAsc(true);
        }
      }}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {sortKey === key && <span className="text-xs text-gray-400">{sortAsc ? "▲" : "▼"}</span>}
      </span>
    </th>
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300">
          <tr>
            {header("id", "ID", "right")}
            {header("symbol", "Symbol")}
            {header("timeframe", "TF")}
            {header("score_col", "Score Col")}
            {header("op", "Op")}
            {header("v1", "v1", "right")}
            {header("v2", "v2", "right")}
            {header("target_side", "Side")}
            {header("action", "Action")}
            {header("priority", "Priority", "right")}
            {header("active", "Active", "center")}
            {header("version", "Version")}
            {header("valid_from", "Valid From")}
            {header("valid_to", "Valid To")}
            {header("user_id", "User")}
            {header("strategy_id", "Strategy")}
            {header("notes", "Notes")}
            <th className="px-3 py-2 text-center">…</th>
          </tr>
        </thead>
        <tbody>
          {pageData.map((r) => (
            <tr key={r.id} className="border-t border-gray-100 dark:border-gray-800">
              <td className="px-3 py-2 text-right tabular-nums">{r.id}</td>
              <td className="px-3 py-2 font-medium">{r.symbol}</td>
              <td className="px-3 py-2">{r.timeframe}</td>
              <td className="px-3 py-2">{r.score_col}</td>
              <td className="px-3 py-2">{r.op}</td>
              <td className="px-3 py-2 text-right">{r.v1 ?? ""}</td>
              <td className="px-3 py-2 text-right">{r.v2 ?? ""}</td>
              <td className="px-3 py-2">
                <Pill tone={r.target_side === "buy" ? "green" : "red"}>{r.target_side}</Pill>
              </td>
              <td className="px-3 py-2">{r.action}</td>
              <td className="px-3 py-2 text-right tabular-nums">{r.priority}</td>
              <td className="px-3 py-2 text-center">{r.active ? <Pill tone="green">true</Pill> : <Pill tone="red">false</Pill>}</td>
              <td className="px-3 py-2">{r.version ?? ""}</td>
              <td className="px-3 py-2 whitespace-nowrap">{fmtJST(r.valid_from)}</td>
              <td className="px-3 py-2 whitespace-nowrap">{r.valid_to ? fmtJST(r.valid_to) : <span className="text-xs text-gray-500">(open)</span>}</td>
              <td className="px-3 py-2">{r.user_id ?? ""}</td>
              <td className="px-3 py-2">{r.strategy_id ?? ""}</td>
              <td className="px-3 py-2 max-w-[220px] truncate" title={r.notes ?? undefined}>
                {r.notes ?? ""}
              </td>
              <td className="px-3 py-2 text-center">
                <RowMenuSkeleton />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* ページネーション */}
      <div className="mt-3 flex items-center justify-between text-sm">
        <div className="text-gray-500">
          {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, sorted.length)} / {sorted.length}
        </div>
        <div className="flex gap-2">
          <button
            className="px-2 py-1 rounded border border-gray-200 dark:border-gray-800 disabled:opacity-50"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            前へ
          </button>
          <div className="px-2 py-1">{page} / {totalPages}</div>
          <button
            className="px-2 py-1 rounded border border-gray-200 dark:border-gray-800 disabled:opacity-50"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
          >
            次へ
          </button>
        </div>
      </div>
    </div>
  );
}
