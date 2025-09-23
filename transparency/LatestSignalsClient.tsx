"use client";

import React, { useEffect, useMemo, useState } from "react";

type SignalRecord = {
  signal_id?: number | string;
  symbol: string;
  side: string;           // BUY / SELL / EXIT-LONG 等
  reason?: string;        // 任意：根拠など
  price?: number;         // 任意
  created_at: string;     // ISO
};

export default function LatestSignalsClient({
  initial,
  days,
  symbol,
}: {
  initial: SignalRecord[];
  days: number;
  symbol?: string;
}) {
  const [items, setItems] = useState<SignalRecord[]>(initial);
  const [loading, setLoading] = useState(false);

  // since を計算（現在 - days日）
  const sinceIso = useMemo(() => {
    const t = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    return t.toISOString();
  }, [days]);

  useEffect(() => {
    let aborted = false;
    (async () => {
      setLoading(true);
      try {
        const qs = new URLSearchParams({ limit: "10", since: sinceIso });
        if (symbol) qs.set("symbol", symbol);
        const res = await fetch(`/api/public/signals?${qs.toString()}`, { cache: "no-store" });
        if (!res.ok) throw new Error(String(res.status));
        const data: SignalRecord[] = await res.json();
        if (!aborted && Array.isArray(data)) {
          setItems(data);
        }
      } catch {
        // 失敗時は initial を維持
      } finally {
        if (!aborted) setLoading(false);
      }
    })();
    return () => { aborted = true; };
  }, [sinceIso, symbol]);

  if (!items?.length) {
    return (
      <div className="text-sm text-gray-500">
        直近 {days} 日でシグナルが見つかりません。
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {items.map((s, i) => (
        <article
          key={String(s.signal_id ?? `${s.symbol}-${s.created_at}-${i}`)}
          className="rounded-xl border p-3"
        >
          <div className="flex items-baseline justify-between">
            <div className="font-medium">{s.symbol} <span className="text-xs text-gray-500">{s.side}</span></div>
            <time className="text-xs text-gray-500">
              {new Date(s.created_at).toLocaleString()}
            </time>
          </div>
          <div className="text-sm text-gray-700 mt-1">
            {s.reason ? s.reason : "—"}
          </div>
          {s.price != null && (
            <div className="text-xs text-gray-500 mt-1">price: {s.price}</div>
          )}
        </article>
      ))}
      {loading && <div className="text-xs text-gray-400">更新中…</div>}
    </div>
  );
}
