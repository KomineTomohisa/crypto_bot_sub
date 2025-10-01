"use client";
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export default function SymbolSelect({ days }: { days: number }) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const router = useRouter();
  const sp = useSearchParams();
  const currentSymbol = sp.get("symbol") || "";

  useEffect(() => {
    const url = `/api/public/symbols?days=${days}`;
    fetch(url, { cache: "no-store" })
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(setSymbols)
      .catch(() => setSymbols([]));
  }, [days]);

  return (
    <div className="space-y-1">
      <label className="block text-sm text-gray-600">Symbol（任意）</label>
      <select
        className="w-full border rounded-xl px-3 py-2"
        value={currentSymbol}
        onChange={(e) => {
          const sym = e.target.value;
          const d = sp.get("days") || String(days);
          const q = new URLSearchParams({ days: d });
          if (sym) q.set("symbol", sym);
          router.push(`/transparency?${q.toString()}`);
        }}
      >
        <option value="">（全体）</option>
        {symbols.map(s => <option key={s} value={s}>{s}</option>)}
      </select>
    </div>
  );
}
