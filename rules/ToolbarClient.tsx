// app/rules/ToolbarClient.tsx
"use client";

import React, { useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";

/* 既存の Field/Input/Select/Button/Toggle 定義は流用 */

// ✅ Props を明示
export type ToolbarClientProps = {
  initialCount: number;
};

// ✅ デフォルトエクスポートはこの関数のみ（searchParams の型は使わない）
export default function ToolbarClient({ initialCount }: ToolbarClientProps) {
  const router = useRouter();
  const sp = useSearchParams();

  const [symbol, setSymbol] = useState<string>(sp.get("symbol") ?? "");
  const [timeframe, setTimeframe] = useState<string>(sp.get("timeframe") ?? "");
  const [active, setActive] = useState<string>(sp.get("active") ?? "");
  const [onlyOpenEnded, setOnlyOpenEnded] = useState<boolean>(
    sp.get("only_open_ended") === "true"
  );
  const [userId, setUserId] = useState<string>(sp.get("user_id") ?? "");
  const [strategyId, setStrategyId] = useState<string>(sp.get("strategy_id") ?? "");
  const [version, setVersion] = useState<string>(sp.get("version") ?? "");
  const [q, setQ] = useState<string>(sp.get("q") ?? "");

  const apply = () => {
    const usp = new URLSearchParams();
    if (symbol) usp.set("symbol", symbol);
    if (timeframe) usp.set("timeframe", timeframe);
    if (active) usp.set("active", active);
    if (onlyOpenEnded) usp.set("only_open_ended", "true");
    if (userId) usp.set("user_id", userId);
    if (strategyId) usp.set("strategy_id", strategyId);
    if (version) usp.set("version", version);
    if (q) usp.set("q", q);
    router.push(`/rules?${usp.toString()}`);
  };

  const clear = () => {
    setSymbol("");
    setTimeframe("");
    setActive("");
    setOnlyOpenEnded(false);
    setUserId("");
    setStrategyId("");
    setVersion("");
    setQ("");
    router.push(`/rules`);
  };

  const exportCsv = () => {
    const current = new URLSearchParams();
    if (symbol) current.set("symbol", symbol);
    if (timeframe) current.set("timeframe", timeframe);
    if (active) current.set("active", active);
    if (onlyOpenEnded) current.set("only_open_ended", "true");
    if (userId) current.set("user_id", userId);
    if (strategyId) current.set("strategy_id", strategyId);
    if (version) current.set("version", version);
    if (q) current.set("q", q);
    window.open(`/rules.csv?${current.toString()}`, "_blank");
  };

  return (
    <div className="space-y-4">
      {/* 以降は既存の JSX そのまま */}
      {/* ... */}
      <div className="text-xs text-gray-500">
        現在のヒット件数（サーバー反映後の件数）: <b>{initialCount}</b>
      </div>
    </div>
  );
}
