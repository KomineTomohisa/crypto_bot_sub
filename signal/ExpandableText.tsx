"use client";
import { useState } from "react";

export default function ExpandableText({
  text,
  maxChars = 140,
  className = "",
}: { text?: string; maxChars?: number; className?: string }) {
  const full = text ?? "";
  const isLong = full.length > maxChars;
  const [expanded, setExpanded] = useState(false);
  const shown = expanded || !isLong ? full : full.slice(0, maxChars) + "…";
  return (
    <div className={className}>
      <div className="whitespace-pre-wrap">{shown || "—"}</div>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(v => !v)}
          className="mt-1 text-xs underline text-blue-600 hover:opacity-80"
          aria-expanded={expanded}
        >
          {expanded ? "閉じる" : "もっと見る"}
        </button>
      )}
    </div>
  );
}
