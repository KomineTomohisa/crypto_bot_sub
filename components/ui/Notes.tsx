// components/ui/Notes.tsx
import React from "react";

/**
 * 共通注記（Notes）コンポーネント
 * - 右肩などに並べて使う想定
 * - 各ノートは一貫した外観（小さめのバッジ＋infoアイコン）
 * - tooltip はネイティブ title を使用（追加ライブラリ不要）
 */
export function Notes({ items }: { items: { label: React.ReactNode; tooltip?: string }[] }) {
  if (!items?.length) return null;
  return (
    <div className="flex flex-wrap gap-2 items-start" aria-label="補足注記">
      {items.map((it, i) => (
        <span
          key={i}
          title={it.tooltip}
          className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
        >
          <InfoIcon className="w-3.5 h-3.5" />
          <span className="leading-none">{it.label}</span>
        </span>
      ))}
    </div>
  );
}

export function InfoNote({ label, tooltip }: { label: React.ReactNode; tooltip?: string }) {
  return (
    <span
      title={tooltip}
      className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
    >
      <InfoIcon className="w-3.5 h-3.5" />
      <span className="leading-none">{label}</span>
    </span>
  );
}

function InfoIcon({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm.75 15.25a.75.75 0 0 1-1.5 0v-6.5a.75.75 0 0 1 1.5 0v6.5ZM12 8.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z" />
    </svg>
  );
}