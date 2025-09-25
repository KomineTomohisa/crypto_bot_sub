import React from "react";

/**
 * レイアウトだけを担う薄いコンポーネント。
 * 左に主要フィルタ、右にサブ情報やアクションを配置する。
 * 実際の <form> 要素や <select> は親側で自由に構成してください。
 */
type Props = {
  left?: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
  ariaLabel?: string;
};

export default function FilterBar({ left, right, className, ariaLabel = "フィルタ" }: Props) {
  return (
    <div
      aria-label={ariaLabel}
      className={[
        "rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 shadow-sm space-y-3",
        className,
      ].filter(Boolean).join(" ")}
    >
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2 flex-wrap">{left}</div>
        <div className="text-xs text-gray-500">{right}</div>
      </div>
    </div>
  );
}
