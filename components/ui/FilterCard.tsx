"use client";

import { ReactNode, useEffect, useRef } from "react";

type Props = {
  title?: ReactNode;
  defaultOpen?: boolean;      // 初期展開（モバイルは閉じ推奨）
  children: ReactNode;
  className?: string;
};

export function FilterCard({
  title = "Filters",
  defaultOpen = false,
  children,
  className = "",
}: Props) {
  const ref = useRef<HTMLDetailsElement>(null);

  // キーボード操作・アクセシビリティ向上（Space/Enterでトグル）
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        el.open = !el.open;
      }
    };

    // ✅ 型を正しく指定することで any を使わずに済む
    el.addEventListener("keydown", onKeyDown as EventListener);

    return () => {
      el.removeEventListener("keydown", onKeyDown as EventListener);
    };
  }, []);

  return (
    <details
      ref={ref}
      className={`group rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm ${className}`}
      {...(defaultOpen ? { open: true } : {})}
    >
      <summary
        className="flex items-center justify-between cursor-pointer select-none list-none px-4 py-3 rounded-2xl
                   focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        aria-label="Toggle filters"
      >
        <div className="font-medium">{title}</div>
        <svg
          className="h-5 w-5 transition-transform duration-200 group-open:rotate-180"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M10 12a1 1 0 0 1-.707-.293l-4-4a1 1 0 1 1 1.414-1.414L10 9.586l3.293-3.293A1 1 0 0 1 14.707 8l-4 4A1 1 0 0 1 10 12z"/>
        </svg>
      </summary>
      <div className="px-4 pb-4 pt-1">{children}</div>
    </details>
  );
}
