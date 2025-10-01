// ------------------------------------------------------------
// components/ui/TablePagination.tsx
import React from "react";
import Link from "next/link";

export function TablePagination({
  prevHref,
  nextHref,
  canPrev,
  canNext,
}: {
  prevHref: string;
  nextHref: string;
  canPrev: boolean;
  canNext: boolean;
}) {
  const base = "rounded-xl px-3 py-2 border";
  return (
    <div className="flex items-center justify-between mt-3">
      <div className="text-sm text-gray-600 dark:text-gray-300">Page Navigation</div>
      <div className="space-x-2">
        <Link
          href={canPrev ? prevHref : "#"}
          className={`${base} ${
            canPrev
              ? "border-gray-300 dark:border-gray-700"
              : "pointer-events-none opacity-40 border-gray-200 dark:border-gray-800"
          }`}
          aria-disabled={!canPrev}
        >
          ← Prev
        </Link>
        <Link
          href={canNext ? nextHref : "#"}
          className={`${base} ${
            canNext
              ? "border-gray-300 dark:border-gray-700"
              : "pointer-events-none opacity-40 border-gray-200 dark:border-gray-800"
          }`}
          aria-disabled={!canNext}
        >
          Next →
        </Link>
      </div>
    </div>
  );
}

