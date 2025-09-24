import React from "react";

type TableRowsProps = { rows?: number; cols?: number; className?: string };
type CardsProps = { count?: number; className?: string };

export function TableRowsSkeleton({ rows = 3, cols = 4, className }: TableRowsProps) {
  return (
    <tbody className={className}>
      {Array.from({ length: rows }).map((_, r) => (
        <tr key={r} className="animate-pulse">
          {Array.from({ length: cols }).map((__, c) => (
            <td key={c} className="py-2">
              <div className="h-4 w-24 bg-gray-200 dark:bg-gray-700 rounded" />
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  );
}

export function CardsSkeleton({ count = 3, className }: CardsProps) {
  return (
    <div className={["grid gap-3", className].filter(Boolean).join(" ")}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="animate-pulse rounded-xl border border-gray-200 dark:border-gray-800 p-4">
          <div className="h-5 w-1/3 bg-gray-200 dark:bg-gray-700 rounded" />
          <div className="h-4 w-1/2 bg-gray-200 dark:bg-gray-700 rounded mt-2" />
          <div className="h-4 w-1/4 bg-gray-200 dark:bg-gray-700 rounded mt-2" />
        </div>
      ))}
    </div>
  );
}

export default function LoadingSkeleton() {
  return (
    <div className="animate-pulse rounded-2xl border border-gray-200 dark:border-gray-800 p-4">
      <div className="h-6 w-1/3 bg-gray-200 dark:bg-gray-700 rounded" />
      <div className="h-4 w-1/2 bg-gray-200 dark:bg-gray-700 rounded mt-2" />
    </div>
  );
}
