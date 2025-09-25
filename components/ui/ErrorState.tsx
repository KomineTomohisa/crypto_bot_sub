import React from "react";
import Link from "next/link";

type Props = {
  error?: unknown;
  summary?: string;
  retry?: { onClick?: () => void; href?: string; label?: string };
  className?: string;
};

export default function ErrorState({ error, summary = "読み込みに失敗しました。", retry, className }: Props) {
  const detail = error instanceof Error ? error.message : (typeof error === "string" ? error : undefined);
  const isLink = retry?.href;
  return (
    <div className={["rounded-2xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 p-4 text-sm", className].filter(Boolean).join(" ")}>
      <div className="text-red-700 dark:text-red-300 font-medium">{summary}</div>
      {detail ? <div className="text-red-600 dark:text-red-400 mt-1 break-all">{detail}</div> : null}
      {retry?.label ? (
        isLink ? (
          <Link href={retry.href!} className="inline-block mt-2 px-3 py-1.5 rounded-lg border border-red-300 dark:border-red-800">
            {retry.label}
          </Link>
        ) : (
          <button onClick={retry.onClick} className="inline-block mt-2 px-3 py-1.5 rounded-lg border border-red-300 dark:border-red-800">
            {retry.label}
          </button>
        )
      ) : null}
    </div>
  );
}
