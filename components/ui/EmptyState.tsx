import React from "react";
import Link from "next/link";

type Props = {
  title?: string;
  message?: React.ReactNode;
  action?: { href: string; label: string } | { onClick: () => void; label: string };
  className?: string;
};

export default function EmptyState({ title = "データがありません", message, action, className }: Props) {
  const isLink = action && "href" in action;
  return (
    <div className={["text-sm text-gray-500 rounded-xl border border-dashed border-gray-300 dark:border-gray-700 p-6 grid place-items-center", className].filter(Boolean).join(" ")}>
      <div className="space-y-2 text-center">
        <div className="font-medium">{title}</div>
        {message ? <div>{message}</div> : null}
        {action ? (
          isLink ? (
            <Link href={action.href} className="inline-block mt-2 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700">
              {action.label}
            </Link>
          ) : (
            <button onClick={action.onClick} className="inline-block mt-2 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700">
              {action.label}
            </button>
          )
        ) : null}
      </div>
    </div>
  );
}
