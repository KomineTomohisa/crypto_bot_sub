import React from "react";

type Props = {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  id?: string; // aria-labelledby 用
  children: React.ReactNode;
  className?: string;
  headerRight?: React.ReactNode; // セクション右上に小さな補助UIを出したいとき
};

export default function Section({ title, subtitle, id, children, className, headerRight }: Props) {
  const headingId = id ?? (typeof title === "string" ? title : undefined);
  return (
    <section
      aria-labelledby={headingId ? `${headingId}-heading` : undefined}
      className={[
        "rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 md:p-6 shadow-sm",
        className,
      ].filter(Boolean).join(" ")}
    >
      {(title || subtitle || headerRight) && (
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title ? (
              <h2 id={headingId ? `${headingId}-heading` : undefined} className="text-lg md:text-xl font-semibold">
                {title}
              </h2>
            ) : null}
            {subtitle ? <p className="text-sm text-gray-500 mt-1">{subtitle}</p> : null}
          </div>
          {headerRight ? <div className="shrink-0">{headerRight}</div> : null}
        </div>
      )}
      {children}
    </section>
  );
}
