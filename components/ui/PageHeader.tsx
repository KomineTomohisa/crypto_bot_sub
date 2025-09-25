import React from "react";

type Props = {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode; // 右上のボタン等
  className?: string;
};

export default function PageHeader({ title, description, actions, className }: Props) {
  return (
    <header className={["space-y-2", className].filter(Boolean).join(" ")}>
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">{title}</h1>
          {description ? (
            <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">{description}</p>
          ) : null}
        </div>
        {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
      </div>
    </header>
  );
}