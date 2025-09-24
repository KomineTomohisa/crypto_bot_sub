import React from "react";

type Props = {
  label: React.ReactNode;
  value: React.ReactNode;
  subtext?: React.ReactNode;
  className?: string;
};

export default function KpiCard({ label, value, subtext, className }: Props) {
  return (
    <div className={["flex-1 rounded-xl border border-gray-200 dark:border-gray-800 p-4", className].filter(Boolean).join(" ")}>
      <div className="text-sm text-gray-500 dark:text-gray-400">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {subtext ? <div className="text-xs text-gray-500 mt-1">{subtext}</div> : null}
    </div>
  );
}
