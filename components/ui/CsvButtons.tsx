import React from "react";
import Link from "next/link";

type LinkItem = {
  href: string;
  label: string;
  download?: boolean;
  target?: "_blank";
  rel?: string;
};

type Props = {
  links: LinkItem[];
  note?: React.ReactNode; // 下部の注意書きなど
  className?: string;
};

/**
 * CSVなどダウンロード系のボタン群を統一する。
 * 内部遷移でない場合（外部CSV等）は <a> ではなく Link + props を利用し、ESLintエラーを避ける。
 * (Next.js は Link に外部URLも渡せる)
 */
export default function CsvButtons({ links, note, className }: Props) {
  return (
    <div className={["space-y-2", className].filter(Boolean).join(" ")}>
      <div className="flex flex-wrap gap-3">
        {links.map((l, i) => (
          <Link
            key={i}
            href={l.href}
            className="px-4 py-2 border rounded-xl border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
            download={l.download}
            target={l.target}
            rel={l.rel}
          >
            {l.label}
          </Link>
        ))}
      </div>
      {note ? <p className="text-xs text-gray-500">{note}</p> : null}
    </div>
  );
}
