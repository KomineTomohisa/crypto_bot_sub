export const revalidate = 0;

import { PageHeader, Section } from "@/components/ui";
import type { Rule } from "./types";
import ToolbarClient from "./ToolbarClient";
import RulesTableClient from "./RulesTableClient";
import StrategyViewClient from "./StrategyViewClient";

/* =========================
   SSR ユーティリティ
   ========================= */
function apiBase() {
  const isServer = typeof window === "undefined";
  return isServer ? process.env.API_BASE_INTERNAL! : process.env.NEXT_PUBLIC_API_BASE!;
}

async function fetchRules(params: URLSearchParams): Promise<Rule[]> {
  const base = apiBase();
  const url = new URL(`${base}/admin/signal-rules`);
  params.forEach((v, k) => url.searchParams.set(k, v));
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) return [];
  return res.json();
}

/* =========================
   ページ本体（RSC）
   ========================= */
export default async function RulesPage({
  searchParams,
}: {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}) {
  const sp = searchParams ? await searchParams : undefined;
  const usp = new URLSearchParams();

  for (const k of [
    "symbol",
    "timeframe",
    "active",
    "only_open_ended",
    "user_id",
    "strategy_id",
    "version",
    "q",
    "sort",
  ]) {
    const v = sp?.[k];
    if (typeof v === "string") usp.set(k, v);
    else if (Array.isArray(v) && v[0]) usp.set(k, v[0]);
  }

  const rules = await fetchRules(usp);

  return (
    <main className="p-6 md:p-8 max-w-4xl mx-auto space-y-8">
      <PageHeader
        title="Signal Rule Thresholds"
        description={
          <>
            DB上の <b>signal_rule_thresholds</b> を一覧表示します。将来的な編集・複製・無効化に対応しやすい設計です。
          </>
        }
      />

      <Section title="フィルタ & 操作">
        <ToolbarClient initialCount={rules.length} />
      </Section>

      <Section title="見やすいビュー（初心者向け）" headerRight={<div className="text-xs text-gray-500">カード表示</div>}>
        <StrategyViewClient data={rules} />
      </Section>

      <Section
        title="ルール一覧"
        headerRight={<div className="text-xs text-gray-500">件数: <b>{rules.length}</b></div>}
      >
        <RulesTableClient initialData={rules} />
      </Section>

      <Section title="ヘルプ">
        <ul className="list-disc pl-5 text-sm text-gray-600 dark:text-gray-300 space-y-1">
          <li>将来的に行右端の <em>…</em> メニューから「編集 / 複製 / 無効化 / 削除」を提供します。</li>
          <li>編集はモーダルで行い、<code>/admin/signal-rules/:id</code> に <code>PUT</code> で送信する想定です。</li>
          <li>CSVエクスポートは現在のフィルタ結果を対象に生成します。</li>
        </ul>
      </Section>
    </main>
  );
}
