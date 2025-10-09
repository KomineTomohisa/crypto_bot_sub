"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

export default function EquityChartClient({
  data,
}: {
  data: Array<{ date: string; equity_yen: number; win_rate: number; dd_pct: number }>;
}) {
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 16, left: 8, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} minTickGap={24} />
          
          {/* 左軸: 資産額 */}
          <YAxis
            yAxisId="yen"
            tickFormatter={(v: number) => `¥${Math.round(v).toLocaleString("ja-JP")}`}
            width={80}
          />
          {/* 右軸: Win Rate */}
          <YAxis
            yAxisId="rate"
            orientation="right"
            tickFormatter={(v: number) => `${v.toFixed(1)}%`}
            width={60}
          />

          <Tooltip
            formatter={(value: number, name: string) => {
              if (name === "資産額") return [`¥${Math.round(value).toLocaleString("ja-JP")}`, name];
              if (name === "勝率") return [`${value.toFixed(2)}%`, name];
              return [String(value), name];
            }}
            labelFormatter={(label: string) => `日付: ${label}`}
          />
          <Legend />

          {/* 資産額 */}
          <Line
            yAxisId="yen"
            type="monotone"
            dataKey="equity_yen"
            name="資産額"
            strokeWidth={2}
            stroke="#3b82f6" // 青
            dot={false}
          />
          {/* 勝率 */}
          <Line
            yAxisId="rate"
            type="monotone"
            dataKey="win_rate"
            name="勝率"
            strokeWidth={2}
            stroke="#10b981" // 緑
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
