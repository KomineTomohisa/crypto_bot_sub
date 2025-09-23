"use client";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function ChartClient({ data }: { data: { date: string; win_rate?: number | null; avg_pnl_pct?: number | null }[] }) {
  const d = data.map(x => ({
    date: x.date.slice(5),                         // MM-DD だけ表示
    win_rate: x.win_rate != null ? Math.round(x.win_rate * 1000) / 10 : null,   // %
    avg_pnl_pct: x.avg_pnl_pct != null ? Math.round(x.avg_pnl_pct * 1000) / 10 : null, // %
  }));
  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={d}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" domain={[0, 100]} tickFormatter={(v)=>`${v}%`} />
          <YAxis yAxisId="right" orientation="right" tickFormatter={(v)=>`${v}%`} />
          <Tooltip formatter={(v:number)=>`${v}%`} />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="win_rate" name="Win Rate (%)" dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="avg_pnl_pct" name="Avg PnL (%)" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
