export type Rule = {
  id: number;
  symbol: string;
  timeframe: string; // 例: "15m"
  score_col: string;
  op: string; // between, <, <=, >, >=, ==, !=, is_null, is_not_null
  v1: string | number | null;
  v2: string | number | null;
  target_side: "buy" | "sell" | string;
  action: string; // disable
  priority: number;
  active: boolean;
  version: string | null;
  valid_from: string | null; // ISO
  valid_to: string | null; // ISO
  user_id?: string | null;
  strategy_id?: string | null;
  notes?: string | null;
};