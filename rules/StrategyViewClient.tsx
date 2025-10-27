"use client";

import { useMemo, useState } from "react";
import type { Rule } from "./types";

/* =========================
   型・ユーティリティ
   ========================= */

type Side = "buy" | "sell";
type Segment = { start: number; end: number; disabled: boolean; raw: Rule };
type Marker = { pos: number; disabled: boolean; label: string; raw: Rule };

type IndicatorBucket = {
  score_col: string;
  byTimeframe: Record<string, { segments: Segment[]; markers: Marker[]; extraNotes: string[] }>;
  totals: { active: number; openEnded: number; count: number };
};

type SideGroup = { side: Side; indicators: IndicatorBucket[] };
type SymbolGroup = { symbol: string; sides: SideGroup[] };

const INDICATOR_ORDER = [
  "bb_score_short","bb_score_long","rsi_score_short","rsi_score_long",
  "adx_score_short","adx_score_long","atr_score_short","atr_score_long",
  "ma_score_short","ma_score_long",
];

function clamp01(x: number){ return Math.max(0, Math.min(1, x)); }
function toNum(v: string | number | null){ if(v==null) return null; const n=Number(v); return Number.isFinite(n)?n:null; }

function buildSegmentsAndMarkers(rule: Rule){
  const disabled = rule.action === "disable";
  const segs: Segment[] = []; const marks: Marker[] = []; const notes: string[] = [];
  const v1n = toNum(rule.v1); const v2n = toNum(rule.v2);

  switch (rule.op) {
    case "between":
      if(v1n==null||v2n==null){notes.push("between値不足");break;}
      segs.push({ start: clamp01(Math.min(v1n,v2n)), end: clamp01(Math.max(v1n,v2n)), disabled, raw: rule });
      break;
    case "<": case "<=":
      if(v1n==null){notes.push(`${rule.op} 値不足`);break;}
      segs.push({ start: 0, end: clamp01(v1n), disabled, raw: rule });
      break;
    case ">": case ">=":
      if(v1n==null){notes.push(`${rule.op} 値不足`);break;}
      segs.push({ start: clamp01(v1n), end: 1, disabled, raw: rule });
      break;
    case "==": case "!=":
      if(v1n==null){notes.push(`${rule.op} 値不足`);break;}
      marks.push({ pos: clamp01(v1n), disabled, label: rule.op==="=="?"=":"≠", raw: rule });
      break;
    case "is_null": notes.push("値なし条件"); break;
    case "is_not_null": notes.push("値あり条件"); break;
    default: notes.push(`未対応op: ${rule.op}`);
  }
  return { segs, marks, notes };
}

function sideColor(side: Side){ return side==="buy" ? "bg-emerald-500" : "bg-rose-500"; }
function pillToneForSide(side: Side){ return side==="buy" ? "green" : "red"; }

/* =========================
   UIパーツ
   ========================= */

function Pill({ children, tone = "default" }:{
  children:React.ReactNode; tone?:"green"|"red"|"blue"|"default";
}){
  const map: Record<string,string> = {
    green:"bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300",
    red:"bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300",
    blue:"bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
    default:"bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };
  return <span className={`inline-flex items-center rounded px-2 py-0.5 text-[11px] font-medium ${map[tone]}`}>{children}</span>;
}

function Legend(){
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-6 rounded bg-emerald-500"/><span>buy 有効領域</span></span>
      <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-6 rounded bg-rose-500"/><span>sell 有効領域</span></span>
      <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-6 rounded bg-gray-400"/><span>無効（disable）</span></span>
      <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-0.5 rounded bg-gray-700"/><span>一致/不一致マーカー（= / ≠）</span></span>
      <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-10 rounded bg-gray-200 border border-gray-300"/><span>0–1 バー（下に 0 / 1 目盛）</span></span>
    </div>
  );
}

/* ========= 範囲ラベル（中央） ========= */
function CenterRangeLabel({
  mode, score,
}:{ mode: "both" | "leftOnly" | "rightOnly" | "disabled"; score: string; }){
  return (
    <span className="font-mono">
      {mode !== "leftOnly" ? "< " : ""}
      {score}
      {mode !== "rightOnly" ? " <" : ""}
    </span>
  );
}

/* ============== 指標ごとの編集モーダル（モック） ============== */

type IndicatorEditModalProps = {
  open: boolean;
  onClose: () => void;
  symbol: string;
  side: Side;
  score_col: string;
  rules: Rule[];                 // この indicator に属するルールのみ
  onCreate: (draft: Omit<Rule,"id">) => void;
  onUpdate: (rule: Rule) => void;
  onDelete: (id: number) => void;
};

function IndicatorEditModal({
  open, onClose, symbol, side, score_col, rules, onCreate, onUpdate, onDelete,
}: IndicatorEditModalProps){
  const [editMap, setEditMap] = useState<Record<number, Rule>>({});
  const [initKey, setInitKey] = useState<string>("");
  const [confirm, setConfirm] = useState<null | {type:"apply"|"delete"; id:number}>(null);

  // 🔸ベースライン（初期値）を保持：モーダルを開いた時点の値
  const [baseline, setBaseline] = useState<Record<number, { v1: number|null; v2: number|null; active: boolean }>>({});

  if (initKey !== `${symbol}|${side}|${score_col}`){
    setEditMap({});
    setConfirm(null);
    setInitKey(`${symbol}|${side}|${score_col}`);
    // 初期スナップショット
    const b: Record<number, {v1:number|null; v2:number|null; active:boolean}> = {};
    for (const r of rules) {
      b[r.id] = { v1: (r.v1 ?? null) as number|null, v2: (r.v2 ?? null) as number|null, active: !!r.active };
    }
    setBaseline(b);
  }
  if (!open) return null;

  // 片側条件のモード
  const rangeModeFor = (r: Rule): "both" | "leftOnly" | "rightOnly" | "disabled" => {
    switch (r.op) {
      case "between": return "both";
      case ">": case ">=": return "leftOnly";   // v1 < score
      case "<": case "<=": return "rightOnly";  // score < v1（UI右入力→内部は v1 に格納）
      case "==": case "!=": case "is_null": case "is_not_null": return "disabled";
      default: return "disabled";
    }
  };

  const edValue = (id:number, key:"v1"|"v2") => (editMap[id]?.[key] ?? rules.find(r=>r.id===id)?.[key]) ?? null;

  const handleNumChange = (id:number, which:"left"|"right") => (e:React.ChangeEvent<HTMLInputElement>)=>{
    const raw = rules.find(r=>r.id===id)!;
    const mode = rangeModeFor(raw);
    const val = e.currentTarget.value === "" ? null : Number(e.currentTarget.value);
    setEditMap(prev=>{
      const base = prev[id] ?? raw;
      // < / <= は右入力→内部 v1 に保存、> / >= は左入力→内部 v1
      if (mode==="rightOnly" || mode==="leftOnly"){
        return { ...prev, [id]: { ...base, v1: val } };
      }
      if (mode==="both"){
        if (which==="left") return { ...prev, [id]: { ...base, v1: val } };
        else return { ...prev, [id]: { ...base, v2: val } };
      }
      return prev;
    });
  };

  const handleActive = (id:number) => (e:React.ChangeEvent<HTMLInputElement>)=>{
    const raw = rules.find(r=>r.id===id)!;
    const v = e.currentTarget.checked;
    setEditMap(prev=>({ ...prev, [id]: { ...(prev[id] ?? raw), active: v }}));
  };

  const applyRow = (id:number)=>{
    const draft = editMap[id] ?? rules.find(r=>r.id===id);
    if (!draft) return;
    onUpdate(draft);
    // ベースライン更新（適用後は未変更状態に戻す）
    setBaseline((b)=>({
      ...b,
      [id]: { v1: (draft.v1 ?? null) as number|null, v2: (draft.v2 ?? null) as number|null, active: !!draft.active }
    }));
    setEditMap(prev=>{ const cp={...prev}; delete cp[id]; return cp; });
  };

  // 新規追加（この指標にひも付けて1行作る）→ 追加直後は「新規」扱い＝適用ボタン常時活性
  const addRow = ()=>{
    const defaultTF = rules[0]?.timeframe ?? "15m";
    onCreate({
      symbol,
      timeframe: defaultTF,
      score_col,
      op: "between",
      v1: 0.2,
      v2: 0.8,
      target_side: side,
      action: "",         // 空なら有効扱い（disable だけが無効）
      priority: 100,
      active: true,
      version: null,
      valid_from: null,
      valid_to: null,
      user_id: null,
      strategy_id: null,
      notes: "",
    });
    // baseline は更新しない → 新規レコードは baseline に存在しないため dirty 扱い
  };

  // 🔸dirty 判定：baseline に無いID = 新規 → true。ある場合は編集対象だけ比較
  const isDirty = (r: Rule): boolean => {
    const mode = rangeModeFor(r);
    const cur = editMap[r.id] ?? r;
    const base = baseline[r.id]; // ない＝新規
    if (!base) return true;
    const v1c = (cur.v1 ?? null) as number|null;
    const v2c = (cur.v2 ?? null) as number|null;
    const act = !!cur.active;

    if (mode === "both") {
      return v1c !== base.v1 || v2c !== base.v2 || act !== base.active;
    }
    if (mode === "leftOnly" || mode === "rightOnly") {
      return v1c !== base.v1 || act !== base.active;
    }
    // disabled（==/!=/null 系）は active のみ比較
    return act !== base.active;
  };

  return (
    <div className="fixed inset-0 z-[60]">
      {/* 背景 */}
      <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden />

      {/* 本体 */}
      <div className="absolute inset-x-0 top-10 mx-auto w-[min(100%,980px)] rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 shadow-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="text-lg font-semibold">
            戦略編集（{symbol} / <span className="capitalize">{side}</span> / <span className="font-mono">{score_col}</span>）※モック
          </div>
          <div className="flex items-center gap-2">
            {/* 新規追加ボタン（閉じるの左） */}
            <button
              className="px-3 py-1.5 text-sm rounded-lg border bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:bg-emerald-900/20 dark:text-emerald-300"
              onClick={addRow}
              title="この指標に新しいルールを追加（モック）"
            >
              新規追加
            </button>
            <button className="px-3 py-1.5 text-sm rounded-lg border" onClick={onClose}>
              閉じる
            </button>
          </div>
        </div>

        <div className="max-h-[65vh] overflow-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300 sticky top-0">
              <tr>
                <th className="px-2 py-2">ID</th>
                {/* tf を中央＆少し大きめ */}
                <th className="px-2 py-2 text-center">tf</th>
                <th className="px-2 py-2">範囲編集</th>
                {/* 操作列も中央寄せの見出し */}
                <th className="px-2 py-2 text-center">active</th>
                <th className="px-2 py-2 text-center">操作</th>
              </tr>
            </thead>
            <tbody>
              {rules.map((r)=>{
                const mode = rangeModeFor(r);
                const leftDisabled  = mode==="rightOnly" || mode==="disabled";
                const rightDisabled = mode==="leftOnly"  || mode==="disabled";

                // 表示用の値（片側条件なら右/左に v1 を映す）
                const leftVal  = mode==="leftOnly"||mode==="both" ? edValue(r.id,"v1") : null;
                const rightVal = mode==="rightOnly"||mode==="both" ? (mode==="rightOnly" ? edValue(r.id,"v1") : edValue(r.id,"v2")) : null;

                const dirty = isDirty(r);

                return (
                  <tr key={r.id} className="border-t border-gray-100 dark:border-gray-800">
                    <td className="px-2 py-2 text-right tabular-nums">{r.id}</td>

                    {/* tf：中央＆少し太字 */}
                    <td className="px-2 py-2 text-center font-medium">
                      <span className="inline-block min-w-[48px]">{r.timeframe}</span>
                    </td>

                    <td className="px-2 py-1">
                      {/* コンパクト幅に制限＋非活性は薄く */}
                      <div
                        className={`mx-auto max-w-[360px] grid grid-cols-[100px_1fr_100px] items-center gap-1 ${
                          mode === "disabled" ? "opacity-60" : ""
                        }`}
                      >
                        {/* 左ボックス */}
                        <input
                          type="number" step="0.001"
                          className={`border rounded px-2 py-1 w-[100px] bg-transparent
                                      disabled:bg-gray-200 disabled:text-gray-500 disabled:border-gray-300 disabled:cursor-not-allowed
                                      dark:disabled:bg-gray-800 dark:disabled:text-gray-400 dark:disabled:border-gray-700`}
                          value={leftVal ?? ""}
                          onChange={handleNumChange(r.id,"left")}
                          disabled={leftDisabled}
                          aria-label="v1-left"
                        />

                        {/* 中央ラベル */}
                        <div className={`text-xs text-center ${mode==="disabled" ? "text-gray-400" : "text-gray-700 dark:text-gray-200"}`}>
                          <CenterRangeLabel mode={mode} score={r.score_col} />
                        </div>

                        {/* 右ボックス */}
                        <input
                          type="number" step="0.001"
                          className={`border rounded px-2 py-1 w-[100px] bg-transparent
                                      disabled:bg-gray-200 disabled:text-gray-500 disabled:border-gray-300 disabled:cursor-not-allowed
                                      dark:disabled:bg-gray-800 dark:disabled:text-gray-400 dark:disabled:border-gray-700`}
                          value={rightVal ?? ""}
                          onChange={handleNumChange(r.id,"right")}
                          disabled={rightDisabled}
                          aria-label="v2-right"
                        />
                      </div>

                      {(r.op==="=="||r.op==="!="||r.op==="is_null"||r.op==="is_not_null") && (
                        <div className="text-[11px] text-amber-600 mt-1">
                          このルールは {r.op} のため範囲編集は無効です。
                        </div>
                      )}
                    </td>

                    {/* active */}
                    <td className="px-2 py-2 text-center">
                      <input
                        type="checkbox"
                        checked={(editMap[r.id]?.active ?? r.active) ? true : false}
                        onChange={handleActive(r.id)}
                        aria-label="active"
                      />
                    </td>

                    {/* 操作：ボタンを少し大きく＆中央配置。dirtyで活性化 */}
                    <td className="px-2 py-2">
                      <div className="flex items-center justify-center gap-2">
                        <button
                          className="px-3 py-1 text-sm rounded-lg border hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                          onClick={()=>setConfirm({type:"apply", id:r.id})}
                          title={dirty ? "変更を適用" : "変更がありません"}
                          disabled={!dirty}
                        >
                          適用
                        </button>
                        <button
                          className="px-3 py-1 text-sm rounded-lg border border-rose-300 text-rose-600 hover:bg-rose-50"
                          onClick={()=>setConfirm({type:"delete", id:r.id})}
                          title="この行を削除"
                        >
                          削除
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {rules.length===0 && (
                <tr><td className="px-2 py-8 text-center text-xs text-gray-500" colSpan={5}>この指標のルールはありません。</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 確認モーダル */}
      {confirm && (
        <div className="fixed inset-0 z-[70]">
          <div className="absolute inset-0 bg-black/50" onClick={()=>setConfirm(null)} aria-hidden />
          <div className="absolute inset-x-0 top-1/3 mx-auto w-[min(92%,480px)] rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 shadow-xl p-5">
            <div className="text-base font-semibold mb-2">
              {confirm.type==="apply" ? "本当に確定してよいですか？" : "本当に削除してよいですか？"}
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
              この操作はモックですが、現在の画面内のデータは更新／削除されます。
            </p>
            <div className="flex justify-end gap-2">
              <button className="px-3 py-1.5 text-sm rounded-lg border" onClick={()=>setConfirm(null)}>キャンセル</button>
              {confirm.type==="apply" ? (
                <button
                  className="px-3 py-1.5 text-sm rounded-lg bg-emerald-600 text-white hover:bg-emerald-700"
                  onClick={()=>{ applyRow(confirm.id); setConfirm(null); }}
                >
                  確定
                </button>
              ) : (
                <button
                  className="px-3 py-1.5 text-sm rounded-lg bg-rose-600 text-white hover:bg-rose-700"
                  onClick={()=>{ onDelete(confirm.id); setConfirm(null); }}
                >
                  削除
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


/* =========================
   本体（表示＋ローカル編集モック）
   ========================= */

export default function StrategyViewClient({ data }: { data: Rule[] }) {
  // ローカル編集用コピー（モック）
  const [localRules, setLocalRules] = useState<Rule[]>(() => data.map(r=>({...r})));

  // symbol > side > indicator
  const groups = useMemo<SymbolGroup[]>(() => {
    const bySymbol = new Map<string, Rule[]>();
    for (const r of localRules){ const k=r.symbol ?? "(unknown)"; if(!bySymbol.has(k)) bySymbol.set(k,[]); bySymbol.get(k)!.push(r); }

    const result: SymbolGroup[] = [];
    for (const [symbol, rules] of bySymbol.entries()){
      const sides: Side[] = ["buy","sell"];
      const sideGroups: SideGroup[] = [];
      for (const side of sides){
        const sideRules = rules.filter(r => (r.target_side ?? "").toLowerCase() === side);
        const byIndicator = new Map<string, Rule[]>();
        for (const r of sideRules){ const k=r.score_col ?? "(unknown)"; if(!byIndicator.has(k)) byIndicator.set(k,[]); byIndicator.get(k)!.push(r); }

        const indicators: IndicatorBucket[] = [];
        for (const [score_col, indRules] of byIndicator.entries()){
          const byTF = new Map<string, {segments:Segment[]; markers:Marker[]; extraNotes:string[];}>();
          for (const r of indRules){
            const tf=r.timeframe ?? "";
            if(!byTF.has(tf)) byTF.set(tf,{segments:[],markers:[],extraNotes:[]});
            const built = buildSegmentsAndMarkers(r);
            byTF.get(tf)!.segments.push(...built.segs);
            byTF.get(tf)!.markers.push(...built.marks);
            byTF.get(tf)!.extraNotes.push(...built.notes);
          }
          const totals = {
            active: indRules.filter(r=>r.active).length,
            openEnded: indRules.filter(r=>r.valid_to==null).length,
            count: indRules.length,
          };
          const entries = Array.from(byTF.entries()).sort((a,b)=>a[0].localeCompare(b[0],"en",{numeric:true}));
          const byTimeframe: IndicatorBucket["byTimeframe"] = {};
          for (const [tf,v] of entries) byTimeframe[tf]=v;
          indicators.push({ score_col, byTimeframe, totals });
        }
        indicators.sort((a,b)=>{const ai=INDICATOR_ORDER.indexOf(a.score_col), bi=INDICATOR_ORDER.indexOf(b.score_col); if(ai!==-1&&bi!==-1) return ai-bi; if(ai!==-1) return -1; if(bi!==-1) return 1; return a.score_col.localeCompare(b.score_col);});
        sideGroups.push({ side, indicators });
      }
      sideGroups.sort((a,b)=>{ const ca=a.indicators.reduce((acc,ib)=>acc+ib.totals.count,0); const cb=b.indicators.reduce((acc,ib)=>acc+ib.totals.count,0); return cb-ca; });
      result.push({ symbol, sides: sideGroups });
    }
    result.sort((a,b)=>{ const ca=a.sides.reduce((acc,sg)=>acc+sg.indicators.reduce((acc2,ib)=>acc2+ib.totals.count,0),0); const cb=b.sides.reduce((acc,sg)=>acc+sg.indicators.reduce((acc2,ib)=>acc2+ib.totals.count,0),0); return cb-ca; });
    return result;
  }, [localRules]);

  /* 折りたたみ状態（通貨ごと） */
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const toggleSymbol = (symbol:string)=> setCollapsed(prev=>({...prev,[symbol]:!prev[symbol]}));

  /* 指標モーダルの状態 */
  const [modal, setModal] = useState<null | {symbol:string; side:Side; score_col:string}>(null);
  const openIndicatorModal = (symbol:string, side:Side, score_col:string)=> setModal({symbol, side, score_col});
  const closeModal = ()=> setModal(null);

  const rulesOfIndicator = (symbol:string, side:Side, score_col:string) =>
    localRules.filter(r => r.symbol===symbol && (r.target_side??"").toLowerCase()===side && r.score_col===score_col);

  // 追加・更新・削除（ローカルのみ）
  const handleCreate = (draft: Omit<Rule,"id">) => {
    const newId = (localRules.reduce((m, r) => Math.max(m, r.id), 0) || 0) + 1;
    const row: Rule = { id: newId, ...draft };
    setLocalRules(rows => [...rows, row]);
  };
  const handleUpdate = (rule: Rule)=> setLocalRules(rows=>rows.map(r=>r.id===rule.id?{...rule}:r));
  const handleDelete = (id:number)=> setLocalRules(rows=>rows.filter(r=>r.id!==id));

  return (
    <div className="space-y-6">
      {/* 凡例 */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-800 p-3 bg-white dark:bg-gray-900"><Legend/></div>

      {groups.map((g)=>{
        const isCollapsed = !!collapsed[g.symbol];
        return (
          <div key={g.symbol} className="rounded-2xl border border-gray-200 dark:border-gray-800 p-4 bg-white dark:bg-gray-900">
            {/* 通貨ヘッダ */}
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="text-lg font-semibold">{g.symbol}</div>
              <div className="flex items-center gap-2 text-xs">
                <Pill>総 {g.sides.reduce((acc,sg)=>acc+sg.indicators.reduce((a,ib)=>a+ib.totals.count,0),0)}</Pill>
                <Pill tone="green">Active {g.sides.reduce((acc,sg)=>acc+sg.indicators.reduce((a,ib)=>a+ib.totals.active,0),0)}</Pill>
                <Pill tone="blue">OpenEnd {g.sides.reduce((acc,sg)=>acc+sg.indicators.reduce((a,ib)=>a+ib.totals.openEnded,0),0)}</Pill>
                <button onClick={()=>toggleSymbol(g.symbol)} className="ml-2 inline-flex items-center gap-1 rounded-lg border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs hover:bg-gray-50 dark:hover:bg-gray-800" aria-expanded={!isCollapsed} aria-controls={`symbol-content-${g.symbol}`}>{isCollapsed?"展開":"折りたたみ"}</button>
              </div>
            </div>

            {!isCollapsed && (
              <div id={`symbol-content-${g.symbol}`} className="mt-3 space-y-4">
                {g.sides.map((sg)=>(
                  <div key={`${g.symbol}__${sg.side}`} className="rounded-xl border border-gray-100 dark:border-gray-800 p-3 bg-gray-50 dark:bg-gray-950">
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-sm font-medium flex items-center gap-2">
                        <Pill tone={pillToneForSide(sg.side)}>{sg.side}</Pill>
                        <span className="text-xs text-gray-500">ルール {sg.indicators.reduce((a,ib)=>a+ib.totals.count,0)}</span>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {sg.indicators.map((ib)=>(
                        <div key={`${g.symbol}__${sg.side}__${ib.score_col}`} className="rounded-lg border border-gray-200 dark:border-gray-800 p-3 bg-white dark:bg-gray-900">
                          <div className="flex items-center justify-between">
                            <div className="text-sm font-medium">{ib.score_col}</div>
                            <div className="flex items-center gap-2 text-xs">
                              <Pill>TF {Object.keys(ib.byTimeframe).length}</Pill>
                              <Pill tone="green">Active {ib.totals.active}</Pill>
                              <Pill tone="blue">OpenEnd {ib.totals.openEnded}</Pill>
                              <button
                                onClick={()=>openIndicatorModal(g.symbol, sg.side, ib.score_col)}
                                className="ml-2 inline-flex items-center gap-1 rounded-lg border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs hover:bg-gray-50 dark:hover:bg-gray-800"
                                title="この指標の戦略を編集（モック）"
                              >
                                編集
                              </button>
                            </div>
                          </div>

                          {/* timeframeごとのバー */}
                          <div className="mt-2 space-y-3">
                            {Object.entries(ib.byTimeframe).map(([tf,v])=>(
                              <div key={`${ib.score_col}__${tf}`}>
                                <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                                  <span>TF: <b>{tf}</b></span>
                                  <span>seg {v.segments.length} / mark {v.markers.length}</span>
                                </div>
                                <div className="relative w-full h-3 rounded bg-gray-200 dark:bg-gray-800 overflow-hidden">
                                  {v.segments.map((s,idx)=>{
                                    const left=`${s.start*100}%`; const width=`${Math.max(0,s.end-s.start)*100}%`;
                                    const color=s.disabled?"bg-gray-400":sideColor(sg.side);
                                    const title=`${ib.score_col} ${s.raw.op} ${s.raw.v1 ?? ""}${s.raw.op==="between" ? ` ~ ${s.raw.v2 ?? ""}` : ""} ${s.disabled?"(無効)":""}`;
                                    return <div key={idx} className={`absolute top-0 h-3 ${color}`} style={{left,width}} title={title} />;
                                  })}
                                  {v.markers.map((m,idx)=>{
                                    const left=`${m.pos*100}%`; const color=m.disabled?"bg-gray-600":"bg-gray-900";
                                    const title=`${ib.score_col} ${m.label} ${m.raw.v1 ?? ""} ${m.disabled?"(無効)":""}`;
                                    return <div key={`mk-${idx}`} className={`absolute top-0 h-3 w-0.5 ${color}`} style={{left}} title={title} />;
                                  })}
                                </div>
                                <div className="mt-1 text-[11px] text-gray-600 dark:text-gray-300 flex items-center justify-between"><span className="font-semibold">0</span><span className="font-semibold">1</span></div>
                                {v.extraNotes.length>0 && <div className="mt-1 text-[11px] text-gray-500">{Array.from(new Set(v.extraNotes)).join(" / ")}</div>}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                      {sg.indicators.length===0 && (
                        <div className="rounded-lg border border-dashed border-gray-300 dark:border-gray-700 p-6 text-center text-sm text-gray-500">
                          {sg.side} 側の指標はありません。
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {groups.length===0 && (
        <div className="rounded-xl border border-dashed border-gray-300 dark:border-gray-700 p-8 text-center text-sm text-gray-500">
          条件に一致するルールがありません。上のフィルタを変更して再検索してください。
        </div>
      )}

      {/* 指標ごとの編集モーダル（モック） */}
      {modal && (
        <IndicatorEditModal
          open={true}
          onClose={closeModal}
          symbol={modal.symbol}
          side={modal.side}
          score_col={modal.score_col}
          rules={rulesOfIndicator(modal.symbol, modal.side, modal.score_col)}
          onCreate={handleCreate}
          onUpdate={handleUpdate}
          onDelete={handleDelete}
        />
      )}
    </div>
  );
}
