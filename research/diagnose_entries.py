"""Read-only diagnosis of frozen research trades; never imports the live bot.

Uses only verified local caches and public candles. No parameter search, order
gateway, database, environment file or production service is accessed.
"""
import ast
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import pandas as pd

from signal_scoring import causal_macd_scores
from strategy_recording import code_fingerprint

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/backtests/regimes_20261122'
RAW = ROOT / 'outputs/backtests/search_20261122/raw'
OUT = ROOT / 'outputs/backtests/entry_diagnosis_20261123'
MONTH = '2026-08'
RECEIPTS = {}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6):
        raise AssertionError((a, b))


def summary(trades):
    total = lambda key: sum(float(t[key]) for t in trades)
    gross, execution = total('reference_gross_profit'), total('execution_cost')
    fees = total('entry_fee') + total('exit_fee')
    carry, net = total('carry_fee'), total('profit')
    close(gross-execution-fees-carry, net)
    gains = sum(t['profit'] for t in trades if t['profit'] > 0)
    losses = -sum(t['profit'] for t in trades if t['profit'] < 0)
    return dict(trades=len(trades), net=net, reference_gross=gross,
                execution_cost=execution, fees=fees, carry=carry,
                wins=sum(t['profit'] > 0 for t in trades),
                profit_factor=gains/losses if losses else None,
                mean_holding_hours=total('holding_hours')/len(trades) if trades else None,
                fee_flipped_losses=sum(t['reference_gross_profit'] > 0 and t['profit'] < 0 for t in trades))


def grouped(trades, key):
    groups = defaultdict(list)
    for trade in trades:
        groups[str(trade[key])].append(trade)
    result = {name: summary(rows) for name, rows in sorted(groups.items())}
    assert sum(v['trades'] for v in result.values()) == len(trades)
    close(sum(v['net'] for v in result.values()), sum(t['profit'] for t in trades))
    return result


def verified_frame(symbol, day, expected_source):
    path = STUDY / 'feature_cache' / f'{symbol}_{day:%Y%m%d}.pkl.gz'
    metadata = path.with_suffix('.meta.json')
    saved = read_json(metadata)
    assert saved['source'] == expected_source
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert saved['sha256'] == digest
    # Pickles are self-generated research artifacts, verified before loading.
    frame = pd.read_pickle(path, compression='gzip')
    assert frame.timestamp.is_monotonic_increasing and not frame.timestamp.duplicated().any()
    RECEIPTS[str(path.relative_to(ROOT))] = digest
    return frame


def raw_bar(symbol, at):
    path = RAW / f'{symbol}_15min_{at:%Y%m%d}.json'
    receipt = RAW_HASHES[path.name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt
    RECEIPTS[str(path.relative_to(ROOT))] = receipt
    candles = next(c['ohlcv'] for c in read_json(path)['data']['candlestick'] if c['type'] == '15min')
    return next(row for row in candles if int(row[5]) == int(at.timestamp()*1000))


def boundary_value(trades, at):
    """Reconstruct last observation of the boundary day, matching daily bins."""
    realized = sum(t['profit'] for t in trades if pd.Timestamp(t['exit_time']) <= at)
    floating = costs = 0.
    positions = []
    for t in trades:
        entry, exit_at = pd.Timestamp(t['entry_time']), pd.Timestamp(t['exit_time'])
        if not entry <= at < exit_at:
            continue
        bar = raw_bar(t['symbol'], at if entry == at else at-pd.Timedelta(minutes=15))
        mark = float(bar[0] if entry == at else bar[3])
        value = (1 if t['type'] == 'long' else -1)*t['size']*(mark-t['entry_price'])
        paid = t['entry_fee'] + sum(e['amount'] for e in t['carry_events'] if pd.Timestamp(e['at']) <= at)
        floating += value
        costs += paid
        positions.append(dict(symbol=t['symbol'], side=t['type'], unrealized=value, paid_costs=paid))
    return dict(realized=realized, unrealized=floating, open_paid_costs=costs,
                net=realized+floating-costs, positions=positions)


def annotate(results, expected_source):
    requests = defaultdict(list)
    for name, result in results.items():
        for index, t in enumerate(result['trades']):
            at = pd.Timestamp(t['entry_time'])
            requests[t['symbol'], at.normalize()].append((name, index, at))
    for (symbol, day), tasks in sorted(requests.items()):
        frame = verified_frame(symbol, day, expected_source)
        for name, index, at in tasks:
            trade = results[name]['trades'][index]
            end = int(frame.timestamp.searchsorted(at, side='left'))
            rows = frame.iloc[end-2:end]
            assert len(rows) == 2
            assert rows.timestamp.iloc[0] == at-pd.Timedelta(minutes=30)
            assert rows.timestamp.iloc[1] == at-pd.Timedelta(minutes=15)
            column = 'buy_signal' if trade['type'] == 'short' else 'sell_signal'
            assert rows[column].all()
            sign = 1 if trade['type'] == 'long' else -1
            def context(first, second):
                values = (rows[first]-rows[second])*sign
                return 'both_follow' if (values > 0).all() else 'both_against' if (values < 0).all() else 'mixed_or_equal'
            trade['di_context'] = context('plus_di14', 'minus_di14')
            trade['ema_cross_context'] = context('EMA_short', 'EMA_long')
            trade['price_ema_context'] = context('close', 'EMA_long')
            trade['symbol_side'] = trade['symbol'] + ':' + trade['type']
            trade['week'] = str((at.day-1)//7+1)
            columns = ['timestamp','close','RSI','CCI','ADX','ATR','EMA_short','EMA_long',
                       'plus_di14','minus_di14','MACD_histogram','macd_score_long','macd_score_short',
                       'buy_score_scaled','sell_score_scaled','buy_signal','sell_signal']
            trade['closed_entry_rows'] = [
                {c: r[c].isoformat() if c == 'timestamp' else bool(r[c]) if c.endswith('_signal') else float(r[c]) for c in columns}
                for _, r in rows.iterrows()]
        del frame


def weight_invariance(source):
    """Execute extracted source only; zero weights is a diagnostic mutation."""
    owner = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'CryptoTradingBot')
    names = ('generate_signals','generate_signals_with_sentiment')
    original = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tables = {}
    for n in original[0].body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('signal_thresholds','indicator_weights_long','indicator_weights_short'):
            tables[n.targets[0].id] = ast.literal_eval(n.value)
    assert all(v == {'buy':0., 'sell':0.} for v in tables['signal_thresholds'].values())
    altered = deepcopy(original)
    for n in altered[0].body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('indicator_weights_long','indicator_weights_short'):
            zero = {s:{k:0. for k in values} for s,values in tables[n.targets[0].id].items()}
            n.value = ast.parse(repr(zero), mode='eval').body
    def bot(nodes):
        ns = dict(np=np,pd=pd,causal_macd_scores=causal_macd_scores)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), 'diagnostic_source_only', 'exec'),ns)
        result = SimpleNamespace(exchange_settings_gmo={'live_trade':True},logger=logging.getLogger('diagnosis'))
        for name in names:
            setattr(result,name,MethodType(ns[name],result))
        return result
    blocked = deepcopy(original)
    for n in blocked[0].body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'signal_thresholds':
            n.value = ast.parse(repr({s:{'buy':1.01,'sell':1.01} for s in tables['signal_thresholds']}), mode='eval').body
    baseline, mutation, positive_control = bot(original), bot(altered), bot(blocked)
    checks = []
    expected = read_json(STUDY/'source_manifest.json')['source']
    for symbol in sorted(tables['signal_thresholds']):
        frame = verified_frame(symbol, pd.Timestamp('2026-08-01',tz='Asia/Tokyo'), expected)
        left = baseline.generate_signals_with_sentiment(symbol,frame.copy(),pd.DataFrame())
        right = mutation.generate_signals_with_sentiment(symbol,frame.copy(),pd.DataFrame())
        control = positive_control.generate_signals_with_sentiment(symbol,frame.copy(),pd.DataFrame())
        assert left[['buy_signal','sell_signal']].equals(frame[['buy_signal','sell_signal']])
        assert left[['buy_signal','sell_signal']].equals(right[['buy_signal','sell_signal']])
        assert not control[['buy_signal','sell_signal']].any().any()
        assert left[['buy_score_scaled','sell_score_scaled']].ge(0).all().all()
        # Warm-up ATR NaN * zero remains NaN before the scaler's fallback.
        assert (right.buy_score.dropna() == 0).all() and (right.sell_score.dropna() == 0).all()
        checks.append(dict(symbol=symbol,rows=len(frame),saved_cache_reproduced=True,
                           weights_zero_signals_unchanged=True,
                           threshold_positive_control_blocks_all=True,
                           buy_signals=int(left.buy_signal.sum()),sell_signals=int(left.sell_signal.sum())))
    return dict(tables=tables,checks=checks,
                interpretation='At zero thresholds, bounded scaled scores always pass. The weighted aggregates do not select entries; downstream component gates still do.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = read_json(STUDY/'source_manifest.json')
    assert code_fingerprint(ROOT) == manifest['source'], 'Production strategy source changed since research'
    source = (ROOT/'crypto_bot_sub.py').read_text(encoding='utf-8-sig')
    global RAW_HASHES
    RAW_HASHES = {r['file']:r['sha256'] for r in read_json(RAW.parent/'raw_checksums.json')}
    results = {p.stem:read_json(p) for p in sorted((STUDY/'continuous').glob('*.json'))}
    input_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((STUDY/'continuous').glob('*.json'))}
    for r in results.values():
        assert r['run_identity'] == manifest['identity'] and not r['open_positions']
        assert len(r['trades']) == r['metrics']['trades']
        close(summary(r['trades'])['net'], r['metrics']['realized_profit'])
    annotate(results,manifest['source'])
    score_audit = weight_invariance(source)
    cases = {}
    for name, result in results.items():
        trades = result['trades']
        august = [t for t in trades if pd.Timestamp(t['exit_time']).strftime('%Y-%m') == MONTH]
        before = boundary_value(trades,pd.Timestamp('2026-07-31 23:45',tz='Asia/Tokyo'))
        after = boundary_value(trades,pd.Timestamp('2026-08-31 23:45',tz='Asia/Tokyo'))
        month_net = after['net']-before['net']
        close(month_net,result['metrics']['monthly_equity_changes'][MONTH])
        close(after['realized']-before['realized'],summary(august)['net'])
        cases[name] = dict(case=result['case'],all_closed=summary(trades),
            august_closed=summary(august),august_monthly_equity=month_net,
            month_bridge=dict(before=before,after=after,delta_unrealized=after['unrealized']-before['unrealized'],
                              delta_open_paid_costs=after['open_paid_costs']-before['open_paid_costs']),
            by_exit_month={m:summary([t for t in trades if t['exit_time'].startswith(m)]) for m in sorted({t['exit_time'][:7] for t in trades})},
            august_by={k:grouped(august,k) for k in ('type','symbol','symbol_side','exit_reason','di_context','ema_cross_context','price_ema_context')},
            all_by={k:grouped(trades,k) for k in ('type','di_context','ema_cross_context','price_ema_context')},
            worst_august=sorted(august,key=lambda t:t['profit'])[:10],
            august_entry_cohort=summary([t for t in trades if t['entry_time'].startswith(MONTH)]))
    checked = sum(len(r['trades']) for r in results.values())
    output = dict(source=manifest['source'],study_identity=manifest['identity'],
        closed_trade_records_checked=checked,score_audit=score_audit,cases=cases,
        checks=dict(accounting_identities=True,all_group_sums=True,all_two_closed_entry_rows=True,
                    all_seven_monthly_equity_bridges=True,seven_source_weight_mutations=True),
        limitations=['All results are retrospective and descriptive; subgroup removal is not a tested strategy.',
                     'Decision rows precede entry by 15 and 30 minutes; outcome groups use exit month JST.',
                     'Monthly equity uses existing research daily bins, including unrealized and accrued open costs.',
                     'Bitbank OHLC proxy, historical SR absent, fee/spread/carry scenario, no tick-level fill evidence.',
                     'Score threshold zero can be intentional; this audit does not establish an intended nonzero threshold.',
                     'No production access, code change, order, new strategy backtest or deployment.'])
    (OUT/'diagnosis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'annotated_trades.json').write_text(json.dumps({k:v['trades'] for k,v in results.items()},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    inputs = dict(inputs=input_hashes,verified_artifacts=RECEIPTS,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source=manifest['source'])
    (OUT/'input_receipts.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    for name, case in cases.items():
        print(json.dumps(dict(case=name,august_closed=case['august_closed'],monthly_equity=case['august_monthly_equity'],
                             by_di=case['august_by']['di_context'],by_ema=case['august_by']['ema_cross_context']),ensure_ascii=False),flush=True)
    print(json.dumps(dict(checked=checked,verified_files=len(RECEIPTS),source_mutations=len(score_audit['checks']),checks=output['checks'])),flush=True)


if __name__ == '__main__':
    main()
