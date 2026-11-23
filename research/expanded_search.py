"""A separate, preregistered sweep. Public cached data only; never deploys."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd

from research.search_backtests import (ROOT, SYMBOLS, DailyDecisionCache, read_frames,
    write_json, code_fingerprint, execute, make_bot, offline_io_guard)
from research.replay_engine import simulate, case_prepare, install_case, compare_core
from strategy_recording import runtime_versions

OUT = ROOT / 'outputs/backtests/expanded_20261122'
OLD = ROOT / 'outputs/backtests/search_20261122'
BASELINE = dict(direction='both', adx_min=0, exit_profile='baseline')
CANDIDATE = dict(direction='both', adx_min=30, exit_profile='no_short_lock')
_FRAMES = _CACHE = _COSTS = _IDENTITY = None
FILES = ['research/expanded_search.py', 'research/search_backtests.py', 'research/replay_engine.py',
         'research/power.py', 'configs/backtest_costs_reference.json']


def cases():
    result = [dict(direction=d, adx_min=a, exit_profile=e) for d,a,e in product(
        ['both','short_only'], [0,20,25,30,35,40], ['baseline','no_short_lock'])]
    # Short-lock changes cannot affect a long-only strategy: do not count duplicates.
    result += [dict(direction='long_only', adx_min=a, exit_profile='no_short_lock') for a in [0,20,25,30,35,40]]
    for d,a,(tp,sl) in product(['both','short_only'], [25,30], [(.8,1.),(1.2,1.),(1.,.8),(1.,1.2)]):
        result.append(dict(direction=d, adx_min=a, exit_profile='no_short_lock', tp_multiplier=tp, sl_multiplier=sl))
    result += [dict(direction=d, adx_min=a, exit_profile='short_lock_1pct') for d,a in product(['both','short_only'],[25,30])]
    assert len(result) == 50
    return result


def key(case):
    return f"{case['direction']}_adx{case['adx_min']}_{case['exit_profile']}_tp{case.get('tp_multiplier',1):g}_sl{case.get('sl_multiplier',1):g}"


def identity():
    values={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}
    values.update(source=code_fingerprint(ROOT), runtime=runtime_versions(),
                  plan=hashlib.sha256((OUT/'study_plan.json').read_bytes()).hexdigest(),
                  raw_receipt=hashlib.sha256((OLD/'raw_checksums.json').read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def read_plan():
    return json.loads((OUT/'study_plan.json').read_text())


def feature_ast(source):
    tree=ast.parse(source)
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CryptoTradingBot')
    names={'build_features','generate_signals','generate_signals_with_sentiment','_apply_rule_thresholds'}
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in names}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(registered_at=datetime.now(timezone.utc).isoformat(), candidates=cases(), candidate_count=50,
        symbols=SYMBOLS, capital=200000, trade_size=40000, train=['2026-05-01','2026-08-01'],
        validation=['2026-08-01','2026-10-01'], historical_audit=['2026-10-01','2026-11-22'],
        continuous=['2026-05-01','2026-11-22'], shortlist=6, finalists=3,
        train_screen='>=30 closed trades, net>0, PF>=1.1, estimated DD<=10%, >=2 positive training months; eligible first then net/DD',
        validation_selection='freeze top 3 by worst(base, double-spread/slippage) net/absolute-DD; no reselection after audit',
        no_untouched_holdout=True, production_changed=False,
        limitations=['All date ranges were inspected in an earlier study; chronological splits are not untouched evidence.',
                     'Bitbank OHLC proxy, no historical SR, full-notional cash reserve, no liquidation model.',
                     'Fixed reference fees/carry; spread/slippage are assumptions; DD uses observed grid only.',
                     'Many trials increase selection bias; no global optimum or future profit claim.'])
    if (OUT/'study_plan.json').exists():
        existing=read_plan();plan['registered_at']=existing['registered_at']
        if plan!=existing:raise RuntimeError('Plan changed; use a different study directory')
    write_json(OUT/'study_plan.json',plan)
    frames=read_frames(OLD/'raw')
    receipts=json.loads((OLD/'raw_checksums.json').read_text())
    for item in receipts:
        if hashlib.sha256((OLD/'raw'/item['file']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Raw candle receipt differs')
    quality=[]
    for (symbol,interval),frame in frames.items():
        start=pd.Timestamp('2026-04-29',tz='Asia/Tokyo');end=pd.Timestamp('2026-11-22',tz='Asia/Tokyo')
        actual=pd.DatetimeIndex(frame.loc[frame.timestamp.ge(start)&frame.timestamp.lt(end),'timestamp'])
        expected=pd.date_range(start,end,freq='15min' if interval=='15min' else '1h',inclusive='left')
        if len(expected.difference(actual)) or actual.has_duplicates:raise RuntimeError('Candle gap or duplicate')
        quality.append(dict(symbol=symbol,interval=interval,bars=len(actual),missing=0))
    write_json(OUT/'data_quality.json',quality)
    old_manifest=json.loads((OLD/'run_manifest.json').read_text())
    old_source=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory=C:/crypto_bot_sub','show','cc6c8479:crypto_bot_sub.py'],cwd=ROOT).decode('utf-8-sig')
    hashes=feature_ast((ROOT/'crypto_bot_sub.py').read_text(encoding='utf-8-sig'))
    if hashes!=feature_ast(old_source) or len(hashes)!=4:raise RuntimeError('Feature methods changed; cannot reuse old features')
    if runtime_versions()!=old_manifest['runtime_versions']:raise RuntimeError('Feature runtime changed; recompute instead')
    # Compare the complete feature-producing cache method, not just its callers.
    old_adapter=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory=C:/crypto_bot_sub','show','eb8c52e8:research/search_backtests.py'],cwd=ROOT).decode('utf-8-sig')
    def day_ast(text):
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='DailyDecisionCache')
        return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='day'),include_attributes=False)
    if day_ast(old_adapter)!=day_ast((ROOT/'research/search_backtests.py').read_text()):raise RuntimeError('Daily feature producer changed')
    cache=DailyDecisionCache(frames,cache_dir=OUT/'feature_cache')
    reused=0;computed=0
    for day in pd.date_range(plan['train'][0],plan['continuous'][1],inclusive='left',tz='Asia/Tokyo'):
        for symbol in SYMBOLS:
            raw=frames[symbol,'15min'];raw=raw.loc[raw.timestamp.ge(day-pd.Timedelta(days=2))&raw.timestamp.lt(day+pd.Timedelta(days=1))]
            input_hash=hashlib.sha256(raw.timestamp.array.asi8.tobytes()+raw[['open','high','low','close','volume']].to_numpy(dtype='float64').tobytes()).hexdigest()
            old=OLD/'feature_cache'/f'{symbol}_{day:%Y%m%d}.pkl.gz';meta=old.with_suffix('.meta.json')
            target=cache.cache_dir/old.name
            saved=json.loads(meta.read_text()) if old.exists() and meta.exists() else {}
            if (saved.get('source')==old_manifest['source'] and saved.get('input_hash')==input_hash
                    and saved.get('sha256')==hashlib.sha256(old.read_bytes()).hexdigest()):
                shutil.copy2(old,target)
                write_json(target.with_suffix('.meta.json'),dict(input_hash=input_hash,source=cache.fingerprint,
                    adapter=cache.adapter_hash,sha256=saved['sha256'],reused_from=str(old.relative_to(ROOT)),feature_ast=hashes))
                reused+=1
            else:
                cache.day(symbol,day);cache.days.clear();computed+=1
        if (reused+computed)%140==0:print(json.dumps(dict(phase='features',reused=reused,computed=computed)),flush=True)
    write_json(OUT/'feature_preparation.json',dict(reused=reused,computed=computed,feature_ast=hashes,
        raw_files_verified=len(receipts),old_runtime_matched=True,old_daily_producer_ast_matched=True))
    write_json(OUT/'source_manifest.json',dict(source=code_fingerprint(ROOT),identity=identity(),runtime_versions=runtime_versions(),
        git_commit=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory=C:/crypto_bot_sub','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        research_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}))
    print(json.dumps(dict(phase='prepared',candidates=50,reused=reused,computed=computed,source=code_fingerprint(ROOT))),flush=True)


def verify():
    frames=read_frames(OLD/'raw');cache=DailyDecisionCache(frames,cache_dir=OUT/'feature_cache')
    costs=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text())
    start=pd.Timestamp('2026-10-20',tz='Asia/Tokyo');end=start+pd.Timedelta(days=3)
    results=[]
    print(json.dumps(dict(phase='uncached_real_reference')),flush=True)
    source,seconds=execute(frames,start,end,costs,symbols=['ltc_jpy'])
    result=simulate(frames,start,end,costs,cache,symbols=['ltc_jpy']);compare_core(source,result)
    results.append(dict(kind='uncached_reference',symbols=1,days=3,trades=len(result['trades']),seconds=seconds))
    selected=[CANDIDATE,dict(direction='both',adx_min=30,exit_profile='no_short_lock',tp_multiplier=.8),
              dict(direction='short_only',adx_min=25,exit_profile='no_short_lock',sl_multiplier=1.2)]
    for case in selected:
        source=make_bot(frames,start,end,prepare=case_prepare(cache,case));install_case(source,case)
        with offline_io_guard():source.backtest(3,costs=costs)
        result=simulate(frames,start,end,costs,cache,case);compare_core(source.captured,result)
        results.append(dict(kind='real_source_account_and_orders',case=case,symbols=7,days=3,trades=len(result['trades'])))
        print(json.dumps(dict(phase='verified_case',case=case,trades=len(result['trades']))),flush=True)
    write_json(OUT/'verification.json',dict(identity=identity(),checks=results,all_matched=True))


def initialize(expected):
    global _FRAMES,_CACHE,_COSTS,_IDENTITY
    if identity()!=expected:raise RuntimeError('Worker source identity differs')
    _IDENTITY=expected;_FRAMES=read_frames(OLD/'raw')
    _CACHE=DailyDecisionCache(_FRAMES,cache_dir=OUT/'feature_cache')
    _COSTS=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text())


def job(case,period,stress):
    if identity()!=_IDENTITY:raise RuntimeError('Sources changed during search')
    costs=deepcopy(_COSTS)
    if stress:
        costs['name']+='_double_friction';costs['spread_bps']*=2;costs['slippage_bps']*=2
    start,end=[pd.Timestamp(x,tz='Asia/Tokyo') for x in period]
    begun=time.monotonic();result=simulate(_FRAMES,start,end,costs,_CACHE,case)
    if identity()!=_IDENTITY:raise RuntimeError('Sources changed during computation')
    result.update(case=case,period=period,stress=stress,cost_scenario=costs,run_identity=_IDENTITY,seconds=time.monotonic()-begun)
    return result


def run_phase(pool,name,candidates,period,stress=False):
    folder=OUT/name;folder.mkdir(exist_ok=True)
    results=[];pending={}
    for case in candidates:
        path=folder/(key(case)+('_stress' if stress else '')+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            if result['run_identity']!=identity() or result['case']!=case or result['period']!=period or result['stress']!=stress:
                raise RuntimeError('Saved checkpoint differs')
            results.append(result)
        else:pending[pool.submit(job,case,period,stress)]=path
    for future in as_completed(pending):
        result=future.result();write_json(pending[future],result);results.append(result)
        progress=dict(phase=name,done=len(results),total=len(candidates),stress=stress,case=result['case'],metrics=result['metrics'],seconds=result['seconds'])
        write_json(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    return results


def score(r):
    m=r['metrics'];return m['net_equity_change']/max(m['max_observation_drawdown'],1)


def eligible(r):
    m=r['metrics'];pf=m['profit_factor']
    positive=sum(value>0 for month,value in m['monthly_equity_changes'].items() if month in ('2026-05','2026-06','2026-07'))
    return m['trades']>=30 and m['net_equity_change']>0 and (pf is not None and pf>=1.1) and m['max_observation_drawdown_pct']<=10 and positive>=2 and not m['negative_cash_observed']


def controls(selected):
    return selected + [case for case in (BASELINE,CANDIDATE) if case not in selected]


def run(workers):
    proof=json.loads((OUT/'verification.json').read_text())
    if not proof['all_matched'] or proof['identity']!=identity():raise RuntimeError('Current-source parity verification required')
    plan=read_plan();manifest=dict(identity=identity(),plan=plan,costs=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text()),workers=workers)
    if (OUT/'run_manifest.json').exists() and json.loads((OUT/'run_manifest.json').read_text())!=manifest:
        raise RuntimeError('Existing study configuration differs')
    write_json(OUT/'run_manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(identity(),)) as pool:
        training=run_phase(pool,'train',cases(),plan['train'])
        training.sort(key=lambda r:(not eligible(r),-score(r),key(r['case'])))
        shortlisted=[r['case'] for r in training[:plan['shortlist']]]
        write_json(OUT/'training_selection.json',dict(selected=shortlisted,ranking=[dict(case=r['case'],eligible=eligible(r),score=score(r),metrics=r['metrics']) for r in training]))
        validation=run_phase(pool,'validation',controls(shortlisted),plan['validation'])
        stressed=run_phase(pool,'validation',controls(shortlisted),plan['validation'],True)
        stress_map={key(r['case']):r for r in stressed}
        ranked=sorted([r for r in validation if r['case'] in shortlisted],key=lambda r:(-min(score(r),score(stress_map[key(r['case'])])),key(r['case'])))
        finalists=[r['case'] for r in ranked[:plan['finalists']]]
        write_json(OUT/'validation_selection.json',dict(finalists=finalists,ranking=[dict(case=r['case'],metrics=r['metrics'],stress_metrics=stress_map[key(r['case'])]['metrics']) for r in ranked]))
        auditing=controls(finalists)
        audit=run_phase(pool,'historical_audit',auditing,plan['historical_audit'])
        audit_stress=run_phase(pool,'historical_audit',auditing,plan['historical_audit'],True)
        continuous=run_phase(pool,'continuous',auditing,plan['continuous'])
    compact=lambda rows:[dict(case=r['case'],metrics=r['metrics']) for r in rows]
    write_json(OUT/'final_results.json',dict(completed=True,production_changed=False,identity=identity(),plan=plan,
        training=compact(training),validation=compact(validation),validation_stress=compact(stressed),
        finalists=finalists,historical_audit=compact(audit),historical_audit_stress=compact(audit_stress),continuous=compact(continuous)))
    print(json.dumps(dict(phase='complete',finalists=finalists)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','verify','run']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('workers must be between 1 and 4')
    from research.power import keep_awake
    with keep_awake():
        if args.phase=='prepare':prepare()
        elif args.phase=='verify':verify()
        else:run(args.workers)


if __name__=='__main__':main()
