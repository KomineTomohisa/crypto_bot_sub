"""Run the preregistered chronological study; no live trading or parameter writes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import pandas as pd

from research.search_backtests import ROOT, SYMBOLS, DailyDecisionCache, read_frames, write_json, code_fingerprint
from research.replay_engine import simulate
from strategy_recording import runtime_versions

OUT=ROOT/'outputs/backtests/search_20261122'
_FRAMES=None;_CACHE=None;_COSTS=None
_IDENTITY=None


def study_identity():
    files=['research/search_backtests.py','research/replay_engine.py','research/run_search.py','research/power.py',
           'configs/backtest_costs_reference.json','outputs/backtests/search_20261122/study_plan.json',
           'outputs/backtests/search_20261122/raw_checksums.json']
    values={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    values['source']=code_fingerprint(ROOT)
    values['runtime']=runtime_versions()
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def initialize(identity):
    global _FRAMES,_CACHE,_COSTS,_IDENTITY
    if study_identity()!=identity:raise RuntimeError('Worker study identity differs')
    _IDENTITY=identity
    _FRAMES=read_frames(OUT/'raw');_CACHE=DailyDecisionCache(_FRAMES)
    _COSTS=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text())


def identifier(case):
    return f"{case['direction']}_adx{case['adx_min']}_{case['exit_profile']}"


def job(case,period,stress):
    if study_identity()!=_IDENTITY:raise RuntimeError('Research sources changed')
    costs=deepcopy(_COSTS)
    if stress:
        costs['name']+='_double_spread_slippage'
        costs['spread_bps']*=2;costs['slippage_bps']*=2
    start,end=[pd.Timestamp(day,tz='Asia/Tokyo') for day in period]
    begun=time.monotonic()
    result=simulate(_FRAMES,start,end,costs,_CACHE,case)
    if study_identity()!=_IDENTITY:raise RuntimeError('Research sources changed during computation')
    result.update(case=case,period=period,cost_scenario=costs,stress=stress,seconds=time.monotonic()-begun,run_identity=_IDENTITY)
    return result


def eligible(result,period):
    m=result['metrics'];p=m['profit_factor']
    months=set(pd.date_range(period[0],period[1],freq='MS',inclusive='left').strftime('%Y-%m'))
    positive=sum(value>0 for month,value in m['monthly_equity_changes'].items() if month in months)
    pf_ok=p>=1.1 if p is not None else m['win_rate_pct']>0
    return (m['trades']>=30 and m['net_equity_change']>0 and pf_ok
            and m['max_observation_drawdown_pct']<=10 and not m['negative_cash_observed'] and positive>=2)


def score(result):
    m=result['metrics']
    return m['net_equity_change']/max(m['max_observation_drawdown'],1)


def run_phase(pool,name,cases,period,stress=False):
    folder=OUT/name;folder.mkdir(exist_ok=True)
    futures={};results=[]
    for case in cases:
        path=folder/(identifier(case)+('_stress' if stress else '')+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            if (result['case']!=case or result['period']!=period or result['stress']!=stress
                    or result.get('run_identity')!=study_identity()):
                raise RuntimeError('Saved study job differs')
            results.append(result)
        else:
            futures[pool.submit(job,case,period,stress)]=path
    for future in as_completed(futures):
        result=future.result();write_json(futures[future],result);results.append(result)
        print(json.dumps(dict(phase=name,done=len(results),total=len(cases),stress=stress,
            case=result['case'],metrics=result['metrics'],seconds=result['seconds'])),flush=True)
        write_json(OUT/'progress.json',dict(phase=name,done=len(results),total=len(cases),stress=stress))
    return results


def validate_data(frames):
    checks=[]
    start=pd.Timestamp('2026-04-29',tz='Asia/Tokyo');end=pd.Timestamp('2026-11-22',tz='Asia/Tokyo')
    for (symbol,interval),frame in frames.items():
        actual=pd.DatetimeIndex(frame.loc[frame.timestamp.ge(start)&frame.timestamp.lt(end),'timestamp'])
        expected=pd.date_range(start,end,freq='15min' if interval=='15min' else '1h',inclusive='left')
        missing=expected.difference(actual);extra=actual.difference(expected)
        checks.append(dict(symbol=symbol,interval=interval,bars=len(actual),missing=len(missing),extra=len(extra)))
        if len(missing) or len(extra):raise RuntimeError('Incomplete public candles: '+symbol+' '+interval)
    write_json(OUT/'data_quality.json',checks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    plan=json.loads((OUT/'study_plan.json').read_text())
    frames=read_frames(OUT/'raw');validate_data(frames)
    expected=json.loads((OUT/'raw_checksums.json').read_text())
    for item in expected:
        if hashlib.sha256((OUT/'raw'/item['file']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Raw file changed')
    if args.prepare_only:
        cache=DailyDecisionCache(frames);done=0
        for day in pd.date_range(plan['data_start'],plan['end_exclusive'],inclusive='left',tz='Asia/Tokyo'):
            for symbol in SYMBOLS:
                cache.day(symbol,day);done+=1
            if done%70==0:
                print(json.dumps(dict(phase='prepare_features',done=done,total=205*7)),flush=True)
                cache.days.clear()
        write_json(OUT/'feature_preparation.json',dict(days=done,source=code_fingerprint(ROOT),complete=True))
        return
    verification=json.loads((OUT/'acceleration_verification.json').read_text())
    if not verification['all_matched'] or verification['source']!=code_fingerprint(ROOT):
        raise RuntimeError('Source verification missing or stale')
    for name,digest in verification['research_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Verified research engine changed')
    c=plan['candidates']
    cases=[dict(direction=d,adx_min=a,exit_profile=e) for d,a,e in product(c['direction'],c['adx_min'],c['exit_profile'])]
    if len(cases)!=36:raise RuntimeError('Candidate grid differs')
    identity=study_identity()
    manifest=dict(plan=plan,source=code_fingerprint(ROOT),engine=verification['research_hashes'],run_identity=identity,
        costs=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text()),workers=args.workers,
        runtime_versions=runtime_versions(),execution_engine='research_event_loop_shared_strategy_and_costs',
        continuous_diagnostic='After finalists are frozen; not used for parameter selection. May 1 through Nov 21 with positions carried.')
    path=OUT/'run_manifest.json'
    if path.exists() and json.loads(path.read_text()).get('run_identity')!=identity:
        raise RuntimeError('Existing study used different sources; use a new output directory')
    write_json(path,manifest)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(identity,)) as pool:
        train=run_phase(pool,'train',cases,plan['train'])
        train.sort(key=lambda r:(not eligible(r,plan['train']),-score(r),identifier(r['case'])))
        selected=[r['case'] for r in train[:5]]
        write_json(OUT/'training_selection.json',dict(ranking=[dict(case=r['case'],eligible=eligible(r,plan['train']),score=score(r),metrics=r['metrics']) for r in train],selected=selected))
        baseline=dict(direction='both',adx_min=0,exit_profile='baseline')
        controls=selected if baseline in selected else selected+[baseline]
        validation_all=run_phase(pool,'validation',controls,plan['validation'])
        stress=run_phase(pool,'validation',controls,plan['validation'],True)
        validation=[r for r in validation_all if r['case'] in selected]
        stressed={identifier(r['case']):r for r in stress}
        validation.sort(key=lambda r:(-(min(score(r),score(stressed[identifier(r['case'])]))),identifier(r['case'])))
        finalists=[r['case'] for r in validation[:3]]
        write_json(OUT/'validation_selection.json',dict(finalists=finalists,
            ranking=[dict(case=r['case'],metrics=r['metrics'],stress_metrics=stressed[identifier(r['case'])]['metrics']) for r in validation]))
        audits=finalists if baseline in finalists else finalists+[baseline]
        audit=run_phase(pool,'historical_audit',audits,plan['historical_audit'])
        audit_stress=run_phase(pool,'historical_audit',audits,plan['historical_audit'],True)
        fresh=run_phase(pool,'fresh_short_audit',audits,plan['fresh_short_audit'])
        continuous=run_phase(pool,'continuous_diagnostic',audits,[plan['data_start'],plan['end_exclusive']])
    write_json(OUT/'final_results.json',dict(plan=plan,train=[dict(case=r['case'],metrics=r['metrics'],eligible=eligible(r,plan['train'])) for r in train],
        validation=[dict(case=r['case'],metrics=r['metrics']) for r in validation],validation_stress=[dict(case=r['case'],metrics=r['metrics']) for r in stress],
        finalists=finalists,audit=[dict(case=r['case'],metrics=r['metrics']) for r in audit],
        baseline_validation=[dict(case=r['case'],metrics=r['metrics']) for r in validation_all if r['case']==baseline],
        audit_stress=[dict(case=r['case'],metrics=r['metrics']) for r in audit_stress],fresh_short_audit=[dict(case=r['case'],metrics=r['metrics']) for r in fresh],
        continuous_diagnostic=[dict(case=r['case'],metrics=r['metrics']) for r in continuous],
        completed=True,production_changed=False,global_optimum_claim=False))
    print(json.dumps(dict(phase='study_complete',finalists=finalists)),flush=True)


if __name__=='__main__':
    from research.power import keep_awake
    with keep_awake():main()
