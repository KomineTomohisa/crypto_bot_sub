"""Broad entry-regime search, training-only refinement and frozen audits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from research import expanded_search as engine
from research.regime_filters import CONTEXTS, BANDS, SESSIONS
from research.search_backtests import ROOT,SYMBOLS,DailyDecisionCache,read_frames,write_json,make_bot,execute,offline_io_guard
from research.replay_engine import case_prepare,install_case,simulate,compare_core
from strategy_recording import code_fingerprint,runtime_versions

OUT=ROOT/'outputs/backtests/regimes_20261122'
PRIOR=ROOT/'outputs/backtests/expanded_20261122'
RAW=ROOT/'outputs/backtests/search_20261122'
BASELINE=dict(direction='both',adx_min=0,exit_profile='baseline')
ANCHORS=[dict(direction='both',adx_min=30,exit_profile='no_short_lock'),
         dict(direction='both',adx_min=35,exit_profile='no_short_lock'),
         dict(direction='short_only',adx_min=30,exit_profile='no_short_lock',tp_multiplier=1.,sl_multiplier=.8)]
EXITS=[(1.,1.),(.8,1.),(1.2,1.),(1.,.8),(1.,1.2)]


class BoundedFeatureCache(DailyDecisionCache):
    """Keep at most three days across seven symbols; disk receipts stay intact."""
    max_days = 21

    def day(self, symbol, observed_at):
        result=super().day(symbol,observed_at)
        while len(self.days)>self.max_days:
            self.days.pop(next(iter(self.days)))
        return result


def case_key(case):
    return (f"{case['direction']}_adx{case['adx_min']}_{case['exit_profile']}_tp{case.get('tp_multiplier',1):g}_sl{case.get('sl_multiplier',1):g}"
            +f"_{case.get('entry_context','none')}_{case.get('atr_band','all')}_{case.get('session','all')}")


def configure():
    engine.OUT=OUT
    engine.key=case_key
    engine.DailyDecisionCache=BoundedFeatureCache
    engine.FILES=['research/regime_search.py','research/regime_filters.py','research/expanded_search.py',
                  'research/search_backtests.py','research/replay_engine.py','research/power.py','configs/backtest_costs_reference.json']


def profiles():
    values=[]
    for anchor,context,band,session in product(ANCHORS,CONTEXTS,BANDS,SESSIONS):
        case=deepcopy(anchor)
        for key,value,default in [('entry_context',context,'none'),('atr_band',band,'all'),('session',session,'all')]:
            if value!=default:case[key]=value
        values.append(case)
    assert len(values)==216 and len({case_key(c) for c in values})==216
    return values


def controls(cases):
    return cases+[c for c in [BASELINE,*ANCHORS] if c not in cases]


def prepare():
    configure();OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(registered_at=datetime.now(timezone.utc).isoformat(),screen_candidates=216,
        anchors=ANCHORS,contexts=CONTEXTS,atr_bands=BANDS,sessions=SESSIONS,
        screen=['2026-05-01','2026-06-01'],train=['2026-05-01','2026-08-01'],validation=['2026-08-01','2026-10-01'],
        historical_audit=['2026-10-01','2026-11-22'],continuous=['2026-05-01','2026-11-22'],
        monthly_diagnostics=[['2026-08-01','2026-09-01'],['2026-09-01','2026-10-01']],
        screen_shortlist=12,exit_refinements=EXITS,training_shortlist=8,finalists=3,
        symbols=SYMBOLS,initial_capital=200000,trade_size=40000,
        screening='>=10 trades, net>0, PF>=1.05, DD<=10%, no negative cash; eligible first then net/DD; keep 12 distinct completed-trade paths',
        training='>=30 trades, net>0, PF>=1.1, DD<=10%, >=2 positive May-July months, no negative cash; eligible first then net/DD',
        validation_selection='Freeze top 3 by eligible-positive first, then worst(normal,double-friction) net/absolute-DD; no audit reselection',
        limitations=['All dates are known from prior studies; no untouched or future-data profitability claim.',
                     'Monthly diagnostics occur after selection and are not untouched walk-forward tests.',
                     'Bitbank OHLC proxy, no historical SR/margin/liquidation, fixed scenario fees and carry, observation-grid DD.',
                     '216 profiles plus training refinement increase multiple-comparison bias; forward shadow evidence remains necessary.'])
    path=OUT/'study_plan.json'
    if path.exists():
        existing=json.loads(path.read_text());plan['registered_at']=existing['registered_at']
        if json.loads(json.dumps(plan))!=existing:raise RuntimeError('Registered plan changed')
    write_json(path,plan)
    prior=json.loads((PRIOR/'session_checkpoint.json').read_text())
    if not prior['study_completed'] or not prior['independent_audit_passed']:raise RuntimeError('Previous verified study required')
    receipt=json.loads((RAW/'raw_checksums.json').read_text())
    for row in receipt:
        if hashlib.sha256((RAW/'raw'/row['file']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Public candle changed')
    old_source=json.loads((PRIOR/'source_manifest.json').read_text())
    if old_source['source']!=code_fingerprint(ROOT) or old_source['runtime_versions']!=runtime_versions():
        raise RuntimeError('Feature/runtime source changed; recompute first')
    current_adapter=hashlib.sha256((ROOT/'research/search_backtests.py').read_bytes()).hexdigest()
    if old_source['research_sha256']['research/search_backtests.py']!=current_adapter:raise RuntimeError('Feature adapter changed')
    destination=OUT/'feature_cache';destination.mkdir(exist_ok=True)
    copied=0
    for file in (PRIOR/'feature_cache').glob('*.pkl.gz'):
        metadata=file.with_suffix('.meta.json');value=json.loads(metadata.read_text())
        if value['source']!=code_fingerprint(ROOT) or value['adapter']!=current_adapter or value['sha256']!=hashlib.sha256(file.read_bytes()).hexdigest():
            raise RuntimeError('Feature-cache receipt differs')
        shutil.copy2(file,destination/file.name);shutil.copy2(metadata,destination/metadata.name);copied+=1
    if copied!=1435:raise RuntimeError('Missing daily features')
    write_json(OUT/'source_manifest.json',dict(identity=engine.identity(),source=code_fingerprint(ROOT),runtime_versions=runtime_versions(),
        public_raw_files_verified=len(receipt),daily_features_copied=copied,
        research_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in engine.FILES},
        base_commit=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory=C:/crypto_bot_sub','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    print(json.dumps(dict(phase='prepared',profiles=216,features=copied,raw_files=len(receipt))),flush=True)


def verify():
    configure();frames=read_frames(RAW/'raw');cache=BoundedFeatureCache(frames,cache_dir=OUT/'feature_cache')
    costs=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text())
    start=pd.Timestamp('2026-10-20',tz='Asia/Tokyo');end=start+pd.Timedelta(days=3)
    checks=[]
    print(json.dumps(dict(phase='uncached_reference')),flush=True)
    source,seconds=execute(frames,start,end,costs,symbols=['ltc_jpy'])
    result=simulate(frames,start,end,costs,cache,symbols=['ltc_jpy']);compare_core(source,result)
    checks.append(dict(kind='uncached_source',days=3,symbols=1,trades=len(result['trades']),seconds=seconds))
    candidates=[dict(ANCHORS[0],entry_context='ema_fade'),
                dict(ANCHORS[1],entry_context='di_follow',atr_band='quiet'),
                dict(ANCHORS[2],atr_band='normal',session='jst_17_01')]
    for case in candidates:
        source=make_bot(frames,start,end,prepare=case_prepare(cache,case));install_case(source,case)
        with offline_io_guard():source.backtest(3,costs=costs)
        result=simulate(frames,start,end,costs,cache,case);compare_core(source.captured,result)
        checks.append(dict(kind='source_account_orders',case=case,trades=len(result['trades']),days=3,symbols=7))
        print(json.dumps(dict(phase='verified_case',case=case,trades=len(result['trades']))),flush=True)
    if not any(c['trades']>0 for c in checks[1:]):raise RuntimeError('Regime verification needs nonempty fills')
    write_json(OUT/'verification.json',dict(identity=engine.identity(),all_matched=True,checks=checks))


def initialize(expected):
    configure();engine.initialize(expected)


def screen_ok(result):
    m=result['metrics'];pf=m['profit_factor']
    return m['trades']>=10 and m['net_equity_change']>0 and pf is not None and pf>=1.05 and m['max_observation_drawdown_pct']<=10 and not m['negative_cash_observed']


def order_path(result):
    rows=[{k:t[k] for k in ('symbol','type','entry_time','exit_time','entry_price','exit_price','size','exit_reason')} for t in result['trades']]
    from strategy_recording import digest
    return digest(rows)


def refine(selected):
    values={}
    for case,(tp,sl) in product(selected,EXITS):
        result=dict(case)
        result['tp_multiplier']=round(result.get('tp_multiplier',1.)*tp,6)
        result['sl_multiplier']=round(result.get('sl_multiplier',1.)*sl,6)
        # Canonical defaults preserve stable case IDs and control matching.
        if result['tp_multiplier']==1 and 'tp_multiplier' not in case:result.pop('tp_multiplier')
        if result['sl_multiplier']==1 and 'sl_multiplier' not in case:result.pop('sl_multiplier')
        values[case_key(result)]=result
    return list(values.values())


def run(workers):
    configure();proof=json.loads((OUT/'verification.json').read_text())
    if not proof['all_matched'] or proof['identity']!=engine.identity():raise RuntimeError('Current-source parity proof required')
    plan=json.loads((OUT/'study_plan.json').read_text())
    manifest=dict(identity=engine.identity(),plan=plan,workers=workers,costs=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text()))
    path=OUT/'run_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Run configuration changed')
    write_json(path,manifest)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(engine.identity(),)) as pool:
        screen=engine.run_phase(pool,'screen',profiles()+[BASELINE],plan['screen'])
        screen.sort(key=lambda r:(not screen_ok(r),-engine.score(r),case_key(r['case'])))
        selected=[];seen=set()
        for row in screen:
            if row['case']==BASELINE:continue
            if row['metrics']['trades']<10 or row['metrics']['negative_cash_observed']:continue
            fingerprint=order_path(row)
            if fingerprint in seen:continue
            seen.add(fingerprint);selected.append(row['case'])
            if len(selected)==plan['screen_shortlist']:break
        write_json(OUT/'screen_selection.json',dict(selected=selected,unique_order_paths=len({order_path(r) for r in screen}),
            ranking=[dict(case=r['case'],eligible=screen_ok(r),score=engine.score(r),metrics=r['metrics']) for r in screen]))
        training=engine.run_phase(pool,'train',controls(refine(selected)),plan['train'])
        training.sort(key=lambda r:(not engine.eligible(r),-engine.score(r),case_key(r['case'])))
        shortlisted=[];seen=set()
        for row in training:
            if row['metrics']['trades']<30 or row['metrics']['negative_cash_observed']:continue
            fingerprint=order_path(row)
            if fingerprint in seen:continue
            seen.add(fingerprint);shortlisted.append(row['case'])
            if len(shortlisted)==plan['training_shortlist']:break
        write_json(OUT/'training_selection.json',dict(selected=shortlisted,ranking=[dict(case=r['case'],eligible=engine.eligible(r),score=engine.score(r),metrics=r['metrics']) for r in training]))
        validation=engine.run_phase(pool,'validation',controls(shortlisted),plan['validation'])
        stress=engine.run_phase(pool,'validation',controls(shortlisted),plan['validation'],True)
        stressed={case_key(r['case']):r for r in stress}
        chosen=[r for r in validation if r['case'] in shortlisted]
        def passes(row):
            m=row['metrics'];sm=stressed[case_key(row['case'])]['metrics']
            return all(x['trades']>=30 and x['net_equity_change']>0 and x['profit_factor'] is not None and x['profit_factor']>=1 and x['max_observation_drawdown_pct']<=10 and not x['negative_cash_observed'] for x in (m,sm))
        chosen.sort(key=lambda r:(not passes(r),-min(engine.score(r),engine.score(stressed[case_key(r['case'])])),case_key(r['case'])))
        finalists=[r['case'] for r in chosen[:plan['finalists']]]
        write_json(OUT/'validation_selection.json',dict(finalists=finalists,ranking=[dict(case=r['case'],passed=passes(r),metrics=r['metrics'],stress_metrics=stressed[case_key(r['case'])]['metrics']) for r in chosen]))
        targets=controls(finalists)
        audit=engine.run_phase(pool,'historical_audit',targets,plan['historical_audit'])
        audit_stress=engine.run_phase(pool,'historical_audit',targets,plan['historical_audit'],True)
        months=[]
        for index,period in enumerate(plan['monthly_diagnostics']):
            months.append(engine.run_phase(pool,'monthly_'+str(index),targets,period))
        continuous=engine.run_phase(pool,'continuous',targets,plan['continuous'])
    compact=lambda rows:[dict(case=r['case'],metrics=r['metrics']) for r in rows]
    write_json(OUT/'final_results.json',dict(completed=True,identity=engine.identity(),production_changed=False,plan=plan,
        screen=compact(screen),training=compact(training),validation=compact(validation),validation_stress=compact(stress),
        finalists=finalists,historical_audit=compact(audit),historical_audit_stress=compact(audit_stress),
        monthly_diagnostics=[compact(rows) for rows in months],continuous=compact(continuous)))
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
