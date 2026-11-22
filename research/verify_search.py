"""Freeze and verify accelerated decisions on real, cached public candles."""
import hashlib
import json
from pathlib import Path
import sys
import time

from research.search_backtests import *
from research.replay_engine import *


def main():
    out=ROOT/'outputs/backtests/search_20261122'
    frames=read_frames(ROOT/'outputs/backtests/20261119_recent30/raw')
    costs=json.loads((ROOT/'configs/backtest_costs_reference.json').read_text())
    cache=DailyDecisionCache(frames)
    start=pd.Timestamp('2026-10-20',tz='Asia/Tokyo');end=start+pd.Timedelta(days=3)
    checks=[]
    print(json.dumps(dict(phase='uncached_reference_check')),flush=True)
    source,seconds=execute(frames,start,end,costs,symbols=['ltc_jpy'])
    replay=simulate(frames,start,end,costs,cache,symbols=['ltc_jpy'])
    compare_core(source,replay)
    checks.append(dict(kind='uncached_source',days=3,symbols=1,trades=len(replay['trades']),seconds=seconds,matched=True))
    cases=[BASELINE,dict(direction='short_only',adx_min=25,exit_profile='no_short_lock'),
           dict(direction='both',adx_min=30,exit_profile='tp_120pct')]
    for index,case in enumerate(cases):
        print(json.dumps(dict(phase='cached_source_case',case=case)),flush=True)
        source=make_bot(frames,start,end,prepare=case_prepare(cache,case))
        install_case(source,case)
        with offline_io_guard():source.backtest(3,costs=costs)
        replay=simulate(frames,start,end,costs,cache,case)
        compare_core(source.captured,replay)
        checks.append(dict(kind='source_account_and_order_path',case=case,days=3,symbols=7,trades=len(replay['trades']),matched=True))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'research/search_backtests.py',ROOT/'research/replay_engine.py']}
    write_json(out/'acceleration_verification.json',dict(checks=checks,all_matched=True,source=code_fingerprint(ROOT),research_hashes=hashes))
    print(json.dumps(dict(phase='verification_complete',checks=len(checks))),flush=True)


if __name__=='__main__':main()
