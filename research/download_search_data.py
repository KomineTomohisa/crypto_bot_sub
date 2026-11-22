"""Download public Bitbank candles only; never read local credentials or dotenv."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import requests

from research.search_backtests import ROOT, SYMBOLS, write_json


def complete_partition(payload,interval,day):
    try:
        rows=next(c['ohlcv'] for c in payload['data']['candlestick'] if c['type']==interval)
        actual=pd.to_datetime([row[5] for row in rows],unit='ms',utc=True)
        start=pd.to_datetime(day,format='%Y%m%d',utc=True)
        expected=pd.date_range(start,start+pd.Timedelta(days=1),freq='15min' if interval=='15min' else '1h',inclusive='left')
        return payload.get('success')==1 and actual.equals(expected)
    except (KeyError,TypeError,ValueError,IndexError,StopIteration):
        return False


def main():
    out = ROOT/'outputs/backtests/search_20261122'
    raw = out/'raw'; raw.mkdir(parents=True, exist_ok=True)
    plan = dict(
        registered_before_search=True, initial_capital=200000, trade_size=40000,
        data_start='2026-05-01', end_exclusive='2026-11-22',
        train=['2026-05-01','2026-08-01'], validation=['2026-08-01','2026-10-01'],
        historical_audit=['2026-10-01','2026-11-19'], fresh_short_audit=['2026-11-19','2026-11-22'],
        candidates=dict(direction=['both','short_only','long_only'], adx_min=[0,25,30],
                        exit_profile=['baseline','no_short_lock','short_lock_1pct','tp_120pct']),
        candidate_count=36, shortlist_from_train=5, finalists_after_validation=3,
        eligibility=dict(min_closed_trades=30,positive_net=True,profit_factor_min=1.1,
                         max_observation_drawdown_pct=10, min_positive_train_months=2),
        ranking='eligible first; net equity change / max(absolute observation-grid equity drawdown,1)',
        stress='Double assumed spread and slippage; same reference fees and carry',
        limitations=['October-November 18 aggregates were already inspected; historical audit is not untouched data',
                     'Only three recent unseen days available; insufficient to certify future profitability',
                     'Bitbank OHLC proxy, approximate costs, no historical SR or margin/liquidation model'],
    )
    plan_path = out/'study_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Registered plan differs; use a new study directory')
    write_json(plan_path, plan)
    for source in (ROOT/'outputs/backtests/20261119_recent30/raw').glob('*.json'):
        target=raw/source.name
        if not target.exists():
            shutil.copy2(source,target)
    days=pd.date_range('2026-04-28','2026-11-22',inclusive='left')
    jobs=[(s,interval,d.strftime('%Y%m%d')) for s in SYMBOLS for interval in ('15min','1hour') for d in days]
    def fetch(job):
        symbol,interval,day=job
        path=raw/f'{symbol}_{interval}_{day}.json'
        valid=path.exists() and complete_partition(json.loads(path.read_text()),interval,day)
        if not valid:
            url=f'https://public.bitbank.cc/{symbol}/candlestick/{interval}/{day}'
            for attempt in range(4):
                try:
                    reply=requests.get(url,timeout=20);reply.raise_for_status()
                    value=reply.json()
                    if value.get('success') != 1:
                        raise ValueError('Public candle response failed')
                    if not complete_partition(value,interval,day):
                        raise ValueError('Incomplete UTC candle partition')
                    write_json(path,value)
                    break
                except (requests.RequestException,ValueError):
                    if attempt==3:raise
                    time.sleep(attempt+1)
                finally:
                    time.sleep(.25)
        return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(fetch,jobs):
            rows.append(row)
            if len(rows)%100==0:
                print(json.dumps(dict(phase='public_download',done=len(rows),total=len(jobs))),flush=True)
    write_json(out/'raw_checksums.json',rows)
    print(json.dumps(dict(phase='download_complete',files=len(rows))),flush=True)


if __name__=='__main__':main()
