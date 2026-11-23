"""Functional public-data replay of fixed entry models, never a profit search."""
from collections import OrderedDict
import hashlib
import json
import math

import pandas as pd

from research.entry_models import LEGACY, TREND, model_spec, prepare_model
from research.replay_engine import simulate, compare_core
from research.search_backtests import (ROOT, SYMBOLS, DailyDecisionCache, make_bot,
                                      read_frames, write_json, offline_io_guard)
from strategy_recording import code_fingerprint, runtime_versions


STUDY = ROOT/'outputs/backtests/regimes_20261122'
PUBLIC = ROOT/'outputs/backtests/search_20261122'
OUT = ROOT/'outputs/backtests/entry_models_20261123/verified'
FILES = ('research/entry_models.py','research/verify_entry_models.py',
         'tests/test_entry_models.py','docs/research-entry-models.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class VerifiedCache:
    """Read-only, bounded view of previously certified self-generated caches."""
    prepare = DailyDecisionCache.prepare

    def __init__(self, frames):
        self.frames=frames;self.days=OrderedDict();self.receipts={}
        self.source=code_fingerprint(ROOT)
        self.adapter=sha(ROOT/'research/search_backtests.py')

    def day(self,symbol,observed_at):
        day=pd.Timestamp(observed_at).normalize();key=(symbol,day)
        if key in self.days:return self.days[key]
        path=STUDY/'feature_cache'/f'{symbol}_{day:%Y%m%d}.pkl.gz'
        receipt=read(path.with_suffix('.meta.json'))
        if receipt['source']!=self.source or receipt['adapter']!=self.adapter or receipt['sha256']!=sha(path):
            raise ValueError('Frozen cache source or hash mismatch')
        raw=self.frames[symbol,'15min']
        raw=raw.loc[raw.timestamp.ge(day-pd.Timedelta(days=2)) & raw.timestamp.lt(day+pd.Timedelta(days=1))]
        input_hash=hashlib.sha256(raw.timestamp.array.asi8.tobytes()+raw[['open','high','low','close','volume']].to_numpy(dtype='float64').tobytes()).hexdigest()
        if input_hash!=receipt['input_hash']:raise ValueError('Frozen cache input mismatch')
        self.receipts[str(path.relative_to(ROOT))]=receipt['sha256']
        self.days[key]=pd.read_pickle(path,compression='gzip')
        while len(self.days)>21:self.days.popitem(last=False)
        return self.days[key]


def near(left,right):
    if not math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-6):
        raise AssertionError((left,right))


def audit(result, model, frames, cache, costs, entries):
    """Independent inequalities and fill arithmetic, not the entry evaluator."""
    friction=costs['spread_bps']/20000+costs['slippage_bps']/10000
    closed=0
    for t in result['trades']:
        symbol,side=t['symbol'],t['type'];direction=1 if side=='long' else -1
        at=pd.Timestamp(t['entry_time']);exit_at=pd.Timestamp(t['exit_time'])
        raw=frames[symbol,'15min'].set_index('timestamp')
        entry=float(raw.loc[at,'open'])*(1+direction*friction)
        exit_price=float(raw.loc[exit_at-pd.Timedelta(minutes=15),'close'])*(1-direction*friction)
        near(entry,t['entry_price']);near(exit_price,t['exit_price'])
        gross=direction*t['size']*(exit_price-entry)
        entry_fee=t['size']*entry*costs['fee_bps'][symbol]/10000
        exit_fee=t['size']*exit_price*costs['fee_bps'][symbol]/10000
        near(gross,t['gross_profit']);near(entry_fee,t['entry_fee']);near(exit_fee,t['exit_fee'])
        carry=sum(e['amount'] for e in t['carry_events'])
        near(carry,t['carry_fee']);near(gross-entry_fee-exit_fee-carry,t['profit'])
        closed+=1
    # Include entries still open at the end, not just eventual winners/losers.
    for symbol,side,at in entries:
        frame=cache.day(symbol,at);index=int(frame.timestamp.searchsorted(at))
        pair=frame.iloc[index-2:index]
        assert len(pair)==2 and pair.timestamp.iloc[0]==at-pd.Timedelta(minutes=30)
        assert pair.timestamp.iloc[1]==at-pd.Timedelta(minutes=15)
        for r in pair.itertuples():
            if model==LEGACY:
                assert getattr(r,'buy_signal' if side=='short' else 'sell_signal')
            else:
                assert 25<=r.ADX<=100
                if side=='long':assert r.close>r.EMA_short>r.EMA_long and r.plus_di14>r.minus_di14 and r.MACD_histogram>0
                else:assert r.close<r.EMA_short<r.EMA_long and r.minus_di14>r.plus_di14 and r.MACD_histogram<0
    near(sum(t['profit'] for t in result['trades']),result['valuation']['realized_profit'])
    valuation=result['valuation']
    near(valuation['realized_profit']+valuation['unrealized_profit']-valuation['open_costs_paid'],valuation['net_equity_change'])
    return dict(closed_checked=closed,entries_including_open_checked=len(entries),independent_fill_and_cost_arithmetic=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(STUDY/'source_manifest.json')
    assert manifest['source']==code_fingerprint(ROOT)
    assert manifest['runtime_versions']==runtime_versions()
    previous=read(STUDY/'session_checkpoint.json')['source_files']
    for name,digest in previous.items():assert sha(ROOT/name)==digest
    raw_receipt=read(PUBLIC/'raw_checksums.json')
    for r in raw_receipt:assert sha(PUBLIC/'raw'/r['file'])==r['sha256']
    plan=dict(purpose='functional_verification_not_strategy_selection',period=['2026-05-01','2026-05-04'],
        models=[model_spec(m) for m in (LEGACY,TREND)],symbols=SYMBOLS,initial_capital=200000,trade_size=40000,
        costs=read(ROOT/'configs/backtest_costs_reference.json'),
        no_parameter_search=True,known_historical_period=True,production_accessed=False,
        source=manifest['source'],source_hashes={f:sha(ROOT/f) for f in FILES},
        runtime=runtime_versions(),public_raw_files_verified=len(raw_receipt),
        checks=['legacy_exact_reference_parity','entry_model_independent_inequalities',
                'exit_history_unchanged','cache_immutable','independent_fill_cost_and_cash_arithmetic'])
    plan_path=OUT/'verification_plan.json'
    if plan_path.exists() and read(plan_path)!=plan:raise RuntimeError('Verification source/plan changed; choose a new output folder')
    write_json(plan_path,plan)
    start,end=[pd.Timestamp(s,tz='Asia/Tokyo') for s in plan['period']]
    checks=[]
    with offline_io_guard():
        frames=read_frames(PUBLIC/'raw');cache=VerifiedCache(frames)
        for model in (LEGACY,TREND):
            prepare=prepare_model(cache,model);observations=[]
            def checked_prepare(bot,symbol,intraday,hourly,at):
                result,hours=prepare(bot,symbol,intraday,hourly,at)
                if pd.Timestamp(at)!=intraday.timestamp.iloc[-1]:
                    original,_=cache.prepare(bot,symbol,intraday,hourly,at)
                    pd.testing.assert_frame_equal(result,original,check_exact=True)
                    observations.append(at)
                return result,hours
            bot=make_bot(frames,start,end,prepare=checked_prepare)
            entries=[]
            bot.log_entry=lambda symbol,side,price,at,*args:entries.append((symbol,side,pd.Timestamp(at)))
            bot.backtest(3,costs=plan['costs'])
            assert bot.captured['trades'] and observations, 'Functional verification requires real completed trades'
            if model==LEGACY:
                reference=simulate(frames,start,end,plan['costs'],cache)
                compare_core(bot.captured,reference)
            audit_result=audit(bot.captured,model,frames,cache,plan['costs'],entries)
            write_json(OUT/f'{model}.json',dict(model_spec=model_spec(model),result=bot.captured,entry_events=entries))
            checks.append(dict(model=model,**audit_result,exit_observations_verified=len(observations),
                               legacy_reference_matched=model==LEGACY,
                               net_equity_change=bot.captured['valuation']['net_equity_change']))
            print(json.dumps(checks[-1]),flush=True)
        for name,digest in cache.receipts.items():assert sha(ROOT/name)==digest
    assert code_fingerprint(ROOT)==manifest['source']
    for name,digest in previous.items():assert sha(ROOT/name)==digest
    for name,digest in plan['source_hashes'].items():assert sha(ROOT/name)==digest
    write_json(OUT/'verification.json',dict(all_passed=True,checks=checks,cache_receipts=cache.receipts,
        production_and_previous_research_sources_unchanged=True,production_accessed=False,
        limitations=['Short known historical period for functional checks only, not profitability evidence.',
                     'Bitbank OHLC proxy, no historical SR, scenario costs, no tick-level fill ordering.']))


if __name__=='__main__':
    main()
