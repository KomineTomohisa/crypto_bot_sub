"""Independently check every recorded fill/cost and build a compact comparison."""
import hashlib
import json
import math

import pandas as pd

from research.search_backtests import ROOT, read_frames, write_json


OUT=ROOT/'outputs/backtests/search_20261122'


def close(a,b,label):
    if not math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-6):
        raise AssertionError((label,a,b))


def main():
    final=json.loads((OUT/'final_results.json').read_text())
    if not final['completed']:raise RuntimeError('Search is not complete')
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    frames=read_frames(OUT/'raw')
    prices={symbol:frame.set_index('timestamp') for (symbol,interval),frame in frames.items() if interval=='15min'}
    jobs=[];trade_count=0
    folders=['train','validation','historical_audit','fresh_short_audit','continuous_diagnostic']
    for folder in folders:
        for path in sorted((OUT/folder).glob('*.json')):
            result=json.loads(path.read_text())
            if result['run_identity']!=manifest['run_identity']:raise AssertionError('Study identity differs')
            costs=result['cost_scenario'];friction=costs['spread_bps']/20000+costs['slippage_bps']/10000
            realized=0.;entry_exit_fees=0.;carry=0.
            for trade in result['trades']:
                trade_count+=1
                at=pd.Timestamp(trade['entry_time']);end=pd.Timestamp(trade['exit_time'])
                if not (pd.Timestamp(result['period'][0],tz='Asia/Tokyo')<=at<end<=pd.Timestamp(result['period'][1],tz='Asia/Tokyo')):
                    raise AssertionError('Trade outside period')
                if trade['type'] not in ('long','short'):raise AssertionError('Unknown direction')
                direction=1 if trade['type']=='long' else -1
                source=prices[trade['symbol']]
                entry_reference=float(source.loc[at,'open'])
                exit_reference=float(source.loc[end-pd.Timedelta(minutes=15),'close'])
                close(trade['entry_price'],entry_reference*(1+direction*friction),'entry reference fill')
                close(trade['exit_price'],exit_reference*(1-direction*friction),'exit reference fill')
                size=trade['size'];rate=costs['fee_bps'][trade['symbol']]/10000
                gross=direction*(trade['exit_price']-trade['entry_price'])*size
                entry_fee=size*trade['entry_price']*rate;exit_fee=size*trade['exit_price']*rate
                close(trade['entry_fee'],entry_fee,'entry fee');close(trade['exit_fee'],exit_fee,'exit fee')
                accrued=0.
                expected_cutoff=at.normalize()+pd.Timedelta(hours=6)
                if expected_cutoff<=at:expected_cutoff+=pd.Timedelta(days=1)
                expected_times=list(pd.date_range(expected_cutoff,end,freq='1D')) if expected_cutoff<=end else []
                if [pd.Timestamp(e['at']) for e in trade['carry_events']]!=expected_times:
                    raise AssertionError('Carry cutoff missing or duplicated')
                for event in trade['carry_events']:
                    cutoff=pd.Timestamp(event['at'])
                    if not at<cutoff<=end or pd.Timestamp(event['reference_at'])>cutoff:
                        raise AssertionError('Noncausal carry event')
                    amount=event['reference_price']*size*costs['carry_bps_per_day']/10000
                    reference_at=pd.Timestamp(event['reference_at'])
                    reference=(source.loc[at,'open'] if reference_at==at
                               else source.loc[reference_at-pd.Timedelta(minutes=15),'close'])
                    close(event['reference_price'],reference,'carry past reference price')
                    close(event['amount'],amount,'carry amount');accrued+=amount
                close(trade['carry_fee'],accrued,'carry total')
                net=gross-entry_fee-exit_fee-accrued
                close(trade['gross_profit'],gross,'gross P/L');close(trade['profit'],net,'net P/L')
                close(trade['profit_pct'],net/(size*trade['entry_price'])*100,'net percent')
                realized+=net;entry_exit_fees+=entry_fee+exit_fee;carry+=accrued
            v=result['valuation']
            close(v['realized_profit'],realized,'closed P/L total')
            open_costs=sum(p['entry_fee']+p['carry_fee'] for p in result['open_positions'].values())
            close(v['open_costs_paid'],open_costs,'open costs')
            close(v['execution_fees_paid'],entry_exit_fees+sum(p['entry_fee'] for p in result['open_positions'].values()),'all execution fees')
            close(v['carry_fees_paid'],carry+sum(p['carry_fee'] for p in result['open_positions'].values()),'all carry fees')
            close(v['ending_available_cash']+v['reserved_capital'],200000+realized-open_costs,'cash identity')
            close(v['ending_equity'],200000+v['net_equity_change'],'equity identity')
            close(sum(result['metrics']['monthly_equity_changes'].values()),v['net_equity_change'],'monthly equity changes')
            jobs.append(dict(file=str(path.relative_to(OUT)),trades=len(result['trades']),verified=True))
    for row in json.loads((OUT/'raw_checksums.json').read_text()):
        if hashlib.sha256((OUT/'raw'/row['file']).read_bytes()).hexdigest()!=row['sha256']:
            raise AssertionError('Raw data changed')
    write_json(OUT/'independent_audit.json',dict(all_verified=True,jobs=len(jobs),recorded_trades_checked=trade_count,
        checks=jobs,raw_files_unchanged=True,production_changed=False,
        audit_source_sha256=hashlib.sha256((ROOT/'research/audit_search_results.py').read_bytes()).hexdigest()))
    winner=final['finalists'][0]
    def select(rows,case):return next(row['metrics'] for row in rows if row['case']==case)
    winner_summary={phase:select(final[phase],winner) for phase in ['validation','validation_stress','audit','audit_stress','fresh_short_audit','continuous_diagnostic']}
    train=next(row for row in final['train'] if row['case']==winner)
    supported=(train['eligible'] and all(winner_summary[k]['net_equity_change']>0 for k in ['validation','validation_stress','audit','audit_stress','continuous_diagnostic']))
    write_json(OUT/'review_summary.json',dict(validation_selected_winner=winner,winner_metrics=winner_summary,
        train_eligible=train['eligible'],positive_across_main_checks=supported,
        conclusion='provisional_research_candidate' if supported else 'no_supported_winner',
        unique_parameter_candidates=36,completed_backtests=len(jobs),independent_audit_passed=True,
        limitations=final['plan']['limitations']+['Period comparisons reset positions and capital; continuous diagnostic preserves positions',
            'Drawdown uses observed quote-grid estimates, not true intrabar or real account equity',
            'The three-day fresh audit is too short for profitability certification',
            'Winner fixed by validation; no re-selection using audit outcomes'] ))
    print(json.dumps(dict(jobs=len(jobs),trades_checked=trade_count,winner=winner,positive_main_checks=supported)),flush=True)


if __name__=='__main__':main()
