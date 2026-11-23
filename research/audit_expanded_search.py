"""Independent arithmetic audit of every recorded trial and a compact report."""
from collections import defaultdict
import hashlib
import json
import math

import pandas as pd

from research.expanded_search import OUT, OLD, BASELINE, CANDIDATE, identity, key, eligible
from research.search_backtests import ROOT, read_frames, write_json


def close(actual, expected, label):
    if not math.isclose(float(actual),float(expected),rel_tol=1e-11,abs_tol=1e-6):
        raise AssertionError(label)


def audit(result, prices):
    costs=result['cost_scenario'];friction=costs['spread_bps']/20000+costs['slippage_bps']/10000
    realized=fees=carry=0.
    for trade in result['trades']:
        at=pd.Timestamp(trade['entry_time']);end=pd.Timestamp(trade['exit_time'])
        assert pd.Timestamp(result['period'][0],tz='Asia/Tokyo')<=at<end<=pd.Timestamp(result['period'][1],tz='Asia/Tokyo')
        direction=1 if trade['type']=='long' else -1
        source=prices[trade['symbol']];size=trade['size'];rate=costs['fee_bps'][trade['symbol']]/10000
        close(trade['entry_price'],source.loc[at,'open']*(1+direction*friction),'entry reference')
        close(trade['exit_price'],source.loc[end-pd.Timedelta(minutes=15),'close']*(1-direction*friction),'exit reference')
        gross=direction*(trade['exit_price']-trade['entry_price'])*size
        ef=size*trade['entry_price']*rate;xf=size*trade['exit_price']*rate
        close(trade['entry_fee'],ef,'entry fee');close(trade['exit_fee'],xf,'exit fee')
        cutoff=at.normalize()+pd.Timedelta(hours=6)
        if cutoff<=at:cutoff+=pd.Timedelta(days=1)
        expected=list(pd.date_range(cutoff,end,freq='1D')) if cutoff<=end else []
        assert [pd.Timestamp(e['at']) for e in trade['carry_events']]==expected
        accrued=0.
        for event in trade['carry_events']:
            cutoff=pd.Timestamp(event['at']);ref_at=pd.Timestamp(event['reference_at'])
            assert at<cutoff<=end and ref_at<=cutoff
            ref=source.loc[at,'open'] if ref_at==at else source.loc[ref_at-pd.Timedelta(minutes=15),'close']
            amount=event['reference_price']*size*costs['carry_bps_per_day']/10000
            close(event['reference_price'],ref,'past carry reference');close(event['amount'],amount,'carry amount')
            accrued+=amount
        close(trade['carry_fee'],accrued,'carry total')
        net=gross-ef-xf-accrued
        close(trade['gross_profit'],gross,'gross profit');close(trade['profit'],net,'net profit')
        close(trade['profit_pct'],net/(size*trade['entry_price'])*100,'profit percent')
        realized+=net;fees+=ef+xf;carry+=accrued
    value=result['valuation'];opened=result['open_positions']
    open_costs=sum(p['entry_fee']+p['carry_fee'] for p in opened.values())
    close(value['realized_profit'],realized,'realized sum')
    close(value['open_costs_paid'],open_costs,'open costs')
    close(value['execution_fees_paid'],fees+sum(p['entry_fee'] for p in opened.values()),'all fees')
    close(value['carry_fees_paid'],carry+sum(p['carry_fee'] for p in opened.values()),'all carry')
    close(value['ending_available_cash']+value['reserved_capital'],200000+realized-open_costs,'cash identity')
    close(value['ending_equity'],200000+value['net_equity_change'],'equity identity')
    close(sum(result['metrics']['monthly_equity_changes'].values()),value['net_equity_change'],'monthly sum')
    return len(result['trades'])


def main():
    final=json.loads((OUT/'final_results.json').read_text())
    assert final['completed'] and final['identity']==identity()
    frames=read_frames(OLD/'raw')
    prices={symbol:frame.set_index('timestamp') for (symbol,interval),frame in frames.items() if interval=='15min'}
    jobs=[];count=0;training_paths=set();contributions={}
    for folder in ('train','validation','historical_audit','continuous'):
        for path in sorted((OUT/folder).glob('*.json')):
            result=json.loads(path.read_text());assert result['run_identity']==identity()
            trades=audit(result,prices);count+=trades
            jobs.append(dict(file=str(path.relative_to(OUT)),trades=trades,verified=True))
            if folder=='train':
                trading_path=[{k:t[k] for k in ('symbol','type','entry_time','exit_time','entry_price','exit_price','size','exit_reason')} for t in result['trades']]
                training_paths.add(hashlib.sha256(json.dumps(trading_path,sort_keys=True).encode()).hexdigest())
            if folder=='continuous':
                values=defaultdict(lambda:dict(trades=0,closed_trade_net_profit=0.,wins=0))
                for trade in result['trades']:
                    entry=values[trade['symbol']];entry['trades']+=1;entry['closed_trade_net_profit']+=trade['profit'];entry['wins']+=trade['profit']>0
                contributions[key(result['case'])]=dict(values)
    for receipt in json.loads((OLD/'raw_checksums.json').read_text()):
        assert hashlib.sha256((OLD/'raw'/receipt['file']).read_bytes()).hexdigest()==receipt['sha256']
    winner=final['finalists'][0]
    select=lambda phase,case:next(r['metrics'] for r in final[phase] if r['case']==case)
    phases=['validation','validation_stress','historical_audit','historical_audit_stress','continuous']
    # The two unchanged controls must reproduce the earlier independent study.
    earlier=json.loads((OLD/'final_results.json').read_text())
    control_regression=[]
    for case in (BASELINE,CANDIDATE):
        old=next(r['metrics'] for r in earlier['continuous_diagnostic'] if r['case']==case)
        current=select('continuous',case)
        for field in ('trades','profit_factor','net_equity_change','realized_profit','max_observation_drawdown_pct'):
            close(current[field],old[field],'unchanged control '+field)
        control_regression.append(dict(case=case,matched=True))
    winner_metrics={phase:select(phase,winner) for phase in phases}
    training=next(r for r in final['training'] if r['case']==winner)
    supported=eligible(training) and all(m['net_equity_change']>0 and m['trades']>=30 and not m['negative_cash_observed'] for m in winner_metrics.values())
    write_json(OUT/'independent_audit.json',dict(all_verified=True,jobs=len(jobs),recorded_trades_checked=count,checks=jobs,raw_files_unchanged=True,
        auditor_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    summary=dict(completed_backtests=len(jobs),unique_parameter_candidates=50,unique_training_order_paths=len(training_paths),independent_audit_passed=True,
        winner_fixed_by_validation=winner,winner_metrics=winner_metrics,
        conclusion='provisional_research_candidate_requires_forward_validation' if supported else 'no_supported_winner',
        positive_main_checks=supported,baseline={p:select(p,BASELINE) for p in phases},
        current_shadow_candidate={p:select(p,CANDIDATE) for p in phases},
        finalists=final['finalists'],continuous=final['continuous'],continuous_symbol_contributions=contributions,unchanged_controls_reproduce_previous_study=control_regression,
        limitations=final['plan']['limitations'],production_changed=False)
    write_json(OUT/'review_summary.json',summary)
    legend='条件名: both=両方向、short_only=ショートのみ、long_only=ロングのみ。adx=確定2本双方のADX下限。no_short_lock=Short利益保護OFF。tp/sl=元の利確・損切りまでの幅に掛ける倍率（0.8は20%縮小、1.2は20%拡大）。'
    lines=['# 条件を変更したバックテストの比較','',f'50条件、計{len(jobs)}回。初期資金20万円、1回4万円、7銘柄。費用込みの推計。',
           '',legend,'','純評価損益は含み損益を含む。未決済建玉の将来の決済費用は含まない。PFは決済済み取引の純利益合計÷純損失合計の絶対値。',
           '','条件選択: 5–7月。期間を分けた確認: 8–9月。追加確認: 10月–11月21日。すべて以前に分析した期間であり、未使用データでの保証ではない。',
           '','|条件|全期間の純評価損益|PF|推定最大DD|決済数|','|---|---:|---:|---:|---:|']
    for row in final['continuous']:
        m=row['metrics'];pf=m['profit_factor']
        lines.append(f"|{key(row['case'])}|{m['net_equity_change']:+,.0f}円|{pf:.3f}|{m['max_observation_drawdown_pct']:.2f}%|{m['trades']}|" if pf is not None else f"|{key(row['case'])}|{m['net_equity_change']:+,.0f}円|—|{m['max_observation_drawdown_pct']:.2f}%|{m['trades']}|")
    lines+=['','## 8〜9月の期間別検証','','|条件|通常費用の純評価損益|スプレッド・滑り2倍|PF（通常）|推定DD（通常）|','|---|---:|---:|---:|---:|']
    stressed={key(r['case']):r['metrics'] for r in final['validation_stress']}
    for row in final['validation']:
        m=row['metrics'];sm=stressed[key(row['case'])];pf=m['profit_factor']
        lines.append(f"|{key(row['case'])}|{m['net_equity_change']:+,.0f}円|{sm['net_equity_change']:+,.0f}円|{pf:.3f}|{m['max_observation_drawdown_pct']:.2f}%|")
    lines+=['',f"検証期間で固定した最上位: `{key(winner)}`。判定: `{summary['conclusion']}`。",
            '',f'独立した約定・手数料・持越料・現金整合の確認: {count:,}取引、全{len(jobs)}回が一致。',
            '','仮定: Bitbankの15分OHLC代理値、過去SRなし、固定費用、板・通信遅延・ロスカット未再現。推定DDは観測点ベース。本番設定は変更していない。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    all_lines=['# 学習期間の全50条件','', '5〜7月だけの比較。ここから上位6条件を固定し、別期間の結果で再選択しない。','',legend,'',
               '|条件|純評価損益|PF|推定最大DD|決済数|学習期間の足切り通過|','|---|---:|---:|---:|---:|---|']
    for row in final['training']:
        m=row['metrics'];pf=m['profit_factor']
        pf_text=f'{pf:.3f}' if pf is not None else '—'
        all_lines.append(f"|{key(row['case'])}|{m['net_equity_change']:+,.0f}円|{pf_text}|{m['max_observation_drawdown_pct']:.2f}%|{m['trades']}|{'はい' if eligible(row) else 'いいえ'}|")
    (OUT/'all_training_conditions.md').write_text('\n'.join(all_lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(jobs=len(jobs),trades_checked=count,winner=winner,positive_main_checks=supported)),flush=True)


if __name__=='__main__':main()
