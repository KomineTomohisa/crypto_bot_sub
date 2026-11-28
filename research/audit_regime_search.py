"""Independent entry-gate and accounting audit, including unchanged controls."""
from collections import defaultdict
import hashlib
import json
import math

import pandas as pd

from research.audit_expanded_search import audit,close
from research import regime_search as study
from research.search_backtests import ROOT,read_frames,write_json


def independent_gate(case, rows, at, side):
    if case['direction']=='short_only' and side!='short':return False
    if case['direction']=='long_only' and side!='long':return False
    if any(float(row['ADX'])<case['adx_min'] for row in rows):return False
    session=case.get('session','all');hour=at.tz_convert('Asia/Tokyo').hour
    if session=='jst_01_09' and not 1<=hour<9:return False
    if session=='jst_09_17' and not 9<=hour<17:return False
    if session=='jst_17_01' and not (hour>=17 or hour<1):return False
    band=case.get('atr_band','all')
    if band!='all':
        lo,hi=(0.,.5) if band=='quiet' else (.25,1.)
        if any(not lo<=100*float(row['ATR'])/float(row['close'])<hi for row in rows):return False
    context=case.get('entry_context','none')
    if context=='adx_rising' and float(rows[0]['ADX'])<float(rows[1]['ADX']):return False
    if context in ('ema_follow','ema_fade','di_follow','di_fade'):
        first,second=('EMA_short','EMA_long') if context.startswith('ema') else ('plus_di14','minus_di14')
        sign=1 if side=='long' else -1
        if context.endswith('fade'):sign*=-1
        if any((float(row[first])-float(row[second]))*sign<0 for row in rows):return False
    return True


def audit_entries(result, cache):
    records=list(result['trades'])
    records.extend(dict(position,symbol=symbol,type=position['side']) for symbol,position in result['open_positions'].items())
    for trade in records:
        at=pd.Timestamp(trade['entry_time']);side=trade['type'];symbol=trade['symbol'];case=result['case']
        day=cache.day(symbol,at)
        index=int(day.timestamp.searchsorted(at,side='left'))
        assert index>=2
        rows=[day.iloc[index-1],day.iloc[index-2]]
        assert rows[0]['timestamp']==at-pd.Timedelta(minutes=15) and rows[1]['timestamp']==at-pd.Timedelta(minutes=30)
        column='buy_signal' if side=='short' else 'sell_signal'
        assert all(bool(row[column]) for row in rows)
        assert independent_gate(case,rows,at,side),'Recorded entry violates its frozen regime'
        if side=='long' and all(bool(row['buy_signal']) for row in rows):
            assert not independent_gate(case,rows,at,'short'),'Short priority was bypassed'


def supports(metrics, min_trades=30):
    return metrics['trades']>=min_trades and metrics['net_equity_change']>0 and metrics['profit_factor'] is not None and metrics['profit_factor']>=1 and metrics['max_observation_drawdown_pct']<=10 and not metrics['negative_cash_observed']


def failure_reasons(metrics):
    values=[]
    if metrics['trades']<30:values.append('fewer_than_30_closed_trades')
    if metrics['net_equity_change']<=0:values.append('nonpositive_net_equity_change')
    if metrics['profit_factor'] is None or metrics['profit_factor']<1:values.append('profit_factor_below_1_or_undefined')
    if metrics['max_observation_drawdown_pct']>10:values.append('observation_drawdown_above_10pct')
    if metrics['negative_cash_observed']:values.append('negative_available_cash')
    return values


def main():
    study.configure();out=study.OUT
    final=json.loads((out/'final_results.json').read_text())
    assert final['completed'] and final['identity']==study.engine.identity()
    frames=read_frames(study.RAW/'raw')
    prices={s:f.set_index('timestamp') for (s,i),f in frames.items() if i=='15min'}
    cache=study.BoundedFeatureCache(frames,cache_dir=out/'feature_cache')
    jobs=[];count=0;paths=set();contributions={}
    for folder in ('screen','train','validation','historical_audit','monthly_0','monthly_1','continuous'):
        files=sorted((out/folder).glob('*.json'))
        for path in files:
            result=json.loads(path.read_text());assert result['run_identity']==study.engine.identity()
            trades=audit(result,prices);audit_entries(result,cache);count+=trades
            jobs.append(dict(file=str(path.relative_to(out)),trades=trades,verified=True))
            if folder=='screen':paths.add(study.order_path(result))
            if folder=='continuous':
                by_symbol=defaultdict(lambda:dict(trades=0,net_closed_profit=0.))
                for t in result['trades']:
                    by_symbol[t['symbol']]['trades']+=1;by_symbol[t['symbol']]['net_closed_profit']+=t['profit']
                contributions[study.case_key(result['case'])]=dict(by_symbol)
        print(json.dumps(dict(phase='independent_audit',folder=folder,jobs=len(jobs),trades_checked=count)),flush=True)
    for row in json.loads((study.RAW/'raw_checksums.json').read_text()):
        assert hashlib.sha256((study.RAW/'raw'/row['file']).read_bytes()).hexdigest()==row['sha256']
    select=lambda phase,case:next(r['metrics'] for r in final[phase] if r['case']==case)
    old=json.loads((study.PRIOR/'final_results.json').read_text());control_checks=[]
    for case in [study.BASELINE,*study.ANCHORS]:
        prior=next(r['metrics'] for r in old['continuous'] if r['case']==case)
        current=select('continuous',case)
        for field in ('trades','profit_factor','net_equity_change','realized_profit','max_observation_drawdown_pct'):
            close(current[field],prior[field],'unchanged control '+field)
        control_checks.append(dict(case=case,matched=True))
    winner=final['finalists'][0]
    phases=['validation','validation_stress','historical_audit','historical_audit_stress','continuous']
    winner_metrics={phase:select(phase,winner) for phase in phases}
    training=next(r for r in final['training'] if r['case']==winner)
    positive=study.engine.eligible(training) and all(supports(m) for m in winner_metrics.values())
    summary=dict(completed_backtests=len(jobs),screen_profiles=216,screen_unique_order_paths=len(paths),
        refinement_candidates=len(final['training']),validation_selected_candidate=winner,winner_metrics=winner_metrics,
        positive_main_checks=positive,conclusion='provisional_candidate_requires_forward_shadow' if positive else 'no_supported_winner',
        continuous=final['continuous'],validation=final['validation'],validation_stress=final['validation_stress'],
        finalists=final['finalists'],controls_reproduced=control_checks,independent_audit_passed=True,
        training_eligibility_passed=study.engine.eligible(training),support_failure_reasons={phase:failure_reasons(m) for phase,m in winner_metrics.items()},
        all_validation_candidates=json.loads((out/'validation_selection.json').read_text())['ranking'],
        recorded_trades_checked=count,continuous_symbol_contributions=contributions,production_changed=False,
        limitations=final['plan']['limitations'])
    write_json(out/'independent_audit.json',dict(all_verified=True,jobs=len(jobs),recorded_trades_checked=count,checks=jobs,
        entry_gates_verified=True,raw_files_unchanged=True,controls_reproduced=control_checks,
        auditor_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    write_json(out/'review_summary.json',summary)
    legend='none=追加の方向条件なし、ema/di_follow=方向に順行、ema/di_fade=逆行、adx_rising=ADX上昇。quiet=ATR/価格が0.5%未満、normal=0.25%以上1%未満。時刻はJSTのエントリー時刻。方向・ATRは確定2本とも判定。tp/slは元の決済幅の倍率。'
    lines=['# 相場・ボラティリティ・時間帯の追加探索','',f'216組み合わせ＋基準を一次比較し、利確・損切りの追加探索を含め計{len(jobs)}回。初期20万円、1回4万円、7銘柄、費用込み推計。',
        '',legend,'','全期間は2026-05-01〜11-21。損益は含み損益込み、未決済分の将来決済費用を除く。推定DDは観測点ベース。',
        '','Mayの一次選択→May–Julyの決済幅調整→Aug–Sepで3候補固定→Oct–Novと全期間で確認。全期間は以前に見たデータで、月別確認も未使用データのウォークフォワード検証ではない。',
        '',f"期間別の選択で固定した候補: `{study.case_key(winner)}`。判定: `{summary['conclusion']}`。",'',
        '|条件|全期間の純評価損益|PF|推定DD|8〜9月|費用増の8〜9月|','|---|---:|---:|---:|---:|---:|']
    for row in final['continuous']:
        case=row['case'];m=row['metrics'];v=select('validation',case);s=select('validation_stress',case)
        pf='—' if m['profit_factor'] is None else f"{m['profit_factor']:.3f}"
        lines.append(f"|{study.case_key(case)}|{m['net_equity_change']:+,.0f}円|{pf}|{m['max_observation_drawdown_pct']:.2f}%|{v['net_equity_change']:+,.0f}円|{s['net_equity_change']:+,.0f}円|")
    lines+=['',f'独立した価格・手数料・持越料・資金整合・エントリー条件の検算: {count:,}取引、{len(jobs)}回が一致。従来の4基準も全期間で再現。',
        '','Bitbank OHLC代理値、過去SRなし、固定費用、証拠金・ロスカット・板・通信遅延は未再現。取引数が少ない条件を選択から除外。多数比較による過剰適合の可能性がある。本番設定は変更していない。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    comparison=['# 全一次比較と決済幅調整','',legend]
    for title,rows in [('一次比較',final['screen']),('3か月の決済幅調整',final['training'])]:
        comparison+=['','## '+title,'','|条件|純評価損益|PF|推定DD|決済数|','|---|---:|---:|---:|---:|']
        for row in rows:
            m=row['metrics'];pf='—' if m['profit_factor'] is None else f"{m['profit_factor']:.3f}"
            comparison.append(f"|{study.case_key(row['case'])}|{m['net_equity_change']:+,.0f}円|{pf}|{m['max_observation_drawdown_pct']:.2f}%|{m['trades']}|")
    (out/'all_conditions.md').write_text('\n'.join(comparison)+'\n',encoding='utf-8')
    print(json.dumps(dict(jobs=len(jobs),trades_checked=count,winner=winner,positive_main_checks=positive)),flush=True)


if __name__=='__main__':main()
