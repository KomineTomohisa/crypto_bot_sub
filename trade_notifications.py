"""Japanese trade receipts built only from persisted, confirmed execution data."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import re
import math

JST = timezone(timedelta(hours=9))


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def amount(value, *, signed=False):
    value = number(value)
    if value is None:
        return '未取得'
    rendered = format(value, '+,.8f' if signed else ',.8f').rstrip('0').rstrip('.')
    return rendered


def when(value):
    if value is None:
        return '未取得'
    try:
        at = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if at.tzinfo is None:
            return '未取得（タイムゾーン不明）'
        return at.astimezone(JST).strftime('%Y-%m-%d %H:%M:%S JST')
    except (TypeError, ValueError, OverflowError):
        return '未取得'


def entry_snapshot(snapshot, indicators):
    """Whitelist numeric data; never persist entire rows, API responses or secrets."""
    snapshot = snapshot if hasattr(snapshot, 'get') else {}
    def finite_float(value):
        value = number(value)
        if value is None:
            return None
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            return None
        return converted if math.isfinite(converted) else None
    features = {
        label: finite_float(indicators.get(key))
        for label, key in [('RSI', 'rsi'), ('ADX', 'adx'), ('ATR', 'atr'),
                           ('DI+', 'di_plus'), ('DI-', 'di_minus'),
                           ('EMA短期', 'ema_fast'), ('EMA長期', 'ema_slow')]
    }
    signals = {}
    for key in ('buy_signal', 'sell_signal'):
        value = snapshot.get(key)
        if type(value).__name__ in ('bool', 'bool_'):
            signals[key] = bool(value)
    return {'indicators': features, 'signals': signals}


def compose_execution_notice(result, *, recovered=False, remaining=None):
    """One receipt per finalized order, including partial fills and old contexts."""
    if not result.get('finalized'):
        raise ValueError('Execution receipt requires a finalized order')
    intent = result['intent']
    context = intent.get('context') or {}
    symbol = str(intent['symbol']).upper()
    if not re.fullmatch(r'[A-Z0-9]+_JPY', symbol):
        raise ValueError('Unexpected receipt symbol')
    opened = intent['settle_type'] == 'OPEN'
    side = intent['side']
    if side not in ('BUY', 'SELL') or intent['settle_type'] not in ('OPEN', 'CLOSE'):
        raise ValueError('Unexpected receipt direction')
    long = (side == 'BUY') if opened else (side == 'SELL')
    direction = 'ロング（買い建て）' if long else 'ショート（売り建て）'
    size = number(result.get('executed_size'))
    requested = number(intent.get('size'))
    if size is None or size < 0 or (requested is not None and size > requested):
        raise ValueError('Receipt execution quantity is unconfirmed')
    filled = size is not None and size > 0
    partial = filled and requested is not None and size < requested
    action = 'エントリー' if opened else '決済'
    state = ('一部約定' if partial else '約定確定') if filled else '約定なし'
    subject = f'【LIVE／{action} {state}】{symbol} {"ロング" if long else "ショート"}'
    trades = result.get('trades') or []
    gross_values = [number(t.get('pnl')) for t in trades]
    gross = sum(gross_values, Decimal(0)) if trades and all(v is not None for v in gross_values) else None
    if not opened and filled and gross is not None:
        subject += f' 売買損益 {amount(gross, signed=True)}円'
    body = [f'{symbol}の{direction}を{action}しました。' if filled else f'{symbol}の{action}注文は約定なしで終了しました。']
    if not opened and filled:
        body.append(f'今回の売買損益: {amount(gross, signed=True)} 円（手数料控除前）')
    body += [
            f'結果: {state}' + ('（残りの注文数量は約定していません）' if partial else ''),
            '確認経路: 応答待ち後の注文照合で確定' if recovered else '確認経路: 注文・約定記録で確定', '',
            '【約定内容】', f'方向: {direction}／{"新規" if opened else "決済"}{"買い" if side == "BUY" else "売り"}',
            f'注文数量: {amount(requested)} {symbol.split("_")[0]}',
            f'約定数量: {amount(size)} {symbol.split("_")[0]}']
    if filled:
        body += [f'平均約定価格: {amount(result.get("execution_price"))} 円',
                 f'約定日時: {when(result.get("executed_at"))}']
        price = number(result.get('execution_price'))
        body.append(f'約定代金: {amount(price * size if price is not None else None)} 円（価格×数量）')
    if opened:
        saved = context.get('notification') or {}
        scores = context.get('scores') or {}
        signal_key = 'sell_signal' if long else 'buy_signal'
        score_key = 'sell_score' if long else 'buy_score'
        signal = (saved.get('signals') or {}).get(signal_key)
        body += ['', '【エントリー判断の記録】', f'判断足の開始日時: {when(context.get("decision_bar"))}',
                 f'判断足の終値: {amount(context.get("decision_price"))} 円']
        body.append(f'確定足の{"ロング" if long else "ショート"}シグナル: ' +
                    ('成立' if signal is True else '不成立' if signal is False else '記録なし'))
        consecutive = False
        try:
            latest_bar = datetime.fromisoformat(str(context.get('decision_bar')))
            previous_bar = datetime.fromisoformat(str(saved.get('previous_bar')))
            consecutive = (signal is True and latest_bar.tzinfo is not None and previous_bar.tzinfo is not None
                           and latest_bar - previous_bar == timedelta(minutes=15))
        except (TypeError, ValueError):
            pass
        if consecutive:
            body += ['エントリー理由: 15分足の確定足2本で同方向のシグナルが連続',
                     f'前の判断足の開始日時: {when(saved["previous_bar"])}']
        total = scores.get(score_key + '_scaled', scores.get(score_key))
        body.append(f'採用方向の総合スコア: {amount(total)}')
        macd = scores.get('macd_score_long' if long else 'macd_score_short')
        body.append(f'採用方向のMACDスコア: {amount(macd)}')
        features = saved.get('indicators') or {}
        if features:
            body += ['判断時の指標:'] + [f'  {key}: {amount(features.get(key))}' for key in ('RSI', 'ADX', 'ATR', 'DI+', 'DI-', 'EMA短期', 'EMA長期')]
            body.append('RSI＝過熱度、ADX＝トレンドの強さ、ATR＝価格の平均変動幅、DI＝上下方向の強さ、EMA＝指数平滑移動平均。')
        else:
            body.append('判断時の指標: 記録なし（旧形式の注文）')
        body.append(f'新規注文の取引手数料: {amount(result.get("execution_fee"))} 円')
        body.append('指標は注文判断時の参考値、価格・数量は実約定の記録です。')
    elif filled:
        reason = str(context.get('exit_reason') or '記録なし').replace('\n', ' ').replace('\r', ' ')[:100]
        body += ['', '【決済の理由・損益】', f'決済理由: {reason}',
                 '通知範囲: この決済注文で確定した約定分',
                 f'売買損益（手数料控除前）: {amount(gross, signed=True)} 円']
        sizes = [number(t.get('size')) for t in trades]
        total_size = sum((s or Decimal(0) for s in sizes), Decimal(0))
        entries = [number(t.get('entry_price')) for t in trades]
        entry = (sum((p * s for p, s in zip(entries, sizes)), Decimal(0)) / total_size
                 if total_size > 0 and all(p is not None for p in entries) and all(s is not None and s > 0 for s in sizes) else None)
        basis = entry * total_size if entry is not None else None
        pct = gross / basis * 100 if gross is not None and basis else None
        body += [f'決済分の平均建値: {amount(entry)} 円',
                 f'売買損益率: {amount(pct, signed=True)} %（決済分の建値×数量に対する比率）']
        for label, key in [('配賦済み新規手数料', 'entry_fee'), ('決済手数料', 'exit_fee'),
                           ('手数料控除後損益（日次費用を除く）', 'net_pnl_before_carry')]:
            values = [number(t.get(key)) for t in trades]
            value = sum(values, Decimal(0)) if trades and all(v is not None for v in values) else None
            body.append(f'{label}: {amount(value, signed=key=="net_pnl_before_carry")} 円' if value is not None else f'{label}: 未確定')
        body.append('日次費用は取引別に配賦していません。上記は日次費用控除後の最終損益ではありません。')
        if trades:
            hours = [number(t.get('holding_hours')) for t in trades]
            if all(h is not None for h in hours):
                body.append(f'保有時間: {amount(min(hours))}〜{amount(max(hours))} 時間' if min(hours)!=max(hours) else f'保有時間: {amount(hours[0])} 時間')
            body.append(f'決済完了日時: {when(max((t["closed_at"] for t in trades if t.get("closed_at")), default=None))}')
        body.append(f'元の決済対象の残数量: {amount(remaining)} {symbol.split("_")[0]}' if remaining is not None else '元の決済対象の残数量: この通知では未取得')
    body += ['', f'取引所の注文状態: {intent.get("status", result.get("status", "不明"))}',
             f'照会用注文番号: {result.get("order_id") or "未取得"}']
    return subject, '\n'.join(body), ('entry' if opened else 'exit')
