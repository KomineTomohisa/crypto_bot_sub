from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import threading
import time

import pandas as pd
import pytest

from market_quotes import MarketQuote, QuoteUnavailable
from strategy_policy import BASELINE, CANDIDATE
from strategy_shadow import METHODS, ShadowComparison, ShadowFile, ShadowWorker, isolated_strategy, verify_shadow_journal
from strategy_recording import digest, code_fingerprint
from test_bot_execution_integration import BOT, method


SYMBOL = 'ltc_jpy'
AT = pd.Timestamp('2026-11-22T12:00:00+09:00')
RULE = dict(sizeStep='1', minOrderSize='1', maxOrderSize='10000')
COSTS = dict(name='test', fee_bps={SYMBOL: 3}, spread_bps=2, slippage_bps=1, carry_bps_per_day=4)


def core():
    return ShadowComparison('offline-test', [SYMBOL], 10000, 1000, COSTS,
                            {name: method(BOT, name)[0] for name in METHODS}, fingerprint='test-source')


def features(at=AT, adx=30):
    return pd.DataFrame(dict(timestamp=pd.date_range(at.floor('15min') - pd.Timedelta(minutes=75), periods=6, freq='15min'),
                             open=100., close=100., high=100., low=100., ATR=1., ADX=adx,
                             buy_signal=True, sell_signal=False, plus_di14=30., minus_di14=20.))


def observe(c, at=AT, price=100, frame=None, zones=None):
    if zones is None:
        zones = {digest(q): [] for q in c.sr_queries(SYMBOL)}
    return c.observe(SYMBOL, features(at) if frame is None else frame,
                     MarketQuote(price, at.to_pydatetime()), at, zones, RULE)


def test_independent_wallets_share_input_but_adx_gate_changes_entries():
    c = core()
    frame = features(adx=29)
    before = frame.copy(deep=True)
    assert observe(c, frame=frame)
    assert SYMBOL in c.books[BASELINE.version]['account'].positions
    assert not c.books[CANDIDATE.version]['account'].positions
    pd.testing.assert_frame_equal(frame, before)
    assert not observe(c, frame=frame)  # exact repeated observation is idempotent
    assert c.observations == 1


def test_profit_lock_differs_while_normal_stops_still_close_candidate():
    c = core()
    observe(c)
    observe(c, AT + pd.Timedelta(seconds=30), 99)
    assert c.books[BASELINE.version]['strategy'].short_profit_lock_armed[SYMBOL]
    assert not c.books[CANDIDATE.version]['strategy'].short_profit_lock_armed[SYMBOL]
    observe(c, AT + pd.Timedelta(seconds=60), 100)
    assert len(c.books[BASELINE.version]['trades']) == 1
    assert not c.books[CANDIDATE.version]['trades']
    observe(c, AT + pd.Timedelta(seconds=90), 110)
    assert len(c.books[CANDIDATE.version]['trades']) == 1
    report = c.report()
    assert report['fills_are_estimates']
    assert report['books'][CANDIDATE.version]['execution_fees_paid'] > 0


def test_restart_restores_cash_carry_lock_cooldown_and_dedup(tmp_path):
    c = core()
    observe(c)
    observe(c, AT + pd.Timedelta(seconds=30), 99)
    file = ShadowFile(tmp_path)
    file.save(c)
    other = core()
    other.restore(file.load())
    assert digest(other.snapshot()) == digest(c.snapshot())
    assert not observe(other, AT)
    observe(c, AT + pd.Timedelta(days=1), 100)
    observe(other, AT + pd.Timedelta(days=1), 100)
    assert digest(other.snapshot()) == digest(c.snapshot())
    assert other.books[CANDIDATE.version]['account'].carry_paid > 0
    file.close()


def test_corrupt_or_changed_state_cannot_silently_reset(tmp_path):
    c = core()
    file = ShadowFile(tmp_path)
    file.save(c)
    saved = file.load()
    saved['identity']['code'] = 'other'
    with pytest.raises(ValueError, match='changed'):
        core().restore(saved)
    payload = json.loads(file.path.read_text())
    payload['state']['observations'] = 100
    file.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Corrupt'):
        file.load()
    with pytest.raises(OSError):
        ShadowFile(tmp_path)
    file.close()


def test_stale_quote_or_absent_sr_does_not_open_virtual_positions():
    c = core()
    with pytest.raises(QuoteUnavailable):
        c.observe(SYMBOL, features(), MarketQuote(100, AT - pd.Timedelta(minutes=2)), AT, {}, RULE)
    observe(c, zones={})
    assert not any(b['account'].positions for b in c.books.values())


def test_sr_snapshot_blocks_both_books_with_actual_guard():
    c = core()
    zones = {digest(q): [dict(zone_type='support', price_low=99, price_high=101, strength=20)] for q in c.sr_queries(SYMBOL)}
    observe(c, zones=zones)
    assert not any(b['account'].positions for b in c.books.values())


def test_missing_features_keeps_time_exit_and_carry():
    c = core()
    observe(c)
    observe(c, AT + pd.Timedelta(hours=73), 100, frame=pd.DataFrame())
    assert all(len(b['trades']) == 1 for b in c.books.values())
    assert c.report()['gaps'] == 1


def test_actual_receipts_are_separate_and_deduplicated():
    c = core()
    observe(c)
    before = [deepcopy(b['account'].positions) for b in c.books.values()]
    payload = dict(event_id='actual1', purpose='entry', symbol=SYMBOL, side='SELL',
                   decision_bar=str(AT - pd.Timedelta(minutes=15)), result={'execution_price': 101},
                   trades=[{'trade_id': 't1', 'pnl': 10}])
    c.actual_result(payload); c.actual_result(payload)
    c.actual_result({**payload, 'event_id': 'actual2'})
    assert c.report()['actual_recorded_pnl'] == 10
    assert c.report()['entry_fill_differences'][0]['same_signal_entry']
    assert before == [b['account'].positions for b in c.books.values()]


def test_slow_read_and_full_queue_never_block_caller(tmp_path):
    started, release = threading.Event(), threading.Event()
    class Reader:
        def read(self, symbol):
            started.set()
            release.wait(5)
            raise ValueError('offline')
        def close(self):
            pass
    worker = ShadowWorker(core, tmp_path, Reader(), lambda **kw: [], logging.getLogger('test'), capacity=1)
    try:
        assert worker.offer('observation', (SYMBOL, features()))
        assert started.wait(2)
        before = time.monotonic()
        assert worker.offer('observation', (SYMBOL, features()))
        assert not worker.offer('observation', (SYMBOL, features()))
        assert time.monotonic() - before < .5
        worker.close(.01)
        assert worker.thread.is_alive()
    finally:
        release.set()
        worker.close(3)
    assert not worker.thread.is_alive()


def test_strategy_state_has_no_gateway_or_private_api():
    state = next(iter(core().books.values()))['strategy']
    assert not any(hasattr(state, name) for name in ('gmo_api', '_execution', 'exchange_settings_gmo', 'execute_order'))
    assert 'HTTP_SESSION' not in state.calculate_dynamic_exit_levels.__func__.__globals__


def test_worker_journal_replays_same_wallets_offline_including_sr_failure(tmp_path):
    class Reader:
        def read(self, symbol):
            at = pd.Timestamp.now(tz='Asia/Tokyo')
            return MarketQuote(100, at.to_pydatetime()), at, RULE
        def close(self):
            pass
    def build():
        result = core()
        result.identity['code'] = code_fingerprint()
        return result
    calls = []
    def sr(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError('offline snapshot unavailable')
        return []
    worker = ShadowWorker(build, tmp_path, Reader(), sr, logging.getLogger('test'))
    assert worker.offer('observation', (SYMBOL, features(pd.Timestamp.now(tz='Asia/Tokyo'))))
    worker.close(5)
    assert not worker.thread.is_alive() and not worker.failed
    journal = next((tmp_path / 'observations').glob('*.jsonl'))
    verified = verify_shadow_journal(journal)
    assert verified['checkpoints'] == 1
    assert verified['report']['gaps'] == 1
    assert verified['report']['books'][BASELINE.version]['open_positions'] == 0


def test_optional_shadow_failure_does_not_interrupt_actual_monitoring(monkeypatch):
    from test_display_monitoring_isolation import display_bot, wire_live_loop, SYMBOLS
    bot = display_bot()
    bot.entry_sizes = {s: 5. for s in SYMBOLS}
    bot.short_profit_lock_armed = {}; bot.reentry_block_until = {}
    bot.verify_positions = lambda: None
    bot._refresh_live_stats = lambda *a: None
    bot.send_notification = lambda *a, **kw: None
    bot.get_current_price = lambda _: 100.
    bot.shadow_costs = COSTS
    class BadWorker:
        def offer(self, *a):
            raise OSError('disk failed')
        def close(self, timeout):
            pass
    bot._shadow_worker = BadWorker()
    run, actions, waits = wire_live_loop(bot)
    run()
    assert [('monitor', s, cycle) for cycle in (0, 1) for s in SYMBOLS] == [a for a in actions if a[0] == 'monitor']
    assert len(waits) == 2
