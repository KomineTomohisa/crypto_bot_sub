"""Versioned strategy settings. Missing historical settings always mean legacy."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class StrategyPolicy:
    version: str = 'baseline_v1'
    entry_adx_min: float = 0.0
    short_profit_lock: bool = True
    lock_trigger: float = 0.75
    lock_level: float = 0.25

    def __post_init__(self):
        if self.version not in ('baseline_v1', 'adx30_no_short_lock_v1'):
            raise ValueError('Unknown strategy policy version')
        if type(self.short_profit_lock) is not bool:
            raise ValueError('Profit lock setting must be boolean')
        for name in ('entry_adx_min', 'lock_trigger', 'lock_level'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Policy values must be finite numbers')
        if not 0 <= self.entry_adx_min <= 100 or not 0 < self.lock_level <= self.lock_trigger < 100:
            raise ValueError('Invalid strategy policy thresholds')
        expected = (0, True) if self.version == 'baseline_v1' else (30, False)
        if (self.entry_adx_min, self.short_profit_lock) != expected:
            raise ValueError('Policy version does not match its entry/exit rules')

    def snapshot(self):
        return asdict(self)


BASELINE = StrategyPolicy()
CANDIDATE = StrategyPolicy('adx30_no_short_lock_v1', 30, False)


def policy(value=None):
    if value is None:
        return BASELINE
    if isinstance(value, StrategyPolicy):
        return value
    if isinstance(value, str):
        if value == BASELINE.version:
            return BASELINE
        if value == CANDIDATE.version:
            return CANDIDATE
        raise ValueError('Unknown strategy policy')
    if not isinstance(value, dict) or set(value) != set(BASELINE.snapshot()):
        raise ValueError('Incomplete persisted strategy policy')
    return StrategyPolicy(**value)


def active_policy(bot):
    return policy(getattr(bot, 'strategy_policy', None))


def position_policy(bot, symbol):
    # Never use the active entry policy as a fallback for an existing position.
    try:
        return policy(getattr(bot, 'position_policies', {}).get(symbol))
    except (ValueError, TypeError):
        return BASELINE


def adx_entry_ready(latest, previous, selected=None):
    minimum = policy(selected).entry_adx_min
    if minimum == 0:
        return True
    for row in (latest, previous):
        value = row.get('ADX')
        if isinstance(value, bool) or type(value).__name__ == 'bool_':
            return False
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            return False
        if not math.isfinite(number) or not minimum <= number <= 100:
            return False
    return True
