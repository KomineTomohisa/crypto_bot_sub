"""Account-scoped, atomic exit-state snapshots. No API/DB access on import."""
from copy import deepcopy
from datetime import timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from trading_execution import decimal, timestamp


def position_basis(rows):
    """Use immutable position identity and original entry, with remaining size."""
    result = {}
    for row in rows:
        pid = str(row['positionId'])
        if not pid or pid == 'None' or pid in result:
            raise ValueError('Invalid protection position identity')
        if row['side'] not in ('BUY', 'SELL'):
            raise ValueError('Invalid protection position side')
        result[pid] = {
            'symbol': row['symbol'], 'side': row['side'],
            'price': format(decimal(row['price'], positive=True).normalize(), 'f'),
            'size': format(decimal(row['size'], positive=True).normalize(), 'f'),
            'opened_at': timestamp(row['timestamp']).astimezone(timezone.utc).isoformat(),
        }
    if not result:
        raise ValueError('Protection requires verified positions')
    return result


def checked_price(value):
    if isinstance(value, bool):
        raise ValueError('Boolean protection price')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid protection price')
    return value


def checked_levels(levels, side, entry):
    if levels is None:
        return None
    if not isinstance(levels, dict):
        raise ValueError('Invalid protection levels')
    result = {}
    allowed = ('take_profit_price', 'stop_loss_price', 'take_profit_ratio',
               'stop_loss_ratio', 'atr_value', 'risk_reward_ratio')
    for key in allowed:
        if key in levels:
            result[key] = checked_price(levels[key])
    tp, sl = result['take_profit_price'], result['stop_loss_price']
    if not ((side == 'long' and sl < entry < tp) or
            (side == 'short' and tp < entry < sl)):
        raise ValueError('Protection levels have wrong direction')
    return result


def make_snapshot(account, symbol, rows, side, entry, narrow, profit_lock, levels):
    entry = checked_price(entry)
    basis = position_basis(rows)
    if side not in ('long', 'short') or type(narrow) is not bool or type(profit_lock) is not bool:
        raise ValueError('Invalid protection flags')
    exchange_side = 'BUY' if side == 'long' else 'SELL'
    if any(row['side'] != exchange_side for row in basis.values()):
        raise ValueError('Mixed protection directions')
    total = sum(decimal(row['size']) for row in basis.values())
    average = sum(decimal(row['size']) * decimal(row['price']) for row in basis.values()) / total
    if float(average) != entry:
        raise ValueError('Protection entry differs from verified positions')
    return {'account': account, 'symbol': symbol, 'side': side, 'entry_price': entry,
            'positions': basis, 'opposite_narrow': narrow,
            'short_profit_lock': profit_lock if side == 'short' else False,
            'levels': checked_levels(levels, side, entry)}


def restore_snapshot(saved, account, symbol, rows, side, entry):
    """Only unchanged positions or reductions of the saved position set qualify."""
    if not isinstance(saved, dict):
        return None
    if (saved.get('account'), saved.get('symbol'), saved.get('side')) != (account, symbol, side):
        return None
    current = position_basis(rows)
    prior = saved['positions']
    if not isinstance(prior, dict) or not set(current) <= set(prior):
        return None
    for pid, row in current.items():
        old = prior[pid]
        if any(row[key] != old[key] for key in ('symbol', 'side', 'price', 'opened_at')):
            return None
        if decimal(row['size']) > decimal(old['size'], positive=True):
            return None
    if type(saved['opposite_narrow']) is not bool or type(saved['short_profit_lock']) is not bool:
        raise ValueError('Invalid saved protection flags')
    # A different weighted entry invalidates price-based cache, but not the
    # surviving positions' signal lock. Never relabel old levels as a new basis.
    old_entry = checked_price(saved['entry_price'])
    levels = checked_levels(saved['levels'], side, old_entry)
    return {'opposite_narrow': saved['opposite_narrow'],
            'short_profit_lock': saved['short_profit_lock'] if side == 'short' else False,
            'levels': levels if old_entry == checked_price(entry) else None}


class ExitProtectionFile:
    """Single live-service writer; retain untouched symbols on partial API failure."""
    def __init__(self, directory, account):
        self.account = account
        digest = hashlib.sha256(account.encode('utf-8')).hexdigest()[:24]
        self.path = Path(directory) / ('exit_protection_real_' + digest + '.json')
        self.snapshots = {}
        self.last_saved = None

    def load(self):
        try:
            with self.path.open('rb') as stream:
                raw = stream.read(2 * 1024 * 1024 + 1)
        except FileNotFoundError:
            return
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('Protection file exceeds limit')
        data = json.loads(raw)
        if (type(data.get('version')) is not int or data['version'] != 1 or
                data.get('account') != self.account or not isinstance(data.get('symbols'), dict)):
            raise ValueError('Protection file identity/version mismatch')
        self.snapshots = data['symbols']
        self.last_saved = self._encoded()

    def _encoded(self):
        return json.dumps({'version': 1, 'account': self.account, 'symbols': self.snapshots},
                          sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

    def update(self, symbol, snapshot):
        if snapshot is None:
            self.snapshots.pop(symbol, None)
        else:
            self.snapshots[symbol] = deepcopy(snapshot)

    def save(self):
        data = self._encoded()
        if data == self.last_saved:
            return False
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', prefix='.exit-protection-', suffix='.tmp',
                                             dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            if os.name != 'nt':
                directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            self.last_saved = data
            return True
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
