"""Validate public ticker observations without network or environment access."""
from dataclasses import dataclass
from datetime import datetime, timezone
import math

from trading_execution import ExecutionError, decimal, timestamp


MAX_QUOTE_AGE_SECONDS = 90


class QuoteUnavailable(ExecutionError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__('Quote unavailable: ' + reason)


def price_value(value):
    """Return a finite, positive float; booleans and overflow are invalid."""
    try:
        price = float(decimal(value, positive=True))
    except (ExecutionError, OverflowError, ValueError, TypeError):
        return None
    return price if math.isfinite(price) and price > 0 else None


@dataclass(frozen=True)
class MarketQuote:
    price: float
    quoted_at: datetime


def checked_quote(quote, observed_at):
    if not isinstance(quote, MarketQuote):
        raise QuoteUnavailable('missing_quote_metadata')
    price = price_value(quote.price)
    if price is None:
        raise QuoteUnavailable('invalid_price')
    try:
        quoted_at = timestamp(quote.quoted_at)
        observed_at = timestamp(observed_at)
    except (ExecutionError, OverflowError, ValueError, TypeError):
        raise QuoteUnavailable('invalid_timestamp') from None
    age = (observed_at - quoted_at).total_seconds()
    if age < 0:
        raise QuoteUnavailable('future_timestamp')
    if age > MAX_QUOTE_AGE_SECONDS:
        raise QuoteUnavailable('stale_timestamp')
    return MarketQuote(price, quoted_at.astimezone(timezone.utc))


def parse_ticker(response, symbol, observed_at):
    if not isinstance(response, dict) or type(response.get('status')) is not int or response['status'] != 0:
        raise QuoteUnavailable('ticker_status')
    rows = response.get('data')
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise QuoteUnavailable('ticker_shape')
    matches = [row for row in rows if row.get('symbol') == symbol]
    if len(matches) != 1:
        raise QuoteUnavailable('ticker_identity')
    row = matches[0]
    return checked_quote(MarketQuote(row.get('last'), row.get('timestamp')), observed_at)
