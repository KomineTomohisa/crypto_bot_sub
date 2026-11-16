"""Causal signal normalization, independent of live/paper/backtest execution."""

import numpy as np
import pandas as pd

MACD_WINDOW = 50
MACD_QUANTILES = (0.2, 0.8)
MACD_SCORING_VERSION = "macd_prior50_q20_q80_v1"


def macd_scoring_spec():
    return {
        "version": MACD_SCORING_VERSION,
        "history_bars": MACD_WINDOW,
        "quantiles": list(MACD_QUANTILES),
        "exclude_current": True,
        "invalid_action": "block_entry",
    }


def _prior_quantile_score(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    history = values.shift(1).rolling(MACD_WINDOW, min_periods=MACD_WINDOW)
    low = history.quantile(MACD_QUANTILES[0], interpolation="linear")
    high = history.quantile(MACD_QUANTILES[1], interpolation="linear")
    width = high - low
    # An absent/constant reference population is unknown, not a directional score.
    return ((values - low) / width.where(width.gt(0))).clip(0, 1)


def causal_macd_scores(frame):
    histogram = _prior_quantile_score(frame["MACD_histogram"])
    macd = pd.to_numeric(frame["MACD"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    signal = pd.to_numeric(frame["MACD_signal"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    crossover = _prior_quantile_score(macd - signal)
    ready = histogram.notna() & crossover.notna()
    return pd.DataFrame(
        {
            "macd_score_long": (histogram * 0.7 + crossover * 0.3).where(ready),
            "macd_score_short": ((1 - histogram) * 0.7 + (1 - crossover) * 0.3).where(
                ready
            ),
            "macd_score_ready": ready,
        },
        index=frame.index,
    )
