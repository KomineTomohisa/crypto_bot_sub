import pandas as pd
import pytest

from research.download_search_data import complete_partition


@pytest.mark.parametrize('interval,frequency',[('15min','15min'),('1hour','1h')])
def test_only_a_complete_utc_day_can_be_reused(interval,frequency):
    times=pd.date_range('2026-11-18','2026-11-19',freq=frequency,inclusive='left',tz='UTC')
    rows=[[1,1,1,1,0,int(t.value//1000000)] for t in times]
    payload={'success':1,'data':{'candlestick':[{'type':interval,'ohlcv':rows}]}}
    assert complete_partition(payload,interval,'20261118')
    assert not complete_partition(payload,interval,'20261119')
    payload['data']['candlestick'][0]['ohlcv']=rows[:len(rows)//2]
    assert not complete_partition(payload,interval,'20261118')
    payload['data']['candlestick'][0]['ohlcv']=rows[:-1]+rows[-2:-1]
    assert not complete_partition(payload,interval,'20261118')
