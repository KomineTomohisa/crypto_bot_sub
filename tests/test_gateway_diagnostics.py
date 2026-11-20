from types import SimpleNamespace
import pytest
from trading_execution import ExchangeGateway,ExecutionError


def test_failed_read_reports_only_error_code_and_endpoint():
    gateway=ExchangeGateway(SimpleNamespace(_request=lambda *a,**k:{'status':1,'messages':[{'message_code':'ERR-5012','message_string':'secret-must-not-leak'}]}))
    with pytest.raises(ExecutionError) as exc:
        gateway.request('/v1/openPositions')
    assert 'ERR-5012' in str(exc.value)
    assert 'secret-must-not-leak' not in str(exc.value)


def test_malformed_list_reports_types_without_values():
    gateway=ExchangeGateway(SimpleNamespace(_request=lambda *a,**k:{'status':0,'data':{'private-value':'secret-must-not-leak'}}))
    with pytest.raises(ExecutionError) as exc:
        gateway.pages('/v1/openPositions',{'symbol':'LTC_JPY'})
    assert 'data_type=dict list_type=NoneType' in str(exc.value)
    assert 'secret-must-not-leak' not in str(exc.value)


def test_invalid_error_code_is_not_logged_as_diagnostics():
    gateway=ExchangeGateway(SimpleNamespace(_request=lambda *a,**k:{'status':1,'messages':[{'message_code':'secret-must-not-leak'}]}))
    with pytest.raises(ExecutionError) as exc:
        gateway.request('/v1/openPositions')
    assert 'secret-must-not-leak' not in str(exc.value)


@pytest.mark.parametrize('data',[{}, {'pagination':{'currentPage':1,'count':100}}])
def test_successful_empty_positions_may_omit_list(data):
    gateway=ExchangeGateway(SimpleNamespace(_request=lambda *a,**k:{'status':0,'data':data}))
    assert gateway.positions('LTC_JPY') == []


@pytest.mark.parametrize('data',[
    None,{'pagination':{}},{'pagination':{'currentPage':2,'count':100}},
    {'pagination':{'currentPage':1,'count':100},'list':None},
    {'pagination':{'currentPage':1,'count':100},'other':'unexpected'},
    {'pagination':{'currentPage':True,'count':100}},
])
def test_ambiguous_empty_positions_are_not_treated_as_no_positions(data):
    gateway=ExchangeGateway(SimpleNamespace(_request=lambda *a,**k:{'status':0,'data':data}))
    with pytest.raises(ExecutionError):gateway.positions('LTC_JPY')


@pytest.mark.parametrize('data',[{}, {'pagination':{'currentPage':1,'count':100}}])
def test_omitted_empty_list_does_not_clear_a_recorded_open_position(data):
    import logging
    from live_execution import LiveExecutionMixin
    gateway=ExchangeGateway(SimpleNamespace(_request=lambda *a,**k:{'status':0,'data':data}))
    service=SimpleNamespace(gateway=gateway,account='test',store=SimpleNamespace(open_position_ids=lambda *a:{'recorded-position'}))
    bot=SimpleNamespace(_execution=lambda:service,symbols=['x'],symbol_mapping={'x':'LTC_JPY'},positions={'x':'long'},logger=logging.getLogger('test'),save_positions=lambda:None)
    LiveExecutionMixin._sync_live_positions(bot)
    assert bot.positions['x']=='long'
    assert 'x' in bot._position_sync_failed
    assert not LiveExecutionMixin._execute_live_entry(bot,'x','buy',1)['success']
