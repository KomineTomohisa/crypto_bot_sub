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
