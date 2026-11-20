"""All exchange I/O is fake. PostgreSQL tests only target a dedicated localhost database."""

from pathlib import Path
import uuid
import pytest
import requests
import smtplib
from sqlalchemy import create_engine
from execution_store import ExecutionStore

ROOT = Path(__file__).resolve().parents[1]
TEST_URL = (
    "postgresql+psycopg2://postgres:local-test-only@127.0.0.1:55439/execution_test"
)


@pytest.fixture(autouse=True)
def no_exchange_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "HTTP must be mocked; never access a live exchange in tests"
        )

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(smtplib, "SMTP", forbidden)


@pytest.fixture
def store():
    admin = create_engine(TEST_URL)
    schema = "t_" + uuid.uuid4().hex
    with admin.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA " + schema)
    engine = create_engine(
        TEST_URL, connect_args={"options": "-c search_path=" + schema}
    )
    base = """
    CREATE TABLE orders(order_id varchar PRIMARY KEY,symbol varchar,side varchar,type varchar,size numeric,status varchar,requested_at timestamptz,placed_at timestamptz,raw jsonb);
    CREATE TABLE fills(fill_id varchar PRIMARY KEY,order_id varchar,price numeric,size numeric,fee numeric,executed_at timestamptz,raw jsonb);
    CREATE TABLE positions(position_id varchar PRIMARY KEY,symbol varchar,side varchar,size numeric,avg_entry_price numeric,opened_at timestamptz,updated_at timestamptz,raw jsonb,source text,user_id bigint,open_signal_id uuid);
    CREATE TABLE trades(trade_id varchar PRIMARY KEY,symbol varchar,side varchar,entry_position_id varchar,exit_order_id varchar,entry_price numeric,exit_price numeric,size numeric,pnl numeric,pnl_pct numeric,holding_hours numeric,closed_at timestamptz,raw jsonb,source text,user_id bigint);
    """
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(base)
        conn.exec_driver_sql(
            (ROOT / "migrations/001_execution_accounting.sql").read_text()
        )
        conn.exec_driver_sql(
            (ROOT / "migrations/002_durable_close_batches.sql").read_text()
        )
        conn.exec_driver_sql(
            (ROOT / "migrations/003_close_stall_notifications.sql").read_text()
        )
    yield ExecutionStore(engine)
    engine.dispose()
    with admin.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA " + schema + " CASCADE")
    admin.dispose()
