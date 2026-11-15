BEGIN;

CREATE TABLE IF NOT EXISTS bot_order_intents (
    client_order_id varchar(36) PRIMARY KEY,
    account_key text NOT NULL,
    symbol text NOT NULL,
    side text NOT NULL CHECK (side IN ('BUY','SELL')),
    settle_type text NOT NULL CHECK (settle_type IN ('OPEN','CLOSE')),
    size numeric NOT NULL CHECK (size > 0),
    position_id text,
    decision_key text,
    request jsonb NOT NULL,
    context jsonb NOT NULL,
    status text NOT NULL DEFAULT 'PREPARED',
    exchange_order_id text,
    finalized boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (account_key, decision_key)
);
CREATE UNIQUE INDEX IF NOT EXISTS bot_one_pending_symbol
    ON bot_order_intents(account_key, symbol) WHERE NOT finalized;
CREATE UNIQUE INDEX IF NOT EXISTS bot_one_pending_account
    ON bot_order_intents(account_key) WHERE NOT finalized;

CREATE TABLE IF NOT EXISTS bot_execution_fills (
    account_key text NOT NULL,
    execution_id text NOT NULL,
    client_order_id varchar(36) NOT NULL REFERENCES bot_order_intents(client_order_id),
    order_id text NOT NULL,
    position_id text NOT NULL,
    symbol text NOT NULL,
    side text NOT NULL,
    settle_type text NOT NULL,
    price numeric NOT NULL CHECK (price > 0),
    size numeric NOT NULL CHECK (size > 0),
    fee numeric,
    loss_gain numeric,
    executed_at timestamptz NOT NULL,
    raw jsonb NOT NULL,
    PRIMARY KEY(account_key, execution_id)
);
CREATE INDEX IF NOT EXISTS bot_fills_position ON bot_execution_fills(account_key, position_id);

CREATE TABLE IF NOT EXISTS bot_position_baselines (
    account_key text NOT NULL,
    position_id text NOT NULL,
    symbol text NOT NULL,
    side text NOT NULL,
    size numeric NOT NULL CHECK(size > 0),
    price numeric NOT NULL CHECK(price > 0),
    opened_at timestamptz NOT NULL,
    PRIMARY KEY(account_key, position_id)
);

CREATE TABLE IF NOT EXISTS bot_account_fees (
    account_key text NOT NULL,
    symbol text NOT NULL,
    business_date date NOT NULL,
    fee_type text NOT NULL,
    delivery_amount numeric NOT NULL,
    delivered_at timestamptz NOT NULL,
    raw jsonb NOT NULL,
    PRIMARY KEY(account_key, symbol, business_date, fee_type)
);
CREATE TABLE IF NOT EXISTS bot_fee_sync (
    account_key text PRIMARY KEY,
    through_date date NOT NULL
);

-- Existing pnl remains gross trading P/L. Historical estimates are not rewritten.
ALTER TABLE trades ADD COLUMN IF NOT EXISTS gross_pnl numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_fee numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_fee numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS net_pnl_before_carry numeric;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS accounting_version text NOT NULL DEFAULT 'legacy_estimate';
ALTER TABLE trades ADD COLUMN IF NOT EXISTS cost_status text NOT NULL DEFAULT 'unverified';
COMMIT;
