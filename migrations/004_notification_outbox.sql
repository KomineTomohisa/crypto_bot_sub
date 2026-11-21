BEGIN;

-- No historical backfill. New execution confirmations register delivery here
-- in the same transaction as fills/accounting, using only non-secret metadata.
CREATE TABLE IF NOT EXISTS bot_notification_outbox (
    job_id varchar(36) PRIMARY KEY,
    account_key text NOT NULL,
    event_key text NOT NULL,
    client_order_id varchar(32) REFERENCES bot_order_intents(client_order_id),
    channel text NOT NULL CHECK (channel IN ('email','line')),
    kind text NOT NULL CHECK (kind IN ('entry','exit','error','info','daily_report')),
    recovered boolean NOT NULL DEFAULT false,
    subject text,
    body text,
    route_hash text,
    created_at timestamptz NOT NULL DEFAULT now(),
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    first_attempt_at timestamptz,
    delivered_at timestamptz,
    terminal_reason text,
    last_error_type text,
    claim_token varchar(36),
    claim_until timestamptz,
    UNIQUE (account_key,event_key,channel),
    CHECK ((subject IS NULL) = (body IS NULL))
);
CREATE INDEX IF NOT EXISTS bot_notification_outbox_pending
    ON bot_notification_outbox(account_key,next_attempt_at,created_at)
    WHERE delivered_at IS NULL AND terminal_reason IS NULL;

COMMIT;
