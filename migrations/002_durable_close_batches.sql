BEGIN;

-- A decision can wait here even while the account's single order slot is busy.
CREATE TABLE IF NOT EXISTS bot_close_batches (
    batch_id varchar(36) PRIMARY KEY,
    account_key text NOT NULL,
    symbol text NOT NULL,
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    completed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    CHECK (payload->>'id' = batch_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS bot_one_active_close_batch_symbol
    ON bot_close_batches(account_key, symbol) WHERE NOT completed;

-- Preserve unfinished decisions written by 2d0d4d60 before the first order.
-- Runtime fallback also covers an old process writing after this migration.
INSERT INTO bot_close_batches(batch_id, account_key, symbol, payload, created_at)
SELECT batch_id, account_key, symbol, payload, created_at
FROM (
    SELECT DISTINCT ON (account_key, context #>> '{close_batch,id}')
        context #>> '{close_batch,id}' AS batch_id, account_key, symbol,
        context->'close_batch' AS payload, created_at
    FROM bot_order_intents
    WHERE settle_type='CLOSE' AND context ? 'close_batch'
      AND NOT (context @> '{"close_batch_done": true}'::jsonb)
    ORDER BY account_key, context #>> '{close_batch,id}', created_at
) AS legacy
ON CONFLICT (batch_id) DO NOTHING;

COMMIT;
