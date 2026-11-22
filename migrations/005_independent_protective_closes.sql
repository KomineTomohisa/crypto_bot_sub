BEGIN;
-- Keep symbol-wide exclusion, including an unknown OPEN on that symbol.
-- The account advisory lock in prepare still blocks all new exposure while
-- any account order is unresolved. Only verified durable closes bypass it.
LOCK TABLE bot_order_intents IN SHARE ROW EXCLUSIVE MODE;
CREATE UNIQUE INDEX IF NOT EXISTS bot_one_pending_open_account
    ON bot_order_intents(account_key) WHERE NOT finalized AND settle_type='OPEN';
DROP INDEX IF EXISTS bot_one_pending_account;
COMMIT;
