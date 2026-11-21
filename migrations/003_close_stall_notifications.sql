BEGIN;

-- Separate notification metadata from the immutable order/target payload.
ALTER TABLE bot_close_batches ADD COLUMN IF NOT EXISTS alert_last_attempt_at timestamptz;
ALTER TABLE bot_close_batches ADD COLUMN IF NOT EXISTS alert_last_attempt_kind text;
ALTER TABLE bot_close_batches ADD COLUMN IF NOT EXISTS alert_warning_sent_at timestamptz;
ALTER TABLE bot_close_batches ADD COLUMN IF NOT EXISTS alert_recovery_sent_at timestamptz;
ALTER TABLE bot_close_batches ADD COLUMN IF NOT EXISTS alert_claim_token varchar(36);
ALTER TABLE bot_close_batches ADD COLUMN IF NOT EXISTS alert_claim_until timestamptz;

COMMIT;
