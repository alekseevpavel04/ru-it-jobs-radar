-- Telegram versions the corpus run never extracts, with the rule that drops them, written by
-- `python -m radar.extract.corpus_run dropped` (radar.extract.corpus_drops). Every statement is idempotent.
--
-- One row per dropped version (source, source_id, content_hash) and check. A check inserts all its rows
-- at once and then deletes the rows of earlier checks; read the rows of max(check_id).

CREATE DATABASE IF NOT EXISTS extract;

CREATE TABLE IF NOT EXISTS extract.vacancy_dropped
(
    source LowCardinality(String),
    source_id String,
    content_hash String,
    dt Date COMMENT 'raw partition of the version',
    reason LowCardinality(String) COMMENT 'digest, link_stub or channel_lang (radar.extract.telegram_corpus.drop_reason)',
    rules_sha256 String COMMENT 'sha256 of min_content_chars and the channel languages the check used',
    check_id String COMMENT 'the check that wrote the row, UTC timestamp yyyymmddThhmmss',
    checked_at DateTime64(3, 'UTC'),
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
ORDER BY (source, source_id, content_hash, check_id);
