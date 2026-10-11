-- Read-only account of the analyst agent, its only definition. Two callers apply it with
-- the full-rights account: infra/clickhouse/init.sh on every `up` (the clickhouse-init service) and
-- scripts/apply_agent_user.py from the host. Both substitute {ro_user} and {ro_password} from
-- CLICKHOUSE_RO_USER and CLICKHOUSE_RO_PASSWORD after checking their alphabet, since CREATE USER
-- takes no query parameters. Every statement is idempotent, so either may run any number of times.
--
-- readonly = 1 forbids writes, DDL and changing any setting from the query, the URL or a SETTINGS
-- clause, so the limits below hold for whatever SQL reaches the server. The agent checks the same
-- things in code first (radar.agent.sqlcheck); this file is the layer that does not trust that code.
CREATE SETTINGS PROFILE OR REPLACE agent_readonly SETTINGS
    readonly = 1,
    max_execution_time = 10,
    timeout_overflow_mode = 'throw',
    max_result_rows = 10000,
    max_result_bytes = 50000000,
    result_overflow_mode = 'throw',
    max_rows_to_read = 100000000,
    read_overflow_mode = 'throw',
    max_memory_usage = 1000000000,
    max_threads = 2;

CREATE USER IF NOT EXISTS {ro_user};

ALTER USER {ro_user} IDENTIFIED WITH sha256_password BY '{ro_password}' SETTINGS PROFILE 'agent_readonly';

REVOKE ALL ON *.* FROM {ro_user};

-- marts: what the agent answers from. agent_snapshot: the frozen copy of the marts the control set
-- is scored on (radar.agent.snapshot); the evaluation runs the agent's SQL there as this same account.
GRANT SELECT ON marts.* TO {ro_user};

GRANT SELECT ON agent_snapshot.* TO {ro_user};
