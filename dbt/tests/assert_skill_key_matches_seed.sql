-- The ClickHouse folding and hash of a skill spelling (macros skill_key, skill_hash) give the key and the
-- hash radar.extract.skills.key and dbt/skill_seed.py gave the dictionary's own spellings, so the marts
-- find every name and alias, and corpus spellings by the same hash.
SELECT spelling, spelling_key, key_hash, {{ skill_key('spelling') }} AS folded, {{ skill_hash('spelling') }} AS hashed
FROM {{ ref('skill_dictionary_v1') }}
WHERE origin = 'dictionary' AND (folded != spelling_key OR hashed != key_hash)
