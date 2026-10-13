-- The ClickHouse folding of a skill spelling (macro skill_key) gives the key radar.extract.skills.key
-- gave the same spelling when dbt/skill_seed.py wrote the seed, so the marts find every name and alias.
SELECT spelling, spelling_key, {{ skill_key('spelling') }} AS folded
FROM {{ ref('skill_dictionary_v1') }}
WHERE folded != spelling_key
