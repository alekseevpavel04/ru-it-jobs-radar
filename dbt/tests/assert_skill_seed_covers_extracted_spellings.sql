{{ config(enabled=var('extraction_model') != '', severity='warn') }}

-- Skill spellings the extraction model wrote after dbt/skill_seed.py last ran: the seed has not resolved
-- them, so they count in no skill. A warning, not an error: the corpus run keeps writing new spellings
-- between seed refreshes, and a nightly build should not fail on that. Fix: python dbt/skill_seed.py,
-- then rebuild from the seed.
SELECT source, sum(skills_unchecked_n) AS unchecked, sum(skills_written_n) AS written
FROM {{ ref('int_extracted_fields') }}
GROUP BY source
HAVING unchecked > 0
