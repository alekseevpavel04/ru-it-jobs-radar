"""The site's Telegram skill data kept up to date without the local stack.

Weekly counts live in git under aggregates/telegram/ (radar.autopilot.aggregates); a daily GitHub
Actions run (radar.autopilot.daily) adds the new posts to them and radar.site.skill_trends reads them
with --from-aggregates. radar.autopilot.seed writes the history once from the local ClickHouse.
"""
