# Data



Expected columns: `subscription_id, created_at, channel, utm_campaign, plan, canceled_at, ended_at, end_reason`.

Only the aggregated output (`web/data.js`, counts by cohort, source, plan and payment number) is committed. It contains no subscription ids or dates for individual subscribers.
