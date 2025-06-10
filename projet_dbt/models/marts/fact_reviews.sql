-- File: models/star/fact_reviews.sql
{{ config(
    materialized='table',
    schema='star',
    unique_key='review_id'
) }}

SELECT
    ROW_NUMBER() OVER () AS review_id,
    db.bank_id,
    dbr.branch_id,
    dl.location_id,
    ds.sentiment_id,
    dlng.language_id,
    t.review AS review_text,
    t.rating,
    t.date AS review_date
FROM {{ source('public', 'stg_avis_bancaires_transformed') }} AS t
JOIN {{ ref('dim_bank') }} AS db ON t.bank = db.bank_name
JOIN {{ ref('dim_branch') }} AS dbr ON t.branch = dbr.branch_name
JOIN {{ ref('dim_location') }} AS dl ON t.city = dl.city AND t.location = dl.location_detail
JOIN {{ ref('dim_sentiment') }} AS ds ON t.sentiment = ds.sentiment_label
JOIN {{ ref('dim_language') }} AS dlng ON t.language = dlng.language_label
WHERE t.review IS NOT NULL
AND t.rating IS NOT NULL
AND t.date IS NOT NULL