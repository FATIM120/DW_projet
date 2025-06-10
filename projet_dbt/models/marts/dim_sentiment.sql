-- File: models/star/dim_sentiment.sql
{{ config(
    materialized='table',
    schema='star'
) }}

SELECT
    ROW_NUMBER() OVER () AS sentiment_id,
    sentiment_label
FROM (
    SELECT DISTINCT sentiment AS sentiment_label
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE sentiment IS NOT NULL
) AS unique_sentiments