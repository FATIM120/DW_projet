-- File: models/star/bridge_review_topic.sql
{{ config(
    materialized='table',
    schema='star'
) }}

WITH split_topics AS (
    SELECT
        ROW_NUMBER() OVER () AS review_id,
        TRIM(UNNEST(STRING_TO_ARRAY(topics, ','))) AS topic
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE topics IS NOT NULL
)
SELECT
    r.review_id,
    dt.topic_id
FROM split_topics st
JOIN {{ ref('fact_reviews') }} r ON r.review_id = st.review_id
JOIN {{ ref('dim_topic') }} dt ON st.topic = dt.topic_label
WHERE st.topic != ''