-- File: models/star/dim_topic.sql
{{ config(
    materialized='table',
    schema='star',
    unique_key='topic_id'
) }}

WITH split_topics AS (
    SELECT DISTINCT
        TRIM(UNNEST(STRING_TO_ARRAY(topics, ','))) AS topic
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE topics IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER () AS topic_id,
    topic AS topic_label
FROM split_topics
WHERE topic != ''