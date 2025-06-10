-- File: cautions/star/dim_language.sql
{{ config(
    materialized='table',
    schema='star',
    unique_key='language_id'
) }}

WITH distinct_languages AS (
    SELECT DISTINCT language
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE language IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER () AS language_id,
    language AS language_label
FROM distinct_languages