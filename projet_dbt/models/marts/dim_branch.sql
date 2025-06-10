-- File: models/star/dim_branch.sql
{{ config(
    materialized='table',
    schema='star'
) }}

SELECT
    ROW_NUMBER() OVER () AS branch_id,
    t.branch_name
FROM (
    SELECT DISTINCT branch AS branch_name, bank
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE branch IS NOT NULL
) AS t