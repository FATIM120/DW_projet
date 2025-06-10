-- File: models/star/dim_bank.sql
{{ config(
    materialized='table',
    schema='star'
) }}

SELECT
    ROW_NUMBER() OVER () AS bank_id,
    bank_name
FROM (
    SELECT DISTINCT bank AS bank_name
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE bank IS NOT NULL
) AS unique_banks