-- File: models/star/dim_location.sql
{{ config(
    materialized='table',
    schema='star'
) }}

SELECT
    ROW_NUMBER() OVER () AS location_id,
    city,
    location_detail,
    city || ' - ' || location_detail AS location_name
FROM (
    SELECT DISTINCT city, location AS location_detail
    FROM {{ source('public', 'stg_avis_bancaires_transformed') }}
    WHERE city IS NOT NULL AND location IS NOT NULL
) AS unique_locations