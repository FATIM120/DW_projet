{{ config(materialized='table') }}
WITH enriched_data AS (
    SELECT bank, city, branch, location, review, rating, date, language, sentiment, topics
    FROM {{ source('public','stg_avis_bancaires_enriched') }}
),
transformed_data AS (
    SELECT
        bank,
        city,
        -- Nettoyer la location en supprimant les motifs comme "X5X2+P4X", les caractères spéciaux comme "", et la virgule initiale
        TRIM(LEADING ',' FROM TRIM(REGEXP_REPLACE(
            REGEXP_REPLACE(location, '[]', '', 'g'), -- Supprimer les caractères spéciaux comme ""
            '[A-Z0-9]{4}\+[A-Z0-9]{3}', '', 'g'      -- Supprimer les motifs comme "X5X2+P4X"
        ))) AS location_cleaned,
        branch,
        review,
        rating,
        date,
        language,
        sentiment,
        topics
    FROM enriched_data
),
final_transformed_data AS (
    SELECT
        bank,
        city,
        -- Concaténer branch et location_cleaned pour former la nouvelle valeur de branch
        TRIM(branch || ' - ' || location_cleaned) AS branch,
        location_cleaned AS location,
        REGEXP_REPLACE(LOWER(review), '[[:punct:]]', '', 'g') AS review_cleaned,
        COALESCE(CAST(rating AS INTEGER), 0) AS rating_clean,
        -- Extraire uniquement l'année de date_clean
        EXTRACT(YEAR FROM (
            CASE 
                WHEN date ~ 'il y a un[[:space:]\xA0]*an' THEN CAST(CURRENT_DATE - INTERVAL '1 year' AS DATE)
                WHEN date ~ 'il y a ([0-9]+)[[:space:]\xA0]*ans?' THEN CAST(CURRENT_DATE - (INTERVAL '1 year' * SUBSTRING(date FROM 'il y a ([0-9]+)')::INT) AS DATE)
                ELSE CURRENT_DATE END
        ))::INTEGER AS date_clean,
        language,
        sentiment,
        topics
    FROM transformed_data
)
SELECT bank, city, branch, location, review_cleaned AS review, rating_clean AS rating, date_clean AS date, language, sentiment, topics
FROM final_transformed_data