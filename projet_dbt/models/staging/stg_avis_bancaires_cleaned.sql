{{ config(
    materialized = 'table'
)}}


SELECT
"Bank",
"City",
"Branch",
"Location",
"Review",
"Rating",
"Date"
FROM {{ source('public','avis_bancaires')}}
WHERE 
    "Review" IS NOT NULL 
    AND "Review" != ''
    AND LOWER("Review") NOT IN ('pas de text' , '', 'null')