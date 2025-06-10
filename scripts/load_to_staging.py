import pandas as pd
import psycopg2
from db_config import get_db_connection, close_db_connection

def load_csv_to_staging():
    """Charge les données du CSV directement dans PostgreSQL sans transformation."""
    try:
        # Établir la connexion
        conn = get_db_connection()
        if conn is None:
            raise Exception("Impossible de se connecter à PostgreSQL")

        # Lire le fichier CSV pour voir les colonnes
        df = pd.read_csv('/home/fatim/data/avies.csv', encoding='utf-8')
        print(f"Données lues : {len(df)} lignes")
        print(f"Colonnes trouvées : {list(df.columns)}")

        # Créer la table basée sur les colonnes réelles du CSV
        cursor = conn.cursor()
        
        # Supprimer la table si elle existe
        cursor.execute("DROP TABLE IF EXISTS public.avis_bancaires;")
        
        # Créer la table avec toutes les colonnes en TEXT pour éviter les problèmes de type
        columns_def = ', '.join([f'"{col}" TEXT' for col in df.columns])
        create_table_query = f"""
        CREATE TABLE public.avis_bancaires (
            {columns_def}
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        print("Table avis_bancaires créée.")

        # Préparer la requête d'insertion
        columns_list = ', '.join([f'"{col}"' for col in df.columns])
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_query = f"""
        INSERT INTO public.avis_bancaires ({columns_list})
        VALUES ({placeholders})
        """

        # Convertir les données exactement comme elles sont
        data_tuples = []
        for _, row in df.iterrows():
            # Convertir chaque valeur en string, garder None pour les NaN
            tuple_row = tuple(None if pd.isna(val) else str(val) for val in row)
            data_tuples.append(tuple_row)
        
        # Insertion des données
        cursor.executemany(insert_query, data_tuples)
        conn.commit()
        
        print(f"Chargement réussi : {len(df)} lignes insérées dans avis_bancaires")

        # Vérification rapide
        cursor.execute("SELECT COUNT(*) FROM public.avis_bancaires;")
        count = cursor.fetchone()[0]
        print(f"Vérification : {count} lignes dans la table")

    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
        raise
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        close_db_connection(conn)

if __name__ == "__main__":
    print("Début du chargement des données brutes dans la table de staging...")
    load_csv_to_staging()
    print("Chargement terminé.")