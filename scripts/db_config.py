import psycopg2

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="bank_reviews",
            user="fatim",
            password="fatim"  # Remplacez par votre mot de passe
        )
        return conn
    except psycopg2.Error as e:
        print(f"Erreur de connexion à PostgreSQL : {e}")
        return None

def close_db_connection(conn):
    if conn:
        conn.close()