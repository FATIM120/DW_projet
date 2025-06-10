import pandas as pd
import numpy as np
from transformers import pipeline
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
import nltk
import re
from db_config import get_db_connection
import warnings
import torch
import psycopg2
from collections import defaultdict, Counter
import string
import spacy

# --- Download required NLTK data ---
def download_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"), 
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")
    ]
    for resource_path, resource_id in resources:
        try:
            nltk.data.find(resource_path)
            print(f"Resource {resource_id} already downloaded.")
        except LookupError:
            print(f"Resource {resource_id} not found. Downloading...")
            nltk.download(resource_id, quiet=True)
            print(f"Resource {resource_id} downloaded.")

download_nltk_data()

# Ignorer les avertissements spécifiques de pandas liés à la connexion DBAPI2
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable")

# Vérifier la détection du GPU avec PyTorch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("GPU Device:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected, using CPU.")

# Initialiser les pipelines avec PyTorch et GPU (ou CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"Initializing pipelines on device: {device}")
language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", framework="pt", device=device, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="pt", device=device, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# Extraction des données nettoyées depuis PostgreSQL
conn = get_db_connection()
if conn is None:
    raise Exception("Impossible de se connecter à la base de données.")
cursor = conn.cursor()

df = pd.read_sql("SELECT * FROM stg_avis_bancaires_cleaned", conn, coerce_float=True)

# Remplacer les valeurs NaN ou non-string dans 'review' par une chaîne vide
df["review"] = df["review"].fillna("").astype(str)

# Convertir en Dataset pour traitement par lots (ne garder que 'review' pour l'enrichissement)
dataset = Dataset.from_pandas(df[["review"]])

# Fonction pour détecter la langue (compatible avec map)
def detect_language_batch(batch):
    valid_reviews = [r for r in batch["review"] if isinstance(r, str) and r.strip()]
    if not valid_reviews:
        return {"language": ["unknown"] * len(batch["review"])}
    
    try:
        results = language_detector(valid_reviews, truncation=True, max_length=256, batch_size=32)
        lang_map = {review: result["label"] for review, result in zip(valid_reviews, results)}
        final_languages = [lang_map.get(review, "unknown") for review in batch["review"]]
        return {"language": final_languages}
    except Exception as e:
        print(f"Erreur lors de la détection de langue : {e}")
        return {"language": ["unknown"] * len(batch["review"])}

# Fonction pour analyser le sentiment (compatible avec map)
def analyze_sentiment_batch(batch):
    valid_reviews = [r for r in batch["review"] if isinstance(r, str) and r.strip()]
    if not valid_reviews:
        return {"sentiment": ["Neutral"] * len(batch["review"])}
        
    try:
        results = sentiment_analyzer(valid_reviews, truncation=True, max_length=256, batch_size=32)
        sentiment_map = {}
        for review, result in zip(valid_reviews, results):
            label = result["label"]
            if "5 stars" in label or "4 stars" in label:
                sentiment_map[review] = "Positive"
            elif "1 star" in label or "2 stars" in label:
                sentiment_map[review] = "Negative"
            else:
                sentiment_map[review] = "Neutral"
        
        final_sentiments = [sentiment_map.get(review, "Neutral") for review in batch["review"]]
        return {"sentiment": final_sentiments}
    except Exception as e:
        print(f"Erreur lors de l'analyse de sentiment : {e}")
        return {"sentiment": ["Neutral"] * len(batch["review"])}

# Appliquer les fonctions de mappage
print("Détection de la langue...")
dataset = dataset.map(detect_language_batch, batched=True, batch_size=32)
print("Analyse des sentiments...")
dataset = dataset.map(analyze_sentiment_batch, batched=True, batch_size=32)

# Ajouter les résultats au DataFrame
df["language"] = dataset["language"]
df["sentiment"] = dataset["sentiment"]

# --- FILTRAGE : Garder seulement les avis en français et anglais ---
print(f"Nombre total d'avis avant filtrage: {len(df)}")
df_filtered = df[df["language"].isin(["fr", "en"])].copy()
print(f"Nombre d'avis après filtrage (français et anglais): {len(df_filtered)}")

if len(df_filtered) == 0:
    print("Aucun avis en français ou anglais trouvé!")
    raise Exception("Pas de données à traiter après filtrage des langues.")

# ===================================================================
# NOUVELLE CLASSE POUR L'EXTRACTION DE TOPICS SPÉCIFIQUES
# ===================================================================
class BankingTopicExtractor:
    def __init__(self):
        # Patterns spécifiques au domaine bancaire
        self.banking_patterns = {
            # SERVICE ET ACCUEIL
            'service': {
                'positive': [
                    r'(excellent|bon|super|parfait|génial|top)\s+(service|accueil)',
                    r'(service|accueil)\s+(excellent|parfait|impeccable|remarquable)',
                    r'très\s+(bon|bien)\s+(service|accueil)',
                    r'(service|accueil)\s+de\s+qualité',
                    r'personnel\s+(aimable|sympa|gentil|professionnel)',
                    r'(excellent|great|amazing|perfect)\s+(service|customer service)',
                    r'friendly\s+(staff|service)',
                    r'professional\s+(service|staff)'
                ],
                'negative': [
                    r'(mauvais|horrible|nul|catastrophique)\s+(service|accueil)',
                    r'(service|accueil)\s+(décevant|lamentable|inadmissible)',
                    r'très\s+(mauvais|mal)\s+(service|accueil)',
                    r'aucun\s+(service|accueil)',
                    r'personnel\s+(désagréable|impoli|incompétent)',
                    r'(terrible|awful|bad|poor)\s+(service|customer service)',
                    r'rude\s+(staff|service)',
                    r'unprofessional\s+(service|staff)'
                ]
            },
            
            # FRAIS ET COÛTS
            'frais': {
                'negative': [
                    r'(frais|commission|coût)\s+(élevé|cher|excessif|abusif)',
                    r'trop\s+(cher|de frais)',
                    r'(frais|commission)\s+cachés?',
                    r'beaucoup\s+de\s+frais',
                    r'(expensive|high|hidden)\s+(fees|charges|costs)',
                    r'too\s+(expensive|costly)',
                    r'outrageous\s+(fees|charges)'
                ],
                'positive': [
                    r'(frais|commission)\s+(raisonnable|correct|acceptable)',
                    r'pas\s+de\s+frais',
                    r'sans\s+frais',
                    r'(reasonable|fair|low)\s+(fees|charges)',
                    r'no\s+(fees|charges)',
                    r'fee-free'
                ]
            },
            
            # ATTENTE ET RAPIDITÉ
            'attente': {
                'negative': [
                    r'(longue|trop)\s+d?\'?attente',
                    r'attendre\s+(longtemps|trop)',
                    r'(queue|file)\s+(interminable|longue)',
                    r'(long|endless)\s+(wait|queue|line)',
                    r'waiting\s+(forever|too long)',
                    r'slow\s+(service|processing)'
                ],
                'positive': [
                    r'(rapide|vite|efficace)',
                    r'pas\s+d\'attente',
                    r'service\s+rapide',
                    r'(quick|fast|efficient)\s+(service|processing)',
                    r'no\s+wait',
                    r'prompt\s+service'
                ]
            },
            
            # CONSEILLERS
            'conseiller': {
                'positive': [
                    r'(bon|excellent|compétent)\s+(conseiller|banquier)',
                    r'conseiller\s+(aimable|professionnel|à l\'écoute)',
                    r'très\s+bien\s+conseillé',
                    r'(helpful|knowledgeable|professional)\s+(advisor|banker)',
                    r'great\s+(advice|advisor)'
                ],
                'negative': [
                    r'(mauvais|incompétent)\s+(conseiller|banquier)',
                    r'conseiller\s+(désagréable|inutile)',
                    r'mal\s+conseillé',
                    r'(unhelpful|rude|incompetent)\s+(advisor|banker)',
                    r'poor\s+(advice|advisor)'
                ]
            },
            
            # DISTRIBUTEURS / ATM
            'distributeur': {
                'negative': [
                    r'(distributeur|dab|atm)\s+(en panne|cassé|hors service)',
                    r'problème\s+(distributeur|dab|atm)',
                    r'(atm|machine)\s+(broken|out of order|not working)'
                ],
                'positive': [
                    r'(distributeur|dab|atm)\s+(pratique|accessible)',
                    r'bon\s+(distributeur|dab)',
                    r'(convenient|accessible)\s+(atm|machine)'
                ]
            },
            
            # HORAIRES
            'horaires': {
                'negative': [
                    r'horaires?\s+(limité|court|insuffisant)',
                    r'pas\s+assez\s+ouvert',
                    r'fermé\s+trop\s+tôt',
                    r'(limited|short|bad)\s+(hours|opening times)',
                    r'closed\s+too\s+early'
                ],
                'positive': [
                    r'horaires?\s+(pratique|étendu|large)',
                    r'bien\s+ouvert',
                    r'(convenient|extended|good)\s+(hours|opening times)'
                ]
            },
            
            # APPLICATION MOBILE / DIGITAL
            'application': {
                'positive': [
                    r'(application|app)\s+(pratique|bien|super)',
                    r'bonne\s+(application|app)',
                    r'(great|good|useful)\s+(app|application)',
                    r'easy\s+to\s+use\s+(app|application)'
                ],
                'negative': [
                    r'(application|app)\s+(nulle|bugge|plante)',
                    r'problème\s+(application|app)',
                    r'(terrible|buggy|slow)\s+(app|application)',
                    r'(app|application)\s+(crashes|freezes)'
                ]
            }
        }
        
        # Modificateurs d'intensité
        self.intensifiers = {
            'très': 1.5, 'vraiment': 1.3, 'super': 1.4, 'extrêmement': 1.6,
            'plutôt': 0.8, 'assez': 0.9,
            'very': 1.5, 'really': 1.3, 'extremely': 1.6, 'quite': 0.9
        }
        
        # Negateurs
        self.negators = ['ne', 'pas', 'not', 'never', 'jamais', 'aucun', 'aucune']

    def extract_topics_from_text(self, text, language='fr'):
        """Extrait les topics d'un texte donné"""
        if not isinstance(text, str) or not text.strip():
            return "aucun topic"
            
        text = text.lower()
        extracted_topics = []
        
        # Pour chaque catégorie de patterns
        for category, sentiments in self.banking_patterns.items():
            for sentiment, patterns in sentiments.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        # Construire le topic
                        topic_base = self._get_category_label(category, language)
                        sentiment_label = 'positif' if sentiment == 'positive' else 'négatif'
                        
                        if language == 'en':
                            sentiment_label = 'positive' if sentiment == 'positive' else 'negative'
                        
                        # Créer le topic final
                        if sentiment == 'positive':
                            if category == 'service':
                                topic = "bon service" if language == 'fr' else "good service"
                            elif category == 'frais':
                                topic = "frais raisonnables" if language == 'fr' else "reasonable fees"
                            elif category == 'attente':
                                topic = "service rapide" if language == 'fr' else "fast service"
                            elif category == 'conseiller':
                                topic = "bon conseiller" if language == 'fr' else "helpful advisor"
                            elif category == 'distributeur':
                                topic = "distributeur pratique" if language == 'fr' else "convenient ATM"
                            elif category == 'horaires':
                                topic = "bons horaires" if language == 'fr' else "good hours"
                            elif category == 'application':
                                topic = "bonne application" if language == 'fr' else "good app"
                            else:
                                topic = f"{topic_base} {sentiment_label}"
                        else:  # negative
                            if category == 'service':
                                topic = "mauvais service" if language == 'fr' else "bad service"
                            elif category == 'frais':
                                topic = "frais élevés" if language == 'fr' else "high fees"
                            elif category == 'attente':
                                topic = "longue attente" if language == 'fr' else "long wait"
                            elif category == 'conseiller':
                                topic = "mauvais conseiller" if language == 'fr' else "poor advisor"
                            elif category == 'distributeur':
                                topic = "distributeur en panne" if language == 'fr' else "broken ATM"
                            elif category == 'horaires':
                                topic = "horaires limités" if language == 'fr' else "limited hours"
                            elif category == 'application':
                                topic = "application défaillante" if language == 'fr' else "buggy app"
                            else:
                                topic = f"{topic_base} {sentiment_label}"
                        
                        extracted_topics.append(topic)
        
        # Si aucun topic spécifique trouvé, essayer extraction générale
        if not extracted_topics:
            general_topic = self._extract_general_topic(text, language)
            extracted_topics.append(general_topic)
        
        # Retourner le topic le plus pertinent ou combiner
        if len(extracted_topics) == 1:
            return extracted_topics[0]
        elif len(extracted_topics) > 1:
            # Prioriser certains topics
            priority_topics = [t for t in extracted_topics if any(word in t.lower() 
                             for word in ['service', 'frais', 'attente', 'conseiller'])]
            if priority_topics:
                return priority_topics[0]
            else:
                return extracted_topics[0]
        
        return "expérience générale" if language == 'fr' else "general experience"

    def _get_category_label(self, category, language):
        """Convertit les catégories en labels lisibles"""
        labels = {
            'fr': {
                'service': 'service',
                'frais': 'frais',
                'attente': 'attente',
                'conseiller': 'conseil',
                'distributeur': 'distributeur',
                'horaires': 'horaires',
                'application': 'application'
            },
            'en': {
                'service': 'service',
                'frais': 'fees',
                'attente': 'waiting',
                'conseiller': 'advisor',
                'distributeur': 'ATM',
                'horaires': 'hours',
                'application': 'app'
            }
        }
        return labels.get(language, labels['fr']).get(category, category)

    def _extract_general_topic(self, text, language):
        """Extraction de topic général si aucun pattern spécifique"""
        # Mots-clés généraux positifs/négatifs
        positive_words_fr = ['bon', 'bien', 'super', 'excellent', 'parfait', 'satisfait', 'content']
        negative_words_fr = ['mauvais', 'mal', 'nul', 'horrible', 'décevant', 'mécontent']
        
        positive_words_en = ['good', 'great', 'excellent', 'perfect', 'satisfied', 'happy']
        negative_words_en = ['bad', 'terrible', 'awful', 'poor', 'disappointed', 'unhappy']
        
        if language == 'fr':
            if any(word in text for word in positive_words_fr):
                return "expérience positive"
            elif any(word in text for word in negative_words_fr):
                return "expérience négative"
        else:
            if any(word in text for word in positive_words_en):
                return "positive experience"
            elif any(word in text for word in negative_words_en):
                return "negative experience"
        
        return "expérience générale" if language == 'fr' else "general experience"

    def extract_topics_batch(self, reviews, languages):
        """Traite un lot de reviews"""
        topics = []
        for i, review in enumerate(reviews):
            lang = languages[i] if i < len(languages) else 'fr'
            topic = self.extract_topics_from_text(review, lang)
            topics.append(topic)
        return topics

# ===================================================================
# FONCTION D'EXTRACTION DE TOPICS (REMPLACE LDA)
# ===================================================================
def extract_meaningful_topics_with_lda(texts, languages, n_topics=5, words_per_topic=2):
    """
    Version remplacée qui utilise des patterns au lieu de LDA
    pour obtenir des topics comme 'mauvais service', 'frais élevés', etc.
    """
    extractor = BankingTopicExtractor()
    topics = []
    
    print("Extraction des topics avec patterns linguistiques spécialisés...")
    
    for i, text in enumerate(texts):
        lang = languages[i] if i < len(languages) else 'fr'
        topic = extractor.extract_topics_from_text(text, lang)
        topics.append(topic)
    
    # Statistiques
    from collections import Counter
    topic_counts = Counter(topics)
    
    print(f"\n=== TOPICS SPÉCIFIQUES DÉCOUVERTS ===")
    print(f"Exemples de topics générés:")
    for topic, count in topic_counts.most_common(10):
        percentage = (count / len(topics)) * 100
        print(f"• {topic}: {count} avis ({percentage:.1f}%)")
    
    return topics

# Extraction des topics avec la nouvelle méthode
print("Extraction des topics significatifs...")
topics = extract_meaningful_topics_with_lda(
    df_filtered["review"].tolist(), 
    df_filtered["language"].tolist(),
    n_topics=5, 
    words_per_topic=2
)

print("Exemples de topics générés:", topics[:10])
df_filtered["topics"] = topics

# Nettoyage supplémentaire des ratings
def clean_rating(rating):
    if isinstance(rating, (int, float)):
        return int(rating)
    if isinstance(rating, str):
        match = re.search(r"\d", rating)
        if match:
            return int(match.group(0))
    return np.nan

df_filtered["rating"] = df_filtered["rating"].apply(clean_rating)
df_filtered.dropna(subset=["rating"], inplace=True)
df_filtered["rating"] = df_filtered["rating"].astype(int)

# Créer ou remplacer la table avec psycopg2
print("Préparation de la base de données...")
cursor.execute("""
    DROP TABLE IF EXISTS public.stg_avis_bancaires_enriched;
    CREATE TABLE public.stg_avis_bancaires_enriched (
        bank TEXT,
        city TEXT,
        branch TEXT,
        location TEXT,
        review TEXT,
        language TEXT,
        sentiment TEXT,
        topics TEXT,
        rating INTEGER,
        date VARCHAR
    );
""")
conn.commit()

# Insérer les données dans la table
print(f"Insertion de {len(df_filtered)} lignes enrichies...")
insert_query = """
    INSERT INTO public.stg_avis_bancaires_enriched (bank, city, branch, location, review, language, sentiment, topics, rating, date)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""
data_to_insert = []
for row in df_filtered.to_dict("records"):
    data_to_insert.append((
        row.get("bank") or "",
        row.get("city") or "",
        row.get("branch") or "",
        row.get("location") or "",
        row.get("review") or "",
        row.get("language") or "unknown",
        row.get("sentiment") or "Neutral",
        row.get("topics") or "Aucun topic",
        int(row["rating"]), 
        row.get("date")
    ))

try:
    cursor.executemany(insert_query, data_to_insert)
    conn.commit()
    print("Données insérées avec succès.")
except Exception as e:
    print(f"Erreur lors de l'insertion des données : {e}")
    conn.rollback()

# Fermer la connexion
cursor.close()
conn.close()

# Analyses et statistiques détaillées
print(f"\n=== STATISTIQUES FINALES DÉTAILLÉES ===")
print(f"Nombre total d'avis traités: {len(df_filtered)}")

# Répartition par langue
language_counts = df_filtered["language"].value_counts().to_dict()
print(f"Répartition par langue: {language_counts}")

# Répartition des sentiments
sentiment_counts = df_filtered["sentiment"].value_counts().to_dict()
print(f"Répartition des sentiments: {sentiment_counts}")

# Analyse détaillée des topics
print(f"\n=== ANALYSE DES TOPICS ===")
topic_counts = df_filtered["topics"].value_counts()
print(f"Nombre total de topics uniques générés: {len(topic_counts)}")

print(f"\nTop 10 des topics les plus fréquents:")
for topic, count in topic_counts.head(10).items():
    percentage = (count / len(df_filtered)) * 100
    print(f"  • {topic}: {count} avis ({percentage:.1f}%)")

# Corrélation topics/sentiments
print(f"\n=== CORRÉLATION TOPICS/SENTIMENTS ===")
topic_sentiment = df_filtered.groupby(['topics', 'sentiment']).size().unstack(fill_value=0)
if not topic_sentiment.empty:
    print("Répartition des sentiments par topic principal:")
    for topic in topic_counts.head(5).index:
        if topic in topic_sentiment.index:
            row = topic_sentiment.loc[topic]
            total = row.sum()
            if total > 0:
                print(f"  {topic}:")
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    if sentiment in row:
                        pct = (row[sentiment] / total) * 100
                        print(f"    - {sentiment}: {row[sentiment]} ({pct:.1f}%)")

print("\nEnrichissement terminé avec extraction de topics spécifiques !")