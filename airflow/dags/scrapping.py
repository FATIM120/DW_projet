import time
import json
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import random
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Liste des banques et villes
BANKS = ["Al Barid Bank","ATTIJARIWAFA BANK","BANK OF AFRICA","BMCI","SOCIÉTÉ GÉNÉRALE MAROC","CIH BANK","CREDIT DU MAROC","BANK AL YOUSR","BANK ASSAFA","UMNIA BANK","BANQUE CENTRALE POPULAIRE"]

CITIES = ["Casablanca","Rabat","Fes","Tanger","Agadir","Marrakech","sale","Meknes","AL hoceima","Mohammadia","Settat","Arfoud","Nador","Essaouira","Errachidia","Berkane","Asila","Tetouan","Taroudant"]

BASE_URL = "https://www.google.com/maps/search/"
MIN_REVIEWS_PER_BRANCH = 100
MAX_AGENCIES_PER_BANK = 80   


def random_delay(min_seconds=1, max_seconds=3):
    """Attendre un temps aléatoire pour simuler un comportement humain"""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)
    return delay

def handle_consent_popup(page):
    """Gère les popups de consentement de cookies"""
    try:
        # Différents sélecteurs possibles pour les boutons de consentement
        consent_buttons = [
            'button:has-text("Tout accepter")',
            'button:has-text("J\'accepte")',
            'button:has-text("Accepter tout")',
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Agree")'
        ]
        
        for selector in consent_buttons:
            if page.locator(selector).count() > 0:
                logger.info(f"Popup de consentement détecté. Clique sur '{selector}'")
                page.locator(selector).click()
                random_delay(2, 4)
                return True
                
        return False
    except Exception as e:
        logger.warning(f"Erreur lors de la gestion du popup de consentement: {e}")
        return False

def scroller_banques(page):
    """Scrolle la liste des résultats pour charger plus d'agences"""
    try:
        # Utilisation d'un sélecteur CSS plus générique
        results_container = page.locator('div[role="feed"]')
        if results_container.count() > 0:
            logger.info("Scroll des résultats pour charger plus d'agences...")
            for i in range(5):  # Réduit à 5 scrolls
                page.mouse.wheel(0, 1000)
                delay = random_delay(1.5, 3)
                logger.info(f"Scroll {i+1}/5 - Attente de {delay:.1f}s")
            return True
        else:
            logger.warning("Conteneur de résultats non trouvé")
            return False
    except Exception as e:
        logger.error(f"Erreur lors du scroll des résultats: {e}")
        return False

# Dans la fonction cliquer_sur_avis, remplacez le code par cette version améliorée :

def cliquer_sur_avis(page):
    """Clique sur l'onglet Avis d'une agence bancaire"""
    try:
        # Vérifier si nous sommes déjà sur l'onglet des avis
        # Si des éléments d'avis sont visibles, on est déjà sur le bon onglet
        if page.locator('div[data-review-id]').count() > 0:
            logger.info("Déjà sur l'onglet des avis")
            return True
            
        # Essayer d'abord avec un sélecteur plus spécifique pour le tab "Avis"
        reviews_tab = page.locator('button[role="tab"][aria-label*="Avis"]').first
        if reviews_tab.count() > 0:
            logger.info("Clic sur l'onglet Avis (tab)")
            reviews_tab.click()
            random_delay(2, 4)
            return True
        
        # Essayer avec le bouton qui affiche le nombre d'avis
        reviews_count_button = page.locator('button.GQjSyb:has-text("avis")').first
        if reviews_count_button.count() > 0:
            logger.info("Clic sur le bouton de nombre d'avis")
            reviews_count_button.click()
            random_delay(2, 4)
            return True
            
        # Essayer avec "Plus d'avis"
        more_reviews = page.locator('button[aria-label*="Plus d\'avis"]').first
        if more_reviews.count() > 0:
            logger.info("Clic sur 'Plus d'avis'")
            more_reviews.click()
            random_delay(2, 4)
            return True
            
        # En dernier recours, essayer de cliquer sur la note moyenne
        rating_element = page.locator('button.fontDisplayLarge').first
        if rating_element.count() > 0:
            logger.info("Clic sur l'élément de notation")
            rating_element.click()
            random_delay(2, 4)
            return True
            
        logger.warning("Aucun élément d'avis n'a pu être trouvé ou cliqué")
        return False
    except Exception as e:
        logger.error(f"Erreur lors du clic sur l'onglet Avis: {e}")
        return False

def wait_for_reviews_to_load(page):
    """Attend que les avis se chargent"""
    try:
        logger.info("Attente du chargement des avis...")
        review_items = page.locator('div[data-review-id]')
        # Attendre jusqu'à 10 secondes que les avis apparaissent
        review_items.first.wait_for(timeout=10000)
        count = review_items.count()
        logger.info(f"{count} avis chargés initialement")
        return count > 0
    except Exception as e:
        logger.warning(f"Erreur ou timeout lors de l'attente des avis: {e}")
        return False

def charger_et_extraire_avis(page):
    """Charge et extrait les avis sans duplication"""
    if not wait_for_reviews_to_load(page):
        return []
    
    try:
        # Scrolle pour charger plus d'avis
        reviews_container = page.locator('div[role="feed"]')
        if reviews_container.count() > 0:
            logger.info("Scroll pour charger plus d'avis...")
            for i in range(8):
                page.mouse.wheel(0, 1000)
                random_delay(1, 2)
                
        # Développer tous les avis (cliquer sur "Lire la suite")
        more_buttons = page.locator('button.w8nwRe.kyuRq, button[jsaction*="pane.review.expandReview"]')
        count = more_buttons.count()
        if count > 0:
            logger.info(f"Développement de {count} boutons 'plus'...")
            for i in range(count):  # Augmenter pour cliquer sur tous les boutons "lire la suite"
                try:
                    # Essayer de cliquer sur chaque bouton visible
                    if more_buttons.nth(i).is_visible():
                        more_buttons.nth(i).click()
                        random_delay(0.5, 1)
                except Exception as e:
                    logger.warning(f"Erreur en cliquant sur le bouton 'plus' #{i}: {e}")
                    
        # Attendre un peu que tous les avis se développent
        random_delay(2, 3)
                
        # Maintenant, on extrait les avis depuis la page complète
        soup = BeautifulSoup(page.content(), "html.parser")
        
        # Essai avec plusieurs sélecteurs possibles
        review_containers = soup.select('div[data-review-id]') or soup.select('.jftiEf')
        
        if not review_containers:
            logger.warning("Aucun avis trouvé avec les sélecteurs utilisés")
            return []
            
        logger.info(f"Extraction de {len(review_containers)} avis...")
        
        extracted_reviews = []
        seen_reviews = set()  # Ensemble pour suivre les avis déjà vus
        
        for review in review_containers:
            try:
                # Texte de l'avis
                review_text_element = (
                    review.select_one('.wiI7pd') or 
                    review.select_one('.MyEned') or
                    review.select_one('span[jsan*="reviews.snippet.text"]')
                )

                review_text = review_text_element.text.strip() if review_text_element else "Pas de texte"
                
                # Note (étoiles)
                rating_element = (
                    review.select_one('span[aria-label*="étoile"]') or
                    review.select_one('span[aria-label*="star"]') or
                    review.select_one('.kvMYJc')
                )
                rating = rating_element["aria-label"] if rating_element and rating_element.has_attr("aria-label") else "Pas de note"
                
                # Date
                date_element = (
                    review.select_one('.rsqaWe') or
                    review.select_one('span[jsan*="reviews.snippet.publish_date"]')
                )
                date = date_element.text.strip() if date_element else "Pas de date"
                
                # Créer un hachage unique pour cet avis
                review_hash = hash(f"{review_text}|{rating}|{date}")
                
                # Vérifier si cet avis a déjà été vu
                if review_hash not in seen_reviews:
                    seen_reviews.add(review_hash)
                    extracted_reviews.append({
                        "Review": review_text,
                        "Rating": rating,
                        "Date": date
                    })
            except Exception as e:
                logger.warning(f"Erreur lors de l'extraction d'un avis: {e}")
        
        logger.info(f"{len(extracted_reviews)} avis uniques extraits avec succès")
        return extracted_reviews
    except Exception as e:
        logger.error(f"Erreur générale lors de l'extraction des avis: {e}")
        return []
def get_location_details(page, soup):
    """Extrait les détails de l'agence bancaire"""
    try:
        # Nom de l'agence
        branch_elements = [
            soup.select_one('h1.DUwDvf'),
            soup.select_one('h1[jsan*="header-title"]'),
            soup.select_one('h1.fontHeadlineLarge')
        ]
        branch = next((el.text.strip() for el in branch_elements if el), "Nom inconnu")
        
        # Adresse
        address_elements = [
            soup.select_one('button[data-item-id="address"]'),
            soup.select_one('button[jsaction*="pane.address.click"]'),
            soup.select_one('.Io6YTe')
        ]
        location = next((el.text.strip() for el in address_elements if el), "Adresse inconnue")
        
        return branch, location
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des détails de l'agence: {e}")
        return "Nom inconnu", "Adresse inconnue"

# 1. Fonction alternative pour accéder aux détails de l'agence sans cliquer sur l'onglet Avis
def get_agency_reviews_direct(page, bank, city, agency_name=None):
    """Approche alternative pour obtenir les avis sans utiliser les onglets"""
    try:
        # Construire une URL directe pour les avis Google Maps
        search_term = f"{bank} {agency_name or ''} {city} Maroc avis"
        direct_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}"
        
        logger.info(f"Tentative d'accès direct aux avis via: {direct_url}")
        page.goto(direct_url, timeout=30000)
        random_delay(3, 5)
        
        # Gérer le popup de consentement
        handle_consent_popup(page)
        
        # Chercher et cliquer sur le lien "Avis Google"
        google_reviews_link = page.locator('a:has-text("Avis Google")').first
        if google_reviews_link.count() > 0:
            logger.info("Clic sur 'Avis Google'")
            google_reviews_link.click()
            random_delay(3, 5)
            
            # Vérifier si des avis sont chargés
            if wait_for_reviews_to_load(page):
                return True
        
        return False
    except Exception as e:
        logger.error(f"Erreur lors de l'accès direct aux avis: {e}")
        return False

# 2. Améliorez la fonction wait_for_reviews_to_load pour être plus flexible
def wait_for_reviews_to_load(page):
    """Attend que les avis se chargent avec différents sélecteurs possibles"""
    try:
        logger.info("Attente du chargement des avis...")
        
        # Essayer différents sélecteurs pour les avis
        selectors = [
            'div[data-review-id]',  # Sélecteur standard pour les avis
            '.gws-localreviews__google-review',  # Alternative sur la page de recherche Google
            '.WMbnJf',  # Alternative possible
            'div[jsaction*="reviewerLink"]'  # Alternative basée sur le jsaction
        ]
        
        for selector in selectors:
            try:
                review_items = page.locator(selector)
                # Attendre jusqu'à 5 secondes
                review_items.first.wait_for(timeout=5000)
                count = review_items.count()
                if count > 0:
                    logger.info(f"{count} avis chargés avec le sélecteur '{selector}'")
                    return True
            except Exception:
                continue
                
        logger.warning("Aucun avis trouvé avec les sélecteurs disponibles")
        return False
    except Exception as e:
        logger.warning(f"Erreur lors de l'attente des avis: {e}")
        return False

# 3. Modifiez get_reviews_for_bank pour utiliser l'approche alternative si nécessaire
def get_reviews_for_bank(bank, city, page):
    """Récupère les avis pour une banque dans une ville avec méthode de secours"""
    search_url = BASE_URL + f"{bank} {city} Maroc".replace(" ", "+")
    logger.info(f"Navigation vers: {search_url}")
    
    try:
        page.goto(search_url, timeout=60000)
        random_delay(3, 5)
        
        # Gérer le popup de consentement si présent
        handle_consent_popup(page)
        
        # Scroller pour charger plus d'agences
        scroller_banques(page)
        
        # Sélecteur plus générique pour les liens d'agences
        agency_links = page.locator('a[href*="/maps/place/"]')
        count = agency_links.count()
        
        if count == 0:
            logger.warning(f"Aucune agence trouvée pour {bank} à {city}")
            return []
            
        logger.info(f"{count} agences trouvées pour {bank} à {city}")
        
        results = []
        agencies_to_check = min(count, MAX_AGENCIES_PER_BANK)
        
        for i in range(agencies_to_check):
            try:
                logger.info(f"Traitement de l'agence {i+1}/{agencies_to_check}")
                
                # Extraire le nom de l'agence si possible
                agency_element = agency_links.nth(i)
                agency_name = agency_element.inner_text() if agency_element.count() > 0 else None
                
                # Approche principale: cliquer sur l'agence
                agency_element.click()
                random_delay(3, 5)
                
                # Cliquer sur l'onglet Avis
                avis_clicked = cliquer_sur_avis(page)
                
                # Si échec avec l'approche principale, essayer l'approche alternative
                if not avis_clicked:
                    logger.warning(f"Échec de l'approche principale pour l'agence {i+1}, essai avec approche alternative")
                    page.go_back()
                    random_delay(2, 3)
                    
                    if get_agency_reviews_direct(page, bank, city, agency_name):
                        logger.info("Approche alternative réussie pour accéder aux avis")
                        avis_clicked = True
                    else:
                        logger.warning(f"Impossible d'accéder aux avis pour l'agence {i+1}, passage à la suivante")
                        page.goto(search_url, timeout=30000)
                        random_delay(3, 5)
                        scroller_banques(page)
                        continue
                
                # Charger et extraire les avis
                extracted_reviews = charger_et_extraire_avis(page)
                
                if extracted_reviews:
                    # Extraire les détails de l'agence
                    soup = BeautifulSoup(page.content(), "html.parser")
                    branch, location = get_location_details(page, soup)
                    
                    # Ajouter les détails à chaque avis
                    for review in extracted_reviews:
                        results.append({
                            "Bank": bank,
                            "City": city,
                            "Branch": branch,
                            "Location": location,
                            **review
                        })
                    
                    logger.info(f"{len(extracted_reviews)} avis ajoutés pour {branch}")
                else:
                    logger.warning(f"Aucun avis extrait pour l'agence {i+1}")
                
                # Retour aux résultats
                page.goto(search_url, timeout=30000)
                random_delay(3, 5)
                scroller_banques(page)
                
            except Exception as e:
                logger.error(f"Erreur avec l'agence {i+1}: {e}")
                # Revenir aux résultats
                try:
                    page.goto(search_url, timeout=30000)
                    random_delay(3, 5)
                    scroller_banques(page)
                except Exception as nav_error:
                    logger.error(f"Erreur de navigation: {nav_error}")
        
        return results
    except Exception as e:
        logger.error(f"Erreur générale pour {bank} à {city}: {e}")
        return []

def scrape_reviews():
    """Fonction principale de scraping"""
    all_reviews = []
    
    # Réduire le nombre de villes et banques pour les tests
    test_cities = CITIES  # Seulement les 3 premières villes
    test_banks = BANKS   # Seulement les 3 premières banques
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()
        
        # Paramètres de page pour ressembler davantage à un utilisateur humain
        page.set_viewport_size({"width": 1280, "height": 800})
        
        try:
            for city in test_cities:
                for bank in test_banks:
                    logger.info(f"🔍 Scraping {bank} à {city}")
                    reviews = get_reviews_for_bank(bank, city, page)
                    all_reviews.extend(reviews)
                    
                    # Sauvegarder les résultats intermédiaires
                    if reviews:
                        logger.info(f"💾 Sauvegarde intermédiaire: {len(all_reviews)} avis au total")
                        try:
                            with open(f"avis_intermediaire_1{len(all_reviews)}.json", "w", encoding="utf-8") as json_file:
                                json.dump(all_reviews, json_file, ensure_ascii=False, indent=4)
                        except Exception as save_error:
                            logger.error(f"Erreur lors de la sauvegarde intermédiaire: {save_error}")
                    
                    # Pause entre les banques
                    random_delay(5, 10)
        except Exception as e:
            logger.error(f"Erreur générale de scraping: {e}")
        finally:
            browser.close()
            
    return all_reviews

# Exécuter le scraping
def main() :
    logger.info("🚀 Démarrage du scraping des avis bancaires")
    reviews = scrape_reviews()
    
    if reviews:
        logger.info(f"✅ Scraping terminé ! {len(reviews)} avis collectés.")
        
        # Sauvegarde JSON
        try:
            with open("avies.json", "w", encoding="utf-8") as json_file:
                json.dump(reviews, json_file, ensure_ascii=False, indent=4)
            logger.info("💾 Fichier JSON sauvegardé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde JSON: {e}")
        
        # Sauvegarde CSV
        try:
            pd.DataFrame(reviews).to_csv("avies.csv", index=False, encoding="utf-8")
            logger.info("💾 Fichier CSV sauvegardé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde CSV: {e}")
    else:
        logger.error("❌ Aucun avis collecté!")
        
if __name__ == "__main__":
    main()