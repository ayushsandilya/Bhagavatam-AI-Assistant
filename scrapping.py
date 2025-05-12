import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm

BASE_URL = "https://vedabase.io/en/library/sb/{canto}/{chapter}/advanced-view/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_verses_data(canto, chapter):
    """Extracts all Sanskrit, Transliteration, English Translation, and Purport from a chapter."""
    url = BASE_URL.format(canto=canto, chapter=chapter)

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Raise error if request fails
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    verses_data = []

    # Extracting all verses
    sanskrit_verses = soup.find_all("div", class_="av-devanagari")
    transliterations = soup.find_all("div", class_="av-verse_text")
    translations = soup.find_all("div", class_="av-translation")
    purports = soup.find_all("div", class_="av-purport")

    # Ensure all lists have the same length (sometimes purports may be missing)
    max_length = max(len(sanskrit_verses), len(transliterations), len(translations), len(purports))

    for i in range(max_length):
        sanskrit = sanskrit_verses[i].get_text(strip=True) if i < len(sanskrit_verses) else "N/A"
        transliteration = transliterations[i].get_text(strip=True) if i < len(transliterations) else "N/A"
        translation = translations[i].get_text(strip=True) if i < len(translations) else "N/A"
        purport = purports[i].get_text(strip=True) if i < len(purports) else "N/A"

        verses_data.append([canto, chapter, sanskrit, transliteration, translation, purport])

    return verses_data

def scrape_bhagavatam():
    """Scrapes all Cantos & Chapters."""
    all_data = []

    for canto in tqdm(range(1, 13), desc="Scraping Cantos"):
        for chapter in range(1, 100):  # Assumed max 100 chapters per canto
            chapter_data = get_verses_data(canto, chapter)
            if not chapter_data:
                break  # Stop if chapter doesn't exist
            all_data.extend(chapter_data)

            # Avoid getting blocked
            time.sleep(2)

    # Save to CSV
    df = pd.DataFrame(all_data, columns=["Canto", "Chapter", "Sanskrit", "Transliteration", "Translation", "Purport"])
    df.to_csv("bhagavatam_data.csv", index=False, encoding="utf-8-sig")
    print("✅ Scraping completed. Data saved as 'bhagavatam_data.csv'")

# Run scraper
scrape_bhagavatam()
