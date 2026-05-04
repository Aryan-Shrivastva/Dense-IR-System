"""
Builds a Wikipedia corpus using the wikipedia-api library.
Better search coverage than MS MARCO for live demos.

Output: data/wiki_corpus.csv

Usage:
  python data/download_wikipedia.py
  python data/download_wikipedia.py --size 25000
"""

import argparse
import os
import sys
import time
import random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

WIKI_SIZE = int(os.getenv("WIKI_SIZE", 20000))
DATA_DIR = "data"


def chunk_text(text, chunk_size=250):
    words = text.split()
    chunks = [" ".join(words[j:j+chunk_size]) for j in range(0, len(words), chunk_size)]
    return [c for c in chunks if len(c) > 100]


SEED_TITLES = [
    # Computing & AI
    "Artificial intelligence", "Machine learning", "Deep learning", "Neural network",
    "Computer science", "Algorithm", "Data structure", "Programming language",
    "Software engineering", "Database", "Internet", "World Wide Web", "HTTP",
    "Email", "Search engine", "Natural language processing", "Computer vision",
    "Quantum computing", "Robotics", "Automation", "Blockchain", "Cryptocurrency",
    "Big data", "Cloud computing", "Cybersecurity", "Malware", "Encryption",
    "Python", "Java", "C++", "JavaScript", "Open-source", "Linux",
    "Internet of things", "Virtual reality", "Augmented reality", "3D printing",
    "Wireless", "5G", "Mobile computing", "Operating system", "Compiler",
    "Computer network", "Router", "Server", "Data center", "API",
    "Git", "GitHub", "Stack Overflow", "Apache Hadoop", "MapReduce",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Nvidia", "GPU computing",
    "Artificial neural network", "Convolutional neural network", "Recurrent neural network",
    "Transformer", "ChatGPT", "Large language model", "GPT-3", "BERT",
    "Reinforcement learning", "Supervised learning", "Unsupervised learning",
    "Computer graphics", "Image processing", "Human-computer interaction",
    "Data mining", "Information retrieval", "Recommender system", "NLP",
    "Compiler", "Interpreter", "Assembly language", "Functional programming",
    "Object-oriented programming", "Design pattern", "Software architecture",
    "Microprocessor", "CPU", "RAM", "SSD", "Hard disk", "Cache",
    "Firmware", "Device driver", "Embedded system", "Real-time computing",

    # Science
    "Science", "Physics", "Chemistry", "Biology", "Mathematics",
    "Quantum mechanics", "Relativity", "Thermodynamics", "Electromagnetism",
    "Particle physics", "Standard Model", "Higgs boson", "Quark", "Neutrino",
    "Nuclear physics", "Astrophysics", "Cosmology", "Big Bang", "Black hole",
    "Galaxy", "Nebula", "Star", "Supernova", "Pulsar", "Quasar",
    "Planet", "Asteroid", "Comet", "Moon", "Mars", "Jupiter", "Saturn",
    "Exoplanet", "Astrobiology", "SETI", "Space exploration", "NASA",
    "SpaceX", "International Space Station", "Spacecraft", "Rocket",
    "Evolution", "Natural selection", "Genetics", "DNA", "RNA", "Protein",
    "Gene", "Chromosome", "Cell", "Mitochondria", "Photosynthesis",
    "Bacteria", "Archaea", "Virus", "Fungi", "Eukaryote", "Prokaryote",
    "Ecology", "Ecosystem", "Biodiversity", "Climate", "Weather",
    "Geology", "Mineral", "Rock", "Plate tectonics", "Volcano", "Earthquake",
    "Oceanography", "Marine biology", "Atmosphere", "Greenhouse effect",
    "Periodic table", "Atom", "Electron", "Proton", "Neutron", "Molecule",
    "Chemical bond", "Organic chemistry", "Biochemistry", "Pharmacology",
    "Botany", "Zoology", "Entomology", "Ornithology", "Marine life",

    # Medicine & Health
    "Medicine", "Pharmaceutical drug", "Vaccine", "Antibiotic", "Anesthesia",
    "Cancer", "Oncology", "Chemotherapy", "Radiation therapy", "Immunotherapy",
    "Heart disease", "Cardiology", "Stroke", "Hypertension", "Cholesterol",
    "Diabetes", "Obesity", "Malnutrition", "Asthma", "Alzheimer disease",
    "Parkinson disease", "Epilepsy", "Depression", "Schizophrenia", "Autism",
    "Epidemic", "Pandemic", "COVID-19", "Influenza", "Malaria", "HIV", "AIDS",
    "Blood", "Circulatory system", "Respiratory system", "Nervous system",
    "Brain", "Spinal cord", "Surgery", "Organ transplant", "Prosthetic",
    "MRI", "CT scan", "X-ray", "Ultrasound", "Blood test", "Biopsy",
    "Clinical trial", "Drug development", "Clinical research", "Epidemiology",
    "Public health", "WHO", "CDC", "Nutrition", "Mental health", "Yoga",
    "Physical exercise", "Meditation", "Sleep", "Aging", "Gerontology",

    # History & Politics
    "History", "World War II", "World War I", "Cold War", "Vietnam War",
    "French Revolution", "American Revolution", "Roman Empire", "Ancient Greece",
    "Ancient Egypt", "Mesopotamia", "China", "India", "Japan",
    "Industrial Revolution", "Renaissance", "Reformation", "Enlightenment",
    "Colonialism", "Imperialism", "Slavery", "Civil rights movement",
    "Democracy", "Republic", "Monarchy", "Dictatorship", "Anarchism",
    "Capitalism", "Socialism", "Communism", "Fascism", "Globalization",
    "United Nations", "European Union", "NATO", "WHO", "UNESCO",
    "War", "Treaty", "Diplomacy", "Human rights", "Democratization",
    "USA", "United Kingdom", "France", "Germany", "Russia", "China",
    "India", "Japan", "Brazil", "Australia", "Canada", "Mexico",
    "Afghanistan", "Iran", "Iraq", "Saudi Arabia", "Israel", "Pakistan",
    "Ukraine", "Poland", "Turkey", "South Africa", "Nigeria", "Egypt",

    # Culture & Society
    "Literature", "Poetry", "Novel", "Play", "Shakespeare", "Homer",
    "Tolstoy", "Dostoevsky", "Austen", "Dickens", "Fiction", "Nonfiction",
    "Publishing", "Book", "Library", "Journalism", "Newspaper", "Magazine",
    "Painting", "Sculpture", "Photography", "Architecture", "Design",
    "Renaissance art", "Impressionism", "Cubism", "Modernism", "Pop art",
    "Music", "Classical music", "Jazz", "Rock", "Pop music", "Hip hop",
    "Electronic music", "Opera", "Symphony", "Guitar", "Piano", "Violin",
    "Film", "Cinema", "Hollywood", "Bollywood", "Anime", "Documentary",
    "Television", "Streaming", "Radio", "Podcast", "Theater", "Dance",
    "Philosophy", "Ethics", "Metaphysics", "Epistemology", "Logic",
    "Stoicism", "Existentialism", "Buddhism", "Christianity", "Islam",
    "Hinduism", "Judaism", "Atheism", "Agnosticism", "Secularism",
    "Education", "University", "School", "Learning", "Research",
    "Psychology", "Sociology", "Anthropology", "Archaeology", "Linguistics",

    # General
    "Economics", "Macroeconomics", "Microeconomics", "Inflation", "GDP",
    "Finance", "Banking", "Investment", "Stock market", " Cryptocurrency",
    "Marketing", "Management", "Leadership", "Entrepreneurship", "Startup",
    "Game theory", "Statistics", "Probability", "Combinatorics", "Topology",
    "Calculus", "Linear algebra", "Number theory", "Geometry", "Trigonometry",
    "Agriculture", "Food", "Cooking", "Nutrition", "Restaurant", "Wine",
    "Fashion", "Clothing", "Textile", "Sport", "Football", "Basketball",
    "Cricket", "Tennis", "Olympics", "Marathon", "Swimming", "Cycling",
    "Travel", "Tourism", "Hotel", "Restaurant", "Country", "City", "Capital",
    "Language", "English", "Spanish", "Chinese", "Arabic", "French",
]


def download_wiki(limit: int):
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia(
        user_agent="IRProjectResearch/1.0 (college project; akash@example.com)",
        language="en",
    )

    titles = list(set(SEED_TITLES))
    random.shuffle(titles)
    seen = set()
    docs = []
    doc_id = 0

    for title in titles:
        if len(docs) >= limit:
            break

        page = wiki.page(title)
        if not page.exists() or page.title in seen:
            continue

        text = page.text
        if len(text) < 300:
            continue

        seen.add(page.title)
        chunks = chunk_text(text)

        for k, chunk in enumerate(chunks[:6]):
            docs.append({
                "docid": f"wiki_{doc_id:06d}_{k}",
                "title": page.title,
                "body": chunk,
                "url": page.fullurl,
            })
            doc_id += 1

        if len(docs) % 500 == 0:
            print(f"  {len(docs)} chunks from {len(seen)} articles...")

        time.sleep(0.02)

    # Expand via links from collected articles
    print(f"Expanding via links ({len(docs)} chunks so far)...")
    for title in list(seen)[:50]:
        if len(docs) >= limit:
            break
        page = wiki.page(title)
        if not page.exists():
            continue
        for link in list(page.links.values())[:40]:
            if len(docs) >= limit:
                break
            if link.title in seen:
                continue
            p = wiki.page(link.title)
            if not p.exists() or len(p.text) < 300:
                continue
            seen.add(p.title)
            chunks = chunk_text(p.text)
            for k, chunk in enumerate(chunks[:4]):
                docs.append({
                    "docid": f"wiki_{doc_id:06d}_{k}",
                    "title": p.title,
                    "body": chunk,
                    "url": p.fullurl,
                })
                doc_id += 1
            time.sleep(0.02)

    print(f"Done: {len(docs)} chunks from {len(seen)} Wikipedia articles")
    return docs[:limit]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None,
                        help="Max chunks (default: WIKI_SIZE env / 20000)")
    args = parser.parse_args()

    size = args.size if args.size else WIKI_SIZE
    print(f"Building Wikipedia corpus (target: {size} chunks)...")

    os.makedirs(DATA_DIR, exist_ok=True)
    docs = download_wiki(limit=size)

    df = pd.DataFrame(docs)
    out = os.path.join(DATA_DIR, "wiki_corpus.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} chunks -> {out}")