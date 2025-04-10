import os
import json
import requests
from hashlib import md5
import re
import spacy
from images import fetch_image_url, fetch_random_image

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")


# cleans article text
def clean_text(text):
    # matches weird text at start of some articles' text
    pattern = (r'^[A-Za-z]+ \| [A-Za-z]{3} \w{3} \d{1,2} , \d{4} \d{1,2}:\d{2} (am|pm) [A-Za-z]+ (?:[A-Za-z]+ )?-LRB- '
               r'Reuters -RRB- -')
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()


def clean_phrase(phrase):
    phrase = phrase.lower()
    phrase = re.sub(r'[^a-zA-Z\s]', '', phrase)  # remove weird chars
    phrase = phrase.strip()
    return phrase


# extracts keywords from articles
def extract_keywords(text, max_keywords=5):
    doc = nlp(text)
    candidates = []

    for chunk in doc.noun_chunks:
        cleaned = clean_phrase(chunk.text)
        if not nlp.vocab[cleaned].is_stop and len(cleaned.split()) <= 4 and cleaned:
            candidates.append(cleaned)

    for ent in doc.ents:
        if ent.label_ in {"ORG", "GPE", "EVENT", "PERSON"}:
            cleaned = clean_phrase(ent.text)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

    # Frequency count
    freq = {}
    for phrase in candidates:
        if phrase in freq:
            freq[phrase] += 1
        else:
            freq[phrase] = 1

    # Sort and return top keywords
    keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in keywords[:max_keywords]]


# downloads and cache's images
def download_and_cache_image(image_url, image_dir, cache_key):
    # Create a unique filename using a hash of the URL or keyword
    filename = f"{cache_key}.jpg"
    filepath = os.path.join(image_dir, filename)

    # Download if it doesn't already exist
    if not os.path.exists(filepath):
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
    return filepath


# converts PropaNews dataset (.jsonl file) to .json file that FaKnow code can use
def convert_jsonl_to_faknow_array(input_path, output_path, image_dir, default_image="default.jpg", domain=0,
                                  random_image=False):
    os.makedirs(image_dir, exist_ok=True)
    converted_data = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line)
            text = record.get("txt", "").strip()
            label = int(not record.get("label", False))  # True -> 0 (real), False -> 1 (fake)

            cleaned_text = clean_text(text)
            keywords = extract_keywords(cleaned_text, max_keywords=3)

            # Determine local image path
            path = record.get("path", "")
            filename_stem = os.path.basename(path).replace(".rsd.txt", "")
            image_path = None

            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(image_dir, filename_stem + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            # Fallback to fetch image if not found locally
            if not image_path:
                keyword = "+".join(keywords) if keywords else "news"
                image_url = fetch_random_image() if random_image else fetch_image_url(keyword)
                print("image_url", image_url)
                if image_url:
                    cache_key = md5(image_url.encode()).hexdigest()[:10]
                    image_path = download_and_cache_image(image_url, image_dir, cache_key)
                else:
                    image_path = default_image

            converted_data.append({
                "text": cleaned_text,
                "label": label,
                "domain": domain,
                "image": image_path
            })

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(converted_data, outfile, indent=4)


# creating json's for keyword generated images
print("Creating keyword generated image dataset...")
convert_jsonl_to_faknow_array("./data/train_med.jsonl", "train_keyword.json", "./data/images",
                              default_image="default.jpg", domain=0)
convert_jsonl_to_faknow_array("./data/test_med.jsonl", "test_keyword.json", "./data/images",
                              default_image="default.jpg", domain=0)
convert_jsonl_to_faknow_array("./data/val_med.jsonl", "val_keyword.json", "./data/images",
                              default_image="default.jpg", domain=0)
print("Generated keyword image dataset!")

# creating json's for random generated images
print("Creating random generated image dataset...")
convert_jsonl_to_faknow_array("./data/train_med.jsonl", "train_random.json", "./data/images",
                              default_image="default.jpg", domain=0, random_image=True)
convert_jsonl_to_faknow_array("./data/test_med.jsonl", "test_random.json", "./data/images", default_image="default.jpg",
                              domain=0, random_image=True)
convert_jsonl_to_faknow_array("./data/val_med.jsonl", "val_random.json", "./data/images",
                              default_image="default.jpg", domain=0, random_image=True)
print("Generated random image dataset!")

# !!! DO NOT RUN UNLESS YOU PLAN TO HAND-PICK NEW IMAGES !!!
# print("Creating handpicked generated image dataset...")
# convert_jsonl_to_faknow_array("./data/train_med.jsonl", "train_human.json", "./data/images", default_image="default.jpg", domain=0)
# convert_jsonl_to_faknow_array("./data/test_med.jsonl", "test_human.json", "./data/images", default_image="default.jpg", domain=0)
# convert_jsonl_to_faknow_array("./data/val_med.jsonl", "val_human.json", "./data/images",
#                               default_image="default.jpg", domain=0)
# print("Generated handpicked image dataset!")
