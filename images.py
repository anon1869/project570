import requests
import random
from dotenv import load_dotenv
import os

# load env
load_dotenv()

pexels_key = os.getenv("PEXELS_API_KEY")

if not pexels_key:
    raise ValueError("PEXELS_API_KEY not found in environment variables.")

headers = {
    'Authorization': pexels_key
}

# fetch image url using keyword
def fetch_image_url(keyword):
    response = requests.get(
        f'https://api.pexels.com/v1/search?query={keyword}&per_page=1',
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        if data['photos']:
            return data['photos'][0]['src']['medium']
    return None


# fetch random image
def fetch_random_image():
    random_page = random.randint(1, 50)

    response = requests.get(
        'https://api.pexels.com/v1/curated',
        headers=headers,
        params={
            'per_page': 1,
            'page': random_page
        }
    )
    print("response", response)
    if response.status_code == 200:
        data = response.json()
        if data['photos']:
            return data['photos'][0]['src']['medium']
    return None
