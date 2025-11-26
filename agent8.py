import requests
import lancedb
import openai
from openai import OpenAI
import datetime
from datetime import timezone,timedelta,datetime
import requests
import json
import pathlib
from fastapi import APIRouter, Body, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import BaseModel
import time
import os

api_key = "bcbe76132dcf615504d6b69af3145f65b5ecfc43501d4e813b60c99337e44312"
client = OpenAI(api_key=os.getenv("OPENAI_API"))


class ShoppingAgent():
    def __init__(self, api_key, client, vectors,user_prompt):
        self.api_key = api_key
        self.client = client
        self.vectors = vectors
        self.keywords = None
        self.products = None
        self.user_prompt = user_prompt
    
    def llm_chat(self,model: str = "gpt-5-mini"):
        time = datetime.now(timezone.utc)
        
        prompt = f"""You are an assistant that generates shopping keywords based on a person's style, preferences, and lifestyle.  

    {self.user_prompt} This is the prompt that is given by user and give at most importance to this and if there is missiong any relevant info take the help of the below context and order history of the user by any chance if the prompt misses any relevent information.

    Here is the user prompt
    Prompt: {self.user_prompt}


    Task: Based on the above context, generate a list of fashion items suitable for this person. Organize the keywords into the following JSON structure:  

    {{
    "top": [ /* list of tops  ] not more than 3,
    "bottom": [ /* list of bottoms  ] not more than 3,
    "footwear": [ /* list of shoes, sneakers, boots */ ] not more than 3,
    "accessories": [ /* list of accessories such as watches, bags, belts */ ] not more than 3,
    "gender: /* Gender of the user */
    "min_-price": ..,
    "max_price": ..,
    "location": Country,
    }}

    Requirements:
    - Output **only** the data in this structure. Do not include explanations, commentary, or any mention of JSON or code blocks.  
    - Include practical, minimalist, and modern items.  
    - Include neutral colors and versatile styles.  
    - Focus on items suitable for casual, professional, and leisure settings.  
    - Output valid JSON only; do not include extra commentary.
    - Return the min_price and max_price in local currency

    """

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if isinstance(response.choices[0].message.content.strip(), str):
                return response.choices[0].message.content.strip()
            else:
                return None
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return self.llm_chat()

    def get_shop(self):
        return self.llm_chat()

    def get_products(self):
        try:
            shop = json.loads(self.get_shop())
            self.keywords = shop
        except Exception as e:
            print(e)
        products = {}
        gender = shop["gender"]
        for category, keywords in self.keywords.items():
            if not isinstance(keywords, list):
                continue
            products[category] = {}
            for keyword in keywords:
                products[category][keyword] = {"items": []}
                params = {
                    "api_key": self.api_key,
                    "location": shop["location"]
                }
                try:
                    response = requests.get(
                        f"https://serpapi.com/search.json?engine=google_shopping_light&q={keyword} for {gender}",
                        params=params
                    )
                    time.sleep(1)
                    response.raise_for_status()
                    data = response.json()
                    shopping_results = data["shopping_results"]
                except Exception as e:
                    print(e)
                try:
                    for product in shopping_results[:3]:
                        item = {
                            "title": product["title"],
                            "source": product["source"],
                            "price": product["price"],
                            "rating": product["rating"],
                            "image_url": product["thumbnail"]
                        }
                        products[category][keyword]["items"].append(item)
                except Exception as e:
                    print(e)
        return products

    def get_product_link(self):
        page_token = ""
        params = {
            "api_key":api_key
        }
        response = requests.get(f"https://serpapi.com/search.json?engine=google_immersive_product&page_token={page_token}",params=params)
        if response.status_code == 200:
            return response.json()["product_results"]
        else:
            raise Exception(f"Error fetching Local Info: {response.status_code}, {response.text}")


def sync_fashion():
    shop = ShoppingAgent(api_key=api_key,client=client)
    products = shop.get_products()
    return products
            
   
