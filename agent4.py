import requests
import json
import os
from datetime import timezone, datetime
from openai import OpenAI
import time

# SERPAPI_KEY = "88e9df8bf5192113827878d7de00a1f19f57e04ff7d49e944acdd7d9a9f4a653"
SERPAPI_KEY ="bcbe76132dcf615504d6b69af3145f65b5ecfc43501d4e813b60c99337e44312"
client = OpenAI(
    api_key=os.getenv("OPENAI_API")
)
MODEL_NAME = "gpt-5-mini"

class FoodPlanner:
    def __init__(self, serpapi_key):
        self.serpapi_key = serpapi_key

    def search_food_trends(self, dish, location="San Francisco, CA"):
        """Search trending food items via SerpAPI"""
        try:
            url = "https://serpapi.com/search.json"
            params = {"engine": "google", "q": dish, "api_key": self.serpapi_key, "num": 10}
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            trends = []
            for r in data.get("organic_results", [])[:5]:
                if any(k in r.get("title", "").lower() for k in ["food", "dish", "restaurant", "cuisine", "meal"]):
                    trends.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "link": r.get("link", "")
                    })
            return trends
        except Exception as e:
            print(f"Error searching food trends: {e}")
            return []

    def search_nutrition_info(self, dish_name):
        """Search nutritional info"""
        try:
            url = "https://serpapi.com/search.json"
            params = {
                "engine": "google",
                "q": f"{dish_name} nutrition facts calories protein",
                "api_key": self.serpapi_key,
                "num": 5
            }
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            results = resp.json().get("organic_results", [])
            return [{"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")} for r in results[:3]]
        except Exception as e:
            print(f"Error fetching nutrition info: {e}")
            return []


    def fetch_restaurants(self, dish_name, location, dietary_prefs=None):
        """Fetch restaurants"""
        try:
            time.sleep(2)
            url = f"https://serpapi.com/search.json?engine=google_local&q={dish_name}&ll=@{location}z&type=search"
            params = {
                "q":dish_name,
                "location":location,
                "api_key": self.serpapi_key
            }
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            
            data = resp.json()
            local_results = data["local_results"][:3]

            return local_results

        except Exception as e:
            print(f"Error fetching restaurants: {e}")
            return []


    def fetch_image(self, dish_name):
        """Fetch dish image"""
        try:
            time.sleep(2)
            url = "https://serpapi.com/search.json"
            params = {"engine": "google_images", "q": dish_name, "api_key": self.serpapi_key, "num": 1}
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            images = resp.json().get("images_results", [])
            if images:
                return images[0].get("thumbnail", "")
        except Exception as e:
            print(f"Error fetching image: {e}")
        return ""

    def parse_user_prompt(self, user_prompt):
        time = datetime.now()
        prompt = f"""You are an expert meal planning assistant. Based on the user context and current time, recommend ONE appropriate meal.

            ## Analysis Requirements
            From the context, identify:
            - Dietary type (vegetarian/non-vegetarian/vegan)
            - Food preferences and cuisine likes
            - Location and regional food availability
            - Current time (UTC {time}) to determine appropriate meal type (breakfast/lunch/dinner/snack)
            - Veg or Nonveg: {{veg_nonveg}}

            ## Meal Recommendation
            Provide ONE meal that:
            - Suits the current time of day
            - Matches their dietary preferences
            - Is easily available for online ordering in their location
            - Includes complete nutritional information

            ## Context
            {user_prompt}

            ## Output Format
            - Output **only** the data in this structure. Do not include explanations, commentary, or any mention of JSON or code blocks.  
            Return ONLY a valid JSON object with this exact structure:
            {{
            "meal_type": "breakfast/lunch/dinner/snack",
            "local_time": "HH:MM",
            "location":"Place,City,State,Country"
            "dish": {{
                "name": "dish name",
                "cuisine": "cuisine type",
                "description": "brief description"
            }},
            "veg_nonveg":"veg or nonveg",
            "nutrients": {{
                "calories": "kcal",
                "protein": "g",
                "carbohydrates": "g",
                "fats": "g",
                "fiber": "g"
            }},
            "dietary_type": "vegetarian/non-vegetarian/vegan",
            "ordering_tip": "where/how to order this in their location"
            }}"""
    
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            data = response.choices[0].message.content.strip()
            return json.loads(data)
        except Exception as e:
            print(f"Error: {str(e)}")
        return None

    def generate_meal_recommendation(self, user_prompt):
        parsed = self.parse_user_prompt(user_prompt)
        dietary_prefs =  parsed["veg_nonveg"] 

        dish_name = parsed["dish"]["name"]
        cuisine_type = parsed["dish"]["cuisine"]
        location = parsed["location"]

        nutrition_info = parsed["nutrients"]
        restaurants = self.fetch_restaurants(dish_name, location, dietary_prefs)
        image_url = self.fetch_image(dish_name)
        restaurants = []
        image_url = ""
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        meal_type = "breakfast" if 6 <= hour < 11 else "lunch" if 11 <= hour < 15 else "dinner" if 15 <= hour < 21 else "snack"

        return {
            "meal_type": meal_type,
            "local_time": f"{hour:02d}:{current_time.minute:02d}",
            "dish": {
                "name": dish_name,
                "cuisine": cuisine_type,
                "description": "Delicious requested dish",
                "image_url": image_url
            },
            "nutrition": nutrition_info,
            "restaurants": restaurants,
            "dietary_notes": dietary_prefs if dietary_prefs else ["none specified"]
        }


# Example usage
if __name__ == "__main__":
    planner = FoodPlanner(SERPAPI_KEY)
    user_prompt = "I want some tangy, creamy, savoury, flavourful nonveg dishes in Kharagpur"
    recommendation = planner.generate_meal_recommendation(user_prompt)
    print(json.dumps(recommendation, indent=2))
