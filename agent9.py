import requests
import json
import lancedb
from openai import OpenAI
from datetime import datetime,timedelta
import pathlib
from fastapi import APIRouter, Body, Path
from pydantic import BaseModel

api_key = "bcbe76132dcf615504d6b69af3145f65b5ecfc43501d4e813b60c99337e44312"
client = OpenAI(api_key="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")


class TravelPlanner():
    def __init__(self, api_key, client,user_prompt):
        self.api_key = api_key
        self.client = client
        self.destinations = None
        self.dates = None
        self.flights = None
        self.hotels = None
        self.places = None
        self.user_prompt = user_prompt
    def llm_chat(self,model: str = "gpt-5-mini"):
        prompt = f"""You are an assistant that generates city recommendations based on a person's travel preferences, style, and lifestyle.
    {self.user_prompt} This is the prompt that is given by user and give at most importance to this and if there is missiong any relevant info take the help of the below context and order history of the user by any chance if the prompt misses any relevent information.

    Task: Based on the above context, generate a list of cities suitable for this person. Organize the recommendations into the following JSON structure:

    {{
    "cities": [ /* list of up to 1 city suitable with countries for this user in IATA code*/ ],
    "city_names":[ /* names of the city],
    "city_description": [ /* Corresponding city description]
    "recommended_duration_days": [ /* suggested number of days to spend in each city */ ],
    "origin": [ /* where the person stats journey from in IATA code or the nearest famous airport code to the origin],
    "currency":[ /* The local currency which the user resides in and uses]
    }}

    Requirements:
    - Output valid JSON only; do not include extra commentary.
    - The "origin" IATA code must never match any city in "cities", ensuring each listed destination is distinct
    - Include cities that match the user's preferred travel style, activities, climate, and pace.
    - Prioritize cities that are versatile for leisure, culture, adventure, or relaxation depending on the context.
    - Include a mix of popular and off-the-beaten-path destinations if suitable.
    - Ensure the number of cities does not exceed 1.
    - Make sure all cities are in the same country
    - Provide all the places in IATA codes used in Airports
    - Make sure origin and cities dont match
    - Make sure the output is in the given format exactly and no fields are missed
    - Output **only** the data in this structure. Do not include explanations, commentary, or any mention of JSON or code blocks.
    """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if isinstance(response.choices[0].message.content.strip(), str):
                return response.choices[0].message.content.strip()
            else:
                return self.llm_chat()
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def fetch_dest(self):
        try:
            return self.llm_chat()
        except Exception as e:
            print(e)
            return None

    def fetch_weather(self,city):
        base_url = f"https://serpapi.com/search.json?q=What's+the+weather+in+{city}?&hl=en&gl=us"
        params = {"api_key":api_key}
        response = requests.get(base_url,params=params)
        if response.status_code == 200:
            data=response.json()["answer_box"]["forecast"]
            data = [{'day': entry['day'], 'weather': entry['weather']} for entry in data]
            start_date = datetime.now().date()
            date_weather = []
            for i, entry in enumerate(data):
                forecast_date = start_date + timedelta(days=i)
                date_weather.append({'date': forecast_date.strftime('%Y-%m-%d'), 'weather': entry['weather']})
            return date_weather
        else:
            raise Exception(f"Error fetching Weather Info: {response.status_code}, {response.text}")
        
    def fetch_dates(self,model: str = "gpt-5-mini"):
        self.destinations = json.loads(self.fetch_dest())
        cities = self.destinations["city_names"]
        weather_info = {city: self.fetch_weather(city) for city in cities}

        prompt = f"""You are an assistant that generates city travel recommendations based on a person's calendar availability and weather conditions in various cities.

{self.user_prompt} This is the prompt that is given by user and give at most importance to this and if there is missiong any relevant info take the help of the below context and order history of the user by any chance if the prompt misses any relevent information.

Here is the context of user    

Today's date is {datetime.now().date()}

{weather_info}

Task: Analyze the user's calendar and the weather data for each city. Generate a list of cities suitable for this person, along with recommended travel dates (max 2-3 dates per city) and a suggested duration of stay.

Requirements:
- Use the provided calendar data to avoid conflicting dates
- Use the provided weather data to suggest optimal travel periods
- Prioritize cities that match the user's travel style, activities, climate, and pace
- Include a mix of popular and off-the-beaten-path destinations if suitable
- Ensure the number of cities does not exceed 4
- Make sure all cities are in the same country
- For each city, suggest 2-3 possible date ranges based on the user's availability and favorable weather
- Use IATA codes for city names
- Don't give today's date

Output ONLY valid JSON in this exact structure with no additional text, explanations, or markdown:
{{"city_code": ["YYYY-MM-DD to YYYY-MM-DD", "YYYY-MM-DD to YYYY-MM-DD", "YYYY-MM-DD to YYYY-MM-DD"]}}

Example:
{{"PNQ": ["2025-10-09 to 2025-10-12", "2025-10-11 to 2025-10-14", "2025-10-13 to 2025-10-16"], "GOI": ["2025-10-12 to 2025-10-15", "2025-10-13 to 2025-10-16", "2025-10-14 to 2025-10-17"], "HYD": ["2025-10-09 to 2025-10-12", "2025-10-11 to 2025-10-14", "2025-10-15 to 2025-10-18"], "BLR": ["2025-10-15 to 2025-10-18", "2025-10-16 to 2025-10-19"]}}

YOUR RESPONSE MUST START WITH {{ AND END WITH }} - NO OTHER TEXT ALLOWED."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if isinstance(response.choices[0].message.content.strip(), str):
                return response.choices[0].message.content.strip()
            else:
                return self.fetch_dates()
        
        except Exception as e:
            return f"Error: {str(e)}"

    def fetch_flights(self, origin, destination,currency,departure_date, return_date=None):
        base_url = "https://serpapi.com/search?engine=google_flights"
        
        params = {
            "engine": "google_flights",
            "departure_id": origin,
            "arrival_id":destination,
            "currency":currency,
            "outbound_date":departure_date,
            "api_key": api_key,
            "type":2
        }
        if origin in self.destinations['cities']:
            return
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            try:
                flights = response.json()["best_flights"][0]
            except Exception as e:
                print(e)
                return None
            simple_flights = []
            for flight_data in [flights]:
                price = flight_data["price"]
                airline_name = flight_data["flights"][0]["airline"]
                flights_list = flight_data["flights"]

                route_info = []
                for leg in flights_list:
                    dep = leg["departure_airport"]
                    arr = leg["arrival_airport"]
                    route_info.append({
                        "from": dep["name"],
                        "departure_time": dep["time"],
                        "to": arr["name"],
                        "arrival_time": arr["time"]
                    })

                simple_flights.append({
                    "airline": airline_name,
                    "price": price,
                    "route": route_info
                })

            return simple_flights
        else:
            raise Exception(f"Error fetching flights: {response.status_code}, {response.text}")

    def fetch_hotels(self,location,check_in_date,check_out_date,currency,adults=2):
        base_url = "https://serpapi.com/search?engine=google_hotels"
        params ={
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": adults,
        "api_key": api_key,
        "q":location,
        "currency":currency
        }
        response = requests.get(base_url,params=params)
        if response.status_code == 200:
            extracted = []
            try:
                properties = response.json()["properties"]
            except Exception as e:
                print(e)
                return None
            for prop in properties:
                info = {}
                
                info['type'] = prop['type'] if 'type' in prop else None
                info['name'] = prop['name'] if 'name' in prop else None
                info['description'] = prop['description'] if 'description' in prop else ''
                
                if 'link' in prop:
                    info['link'] = prop['link']
                elif 'serpapi_property_details_link' in prop:
                    info['link'] = prop['serpapi_property_details_link']
                else:
                    info['link'] = None
                
                info['rate_per_night'] = prop['rate_per_night']['lowest'] if 'rate_per_night' in prop and 'lowest' in prop['rate_per_night'] else None
                info['total_rate'] = prop['total_rate']['lowest'] if 'total_rate' in prop and 'lowest' in prop['total_rate'] else None
                info['amenities'] = prop['amenities'] if 'amenities' in prop else []
                info['location_rating'] = prop['location_rating'] if 'location_rating' in prop else None
                info['overall_rating'] = prop['overall_rating'] if 'overall_rating' in prop else None

                if 'images' in prop:
                    info['images'] = [img['original_image'] for img in prop['images'][:3] if 'original_image' in img]
                else:
                    info['images'] = []

                nearby = []
                if 'nearby_places' in prop:
                    for place in prop['nearby_places']:
                        place_info = {'name': place['name'] if 'name' in place else None}
                        nearby.append(place_info)
                info['nearby_places'] = nearby

                extracted.append(info)

            return extracted
        else:
            raise Exception(f"Error fetching Hotels: {response.status_code}, {response.text}")

    def fetch_transit(self,start_addr,end_addr):
        base_url="https://serpapi.com/search?engine=google_maps_directions"
        params = {
            "start_addr":start_addr,
            "end_addr":end_addr,
            "api_key":api_key,
            "travel_mode":3
        }
        response = requests.get(base_url,params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error fetching Transit Info: {response.status_code}, {response.text}")

    def fetch_popular(self,location,days):
        base_url = f"https://serpapi.com/search.json?engine=google"
        params = {
            "q":location + "Destinations",
            "api_key":api_key
        }
        response = requests.get(base_url,params=params)
        if response.status_code == 200:
            return response.json()["top_sights"]["sights"][:days*3]
        else:
            raise Exception(f"Error fetching Local Info: {response.status_code}, {response.text}")

    def fetch_localplaces(self,q,location):
        base_url = f"https://serpapi.com/search.json?engine=google_local&q={q}&ll=@{location}z&type=search"
        params = {
            "q":q,
            "ll":location,
            "api_key":api_key
        }
        response = requests.get(base_url,params=params)
        if response.status_code == 200:
            return response.json()["top_sights"]["sights"][:9]
        else:
            raise Exception(f"Error fetching Local Info: {response.status_code}, {response.text}")

    def travel(self):
        self.dates = json.loads(self.fetch_dates())
        flight_results = {
            city: [
                    self.fetch_flights(
                        self.destinations["origin"],
                        city,
                        self.destinations["currency"],
                        str(d).split("to")[0].strip()
                    )
                for d in self.dates[city]
            ]
            for city in self.dates
            if city != self.destinations["origin"]
        }
        self.flights = flight_results
        hotels = []
        for city in self.destinations["city_names"]:
            for city_code in self.dates:
                date = self.dates[city_code][0]
                hotels.append({city:self.fetch_hotels(city,date.split("to")[0].strip(),date.split("to")[1].strip(),self.destinations["currency"])[:5]})

        self.hotels = hotels

        itinery = []
        for i,city in enumerate(self.destinations["city_names"]):
            days = self.destinations["recommended_duration_days"][i]
            itinery.append({city:self.make_itenary(city=city,days=days)})
        self.itinery = itinery
    
    def make_plan(self):
        plans = []
        self.travel()
        for i,city in enumerate(self.destinations["city_names"]):
            city_plan = {}
            city_code = self.destinations["cities"][i]
            city_plan["city"] = city
            city_plan["days"] = self.destinations["recommended_duration_days"][i]
            city_plan["description"] = self.destinations["city_description"][i]
            city_plan["flights"] = self.flights[city_code]
            city_plan["hotels"] = self.hotels[i][city]
            city_plan["itinerary"] = self.itinery[i][city]
            plans.append(city_plan)
        print(plans[0])
        return plans[0]

    def make_itenary(self,city,days,model="gpt-5-mini"):
        places = self.fetch_popular(f"Top tourist Attractions in {city}",days)
        prompt = f"""
        Create a detailed multi-day travel itinerary in **JSON format** based on the list of places provided below.  
        Each day should have a unique **theme** (e.g., Heritage, Nature, Adventure, Spiritual, Markets, etc.).  
        For every day, plan **morning**, **afternoon**, and **evening** activities using the given places (or related nearby attractions if needed).  
        If a place lacks a description, create a short, engaging one yourself.  
        Each activity should include a **title**, **description**, **time of day**, and a **thumbnail image URL** (you can use realistic image links or public domain sources).  
        Ensure the order of activities makes geographical and thematic sense.  
        Output should strictly follow this JSON structure:

        {self.user_prompt} This is the prompt that is given by user and give at most importance to this and if there is missiong any relevant info take the help of the below context and order history of the user by any chance if the prompt misses any relevent information.

        {{
        "itinerary": {{
            "days": [
            {{
                "day": 1,
                "theme": "Heritage & Culture",
                "activities": [
                {{
                    "time": "Morning",
                    "title": "Victoria Memorial",
                    "description": "Explore this iconic white marble monument and museum dedicated to Queen Victoria.",
                    "image": "https://example.com/victoria_memorial.jpg"
                }},
                {{
                    "time": "Afternoon",
                    "title": "Indian Museum",
                    "description": "Visit India's oldest and largest museum showcasing art, archaeology, and history.",
                    "image": "https://example.com/indian_museum.jpg"
                }},
                {{
                    "time": "Evening",
                    "title": "Howrah Bridge",
                    "description": "Walk across this historic cantilever bridge and enjoy views of the Hooghly River.",
                    "image": "https://example.com/howrah_bridge.jpg"
                }}
                ]
            }},
            {{
                "day": 2,
                "theme": "Spiritual & Riverside",
                "activities": [
                {{
                    "time": "Morning",
                    "title": "Dakshineswar Kali Temple",
                    "description": "Pay homage at this renowned temple dedicated to Goddess Kali, located along the Hooghly River.",
                    "image": "https://example.com/dakshineswar.jpg"
                }},
                {{
                    "time": "Afternoon",
                    "title": "Belur Math",
                    "description": "Visit the serene headquarters of the Ramakrishna Mission, blending architecture of major faiths.",
                    "image": "https://example.com/belur_math.jpg"
                }},
                {{
                    "time": "Evening",
                    "title": "Princep Ghat",
                    "description": "Enjoy a peaceful boat ride and catch the sunset over the river.",
                    "image": "https://example.com/princep_ghat.jpg"
                }}
                ]
            }}
            ]
        }}
        }}

        Now use the following list of places to create a similar itinerary (ensure creative themes and cohesive flow):

            - Output **only** the data in this structure. Do not include explanations, commentary, or any mention of JSON or code blocks.
        {places}
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(response.choices[0].message.content.strip())
        
        except Exception as e:
            return f"Error: {str(e)}"

def sync_travel(user_prompt):
    travel = TravelPlanner(api_key=api_key,client=client,user_prompt=user_prompt)
    travel = travel.make_plan()
    return travel