import os
import json
import requests
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import praw
from openai import OpenAI


class CallType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ARBITRAGE = "ARBITRAGE"

class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class Position(str, Enum):
    YES = "YES"
    NO = "NO"

class PositionSize(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"

class Sentiment(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"

class CatalystType(str, Enum):
    NEWS = "NEWS"
    REDDIT_TREND = "REDDIT_TREND"
    MISPRICING = "MISPRICING"
    ARBITRAGE = "ARBITRAGE"
    MOMENTUM = "MOMENTUM"

class Urgency(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    NEAR_TERM = "NEAR_TERM"
    MONITOR = "MONITOR"

# ============= PYDANTIC MODELS =============
class MarketOdds(BaseModel):
    yes: float = Field(ge=0, le=1)
    no: float = Field(ge=0, le=1)

class RedditThread(BaseModel):
    url: str
    sentiment: str
    upvotes: int

class TradeCall(BaseModel):
    call_type: CallType
    confidence: Confidence
    market_id: str
    market_question: str
    current_odds: MarketOdds
    recommended_position: Position
    suggested_size: PositionSize
    reasoning: str
    sentiment_score: float = Field(ge=-1, le=1)
    reddit_signal: Sentiment
    catalyst_type: CatalystType
    urgency: Urgency
    risk_factors: List[str]
    key_reddit_threads: List[RedditThread]
    timestamp: str

class SearchKeywords(BaseModel):
    primary_keywords: List[str]
    market_search_terms: List[str]
    reddit_search_queries: List[str]

class PolymarketClient:
    def __init__(self):
        self.base_url = "https://gamma-api.polymarket.com"
    
    def search_markets_by_keywords(self, keywords: List[str], limit=3):
        """Search markets using keywords via API search endpoint"""
        all_markets = []
        
        for keyword in keywords:
            try:
                # Use the search endpoint with query parameter
                response = requests.get(
                    f"{self.base_url}/public-search?q={keyword}",
                    params={
                        "active": True, 
                        "limit": limit,
                        "query": keyword 
                    }
                )
                response.raise_for_status()
                markets = response.json()
                all_markets.extend(markets)
            except Exception as e:
                print(f"Error searching markets with keyword '{keyword}': {e}")
        
        # Remove duplicates based on market ID
        unique_markets = {m.get('id'): m for m in all_markets}.values()
        return list(unique_markets)
    
    def get_market_details(self, market_id):
        """Get specific market details"""
        try:
            response = requests.get(f"{self.base_url}/markets/{market_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching market details: {e}")
            return None

class RedditClient:
    def __init__(self, client_id, client_secret, user_agent):
        """Initialize Reddit client with PRAW"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            redirect_uri="http://localhost:8080"
        )
    
    def search_with_keywords(self, keywords: List[str], subreddits=['politics', 'news', 'worldnews', 'cryptocurrency', 'technology'], limit=3):
        """Search Reddit using multiple keyword queries"""
        all_threads = []
        
        for query in keywords:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(query, limit=limit, sort='relevance', time_filter='week'):
                        all_threads.append({
                            'title': submission.title,
                            'url': f"https://reddit.com{submission.permalink}",
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': submission.created_utc,
                            'selftext': submission.selftext[:500] if submission.selftext else "",
                            'search_query': query
                        })
                except Exception as e:
                    print(f"Error searching r/{subreddit_name} with '{query}': {e}")
        
        # Remove duplicates and sort by score
        unique_threads = {t['url']: t for t in all_threads}.values()
        return sorted(unique_threads, key=lambda x: x['score'], reverse=True)[:15]

class AIAnalyzer:
    def __init__(self, openai_api_key):
        """Initialize OpenAI client"""
        self.client = OpenAI(api_key=openai_api_key)
    
    def extract_keywords_from_query(self, user_query: str) -> SearchKeywords:
        """Use AI to extract search keywords from user query"""
        
        prompt = f"""Extract search keywords from this user query about prediction markets:

USER QUERY: "{user_query}"

Generate:
1. Primary keywords (2-4 core terms)
2. Market search terms (3-5 phrases to find relevant Polymarket markets)
3. Reddit search queries (3-5 specific queries to find relevant discussions)

Be specific and actionable. Focus on entities, events, and topics."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting search keywords for prediction markets and social media analysis."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "extract_keywords",
                "description": "Extract structured search keywords",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "2-4 core keywords"
                        },
                        "market_search_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 phrases to search Polymarket"
                        },
                        "reddit_search_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 queries for Reddit search"
                        }
                    },
                    "required": ["primary_keywords", "market_search_terms", "reddit_search_queries"]
                }
            }],
            function_call={"name": "extract_keywords"}
        )
        
        keywords_data = json.loads(response.choices[0].message.function_call.arguments)
        return SearchKeywords(**keywords_data)
    
    def analyze_market_sentiment(self, market_data, reddit_threads, user_query: str):
        """Use OpenAI to analyze market vs Reddit sentiment and generate trade call"""
        
        market_question = market_data.get('question', 'Unknown')
        outcome_prices = market_data.get('outcomePrices', [0.5, 0.5]) or [0.5, 0.5]
        
        try:
            current_yes_prob = float(outcome_prices[0])
        except Exception:
            try:
                current_yes_prob = float(market_data.get('yesPrice', 0.5))
            except Exception:
                current_yes_prob = 0.5
        
        current_yes_prob = max(0.0, min(1.0, current_yes_prob))
        current_no_prob = 1.0 - current_yes_prob

        raw_volume = market_data.get('volume', 0)
        try:
            volume_numeric = float(raw_volume)
        except Exception:
            volume_numeric = 0.0
        
        reddit_context = "\n".join([
            f"- [{t['score']} upvotes] {t['title']}: {t['selftext'][:200]}"
            for t in reddit_threads[:5]
        ])

        prompt = f"""You are a prediction market analyst responding to this user query:
USER QUERY: "{user_query}"

MARKET DETAILS:
Question: {market_question}
Current YES odds: {current_yes_prob:.2%}
Current NO odds: {current_no_prob:.2%}
Volume: ${volume_numeric:,.0f}

REDDIT DISCUSSIONS (top 5 by upvotes):
{reddit_context if reddit_context else "No relevant discussions found"}

TASK:
Analyze if Reddit sentiment diverges from market odds in context of the user's query. Provide a trade recommendation with:
1. Should we BUY/SELL/HOLD based on sentiment vs odds?
2. Which position (YES/NO)?
3. How confident are you?
4. What's the catalyst?
5. Key risk factors

Be specific and actionable."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are an expert prediction market analyst specializing in sentiment analysis and market inefficiencies."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "generate_trade_call",
                "description": "Generate structured trade recommendation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "call_type": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "ARBITRAGE"]},
                        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "recommended_position": {"type": "string", "enum": ["YES", "NO"]},
                        "suggested_size": {"type": "string", "enum": ["SMALL", "MEDIUM", "LARGE"]},
                        "reasoning": {"type": "string"},
                        "sentiment_score": {"type": "number", "minimum": -1, "maximum": 1},
                        "reddit_signal": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL", "MIXED"]},
                        "catalyst_type": {"type": "string", "enum": ["NEWS", "REDDIT_TREND", "MISPRICING", "ARBITRAGE", "MOMENTUM"]},
                        "urgency": {"type": "string", "enum": ["IMMEDIATE", "NEAR_TERM", "MONITOR"]},
                        "risk_factors": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["call_type", "confidence", "recommended_position", "suggested_size", 
                                "reasoning", "sentiment_score", "reddit_signal", "catalyst_type", 
                                "urgency", "risk_factors"]
                }
            }],
            function_call={"name": "generate_trade_call"}
        )
        
        ai_output = json.loads(response.choices[0].message.function_call.arguments)
        
        trade_call = TradeCall(
            call_type=CallType(ai_output['call_type']),
            confidence=Confidence(ai_output['confidence']),
            market_id=market_data.get('id', 'unknown'),
            market_question=market_question,
            current_odds=MarketOdds(yes=current_yes_prob, no=current_no_prob),
            recommended_position=Position(ai_output['recommended_position']),
            suggested_size=PositionSize(ai_output['suggested_size']),
            reasoning=ai_output['reasoning'],
            sentiment_score=ai_output['sentiment_score'],
            reddit_signal=Sentiment(ai_output['reddit_signal']),
            catalyst_type=CatalystType(ai_output['catalyst_type']),
            urgency=Urgency(ai_output['urgency']),
            risk_factors=ai_output['risk_factors'],
            key_reddit_threads=[
                RedditThread(url=t['url'], sentiment="positive" if t['score'] > 100 else "neutral", upvotes=t['score'])
                for t in reddit_threads[:3]
            ],
            timestamp=datetime.now().isoformat()
        )
        
        return trade_call

class PolymarketRedditAgent:
    def __init__(self, openai_api_key, reddit_client_id, reddit_client_secret, reddit_user_agent):
        self.polymarket = PolymarketClient()
        self.reddit = RedditClient(reddit_client_id, reddit_client_secret, reddit_user_agent)
        self.ai = AIAnalyzer(openai_api_key)
    
    def analyze_from_query(self, user_query: str):
        """
        Main function: User provides a query, we extract keywords, 
        fetch matching markets, search Reddit, and analyze
        """
        print(f"🤖 Processing query: '{user_query}'")
        print("\n" + "="*80)
        
        # STEP 1: Extract keywords using AI
        print("\n📝 STEP 1: Extracting keywords from query...")
        keywords = self.ai.extract_keywords_from_query(user_query)
        print(f"   Primary keywords: {', '.join(keywords.primary_keywords)}")
        print(f"   Market search terms: {', '.join(keywords.market_search_terms)}")
        print(f"   Reddit queries: {', '.join(keywords.reddit_search_queries)}")
        
        # STEP 2: Search Polymarket for matching markets
        print("\n📊 STEP 2: Searching Polymarket for relevant markets...")
        markets = self.polymarket.search_markets_by_keywords(keywords.market_search_terms, limit=3)
        print(f"   Found {len(markets)} matching markets")
        
        if not markets:
            print("   ❌ No markets found matching the query")
            return []
        
        # STEP 3: Search Reddit using keywords
        print("\n🔍 STEP 3: Searching Reddit for related discussions...")
        reddit_threads = self.reddit.search_with_keywords(keywords.reddit_search_queries)
        print(f"   Found {len(reddit_threads)} relevant Reddit threads")
        
        # STEP 4: Analyze each market with AI
        print("\n🧠 STEP 4: Analyzing markets with AI...")
        trade_calls = []
        
        for i, market in enumerate(markets[:5], 1):  # Limit to top 5 markets
            print(f"\n   [{i}/{min(5, len(markets))}] {market.get('question', 'Unknown')[:70]}...")
            
            try:
                trade_call = self.ai.analyze_market_sentiment(market, reddit_threads, user_query)
                trade_calls.append(trade_call)
                print(f"       ✅ {trade_call.call_type.value} {trade_call.recommended_position.value} - {trade_call.confidence.value} confidence")
            except Exception as e:
                print(f"       ❌ Error: {e}")
        
        print("\n" + "="*80)
        print(f"✨ Analysis complete! Generated {len(trade_calls)} trade calls\n")
        
        return trade_calls
    
    def save_calls_to_json(self, trade_calls, filename='trade_calls.json'):
        analysis =    json.dump([call.model_dump() for call in trade_calls], indent=2)
        return analysis
    