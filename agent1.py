import json
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
import os

class TagCategory(str, Enum):
    TECHNOLOGY = "Technology"
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"
    ENERGY = "Energy"
    CONSUMER = "Consumer"
    INDUSTRIAL = "Industrial"
    REAL_ESTATE = "Real Estate"
    MATERIALS = "Materials"
    TELECOMMUNICATIONS = "Telecommunications"
    UTILITIES = "Utilities"
    POLITICS = "Politics"
    ECONOMICS = "Economics"
    REGULATORY = "Regulatory"
    ESG = "ESG"
    CRYPTOCURRENCY = "Cryptocurrency"

class EventType(str, Enum):
    EARNINGS_REPORT = "Earnings Report"
    MERGER_ACQUISITION = "Merger & Acquisition"
    PRODUCT_LAUNCH = "Product Launch"
    PARTNERSHIP = "Partnership"
    REGULATORY_CHANGE = "Regulatory Change"
    LEADERSHIP_CHANGE = "Leadership Change"
    MARKET_MOVEMENT = "Market Movement"
    LEGAL_ISSUE = "Legal Issue"
    FINANCIAL_GUIDANCE = "Financial Guidance"
    NONE = "None"

class ArticleAnalysis(BaseModel):
    sentiment: float = Field(
        description="Sentiment score from -1 (very negative) to 1 (very positive)",
        ge=-1.0,
        le=1.0
    )
    impact: float = Field(
        description="Market impact score from 0 (no impact) to 1 (high impact)",
        ge=0.0,
        le=1.0
    )
    relevance: float = Field(
        description="Relevance score from 0 (not relevant) to 1 (highly relevant)",
        ge=0.0,
        le=1.0
    )
    tags: List[TagCategory] = Field(
        description="List of applicable category tags for the article",
        min_length=1,
        max_length=5
    )
    event_detection: List[EventType] = Field(
        description="List of detected event types in the article",
        max_length=3
    )
    reasoning: str = Field(
        description="Detailed explanation for the scores and classifications assigned"
    )

class NewsAnalysisResponse(BaseModel):
    article_id: str
    article_title: str
    analysis: ArticleAnalysis

class BatchNewsAnalysisResponse(BaseModel):
    analyses: List[NewsAnalysisResponse]

def get_resp(articles):

    articles_data = []
    for idx, article in enumerate(articles, 1):
        article_info = {
            'index': idx,
            'id': article.get('id', f'article_{idx}'),
            'title': article['title'],
            'description': article['description'],
            'publisher': article['publisher']['name'],
            'published': article['published_utc'],
            'tickers': ', '.join(article['tickers']),
            'keywords': ', '.join(article['keywords'])
        }
        articles_data.append(article_info)
    

    articles_text = "\n\n".join([
                f"""Article {a['index']}:
        - ID: {a['id']}
        - Title: {a['title']}
        - Description: {a['description']}
        - Publisher: {a['publisher']}
        - Published: {a['published']}
        - Tickers: {a['tickers']}
        - Keywords: {a['keywords']}"""
                for a in articles_data
            ])
    
    prompt = f"""You are a financial news analyst. Analyze ALL {len(articles)} news articles below and provide structured scoring and classification for EACH article in a single JSON response.

            **Articles to Analyze:**

            {articles_text}

            **Analysis Instructions (apply to EACH article individually):**

            1. **Sentiment** (-1 to 1): Evaluate the market sentiment (bearish to bullish)
            - Bearish (-1 to -0.3): Negative for markets/stocks - rate hikes, recession fears, poor earnings, regulatory crackdowns, geopolitical risks, bear market signals
            - Neutral (-0.3 to 0.3): Mixed signals, uncertain impact, or balanced news
            - Bullish (0.3 to 1): Positive for markets/stocks - rate cuts, economic growth, strong earnings, favorable policy, bull market signals
            
            Focus on: How would this news affect stock prices, investor confidence, and market direction?

            2. **Impact** (0 to 1): Assess how much the stock price would be impacted by this news
            - Low (0 to 0.3): Minimal price movement expected (< 1-2%), background noise, tangential information
            - Medium (0.3 to 0.7): Moderate price movement expected (2-5%), notable but not dramatic
            - High (0.7 to 1): Significant price movement expected (> 5%), major catalyst, game-changing news
            
            Consider: Direct business impact, earnings implications, competitive position changes, material developments

            3. **Relevance** (0 to 1): Determine how directly related this news is to the specific ticker(s) mentioned
            - Low (0 to 0.3): Ticker mentioned in passing, general market news, industry-wide trends with no specific connection
            - Medium (0.3 to 0.7): Ticker affected as part of broader sector/theme, indirect relationship, competitive implications
            - High (0.7 to 1): News directly about the ticker - company-specific announcements, direct mentions, primary focus of article
            
            Focus on: Is this news ABOUT the ticker or just mentioning it? How central is the ticker to the story?

            4. **Tags**: Select 1-5 appropriate category tags from:
            - Technology, Finance, Healthcare, Energy, Consumer, Industrial, Real Estate, Materials, Telecommunications, Utilities, Politics, Economics, Regulatory, ESG, Cryptocurrency

            5. **Event Detection**: Identify up to 3 event types from:
            - Earnings Report, Merger & Acquisition, Product Launch, Partnership, Regulatory Change, Leadership Change, Market Movement, Legal Issue, Financial Guidance, None

            6. **Reasoning**: Provide a concise explanation (2-4 sentences) justifying your scores and classifications. Specifically explain:
            - Why the news is bearish/bullish and the expected sentiment direction
            - What specific price impact magnitude is expected for the ticker and why
            - How directly related this news is to the ticker (primary subject vs. tangential mention)

            **IMPORTANT**: Return a JSON object with an "analyses" array containing exactly {len(articles)} analysis objects, one for each article in the order presented above. Each analysis must include the article_id and article_title from the corresponding article."""

    client = OpenAI(api_key="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")
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a financial news analyst providing structured analysis for {len(articles)} articles simultaneously."},
            {"role": "user", "content": prompt}
        ],
        response_format=BatchNewsAnalysisResponse
    )

    result = completion.choices[0].message.parsed
    print(json.dumps(result.model_dump(), indent=2))
    return result

def get_data(limit,keyword):
    api_key = "MbT_pmEz2Lhn8zRqT7e9KmBhIg3uuayK"
    url = f"https://api.polygon.io/v2/reference/news?ticker={keyword}&order=desc&limit={limit}&sort=published_utc&apiKey={api_key}"
    resp = requests.get(url = url)
    return list(json.loads(resp.text)["results"])
    # url = f"https://serpapi.com/search?engine=google_news_light&q={keyword}"
    # resp = requests.get(url=url,params={"api_key":"bcbe76132dcf615504d6b69af3145f65b5ecfc43501d4e813b60c99337e44312"})
    # return resp.json()['news_results']


def agent_understand(query):

    client = OpenAI(api_key=os.getenv("OPENAI_API"))
    prompt = f"""
    You are a financial AI agent that converts user queries into structured parameters.
    From the following query:
    "{query}"
    
    Extract:
    1. tickers (e.g., AAPL, TSLA, MSFT) not more than 3
    2. analysis_type (choose from: comprehensive, technical, fundamental)
    3. forecast_horizon (choose from: short_term, medium_term, long_term)
    4. Ticker is the company symbol on a particular exchange it can't be empty
    5. Keywords in array not more than 5
    
    Return only valid JSON in this exact format:
    {{
        "ticker": "array of tickers ",
        "keywords: "array of keywords",
        "analysis_type": "<analysis_type>",
        "forecast_horizon": "<forecast_horizon>",
        "currency": "<currency>"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a structured financial query interpreter."},
            {"role": "user", "content": prompt}
        ]
        # response_format=AnalysisRequest
    )

    ai_output = response.choices[0].message.content
    data = json.loads(ai_output)

    return data


class FinNews():
    def fin_news_agent(self,query):
        tickers = agent_understand(query)['ticker']
        all_resp = []
        for ticker in tickers:
            all_data = get_data(2,ticker)
            all_resp.append(get_resp(all_data))
        return all_resp



