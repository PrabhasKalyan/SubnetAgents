import os
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field



class TickerType(str, Enum):
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    AUD = "AUD"
    CAD = "CAD"
    INR = "INR"

class Recommendation(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"



@dataclass
class StockQuery:
    company_name: str
    ticker_symbol: str
    country: str
    currency: Currency
    interval: TickerType
    raw_query: str


@dataclass
class StockPrediction:
    current_price: float
    predicted_price: float
    predicted_change_percent: float
    confidence_score: float
    prediction_horizon_days: int


@dataclass
class StockAnalysis:
    query: StockQuery
    historical_data: pd.DataFrame
    prediction: StockPrediction
    recommendation: Recommendation
    score: float
    reasoning: str
    timestamp: str



class StockInfoResponse(BaseModel):
    company_name: str
    ticker_symbol: str
    country: str
    currency: Literal["USD", "EUR", "GBP", "JPY", "INR", "CNY", "AUD", "CAD"]
    interval: Literal["1h", "1d", "5d", "1wk", "1mo", "3mo"]


class RecommendationResponse(BaseModel):
    recommendation: str
    score: float
    reasoning: str



class StockAnalysisAgent:
    """AI Agent for stock analysis and prediction"""

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API"))
        self.model = "gpt-5-mini"

    def extract_stock_info(self, user_query: str) -> StockQuery:
        """Extract structured stock info from a natural language query"""

        # ✅ Use beta.chat.completions.parse() for structured outputs
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data extraction expert. Extract structured stock information.",
                },
                {"role": "user", "content": f"User query: {user_query}"},
            ],
            response_format=StockInfoResponse,
            temperature=0.1,
        )

        data: StockInfoResponse = response.choices[0].message.parsed

        return StockQuery(
            company_name=data.company_name,
            ticker_symbol=data.ticker_symbol,
            country=data.country,
            currency=Currency(data.currency),
            interval=TickerType(data.interval),
            raw_query=user_query,
        )


    def fetch_historical_data(self, query: StockQuery, lookback_days: int = 60) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance"""

        ticker = yf.Ticker(query.ticker_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        hist_data = ticker.history(start=start_date, end=end_date, interval=query.interval.value)

        if hist_data.empty:
            raise ValueError(f"No historical data found for ticker {query.ticker_symbol}")

        return hist_data


    def predict_with_lagllama(self, historical_data: pd.DataFrame) -> StockPrediction:
        """Simplified LagLlama-style prediction"""

        prices = historical_data["Close"].values
        current_price = float(prices[-1])

        ma_short = np.mean(prices[-5:])
        ma_long = np.mean(prices[-20:])
        trend = (ma_short - ma_long) / ma_long

        prediction_horizon = 5
        predicted_change = trend * 0.5
        predicted_price = current_price * (1 + predicted_change)

        volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        confidence = max(0.5, 1 - volatility * 2)

        return StockPrediction(
            current_price=current_price,
            predicted_price=float(predicted_price),
            predicted_change_percent=float(predicted_change * 100),
            confidence_score=float(confidence),
            prediction_horizon_days=prediction_horizon,
        )

    def generate_recommendation(
        self, query: StockQuery, historical_data: pd.DataFrame, prediction: StockPrediction
    ) -> tuple[Recommendation, float, str]:
        """Generate structured buy/sell recommendation"""

        analysis_prompt = f"""
        Analyze this stock and return structured JSON only.

        Company: {query.company_name} ({query.ticker_symbol})
        Current Price: {prediction.current_price:.2f} {query.currency.value}
        Predicted Price ({prediction.prediction_horizon_days} days): {prediction.predicted_price:.2f}
        Expected Change: {prediction.predicted_change_percent:.2f}%
        Volatility: {np.std(historical_data['Close'][-20:]) / np.mean(historical_data['Close'][-20:]):.2f}
        Confidence: {prediction.confidence_score:.2f}

        Use one of: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL.
        Include reasoning and a 0-100 confidence score.
        """

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior financial analyst. Return structured JSON.",
                },
                {"role": "user", "content": analysis_prompt},
            ],
            response_format=RecommendationResponse,
            temperature=0.3,
        )

        parsed: RecommendationResponse = response.choices[0].message.parsed

        return (
            Recommendation(parsed.recommendation),
            float(parsed.score),
            parsed.reasoning,
        )

    def analyze(self, user_query: str) -> StockAnalysis:
        """Run full analysis pipeline"""

        print(f"🔍 Analyzing query: {user_query}\n")

        # Step 1
        print("📊 Extracting stock information...")
        query = self.extract_stock_info(user_query)
        print(f"   Company: {query.company_name}")
        print(f"   Ticker: {query.ticker_symbol}")
        print(f"   Currency: {query.currency.value}\n")

        # Step 2
        print("📈 Fetching historical data...")
        historical_data = self.fetch_historical_data(query)
        print(f"   Retrieved {len(historical_data)} records.\n")

        # Step 3
        print("🔮 Predicting future prices...")
        prediction = self.predict_with_lagllama(historical_data)
        print(
            f"   Current: {prediction.current_price:.2f} → Predicted: {prediction.predicted_price:.2f} "
            f"({prediction.predicted_change_percent:+.2f}%)"
        )

        # Step 4
        print("\n🤖 Generating AI recommendation...")
        recommendation, score, reasoning = self.generate_recommendation(
            query, historical_data, prediction
        )
        print(f"   Recommendation: {recommendation.value} ({score}/100)\n")

        # Compile results
        return StockAnalysis(
            query=query,
            historical_data=historical_data,
            prediction=prediction,
            recommendation=recommendation,
            score=score,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat(),
        )


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    agent = StockAnalysisAgent(openai_api_key=api_key)

    query = "Should I buy Apple stock? I want daily data in USD"
    analysis = agent.analyze(query)
    return analysis


if __name__ == "__main__":
    main()



    