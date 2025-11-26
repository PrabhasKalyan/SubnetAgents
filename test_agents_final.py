from openai import OpenAI
import json
import os
import asyncio

# === Import all your agents ===
from agent1 import FinNews
from agent2 import run_financial_analysis
from agent3 import (
    create_sample_portfolio,
    create_sample_tax_profile,
    TaxOptimizationEngine,
    TaxReportGenerator,
    PortfolioManager
)
from agent4 import FoodPlanner
from agent5 import StockAnalysisAgent
from agent6 import PolymarketRedditAgent
from fastapi import FastAPI

# === Initialize LLM Client ===
llm = OpenAI(base_url="https://api.groq.com/openai/v1",api_key="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")
app = FastAPI()

# === Step 1: Route query to best agent ===
def route_agent(prompt: str) -> str:
    routing_prompt = f"""
    You are an expert router. Choose the single most relevant agent ID for the user's request.

    AVAILABLE AGENTS:
    - financial_news_analysis → Get or summarize recent financial news.
    - corporate_financial_analysis → Analyze a company’s financial health.
    - tax_optimization_planning → Run personal or portfolio tax optimization.
    - meal_restaurant_recommendation → Suggest meals or restaurants.
    - structured_stock_prediction → Predict stock movement and give recommendations.
    - prediction_market_signal → Analyze prediction markets and Reddit sentiment.

    USER REQUEST: "{prompt}"

    Respond ONLY with the agent ID (e.g., 'meal_restaurant_recommendation').
    """

    resp = llm.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": routing_prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content


# === Step 2: Execute the chosen agent ===
async def run_agent(agent_id: str, query: str):
    print(f"➡️ Selected Agent: {agent_id}")

    if agent_id == "financial_news_analysis":
        agent = FinNews()
        return agent.fin_news_agent()

    elif agent_id == "corporate_financial_analysis":
        return run_financial_analysis(query)

    elif agent_id == "tax_optimization_planning":
        positions = create_sample_portfolio()
        tax_profile = create_sample_tax_profile()
        portfolio = PortfolioManager(positions)
        engine = TaxOptimizationEngine(portfolio, tax_profile)
        optimizations = engine.run_complete_analysis()
        report_generator = TaxReportGenerator()
        return report_generator.generate_report(engine)

    elif agent_id == "meal_restaurant_recommendation":
        SERPAPI_KEY = os.getenv(
            "SERPAPI_KEY",
            "bcbe76132dcf615504d6b69af3145f65b5ecfc43501d4e813b60c99337e44312",
        )
        planner = FoodPlanner(SERPAPI_KEY)
        rec = await planner.generate_meal_recommendation(query)
        return json.dumps(rec, indent=2)

    elif agent_id == "structured_stock_prediction":
        agent = StockAnalysisAgent(openai_api_key=os.getenv("OPENAI_API_KEY"))
        analysis = agent.analyze(query)

        # Capture printed output as string
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            print(analysis)
        return f.getvalue()

    elif agent_id == "prediction_market_signal":
        api_key = os.getenv("POLYMARKET_API_KEY", "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")
        reddit_id = os.getenv("REDDIT_ID", "_3whAB52PZMQLYhc4R3WYg")
        reddit_secret = os.getenv("REDDIT_SECRET", "UBu_lV9pOi__QSsFpUnt1q_7eOxWEw")
        user_agent = "polymarket-agent/1.0"

        agent = PolymarketRedditAgent(api_key, reddit_id, reddit_secret, user_agent)
        trade_calls = agent.analyze_from_query(user_query=query)
        print(trade_calls)
        return (
            json.dumps([call.model_dump() for call in trade_calls], indent=2)
            if trade_calls
            else "No trade calls generated."
        )

    else:
        return f"Unknown agent ID: {agent_id}"



@app.get("/")
def final_agent(query):
    agent_id = route_agent(query)
    result = asyncio.run(run_agent(agent_id, query))
    return result






