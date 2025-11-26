from openai import OpenAI
import requests
import os
import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json



client = OpenAI(api_key=os.getenv("OPENAI_API"))
MODEL_NAME = "gpt-5-mini"
SERP_API_KEY = "bcbe76132dcf615504d6b69af3145f65b5ecfc43501d4e813b60c99337e44312"


class AnalysisType(Enum):
    CASH_FLOW = "cash_flow"
    REVENUE = "revenue"
    RISK = "risk"
    COMPREHENSIVE = "comprehensive"


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TimeHorizon(Enum):
    SHORT_TERM = 30
    MEDIUM_TERM = 90
    LONG_TERM = 180
    ANNUAL = 365


class FinancialMetric(Enum):
    FREE_CASH_FLOW = "freeCashflow"
    OPERATING_CASH_FLOW = "operatingCashflow"
    REVENUE = "totalRevenue"
    NET_INCOME = "netIncome"
    EBITDA = "ebitda"
    TOTAL_DEBT = "totalDebt"
    TOTAL_CASH = "totalCash"


class AnalysisRequest(Enum):
    ticker: str
    analysis_type: AnalysisType
    forecast_horizon: TimeHorizon

class CompanyFinancials:
    def __init__(
        self,
        ticker: str,
        company_name: str,
        sector: str,
        market_cap: float,
        current_price: float,
        pe_ratio: Optional[float],
        debt_to_equity: Optional[float],
        current_ratio: Optional[float],
        historical_prices: pd.DataFrame,
        cash_flow_statement: pd.DataFrame,
        income_statement: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        volatility: float,
        beta: Optional[float],
    ):
        self.ticker = ticker
        self.company_name = company_name
        self.sector = sector
        self.market_cap = market_cap
        self.current_price = current_price
        self.pe_ratio = pe_ratio
        self.debt_to_equity = debt_to_equity
        self.current_ratio = current_ratio
        self.historical_prices = historical_prices
        self.cash_flow_statement = cash_flow_statement
        self.income_statement = income_statement
        self.balance_sheet = balance_sheet
        self.volatility = volatility
        self.beta = beta

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "market_cap": float(self.market_cap) if self.market_cap else None,
            "current_price": float(self.current_price) if self.current_price else None,
            "pe_ratio": float(self.pe_ratio) if self.pe_ratio else None,
            "debt_to_equity": float(self.debt_to_equity) if self.debt_to_equity else None,
            "current_ratio": float(self.current_ratio) if self.current_ratio else None,
            "volatility": float(self.volatility) if self.volatility else None,
            "beta": float(self.beta) if self.beta else None
        }


class ForecastResult:
    def __init__(
        self,
        metric_name: str,
        forecast_df: pd.DataFrame,
        confidence_lower: List[float],
        confidence_upper: List[float],
        predicted_values: List[float],
        trend_direction: str,
        growth_rate: float,
    ):
        self.metric_name = metric_name
        self.forecast_df = forecast_df
        self.confidence_lower = confidence_lower
        self.confidence_upper = confidence_upper
        self.predicted_values = predicted_values
        self.trend_direction = trend_direction
        self.growth_rate = growth_rate

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "metric_name": self.metric_name,
            "trend_direction": self.trend_direction,
            "growth_rate": float(self.growth_rate),
            "predicted_values": [float(v) for v in self.predicted_values] if self.predicted_values else [],
            "confidence_lower": [float(v) for v in self.confidence_lower] if self.confidence_lower else [],
            "confidence_upper": [float(v) for v in self.confidence_upper] if self.confidence_upper else [],
            "average_forecast": float(np.mean(self.predicted_values)) if self.predicted_values else 0.0
        }


class RiskAssessment:
    def __init__(
        self,
        overall_risk: RiskLevel,
        liquidity_risk: RiskLevel,
        market_risk: RiskLevel,
        debt_risk: RiskLevel,
        risk_factors: List[str],
        risk_score: float,
    ):
        self.overall_risk = overall_risk
        self.liquidity_risk = liquidity_risk
        self.market_risk = market_risk
        self.debt_risk = debt_risk
        self.risk_factors = risk_factors
        self.risk_score = risk_score

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "overall_risk": self.overall_risk.value,
            "liquidity_risk": self.liquidity_risk.value,
            "market_risk": self.market_risk.value,
            "debt_risk": self.debt_risk.value,
            "risk_factors": self.risk_factors,
            "risk_score": float(self.risk_score)
        }


class FinancialDataAgent:
    """Fetches real financial data using yfinance"""

    def fetch_company_data(self, ticker: str, period: str = "2y") -> CompanyFinancials:
        """Fetch comprehensive financial data for a company"""
        stock = yf.Ticker(ticker)
        info = stock.info

        hist = stock.history(period=period)
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)

        cash_flow = stock.cashflow
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet

        return CompanyFinancials(
            ticker=ticker,
            company_name=info.get('longName', ticker),
            sector=info.get('sector', 'Unknown'),
            market_cap=info.get('marketCap', 0),
            current_price=info.get('currentPrice', hist['Close'].iloc[-1]),
            pe_ratio=info.get('trailingPE'),
            debt_to_equity=info.get('debtToEquity'),
            current_ratio=info.get('currentRatio'),
            historical_prices=hist,
            cash_flow_statement=cash_flow,
            income_statement=income_stmt,
            balance_sheet=balance_sheet,
            volatility=volatility,
            beta=info.get('beta')
        )

    def extract_metric_series(self, financials: CompanyFinancials,
                             metric: FinancialMetric) -> pd.DataFrame:
        """Extract a specific metric time series"""
        if metric in [FinancialMetric.FREE_CASH_FLOW, FinancialMetric.OPERATING_CASH_FLOW]:
            stmt = financials.cash_flow_statement
        elif metric in [FinancialMetric.REVENUE, FinancialMetric.NET_INCOME, FinancialMetric.EBITDA]:
            stmt = financials.income_statement
        else:
            stmt = financials.balance_sheet

        try:
            data = stmt.loc[metric.value]
            df = pd.DataFrame({
                'ds': pd.to_datetime(data.index),
                'y': data.values
            }).sort_values('ds').reset_index(drop=True)
            return df
        except KeyError:
            return pd.DataFrame(columns=['ds', 'y'])


class ResearchAgent:
    """Fetches market intelligence and news"""

    def fetch_market_intelligence(self, company_name: str, analysis_type: AnalysisType) -> str:
        """Fetch relevant news and analysis"""
        query = f"{company_name} {analysis_type} financial analysis 2025"

        params = {
            "engine": "google",
            "q": query,
            "num": 10,
            "api_key": SERP_API_KEY,
        }

        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
            data = r.json()

            snippets = []
            for item in data.get("organic_results", [])[:5]:
                if "snippet" in item:
                    snippets.append(f"• {item['snippet']}")

            return "\n".join(snippets) if snippets else "No recent news found."
        except Exception as e:
            return f"Research unavailable: {str(e)}"


class MLAnalysisAgent:
    """Performs time series forecasting and anomaly detection"""

    def forecast_metric(self, data: pd.DataFrame,
                        periods: int,
                        metric_name: str) -> ForecastResult:
        """Forecast financial metric using Prophet"""
        if data.empty or len(data) < 2:
            return ForecastResult(
                metric_name=metric_name,
                forecast_df=pd.DataFrame(),
                confidence_lower=[],
                confidence_upper=[],
                predicted_values=[],
                trend_direction="insufficient_data",
                growth_rate=0.0
            )

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )

        model.fit(data)
        future = model.make_future_dataframe(periods=periods, freq='Q')
        forecast = model.predict(future)
        future_forecast = forecast.tail(periods)

        if len(data) >= 2:
            recent_avg = data['y'].tail(4).mean()
            forecast_avg = future_forecast['yhat'].mean()
            growth_rate = ((forecast_avg - recent_avg) / abs(recent_avg)) * 100
        else:
            growth_rate = 0.0

        trend = "increasing" if growth_rate > 2 else "decreasing" if growth_rate < -2 else "stable"

        return ForecastResult(
            metric_name=metric_name,
            forecast_df=future_forecast,
            confidence_lower=future_forecast['yhat_lower'].tolist(),
            confidence_upper=future_forecast['yhat_upper'].tolist(),
            predicted_values=future_forecast['yhat'].tolist(),
            trend_direction=trend,
            growth_rate=growth_rate
        )


class RiskAssessmentAgent:
    """Evaluates financial risks"""

    def assess_risks(self, financials: CompanyFinancials,
                     forecast_results: Dict[str, ForecastResult]) -> RiskAssessment:
        """Comprehensive risk assessment"""
        risk_factors = []
        risk_scores = []

        liquidity_risk = RiskLevel.LOW
        if financials.current_ratio:
            if financials.current_ratio < 1.0:
                liquidity_risk = RiskLevel.HIGH
                risk_factors.append("Current ratio below 1.0 indicates liquidity concerns")
                risk_scores.append(0.8)
            elif financials.current_ratio < 1.5:
                liquidity_risk = RiskLevel.MODERATE
                risk_factors.append("Current ratio below optimal level")
                risk_scores.append(0.5)
            else:
                risk_scores.append(0.2)

        market_risk = RiskLevel.LOW
        if financials.volatility > 0.40:
            market_risk = RiskLevel.HIGH
            risk_factors.append(f"High volatility: {financials.volatility:.2%} annualized")
            risk_scores.append(0.8)
        elif financials.volatility > 0.25:
            market_risk = RiskLevel.MODERATE
            risk_factors.append(f"Moderate volatility: {financials.volatility:.2%}")
            risk_scores.append(0.5)
        else:
            risk_scores.append(0.2)

        debt_risk = RiskLevel.LOW
        if financials.debt_to_equity:
            if financials.debt_to_equity > 2.0:
                debt_risk = RiskLevel.HIGH
                risk_factors.append(f"High debt-to-equity ratio: {financials.debt_to_equity:.2f}")
                risk_scores.append(0.9)
            elif financials.debt_to_equity > 1.0:
                debt_risk = RiskLevel.MODERATE
                risk_factors.append(f"Elevated debt levels: D/E = {financials.debt_to_equity:.2f}")
                risk_scores.append(0.5)
            else:
                risk_scores.append(0.2)

        if 'cash_flow' in forecast_results:
            cf_forecast = forecast_results['cash_flow']
            if cf_forecast.trend_direction == "decreasing":
                risk_factors.append("Declining cash flow trend projected")
                risk_scores.append(0.7)

        avg_risk_score = np.mean(risk_scores) if risk_scores else 0.5

        if avg_risk_score > 0.7:
            overall_risk = RiskLevel.HIGH
        elif avg_risk_score > 0.5:
            overall_risk = RiskLevel.MODERATE
        else:
            overall_risk = RiskLevel.LOW

        return RiskAssessment(
            overall_risk=overall_risk,
            liquidity_risk=liquidity_risk,
            market_risk=market_risk,
            debt_risk=debt_risk,
            risk_factors=risk_factors,
            risk_score=avg_risk_score
        )


class LLMAnalysisAgent:
    """Interprets quantitative results using LLM"""

    def generate_analysis(self, financials: CompanyFinancials,
                         forecasts: Dict[str, ForecastResult],
                         risk_assessment: RiskAssessment,
                         market_intel: str) -> Dict:
        """Generate comprehensive analysis as structured data"""

        forecast_summary = []
        for name, forecast in forecasts.items():
            if forecast and hasattr(forecast, 'forecast_df') and not forecast.forecast_df.empty:
                forecast_summary.append(
                    f"**{name.upper()}**: {forecast.trend_direction} trend, "
                    f"{forecast.growth_rate:+.1f}% projected change"
                )

        prompt = f"""
# FINANCIAL ANALYSIS REQUEST

## Company Profile
- **Company**: {financials.company_name} ({financials.ticker})
- **Sector**: {financials.sector}
- **Market Cap**: ${financials.market_cap:,.0f}
- **Current Price**: ${financials.current_price:.2f}
- **P/E Ratio**: {financials.pe_ratio if financials.pe_ratio else 'N/A'}
- **Beta**: {financials.beta if financials.beta else 'N/A'}
- **Volatility**: {financials.volatility:.2%}

## Forecast Results
{chr(10).join(forecast_summary) if forecast_summary else 'No forecasts available'}

## Risk Assessment
- **Overall Risk**: {risk_assessment.overall_risk.value.upper()}
- **Risk Score**: {risk_assessment.risk_score:.2f}/1.0
- **Key Risk Factors**:
{chr(10).join(f"  - {factor}" for factor in risk_assessment.risk_factors)}

## Market Intelligence
{market_intel}

## ANALYSIS TASK
As a senior financial analyst, provide a structured analysis with these sections:

1. **Executive Summary** (2-3 sentences on overall financial health)
2. **Cash Flow Analysis** (sustainability, trends, concerns)
3. **Risk Evaluation** (interpretation of risk metrics and outlook)
4. **Investment Outlook** (key opportunities and threats)
5. **Strategic Recommendations** (list 3-4 actionable insights)

Provide your response in a clear, professional manner.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a CFA charterholder and senior financial analyst with expertise in quantitative finance and risk management."},
                {"role": "user", "content": prompt}
            ]
        )

        analysis_text = response.choices[0].message.content.strip()
        
        return {
            "full_analysis": analysis_text,
            "summary": analysis_text.split('\n')[0] if analysis_text else ""
        }


class ReportingAgent:
    """Generates structured JSON reports"""

    def generate_json_report(self, financials: CompanyFinancials,
                            forecasts: Dict[str, ForecastResult],
                            risk_assessment: RiskAssessment,
                            llm_analysis: Dict,
                            market_intel: str) -> Dict:
        """Create structured JSON report"""

        report = {
            "metadata": {
                "report_date": datetime.now().isoformat(),
                "analysis_period": "Last 2 years + 6-month forecast",
                "generated_by": "Enterprise Financial Analysis System"
            },
            "company": financials.to_dict(),
            "risk_assessment": risk_assessment.to_dict(),
            "forecasts": {
                name: forecast.to_dict() 
                for name, forecast in forecasts.items() 
                if forecast and hasattr(forecast, 'forecast_df') and not forecast.forecast_df.empty
            },
            "market_intelligence": market_intel,
            "analysis": llm_analysis,
            "disclaimer": "This report is for informational purposes only and should not be considered investment advice. Past performance does not guarantee future results."
        }

        return report


def agent_understand(query):

    prompt = f"""
    You are a financial AI agent that converts user queries into structured parameters.
    From the following query:
    "{query}"
    
    Extract:
    1. ticker (e.g., AAPL, TSLA, MSFT)
    2. analysis_type (choose from: comprehensive, technical, fundamental)
    3. forecast_horizon (choose from: short_term, medium_term, long_term)
    4. Ticker is the company symbol on a particular exchange it can't be empty
    
    Return only valid JSON in this exact format:
    {{
        "ticker": "<ticker>",
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


def run_financial_analysis(query) -> Dict:
    agent_understand(query)

    data = agent_understand(query=query)

    ticker = data['ticker']
    analysis_type = data["analysis_type"]
    forecast_horizon = data["forecast_horizon"]


    print(f"🔍 Fetching financial data for {ticker}...")

    data_agent = FinancialDataAgent()
    financials = data_agent.fetch_company_data(ticker)

    print(f"✅ Loaded data for {financials.company_name}")

    print("📰 Gathering market intelligence...")
    research_agent = ResearchAgent()
    market_intel = research_agent.fetch_market_intelligence(
        financials.company_name,
        analysis_type
    )

    print("📊 Running forecasting models...")
    ml_agent = MLAnalysisAgent()
    forecasts: Dict[str, ForecastResult] = {}

    metrics_to_forecast = [
        (FinancialMetric.FREE_CASH_FLOW, 'cash_flow'),
        (FinancialMetric.REVENUE, 'revenue'),
    ]

    for metric_enum, metric_name in metrics_to_forecast:
        metric_data = data_agent.extract_metric_series(financials, metric_enum)
        if not metric_data.empty:
            forecast = ml_agent.forecast_metric(
                metric_data,
                periods=4,
                metric_name=metric_name
            )
            forecasts[metric_name] = forecast

    print("⚠️  Assessing risks...")
    risk_agent = RiskAssessmentAgent()
    risk_assessment = risk_agent.assess_risks(financials, forecasts)

    print("🤖 Generating AI analysis...")
    llm_agent = LLMAnalysisAgent()
    analysis = llm_agent.generate_analysis(
        financials,
        forecasts,
        risk_assessment,
        market_intel
    )

    print("📝 Compiling final report...")
    reporting_agent = ReportingAgent()
    final_report = reporting_agent.generate_json_report(
        financials,
        forecasts,
        risk_assessment,
        analysis,
        market_intel
    )

    print("✅ Analysis complete!")
    return final_report


if __name__ == "__main__":
    # Example: Tesla analysis with JSON output
    report = run_financial_analysis(
        "I want to analyse beste lectronics company stock for the next 6 months"
    )
    
    # Print formatted JSON to console
    print("\n" + "="*80)
    print("JSON OUTPUT:")
    print("="*80)
    print(json.dumps(report, indent=2))
