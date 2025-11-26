from openai import OpenAI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from scipy.optimize import minimize
import json
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API")
)
MODEL_NAME = "gpt-5-mini"


class TaxFilingStatus(Enum):
    SINGLE = "single"
    MARRIED_JOINT = "married_joint"
    MARRIED_SEPARATE = "married_separate"
    HEAD_OF_HOUSEHOLD = "head_of_household"


class TaxStrategy(Enum):
    TAX_LOSS_HARVEST = "tax_loss_harvest"
    ROTH_CONVERSION = "roth_conversion"
    CHARITABLE_DONATION = "charitable_donation"
    CAPITAL_GAIN_TIMING = "capital_gain_timing"
    RETIREMENT_CONTRIBUTION = "retirement_contribution"


class AccountType(Enum):
    TAXABLE = "taxable"
    TRADITIONAL_IRA = "traditional_ira"
    ROTH_IRA = "roth_ira"
    TRADITIONAL_401K = "traditional_401k"
    ROTH_401K = "roth_401k"
    HSA = "hsa"


class AssetClass(Enum):
    EQUITY = "equity"
    BOND = "bond"
    REAL_ESTATE = "real_estate"
    CRYPTO = "crypto"
    CASH = "cash"


class Position:
    """Individual investment position"""

    def __init__(
        self,
        ticker: str,
        quantity: float,
        purchase_price: float,
        purchase_date: datetime,
        current_price: float,
        account_type: AccountType,
        asset_class: AssetClass,
    ):
        self.ticker = ticker
        self.quantity = quantity
        self.purchase_price = purchase_price
        self.purchase_date = purchase_date
        self.current_price = current_price
        self.account_type = account_type
        self.asset_class = asset_class

    def get_cost_basis(self) -> float:
        return self.quantity * self.purchase_price

    def get_market_value(self) -> float:
        return self.quantity * self.current_price

    def get_unrealized_gain_loss(self) -> float:
        return self.get_market_value() - self.get_cost_basis()

    def get_holding_period_days(self) -> int:
        return (datetime.now() - self.purchase_date).days

    def is_long_term(self) -> bool:
        """Long-term if held > 365 days"""
        return self.get_holding_period_days() > 365


class TaxProfile:
    """User's tax situation"""

    def __init__(
        self,
        filing_status: TaxFilingStatus,
        annual_income: float,
        state: str,
        age: int,
        standard_deduction: float = 0,
        itemized_deductions: float = 0,
        traditional_401k_contributions: float = 0,
        roth_401k_contributions: float = 0,
        traditional_ira_contributions: float = 0,
        roth_ira_contributions: float = 0,
        hsa_contributions: float = 0,
        short_term_gains: float = 0,
        long_term_gains: float = 0,
        carryforward_losses: float = 0,
    ):
        self.filing_status = filing_status
        self.annual_income = annual_income
        self.state = state
        self.age = age
        self.standard_deduction = standard_deduction
        self.itemized_deductions = itemized_deductions
        self.traditional_401k_contributions = traditional_401k_contributions
        self.roth_401k_contributions = roth_401k_contributions
        self.traditional_ira_contributions = traditional_ira_contributions
        self.roth_ira_contributions = roth_ira_contributions
        self.hsa_contributions = hsa_contributions
        self.short_term_gains = short_term_gains
        self.long_term_gains = long_term_gains
        self.carryforward_losses = carryforward_losses

        if self.standard_deduction == 0:
            deductions = {
                TaxFilingStatus.SINGLE: 14600,
                TaxFilingStatus.MARRIED_JOINT: 29200,
                TaxFilingStatus.MARRIED_SEPARATE: 14600,
                TaxFilingStatus.HEAD_OF_HOUSEHOLD: 21900,
            }
            self.standard_deduction = deductions[self.filing_status]

    def copy(self) -> "TaxProfile":
        return TaxProfile(
            filing_status=self.filing_status,
            annual_income=self.annual_income,
            state=self.state,
            age=self.age,
            standard_deduction=self.standard_deduction,
            itemized_deductions=self.itemized_deductions,
            traditional_401k_contributions=self.traditional_401k_contributions,
            roth_401k_contributions=self.roth_401k_contributions,
            traditional_ira_contributions=self.traditional_ira_contributions,
            roth_ira_contributions=self.roth_ira_contributions,
            hsa_contributions=self.hsa_contributions,
            short_term_gains=self.short_term_gains,
            long_term_gains=self.long_term_gains,
            carryforward_losses=self.carryforward_losses,
        )


class TaxOptimization:
    """Recommended tax optimization action"""

    def __init__(
        self,
        strategy: TaxStrategy,
        description: str,
        estimated_tax_savings: float,
        action_items: List[str],
        priority: int,
        deadline: Optional[datetime] = None,
        positions_to_sell: Optional[List[Position]] = None,
        positions_to_buy: Optional[List[str]] = None,
        amount: float = 0,
    ):
        self.strategy = strategy
        self.description = description
        self.estimated_tax_savings = estimated_tax_savings
        self.action_items = action_items
        self.priority = priority
        self.deadline = deadline
        self.positions_to_sell = positions_to_sell or []
        self.positions_to_buy = positions_to_buy or []
        self.amount = amount

# ========== TAX CALCULATORS ==========

class FederalTaxCalculator:
    """Calculate federal income tax"""
    
    # 2024 Tax brackets
    BRACKETS_2024 = {
        TaxFilingStatus.SINGLE: [
            (11600, 0.10),
            (47150, 0.12),
            (100525, 0.22),
            (191950, 0.24),
            (243725, 0.32),
            (609350, 0.35),
            (float('inf'), 0.37)
        ],
        TaxFilingStatus.MARRIED_JOINT: [
            (23200, 0.10),
            (94300, 0.12),
            (201050, 0.22),
            (383900, 0.24),
            (487450, 0.32),
            (731200, 0.35),
            (float('inf'), 0.37)
        ]
    }
    
    # Long-term capital gains brackets
    LTCG_BRACKETS = {
        TaxFilingStatus.SINGLE: [
            (47025, 0.0),
            (518900, 0.15),
            (float('inf'), 0.20)
        ],
        TaxFilingStatus.MARRIED_JOINT: [
            (94050, 0.0),
            (583750, 0.15),
            (float('inf'), 0.20)
        ]
    }
    
    def calculate_ordinary_income_tax(self, taxable_income: float, 
                                     filing_status: TaxFilingStatus) -> float:
        """Calculate federal income tax on ordinary income"""
        brackets = self.BRACKETS_2024.get(filing_status, self.BRACKETS_2024[TaxFilingStatus.SINGLE])
        
        tax = 0
        prev_threshold = 0
        
        for threshold, rate in brackets:
            if taxable_income <= threshold:
                tax += (taxable_income - prev_threshold) * rate
                break
            else:
                tax += (threshold - prev_threshold) * rate
                prev_threshold = threshold
        
        return tax
    
    def calculate_ltcg_tax(self, ltcg: float, taxable_income: float,
                          filing_status: TaxFilingStatus) -> float:
        """Calculate long-term capital gains tax"""
        brackets = self.LTCG_BRACKETS.get(filing_status, self.LTCG_BRACKETS[TaxFilingStatus.SINGLE])
        
        # LTCG brackets are based on total taxable income + LTCG
        total_income = taxable_income + ltcg
        
        tax = 0
        prev_threshold = 0
        remaining_gain = ltcg
        
        for threshold, rate in brackets:
            if total_income <= threshold:
                tax += remaining_gain * rate
                break
            else:
                # Portion in this bracket
                portion = min(remaining_gain, threshold - max(taxable_income, prev_threshold))
                if portion > 0:
                    tax += portion * rate
                    remaining_gain -= portion
                prev_threshold = threshold
        
        return tax
    
    def calculate_total_tax(self, profile: TaxProfile) -> Dict[str, float]:
        """Calculate complete tax liability"""
        # Calculate AGI
        agi = profile.annual_income
        agi -= profile.traditional_401k_contributions
        agi -= profile.traditional_ira_contributions
        agi -= profile.hsa_contributions
        
        # Deductions
        deduction = max(profile.standard_deduction, profile.itemized_deductions)
        taxable_income = max(0, agi - deduction)
        
        # Ordinary income tax
        ordinary_tax = self.calculate_ordinary_income_tax(taxable_income, profile.filing_status)
        
        # Short-term capital gains (taxed as ordinary income)
        net_short_term = profile.short_term_gains
        if net_short_term > 0:
            stcg_tax = self.calculate_ordinary_income_tax(
                taxable_income + net_short_term, profile.filing_status
            ) - ordinary_tax
        else:
            stcg_tax = 0
        
        # Long-term capital gains (preferential rates)
        net_long_term = max(0, profile.long_term_gains - profile.carryforward_losses)
        ltcg_tax = self.calculate_ltcg_tax(net_long_term, taxable_income, profile.filing_status)
        
        # Net Investment Income Tax (3.8% on investment income if AGI > threshold)
        niit_threshold = 200000 if profile.filing_status == TaxFilingStatus.SINGLE else 250000
        if agi > niit_threshold:
            niit = min(agi - niit_threshold, 
                      profile.short_term_gains + profile.long_term_gains) * 0.038
        else:
            niit = 0
        
        total_tax = ordinary_tax + stcg_tax + ltcg_tax + niit
        
        return {
            "agi": agi,
            "taxable_income": taxable_income,
            "ordinary_tax": ordinary_tax,
            "stcg_tax": stcg_tax,
            "ltcg_tax": ltcg_tax,
            "niit": niit,
            "total_federal_tax": total_tax,
            "effective_rate": total_tax / profile.annual_income if profile.annual_income > 0 else 0
        }

# ========== PORTFOLIO MANAGER ==========

class PortfolioManager:
    """Manage investment portfolio and fetch real-time data"""
    
    def __init__(self, positions: List[Position]):
        self.positions = positions
    
    def update_prices(self):
        """Update current prices from yfinance"""
        tickers = list(set(p.ticker for p in self.positions))
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                # Update all positions with this ticker
                for position in self.positions:
                    if position.ticker == ticker:
                        position.current_price = current_price
            except Exception as e:
                print(f"Warning: Could not update price for {ticker}: {e}")
    
    def get_taxable_positions(self) -> List[Position]:
        """Get positions in taxable accounts only"""
        return [p for p in self.positions if p.account_type == AccountType.TAXABLE]
    
    def get_unrealized_losses(self) -> List[Position]:
        """Get positions with unrealized losses"""
        return [p for p in self.get_taxable_positions() 
                if p.get_unrealized_gain_loss() < 0]
    
    def get_unrealized_gains(self) -> List[Position]:
        """Get positions with unrealized gains"""
        return [p for p in self.get_taxable_positions() 
                if p.get_unrealized_gain_loss() > 0]
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio statistics"""
        taxable_positions = self.get_taxable_positions()
        
        total_value = sum(p.get_market_value() for p in taxable_positions)
        total_cost = sum(p.get_cost_basis() for p in taxable_positions)
        unrealized_gain_loss = total_value - total_cost
        
        long_term_gains = sum(p.get_unrealized_gain_loss() for p in taxable_positions 
                             if p.is_long_term() and p.get_unrealized_gain_loss() > 0)
        short_term_gains = sum(p.get_unrealized_gain_loss() for p in taxable_positions 
                              if not p.is_long_term() and p.get_unrealized_gain_loss() > 0)
        
        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "unrealized_gain_loss": unrealized_gain_loss,
            "long_term_gains": long_term_gains,
            "short_term_gains": short_term_gains,
            "num_positions": len(taxable_positions)
        }

# ========== TAX OPTIMIZATION ENGINE ==========

class TaxOptimizationEngine:
    """Main engine for tax optimization strategies"""
    
    def __init__(self, portfolio: PortfolioManager, tax_profile: TaxProfile):
        self.portfolio = portfolio
        self.tax_profile = tax_profile
        self.optimizations: List[TaxOptimization] = []
        self.tax_calc = FederalTaxCalculator()
    
    def analyze_tax_loss_harvesting(self) -> List[TaxOptimization]:
        """Identify tax loss harvesting opportunities"""
        optimizations = []
        
        loss_positions = self.portfolio.get_unrealized_losses()
        
        if not loss_positions:
            return optimizations
        
        # Sort by largest losses first
        loss_positions.sort(key=lambda p: p.get_unrealized_gain_loss())
        
        total_harvestable_losses = 0
        positions_to_harvest = []
        
        for position in loss_positions:
            # Avoid wash sale (need to check if bought similar security in last 30 days)
            # For now, we'll assume no wash sale issues
            
            loss_amount = abs(position.get_unrealized_gain_loss())
            total_harvestable_losses += loss_amount
            positions_to_harvest.append(position)
            
            # Cap at $3,000 ordinary income offset per year
            if total_harvestable_losses >= 3000:
                break
        
        if positions_to_harvest:
            # Calculate tax savings
            # Losses offset gains first, then up to $3k of ordinary income
            current_tax = self.tax_calc.calculate_total_tax(self.tax_profile)
            
            # Create scenario with losses harvested
            test_profile = self.tax_profile.copy()
            test_profile.long_term_gains -= total_harvestable_losses
            test_profile.carryforward_losses += max(0, total_harvestable_losses - 
                                                   (self.tax_profile.long_term_gains + 
                                                    self.tax_profile.short_term_gains + 3000))
            
            new_tax = self.tax_calc.calculate_total_tax(test_profile)
            tax_savings = current_tax['total_federal_tax'] - new_tax['total_federal_tax']
            
            # Find replacement securities (avoid wash sale)
            replacement_tickers = self._find_replacement_securities(positions_to_harvest)
            
            optimization = TaxOptimization(
                strategy=TaxStrategy.TAX_LOSS_HARVEST,
                description=f"Harvest ${total_harvestable_losses:,.0f} in losses from {len(positions_to_harvest)} positions",
                estimated_tax_savings=tax_savings,
                action_items=[
                    f"Sell {p.ticker}: {p.quantity:.2f} shares at ${p.current_price:.2f} (Loss: ${abs(p.get_unrealized_gain_loss()):,.0f})"
                    for p in positions_to_harvest
                ] + [
                    f"Wait 31 days or immediately buy: {', '.join(replacement_tickers[:3])}"
                ],
                priority=1,
                deadline=datetime(datetime.now().year, 12, 31),
                positions_to_sell=positions_to_harvest,
                positions_to_buy=replacement_tickers,
                amount=total_harvestable_losses
            )
            
            optimizations.append(optimization)
        
        return optimizations
    
    def analyze_roth_conversion(self) -> List[TaxOptimization]:
        """Analyze optimal Roth conversion amount"""
        optimizations = []
        
        # Only beneficial if you expect higher tax rates in retirement
        # or if in a low-income year
        
        current_tax = self.tax_calc.calculate_total_tax(self.tax_profile)
        current_bracket_rate = self._get_marginal_rate(self.tax_profile)
        
        # Find optimal conversion amount that fills current bracket
        next_bracket_threshold = self._get_next_bracket_threshold(self.tax_profile)
        room_in_bracket = next_bracket_threshold - current_tax['taxable_income']
        
        if room_in_bracket > 5000 and current_bracket_rate <= 0.22:
            # Worth converting up to the next bracket
            conversion_amount = min(room_in_bracket, 50000)  # Cap at $50k
            
            # Tax on conversion
            conversion_tax = conversion_amount * current_bracket_rate
            
            # Future tax savings (assume 5% higher rate in retirement)
            future_rate = min(current_bracket_rate + 0.05, 0.37)
            future_tax_avoided = conversion_amount * future_rate
            
            # Account for growth (assume 7% annual return over 20 years)
            years_to_retirement = max(1, 65 - self.tax_profile.age)
            growth_multiplier = (1.07 ** years_to_retirement)
            future_value = conversion_amount * growth_multiplier
            future_tax_savings = future_value * future_rate - conversion_tax
            
            optimization = TaxOptimization(
                strategy=TaxStrategy.ROTH_CONVERSION,
                description=f"Convert ${conversion_amount:,.0f} from Traditional IRA to Roth IRA",
                estimated_tax_savings=future_tax_savings,
                action_items=[
                    f"Convert ${conversion_amount:,.0f} before year-end",
                    f"Pay ${conversion_tax:,.0f} in taxes this year",
                    f"Potential future tax-free growth: ${future_value - conversion_amount:,.0f}",
                    "Consider spreading conversion over multiple low-income years"
                ],
                priority=2,
                deadline=datetime(datetime.now().year, 12, 31),
                amount=conversion_amount
            )
            
            optimizations.append(optimization)
        
        return optimizations
    
    def analyze_charitable_giving(self) -> List[TaxOptimization]:
        """Optimize charitable donation timing and asset selection"""
        optimizations = []
        
        # Find highly appreciated long-term positions
        gain_positions = [p for p in self.portfolio.get_unrealized_gains() 
                         if p.is_long_term()]
        
        if not gain_positions or self.tax_profile.itemized_deductions < self.tax_profile.standard_deduction:
            return optimizations
        
        # Sort by highest gains
        gain_positions.sort(key=lambda p: p.get_unrealized_gain_loss(), reverse=True)
        
        best_position = gain_positions[0]
        donation_value = min(best_position.get_market_value(), 10000)  # Example: $10k donation
        
        # Tax savings from deduction
        marginal_rate = self._get_marginal_rate(self.tax_profile)
        deduction_savings = donation_value * marginal_rate
        
        # Avoided capital gains tax
        gain_percentage = best_position.get_unrealized_gain_loss() / best_position.get_cost_basis()
        avoided_gain = donation_value * (gain_percentage / (1 + gain_percentage))
        avoided_tax = avoided_gain * 0.15  # Assume 15% LTCG rate
        
        total_savings = deduction_savings + avoided_tax
        
        optimization = TaxOptimization(
            strategy=TaxStrategy.CHARITABLE_DONATION,
            description=f"Donate ${donation_value:,.0f} of {best_position.ticker} to charity",
            estimated_tax_savings=total_savings,
            action_items=[
                f"Donate {best_position.ticker} shares (not cash) to donor-advised fund",
                f"Claim ${donation_value:,.0f} charitable deduction",
                f"Avoid ${avoided_tax:,.0f} in capital gains tax",
                "Get appraisal if donation > $5,000"
            ],
            priority=3,
            positions_to_sell=[best_position],
            amount=donation_value
        )
        
        optimizations.append(optimization)
        
        return optimizations
    
    def analyze_retirement_contributions(self) -> List[TaxOptimization]:
        """Maximize tax-advantaged retirement contributions"""
        optimizations = []
        
        # 2024 contribution limits
        age_50_plus = self.tax_profile.age >= 50
        
        contrib_401k = self.tax_profile.traditional_401k_contributions + \
                       self.tax_profile.roth_401k_contributions
        contrib_ira = self.tax_profile.traditional_ira_contributions + \
                      self.tax_profile.roth_ira_contributions
        
        limit_401k = 23000 + (7500 if age_50_plus else 0)
        limit_ira = 7000 + (1000 if age_50_plus else 0)
        limit_hsa = 4150  # Family: 8300
        
        room_401k = limit_401k - contrib_401k
        room_ira = limit_ira - contrib_ira
        room_hsa = limit_hsa - self.tax_profile.hsa_contributions
        
        total_room = room_401k + room_ira + room_hsa
        
        if total_room > 0:
            marginal_rate = self._get_marginal_rate(self.tax_profile)
            tax_savings = total_room * marginal_rate
            
            action_items = []
            if room_401k > 0:
                action_items.append(f"Contribute ${room_401k:,.0f} more to 401(k)")
            if room_ira > 0:
                action_items.append(f"Contribute ${room_ira:,.0f} more to Traditional IRA")
            if room_hsa > 0:
                action_items.append(f"Contribute ${room_hsa:,.0f} more to HSA (triple tax advantage!)")
            
            optimization = TaxOptimization(
                strategy=TaxStrategy.RETIREMENT_CONTRIBUTION,
                description=f"Max out retirement contributions: ${total_room:,.0f} remaining",
                estimated_tax_savings=tax_savings,
                action_items=action_items + [
                    f"Immediate tax savings: ${tax_savings:,.0f}",
                    "HSA contributions due by tax filing deadline (April 15)"
                ],
                priority=1,
                deadline=datetime(datetime.now().year + 1, 4, 15),
                amount=total_room
            )
            
            optimizations.append(optimization)
        
        return optimizations
    
    def run_complete_analysis(self) -> List[TaxOptimization]:
        """Run all optimization analyses"""
        print("🔍 Analyzing tax optimization opportunities...\n")
        
        # Update portfolio prices
        self.portfolio.update_prices()
        
        # Run all strategies
        self.optimizations.extend(self.analyze_tax_loss_harvesting())
        self.optimizations.extend(self.analyze_roth_conversion())
        self.optimizations.extend(self.analyze_charitable_giving())
        self.optimizations.extend(self.analyze_retirement_contributions())
        
        # Sort by priority
        self.optimizations.sort(key=lambda x: x.priority)
        
        return self.optimizations
    
    # Helper methods
    def _get_marginal_rate(self, profile: TaxProfile) -> float:
        """Get marginal tax rate"""
        brackets = FederalTaxCalculator.BRACKETS_2024.get(
            profile.filing_status, 
            FederalTaxCalculator.BRACKETS_2024[TaxFilingStatus.SINGLE]
        )
        
        current_tax = self.tax_calc.calculate_total_tax(profile)
        taxable = current_tax['taxable_income']
        
        for threshold, rate in brackets:
            if taxable <= threshold:
                return rate
        return 0.37
    
    def _get_next_bracket_threshold(self, profile: TaxProfile) -> float:
        """Get next tax bracket threshold"""
        brackets = FederalTaxCalculator.BRACKETS_2024.get(
            profile.filing_status,
            FederalTaxCalculator.BRACKETS_2024[TaxFilingStatus.SINGLE]
        )
        
        current_tax = self.tax_calc.calculate_total_tax(profile)
        taxable = current_tax['taxable_income']
        
        for threshold, rate in brackets:
            if taxable <= threshold:
                return threshold
        return float('inf')
    
    def _find_replacement_securities(self, positions: List[Position]) -> List[str]:
        """Find similar securities to avoid wash sale"""
        # Simplified: suggest ETFs as replacements
        replacement_map = {
            "AAPL": ["ARKK", "QQQ", "VGT"],
            "MSFT": ["QQQ", "VGT", "IGV"],
            "TSLA": ["ARKK", "DRIV", "IDRV"],
            "AMZN": ["QQQ", "VGT", "XLY"],
        }
        
        replacements = []
        for pos in positions:
            if pos.ticker in replacement_map:
                replacements.extend(replacement_map[pos.ticker])
            else:
                # Generic tech ETF
                replacements.extend(["SPY", "QQQ", "VTI"])
        
        return list(set(replacements))[:5]


# ========== REPORTING AGENT ==========

class TaxReportGenerator:
    """Generate comprehensive tax optimization reports"""
    
    def generate_report(self, engine: TaxOptimizationEngine) -> str:
        """Generate detailed tax optimization report"""
        profile = engine.tax_profile
        portfolio_summary = engine.portfolio.get_portfolio_summary()
        current_tax = engine.tax_calc.calculate_total_tax(profile) if hasattr(engine, 'tax_calc') else FederalTaxCalculator().calculate_total_tax(profile)
        optimizations = engine.optimizations
        
        total_potential_savings = sum(opt.estimated_tax_savings for opt in optimizations)
        
        report = f"""
# TAX OPTIMIZATION REPORT
**Generated**: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

## TAXPAYER PROFILE

| Attribute | Value |
|-----------|-------|
| Filing Status | {profile.filing_status.value.replace('_', ' ').title()} |
| Annual Income | ${profile.annual_income:,.0f} |
| Age | {profile.age} |
| State | {profile.state} |
| Current Tax Year | {datetime.now().year} |

---

## CURRENT TAX SITUATION

### Income & Deductions
- **Adjusted Gross Income (AGI)**: ${current_tax['agi']:,.0f}
- **Standard Deduction**: ${profile.standard_deduction:,.0f}
- **Taxable Income**: ${current_tax['taxable_income']:,.0f}

### Tax Liability Breakdown
- **Ordinary Income Tax**: ${current_tax['ordinary_tax']:,.0f}
- **Short-Term Capital Gains Tax**: ${current_tax['stcg_tax']:,.0f}
- **Long-Term Capital Gains Tax**: ${current_tax['ltcg_tax']:,.0f}
- **Net Investment Income Tax**: ${current_tax['niit']:,.0f}
- **Total Federal Tax**: ${current_tax['total_federal_tax']:,.0f}
- **Effective Tax Rate**: {current_tax['effective_rate']:.2%}

---

## PORTFOLIO SUMMARY

| Metric | Value |
|--------|-------|
| Total Portfolio Value | ${portfolio_summary['total_value']:,.0f} |
| Total Cost Basis | ${portfolio_summary['total_cost']:,.0f} |
| Unrealized Gain/Loss | ${portfolio_summary['unrealized_gain_loss']:,.0f} |
| Long-Term Gains | ${portfolio_summary['long_term_gains']:,.0f} |
| Short-Term Gains | ${portfolio_summary['short_term_gains']:,.0f} |
| Number of Positions | {portfolio_summary['num_positions']} |

---

## OPTIMIZATION OPPORTUNITIES

**Total Potential Tax Savings**: ${total_potential_savings:,.0f}

"""
        
        # Add each optimization
        for i, opt in enumerate(optimizations, 1):
            priority_emoji = "🔥" if opt.priority == 1 else "⭐" if opt.priority == 2 else "💡"
            deadline_str = f"**Deadline**: {opt.deadline.strftime('%B %d, %Y')}" if opt.deadline else ""
            
            report += f"""
### {priority_emoji} Opportunity #{i}: {opt.strategy.value.replace('_', ' ').title()}

**Description**: {opt.description}  
**Estimated Tax Savings**: ${opt.estimated_tax_savings:,.0f}  
**Priority**: {opt.priority}/5  
{deadline_str}

**Action Items**:
"""
            for action in opt.action_items:
                report += f"- {action}\n"
            
            report += "\n"
        
        # Add year-end checklist
        report += """
---

## YEAR-END TAX CHECKLIST

### Before December 31st
- [ ] Execute all tax loss harvesting transactions
- [ ] Complete Roth conversions
- [ ] Make charitable donations
- [ ] Max out 401(k) contributions
- [ ] Review estimated tax payments

### Before April 15th (Next Year)
- [ ] Make IRA contributions for current tax year
- [ ] Make HSA contributions for current tax year
- [ ] File tax return or extension
- [ ] Pay any remaining tax liability

---

## IMPORTANT DISCLAIMERS

⚠️ **This report is for informational purposes only and does not constitute tax advice.**

- Consult with a licensed tax professional before implementing any strategies
- Tax laws are subject to change and vary by state
- Wash sale rules apply to tax loss harvesting (30-day window)
- Individual circumstances may affect strategy suitability
- Past performance does not guarantee future results

---

## NEXT STEPS

1. **Review** all recommended optimizations with your tax advisor
2. **Prioritize** high-priority items with near-term deadlines
3. **Execute** approved strategies through your brokerage
4. **Track** all transactions for tax reporting
5. **Monitor** quarterly to identify new opportunities

---

*Generated by AI-Powered Tax Optimization Engine*  
*For questions or support, consult your financial advisor*
"""
        
        return report



def create_sample_portfolio() -> List[Position]:
    """Create sample portfolio for demonstration"""
    return [
        # Taxable account positions
        Position(
            ticker="AAPL",
            quantity=100,
            purchase_price=150.0,
            purchase_date=datetime(2022, 1, 15),
            current_price=180.0,  # Will be updated
            account_type=AccountType.TAXABLE,
            asset_class=AssetClass.EQUITY
        ),
        Position(
            ticker="TSLA",
            quantity=50,
            purchase_price=250.0,
            purchase_date=datetime(2023, 6, 1),
            current_price=200.0,  # Loss position
            account_type=AccountType.TAXABLE,
            asset_class=AssetClass.EQUITY
        ),
        Position(
            ticker="MSFT",
            quantity=75,
            purchase_price=300.0,
            purchase_date=datetime(2021, 3, 10),
            current_price=380.0,
            account_type=AccountType.TAXABLE,
            asset_class=AssetClass.EQUITY
        ),
        Position(
            ticker="NVDA",
            quantity=40,
            purchase_price=450.0,
            purchase_date=datetime(2023, 11, 20),
            current_price=400.0,  # Loss position
            account_type=AccountType.TAXABLE,
            asset_class=AssetClass.EQUITY
        ),
        Position(
            ticker="AMZN",
            quantity=60,
            purchase_price=120.0,
            purchase_date=datetime(2022, 8, 5),
            current_price=150.0,
            account_type=AccountType.TAXABLE,
            asset_class=AssetClass.EQUITY
        ),
        # Retirement accounts (not subject to immediate taxation)
        Position(
            ticker="VOO",
            quantity=200,
            purchase_price=350.0,
            purchase_date=datetime(2020, 1, 1),
            current_price=450.0,
            account_type=AccountType.TRADITIONAL_401K,
            asset_class=AssetClass.EQUITY
        ),
    ]


def create_sample_tax_profile() -> TaxProfile:
    """Create sample tax profile"""
    return TaxProfile(
        filing_status=TaxFilingStatus.MARRIED_JOINT,
        annual_income=180000,
        state="CA",
        age=42,
        traditional_401k_contributions=15000,
        roth_401k_contributions=0,
        traditional_ira_contributions=0,
        roth_ira_contributions=0,
        hsa_contributions=2000,
        short_term_gains=5000,
        long_term_gains=12000,
        carryforward_losses=0,
        itemized_deductions=18000
    )


def main():
    """Main execution function"""
    print("=" * 70)
    print("AI-POWERED TAX OPTIMIZATION ENGINE")
    print("=" * 70)
    print()
    
    # Create sample data
    print("📊 Setting up portfolio and tax profile...\n")
    positions = create_sample_portfolio()
    portfolio = PortfolioManager(positions)
    tax_profile = create_sample_tax_profile()
    
    # Initialize optimization engine
    engine = TaxOptimizationEngine(portfolio, tax_profile)
    
    # Run complete analysis
    optimizations = engine.run_complete_analysis()
    
    # Display summary
    print("=" * 70)
    print(f"✅ Analysis Complete! Found {len(optimizations)} optimization opportunities")
    print("=" * 70)
    print()
    
    # Quick summary
    total_savings = sum(opt.estimated_tax_savings for opt in optimizations)
    print(f"💰 Total Potential Tax Savings: ${total_savings:,.0f}\n")
    
    for i, opt in enumerate(optimizations, 1):
        priority_emoji = "🔥" if opt.priority == 1 else "⭐" if opt.priority == 2 else "💡"
        print(f"{priority_emoji} {i}. {opt.description}")
        print(f"   Savings: ${opt.estimated_tax_savings:,.0f}")
        print()
    
    # Generate full report
    print("=" * 70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 70)
    print()
    
    report_generator = TaxReportGenerator()
    full_report = report_generator.generate_report(engine)
    
    # Print report
    print(full_report)
    
    
    # Optional: Generate AI-powered insights
    print("\n" + "=" * 70)
    print("AI-POWERED STRATEGIC INSIGHTS")
    print("=" * 70)
    print()
    
    insights = generate_ai_insights(engine, optimizations)
    print(insights)


def generate_ai_insights(engine: TaxOptimizationEngine, 
                         optimizations: List[TaxOptimization]) -> str:
    """Generate AI-powered strategic tax insights"""
    
    portfolio_summary = engine.portfolio.get_portfolio_summary()
    current_tax = engine.tax_calc.calculate_total_tax(engine.tax_profile) if hasattr(engine, 'tax_calc') else FederalTaxCalculator().calculate_total_tax(engine.tax_profile)
    
    prompt = f"""
You are a senior tax strategist and CPA. Analyze this client's tax situation and provide strategic insights.

## CLIENT PROFILE
- Filing Status: {engine.tax_profile.filing_status.value}
- Annual Income: ${engine.tax_profile.annual_income:,.0f}
- Age: {engine.tax_profile.age}
- Current Tax Liability: ${current_tax['total_federal_tax']:,.0f}
- Effective Tax Rate: {current_tax['effective_rate']:.2%}

## PORTFOLIO
- Total Value: ${portfolio_summary['total_value']:,.0f}
- Unrealized Gains: ${portfolio_summary['unrealized_gain_loss']:,.0f}

## IDENTIFIED OPPORTUNITIES
{chr(10).join(f"- {opt.strategy.value}: ${opt.estimated_tax_savings:,.0f} savings" for opt in optimizations)}

## YOUR TASK
Provide a concise strategic analysis covering:

1. **Immediate Priority Actions** (next 30 days)
2. **Year-End Strategy** (before Dec 31)
3. **Long-Term Tax Planning** (multi-year outlook)
4. **Risk Considerations** (what could go wrong)
5. **One Key Insight** (the most important thing they should know)

Keep it practical, actionable, and under 400 words. Use markdown formatting.
"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a CPA and tax strategist with 20+ years of experience in tax optimization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI insights unavailable: {str(e)}"


# ========== INTERACTIVE MODE ==========

def interactive_mode():
    """Interactive command-line interface for tax optimization"""
    print("\n" + "=" * 70)
    print("INTERACTIVE TAX OPTIMIZATION MODE")
    print("=" * 70)
    print()
    
    # Collect user input
    print("Let's build your tax profile...\n")
    
    # Filing status
    print("Filing Status:")
    print("1. Single")
    print("2. Married Filing Jointly")
    print("3. Married Filing Separately")
    print("4. Head of Household")
    
    status_choice = input("\nEnter choice (1-4): ").strip()
    status_map = {
        "1": TaxFilingStatus.SINGLE,
        "2": TaxFilingStatus.MARRIED_JOINT,
        "3": TaxFilingStatus.MARRIED_SEPARATE,
        "4": TaxFilingStatus.HEAD_OF_HOUSEHOLD
    }
    filing_status = status_map.get(status_choice, TaxFilingStatus.SINGLE)
    
    # Income
    annual_income = float(input("Annual Income ($): ").strip().replace(",", ""))
    age = int(input("Age: ").strip())
    state = input("State (2-letter code, e.g., CA): ").strip().upper()
    
    # Contributions
    print("\nRetirement Contributions (current year):")
    contrib_401k = float(input("401(k) Contributions ($): ").strip().replace(",", "") or "0")
    contrib_ira = float(input("IRA Contributions ($): ").strip().replace(",", "") or "0")
    contrib_hsa = float(input("HSA Contributions ($): ").strip().replace(",", "") or "0")
    
    # Capital gains
    print("\nCapital Gains/Losses (realized this year):")
    st_gains = float(input("Short-Term Gains ($): ").strip().replace(",", "") or "0")
    lt_gains = float(input("Long-Term Gains ($): ").strip().replace(",", "") or "0")
    
    # Create profile
    tax_profile = TaxProfile(
        filing_status=filing_status,
        annual_income=annual_income,
        state=state,
        age=age,
        traditional_401k_contributions=contrib_401k,
        traditional_ira_contributions=contrib_ira,
        hsa_contributions=contrib_hsa,
        short_term_gains=st_gains,
        long_term_gains=lt_gains
    )
    
    # For demo purposes, use sample portfolio
    # In production, would connect to real brokerage via API
    print("\n⚠️  Using sample portfolio for demonstration")
    print("In production, connect to your brokerage account via API\n")
    
    positions = create_sample_portfolio()
    portfolio = PortfolioManager(positions)
    
    # Run analysis
    engine = TaxOptimizationEngine(portfolio, tax_profile)
    optimizations = engine.run_complete_analysis()
    
    # Generate and display report
    report_generator = TaxReportGenerator()
    report = report_generator.generate_report(engine)
    
    print("\n" + "=" * 70)
    print(report)
    return report
    

interactive_mode()
# Income Documents
# Portfolio Brokerage statements
# Deduction & Expense Documentation
# Retirement & Contribution Records