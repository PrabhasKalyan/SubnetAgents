import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import requests
from collections import deque
import time

from river import (
    compose,
    preprocessing,
    metrics,
    stats,
    forest
)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

@dataclass
class OrderBookSnapshot:
    timestamp: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    
@dataclass
class Trade:
    timestamp: float
    price: float
    size: float
    side: str
    
@dataclass
class MarketData:
    symbol: str
    timestamp: float
    mark_price: float
    funding_rate: float
    open_interest: float
    volume_24h: float
    order_book: Optional[OrderBookSnapshot] = None
    recent_trades: List[Trade] = None

@dataclass
class Position:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    leverage: float
    unrealized_pnl: float
    margin_ratio: float
    liquidation_price: float

@dataclass
class TradeSignal:
    timestamp: float
    symbol: str
    signal: SignalStrength
    side: OrderSide
    suggested_entry: float
    suggested_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    reasoning: str
    risk_reward_ratio: float
    model_metrics: Dict

BASE_URL = "https://api.hyperliquid.xyz"

class MarketDataHandler:
    """
    Handles real-time and historical market data.
    FIXED: Uses correct POST endpoints and payload structures for Hyperliquid.
    """
    def __init__(self, symbols):
        self.symbols = symbols
        self.historical_data = {}
        self.current_data = {}

    def _get_pure_symbol(self, symbol: str) -> str:
        """
        Helper: Converts 'BTC-USD' to 'BTC' as required by Hyperliquid.
        """
        if "-" in symbol:
            return symbol.split("-")[0]
        return symbol

    def fetch_realtime_data(self, symbol: str) -> MarketData:
        """
        Fetch real-time market data (Mark price, funding, OI)
        Endpoint: POST /info, type: metaAndAssetCtxs
        """
        pure_symbol = self._get_pure_symbol(symbol)
        url = f"{BASE_URL}/info"
        
        try:
            # Hyperliquid requires a POST request with a JSON body
            payload = {"type": "metaAndAssetCtxs"}
            r = requests.post(url, json=payload, timeout=5)
            data = r.json()
            
            # The API returns [universe_metadata, asset_contexts]
            universe = data[0]['universe']
            asset_ctxs = data[1]
            
            # Find the specific coin index
            coin_index = next((i for i, coin in enumerate(universe) if coin['name'] == pure_symbol), None)
            
            if coin_index is not None and coin_index < len(asset_ctxs):
                ctx = asset_ctxs[coin_index]
                
                # Parse fields (API returns strings for precision)
                mark_px = float(ctx['markPx'])
                funding = float(ctx['funding']) 
                oi = float(ctx['openInterest'])
                # 'dayNtlVlm' is 24h notional volume
                vol24 = float(ctx.get('dayNtlVlm', 0))
            else:
                print(f"[Hyperliquid] Symbol {pure_symbol} not found in universe data.")
                mark_px, funding, oi, vol24 = 0, 0, 0, 0

        except Exception as e:
            print(f"[Hyperliquid] Error fetching realtime data for {symbol}: {e}")
            # Fallback values
            mark_px, funding, oi, vol24 = 0, 0, 0, 0

        data = MarketData(
            symbol=symbol,
            timestamp=datetime.now().timestamp(),
            mark_price=mark_px,
            funding_rate=funding,
            open_interest=oi,
            volume_24h=vol24,
        )

        self.current_data[symbol] = data
        return data


    def fetch_historical_data(self, symbol: str, start_date: datetime,
                              end_date: datetime, interval: str = '1h') -> pd.DataFrame:
        """
        Fetch historical Candles
        Endpoint: POST /info, type: candleSnapshot
        """
        pure_symbol = self._get_pure_symbol(symbol)
        
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        url = f"{BASE_URL}/info"
        
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": pure_symbol,
                "interval": interval,
                "startTime": start_ts,
                "endTime": end_ts
            }
        }

        try:
            r = requests.post(url, json=payload, timeout=30)
            result = r.json() # Returns list of dicts
            
            if not result or not isinstance(result, list):
                print(f"[Hyperliquid] No history found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"[Hyperliquid] Error fetching history: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(result)
        
        df = df.rename(columns={
            "t": "timestamp", 
            "o": "open", 
            "h": "high", 
            "l": "low", 
            "c": "close", 
            "v": "volume"
        })
        
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)
        
        # Convert MS timestamp to Seconds
        df["timestamp"] = df["timestamp"] / 1000.0
        
        # Sort and clean
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Funding rate not in candles, set to 0.0 for backtest context
        df["funding_rate"] = 0.0

        self.historical_data[symbol] = df
        return df


    def update_orderbook(self, symbol: str, snapshot: OrderBookSnapshot = None):
        """
        Fetch L2 Orderbook
        Endpoint: POST /info, type: l2Book
        """
        pure_symbol = self._get_pure_symbol(symbol)
        url = f"{BASE_URL}/info"
        
        try:
            payload = {
                "type": "l2Book",
                "coin": pure_symbol
            }
            r = requests.post(url, json=payload, timeout=5)
            result = r.json()
            
            levels = result.get('levels', [[], []])
            
            if len(levels) >= 2:
                raw_bids = levels[0]
                raw_asks = levels[1]

                bids = [(float(x['px']), float(x['sz'])) for x in raw_bids]
                asks = [(float(x['px']), float(x['sz'])) for x in raw_asks]

                snapshot = OrderBookSnapshot(
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now().timestamp()
                )
                
                if symbol in self.current_data:
                    self.current_data[symbol].order_book = snapshot

        except Exception as e:
            print(f"[Hyperliquid] Error fetching orderbook for {symbol}: {e}")


class OnlineFeatureEngine:
    """Generate features for streaming data using River"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
        self.price_windows: Dict[str, deque] = {}
        self.volume_windows: Dict[str, deque] = {}
        self.funding_windows: Dict[str, deque] = {}
        
        self.stats_trackers: Dict[str, Dict] = {}
        
    def _init_symbol(self, symbol: str):
        """Initialize tracking for a new symbol"""
        if symbol not in self.price_windows:
            self.price_windows[symbol] = deque(maxlen=self.window_size)
            self.volume_windows[symbol] = deque(maxlen=self.window_size)
            self.funding_windows[symbol] = deque(maxlen=self.window_size)
            
            self.stats_trackers[symbol] = {
                'price_mean': stats.Mean(),
                'price_var': stats.Var(),
                'volume_mean': stats.Mean(),
                'volume_var': stats.Var(),
                'funding_mean': stats.Mean(),
                'funding_var': stats.Var(),
                'price_ewm': stats.EWMean(),
                'price_ewm_fast': stats.EWMean(),
            }
    
    def update_and_extract_features(self, symbol: str, candle: Dict) -> Dict:
        """
        Update statistics and extract features from new candle
        """
        self._init_symbol(symbol)
        
        close = candle['close']
        volume = candle['volume']
        funding = candle.get('funding_rate', 0)
        
        self.price_windows[symbol].append(close)
        self.volume_windows[symbol].append(volume)
        self.funding_windows[symbol].append(funding)
        
        trackers = self.stats_trackers[symbol]
        trackers['price_mean'].update(close)
        trackers['price_var'].update(close)
        trackers['volume_mean'].update(volume)
        trackers['volume_var'].update(volume)
        trackers['funding_mean'].update(funding)
        trackers['funding_var'].update(funding)
        trackers['price_ewm'].update(close)
        trackers['price_ewm_fast'].update(close)

        features = {}
        
        # Simple Moving Average
        prices = list(self.price_windows[symbol])
        if len(prices) >= 20:
            features['sma_20'] = np.mean(prices[-20:])
            features['price_std_20'] = np.std(prices[-20:])
            features['price_to_sma'] = close / features['sma_20'] if features['sma_20'] != 0 else 1.0
        else:
            features['sma_20'] = close
            features['price_std_20'] = 0
            features['price_to_sma'] = 1.0
            
        features['ema_fast'] = trackers['price_ewm_fast'].get()
        features['ema_slow'] = trackers['price_ewm'].get()
        features['ema_diff'] = features['ema_fast'] - features['ema_slow']
        
        # Momentum
        if len(prices) >= 10:
            prev_5 = prices[-5]
            prev_10 = prices[-10]
            features['momentum_5'] = (close - prev_5) / prev_5 if prev_5 != 0 else 0
            features['momentum_10'] = (close - prev_10) / prev_10 if prev_10 != 0 else 0
        else:
            features['momentum_5'] = 0
            features['momentum_10'] = 0
        
        # RSI Approximation
        if len(prices) >= 14:
            changes = np.diff(prices[-14:])
            gains = changes[changes > 0].sum() if len(changes[changes > 0]) > 0 else 0.0001
            losses = -changes[changes < 0].sum() if len(changes[changes < 0]) > 0 else 0.0001
            rs = gains / losses
            features['rsi'] = 100 - (100 / (1 + rs))
        else:
            features['rsi'] = 50
        
        # Volatility
        var = trackers['price_var'].get()
        features['volatility'] = var ** 0.5 if var else 0
        
        # Volume Ratio
        volumes = list(self.volume_windows[symbol])
        if len(volumes) >= 20:
            avg_vol = np.mean(volumes[-20:])
            features['volume_ratio'] = volume / avg_vol if avg_vol > 0 else 1
        else:
            features['volume_ratio'] = 1.0
        
        # Funding
        fundings = list(self.funding_windows[symbol])
        if len(fundings) >= 24:
            features['funding_ma'] = np.mean(fundings[-24:])
            features['funding_std'] = np.std(fundings[-24:])
        else:
            features['funding_ma'] = funding
            features['funding_std'] = 0
        
        # Bollinger Bands position
        if features['price_std_20'] > 0:
            features['bb_upper'] = features['sma_20'] + (features['price_std_20'] * 2)
            features['bb_lower'] = features['sma_20'] - (features['price_std_20'] * 2)
            denom = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (close - features['bb_lower']) / denom if denom != 0 else 0.5
        else:
            features['bb_upper'] = close
            features['bb_lower'] = close
            features['bb_position'] = 0.5
        
        features['price_trend'] = 1 if features['ema_fast'] > features['ema_slow'] else 0
        
        return features

class OnlineTradingModel:
    """Online machine learning models using River for streaming data"""
    
    def __init__(self):
        # Adaptive Random Forest Classifier
        self.classifier = compose.Pipeline(
            preprocessing.StandardScaler(),
            forest.ARFClassifier(
                n_models=10,
                max_depth=10,
                seed=42
            )
        )
        
        # Adaptive Random Forest Regressor
        self.regressor = compose.Pipeline(
            preprocessing.StandardScaler(),
            forest.ARFRegressor(
                n_models=10,
                max_depth=8,
                seed=42
            )
        )
        
        self.classification_metric = metrics.Accuracy()
        self.regression_metric = metrics.MAE()
        
        self.n_samples = 0
        self.recent_accuracy = deque(maxlen=100)
        self.recent_mae = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
        
    def learn_one(self, features: Dict, target_direction: int, target_return: float):
        """
        Online learning: update model with one sample
        """
        if self.n_samples > 0:
            # Check performance before update (prequential evaluation)
            y_pred_class = self.classifier.predict_one(features)
            y_pred_reg = self.regressor.predict_one(features)
            
            self.classification_metric.update(target_direction, y_pred_class)
            self.regression_metric.update(target_return, y_pred_reg)
            
            self.recent_accuracy.append(1 if y_pred_class == target_direction else 0)
            self.recent_mae.append(abs(target_return - y_pred_reg))
        
        self.classifier.learn_one(features, target_direction)
        self.regressor.learn_one(features, target_return)
        
        self.n_samples += 1
        
        if self.n_samples % 500 == 0:
            self.logger.info(f"Model updated: {self.n_samples} samples learned")
            self.logger.info(f"Accuracy: {self.classification_metric.get():.4f}")
    
    def predict_proba_one(self, features: Dict) -> Dict[int, float]:
        """Get probability predictions for classification"""
        try:
            proba = self.classifier.predict_proba_one(features)
            # Ensure keys exist
            if 0 not in proba: proba[0] = 0.0
            if 1 not in proba: proba[1] = 0.0
            return proba
        except Exception:
            return {0: 0.5, 1: 0.5}
    
    def predict_one(self, features: Dict) -> Tuple[int, float, float]:
        """
        Make prediction for one sample
        Returns: (direction, confidence, expected_return)
        """
        proba = self.predict_proba_one(features)
        
        # Get direction with highest probability
        direction = max(proba.items(), key=lambda x: x[1])[0]
        confidence = proba[direction]
        
        try:
            expected_return = self.regressor.predict_one(features)
        except:
            expected_return = 0.0
        
        return direction, confidence, expected_return
    
    def get_metrics(self) -> Dict:
        recent_acc = np.mean(self.recent_accuracy) if self.recent_accuracy else 0.5
        recent_mae_val = np.mean(self.recent_mae) if self.recent_mae else 0.0
        
        return {
            'samples_learned': self.n_samples,
            'overall_accuracy': self.classification_metric.get(),
            'recent_accuracy': recent_acc,
            'overall_mae': self.regression_metric.get(),
            'recent_mae': recent_mae_val
        }
    
    def save(self, filepath: str):
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'n_samples': self.n_samples,
            'classification_metric': self.classification_metric,
            'regression_metric': self.regression_metric
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.classifier = model_data['classifier']
        self.regressor = model_data['regressor']
        self.n_samples = model_data['n_samples']
        self.classification_metric = model_data['classification_metric']
        self.regression_metric = model_data['regression_metric']


class RiskManager:
    """Manages risk parameters and position sizing"""
    
    def __init__(self, max_position_size: float = 1000,
                 max_leverage: float = 10.0,
                 max_portfolio_risk: float = 0.02):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_portfolio_risk = max_portfolio_risk 
        self.positions: Dict[str, Position] = {}
        
    def calculate_position_size(self, account_balance: float, 
                                entry_price: float,
                                stop_loss_price: float,
                                confidence: float) -> float:
        """Calculate optimal position size based on Risk (Kelly-ish)"""

        risk_per_unit = abs(entry_price - stop_loss_price) / entry_price
        if risk_per_unit == 0:
            return 0
        
        # Amount to risk in dollars
        risk_amt = account_balance * self.max_portfolio_risk
        
        # Position size in quote currency (e.g. USD)
        max_size_by_risk = risk_amt / risk_per_unit
        
        # Scale by model confidence
        suggested_size = max_size_by_risk * confidence
        
        # Cap at max size
        suggested_size = min(suggested_size, self.max_position_size)
        
        # Convert to units of asset (e.g. BTC)
        size_in_units = suggested_size / entry_price
        
        return size_in_units
    
    def calculate_stop_loss(self, entry_price: float, side: OrderSide, 
                           volatility: float) -> float:
        """Calculate stop loss based on volatility"""
        # Ensure min stop distance
        stop_distance = entry_price * max(volatility * 2, 0.005)
        
        if side == OrderSide.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                             side: OrderSide, risk_reward_ratio: float = 2.5) -> float:
        """Calculate take profit based on risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if side == OrderSide.BUY:
            return entry_price + reward
        else:
            return entry_price - reward


class OnlineStrategyEngine:
    """Strategy engine using online learning"""
    
    def __init__(self, model: OnlineTradingModel, risk_manager: RiskManager):
        self.model = model
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(self, symbol: str, current_price: float,
                       features: Dict, account_balance: float) -> Optional[TradeSignal]:
        """Generate trading signal from online ML model"""
        
        direction, confidence, expected_return = self.model.predict_one(features)
        
        # Filter low confidence signals
        if confidence < 0.55:
            return None
        
        if direction == 1:
            side = OrderSide.BUY
            signal = SignalStrength.STRONG_BUY if confidence > 0.8 else SignalStrength.BUY
        else:
            side = OrderSide.SELL
            signal = SignalStrength.STRONG_SELL if confidence > 0.8 else SignalStrength.SELL
        
        entry_price = current_price
        volatility = features.get('volatility', 0.02)
        
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, side, volatility)
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price, stop_loss, side, risk_reward_ratio=2.5
        )
        
        position_size = self.risk_manager.calculate_position_size(
            account_balance, entry_price, stop_loss, confidence
        )
        
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        reasoning = self._generate_reasoning(features, direction, confidence, expected_return)
        model_metrics = self.model.get_metrics()
        
        return TradeSignal(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            signal=signal,
            side=side,
            suggested_entry=entry_price,
            suggested_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning,
            risk_reward_ratio=rr_ratio,
            model_metrics=model_metrics
        )
    
    def update_model_with_outcome(self, symbol: str, features: Dict, 
                                  actual_return: float):
        """Update model with actual outcome for continuous learning"""
        target_direction = 1 if actual_return > 0 else 0
        self.model.learn_one(features, target_direction, actual_return)
    
    def _generate_reasoning(self, features: Dict, direction: int,
                           confidence: float, expected_return: float) -> str:
        parts = []
        parts.append("Uptrend" if features.get('price_trend', 0) == 1 else "Downtrend")
        
        rsi = features.get('rsi', 50)
        if rsi > 70: parts.append(f"Overbought({rsi:.0f})")
        elif rsi < 30: parts.append(f"Oversold({rsi:.0f})")
        
        dir_str = "BULL" if direction == 1 else "BEAR"
        parts.append(f"ML:{dir_str}({confidence:.1%})")
        
        return " | ".join(parts)


class MonitoringSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_signal(self, signal: TradeSignal):
        self.logger.info(f"\n>>> SIGNAL: {signal.symbol} {signal.side.value.upper()}")
        self.logger.info(f"    Entry: {signal.suggested_entry:.2f}, Size: {signal.suggested_size:.4f}")
        self.logger.info(f"    Reason: {signal.reasoning}")



class HyperliquidOnlineTradingSystem:
    """Main trading system with online learning"""
    
    def __init__(self, symbols: List[str], account_balance: float):
        self.symbols = symbols
        self.account_balance = account_balance
        
        self.market_data_handler = MarketDataHandler(symbols)
        self.feature_engine = OnlineFeatureEngine()
        self.model = OnlineTradingModel()
        self.risk_manager = RiskManager()
        self.strategy_engine = OnlineStrategyEngine(self.model, self.risk_manager)
        self.monitoring = MonitoringSystem()
        
        # Track last features for model updates
        self.last_features: Dict[str, Dict] = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def warm_start(self, symbol: str, start_date: datetime, end_date: datetime):
        """Warm start the online model with historical data"""
        self.logger.info(f"Warm starting model for {symbol}...")
        
        df = self.market_data_handler.fetch_historical_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            self.logger.warning(f"Could not fetch historical data for {symbol}.")
            return self.model.get_metrics()
        
        samples_learned = 0
        
        # Process sequentially
        for i in range(len(df) - 5):
            candle = {
                'close': df.iloc[i]['close'],
                'volume': df.iloc[i]['volume'],
                'funding_rate': df.iloc[i]['funding_rate']
            }
            
            features = self.feature_engine.update_and_extract_features(symbol, candle)
            
            # Future outcome (5 candles ahead)
            future_price = df.iloc[i + 5]['close']
            actual_return = (future_price - candle['close']) / candle['close']
            target_direction = 1 if actual_return > 0 else 0
            
            self.model.learn_one(features, target_direction, actual_return)
            samples_learned += 1
            
        metrics = self.model.get_metrics()
        self.logger.info(f"Warm start complete. Learned {samples_learned} samples.")
        return metrics
    
    def process_new_candle(self, symbol: str, candle: Dict) -> Optional[TradeSignal]:
        """Process new market data candle and generate signal"""
        features = self.feature_engine.update_and_extract_features(symbol, candle)
        self.last_features[symbol] = features.copy()
        
        signal = self.strategy_engine.generate_signal(
            symbol,
            candle['close'],
            features,
            self.account_balance
        )
        
        if signal:
            self.monitoring.log_signal(signal)
        
        return signal
    
    def update_model_with_trade_outcome(self, symbol: str, actual_return: float):
        if symbol in self.last_features:
            self.strategy_engine.update_model_with_outcome(
                symbol,
                self.last_features[symbol],
                actual_return
            )

    def save_model(self, filepath: str):
        self.model.save(filepath)


class OnlineBacktester:
    def __init__(self, system: HyperliquidOnlineTradingSystem):
        self.system = system
        self.logger = logging.getLogger(__name__)
        
    def run_backtest(self, symbol: str, start_date: datetime, 
                    end_date: datetime, initial_balance: float = 10000):
        
        self.logger.info(f"BACKTESTING {symbol} ({start_date.date()} to {end_date.date()})")
        
        df = self.system.market_data_handler.fetch_historical_data(
            symbol, start_date, end_date, interval='1h'
        )
        
        if df.empty:
            self.logger.error("No data for backtest.")
            return {}, [], []

        balance = initial_balance
        trades: List[Dict] = []
        equity_curve = [balance]
        
        # Iterate simulating streaming
        for i in range(50, len(df) - 5):
            candle = {
                'close': df.iloc[i]['close'],
                'volume': df.iloc[i]['volume'],
                'funding_rate': df.iloc[i]['funding_rate']
            }
            
            signal = self.system.process_new_candle(symbol, candle)
            
            if signal and signal.confidence > 0.65:
                entry_price = signal.suggested_entry
                # Use simplified size for backtest
                position_size = (balance * 0.05) / entry_price 
                
                # Check next 5 candles for exit
                future_prices = df.iloc[i+1:i+6]['close'].values
                exit_price = future_prices[-1] # Default exit after time
                
                for price in future_prices:
                    if signal.side == OrderSide.BUY:
                        if price <= signal.stop_loss: 
                            exit_price = signal.stop_loss
                            break
                        elif price >= signal.take_profit:
                            exit_price = signal.take_profit
                            break
                    else:
                        if price >= signal.stop_loss:
                            exit_price = signal.stop_loss
                            break
                        elif price <= signal.take_profit:
                            exit_price = signal.take_profit
                            break
                
                # Calculate PnL
                if signal.side == OrderSide.BUY:
                    pnl = (exit_price - entry_price) * position_size
                else:
                    pnl = (entry_price - exit_price) * position_size
                
                balance += pnl
                trades.append({'pnl': pnl})
                
                # Update model
                actual_ret = (exit_price - entry_price) / entry_price
                if signal.side == OrderSide.SELL: actual_ret = -actual_ret
                self.system.update_model_with_trade_outcome(symbol, actual_ret)
            
            equity_curve.append(balance)
            
        final_return = (balance - initial_balance) / initial_balance
        self.logger.info(f"Final Balance: ${balance:,.2f} | Return: {final_return:.2%} | Trades: {len(trades)}")
        
        return {'return': final_return}, trades, equity_curve

def run_hyperliquid_system_check(symbols: List[str]):
    """
    Initializes and performs warm-start, backtest verification, and live signal 
    check for the Hyperliquid Online Trading System using a list of symbols.
    
    Args:
        symbols: A list of trading symbols (e.g., ['BTC-USD', 'ETH-USD']).
    """
    account_balance = 10000
    print(f"Initializing system for symbols: {symbols}. Account Balance: ${account_balance:,.2f}")
    system = HyperliquidOnlineTradingSystem(symbols, account_balance=account_balance)
    
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for symbol in symbols:
        system.warm_start(symbol, start_date, end_date)
        
    print("\n--- PHASE 2: BACKTEST (Recent Data) ---")
    backtester = OnlineBacktester(system)
    
    bt_start = end_date - timedelta(days=7)
    for symbol in symbols:
        backtester.run_backtest(symbol, bt_start, end_date)
        
    for symbol in symbols:
        md = system.market_data_handler.fetch_realtime_data(symbol)
        
        if md.mark_price > 0:
            print(f"\nProcessing {symbol} @ ${md.mark_price:,.2f}...")
            
            candle = {
                'close': md.mark_price,
                'volume': md.volume_24h,
                'funding_rate': md.funding_rate
            }
            
            signal = system.process_new_candle(symbol, candle)
            
            if not signal:
                print(f"No signal generated for {symbol}. (Model confidence likely too low)")
            else:
                print(f"✅ Signal successfully generated for {symbol}.")
        else:
            print(f"Could not fetch live data for {symbol}")
