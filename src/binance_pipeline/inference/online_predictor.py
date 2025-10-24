import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from collections import deque
from datetime import datetime
import pickle

log = logging.getLogger(__name__)

class StreamingFeatureCalculator:
    """Calculates features incrementally from streaming tick data."""
    def __init__(self, config: Dict):
        self.config = config
        self.price_buffer = deque(maxlen=200)
        self.ewma_states = {}
        self.ewma_spans_ms = {'5s': 5000, '15s': 15000, '1m': 60000, '3m': 180000, '15m': 900000}
        features_to_track = ['mid_price', 'spread_bps', 'microprice', 'taker_flow', 'ofi', 'book_imbalance']
        for feat in features_to_track:
            self.ewma_states[feat] = {span: None for span in self.ewma_spans_ms.keys()}
        log.info("StreamingFeatureCalculator initialized.")

    def _update_ewma(self, feature_name: str, value: float):
        """Updates EWMA state incrementally."""
        for span_name, span_ms in self.ewma_spans_ms.items():
            alpha = 1 - np.exp(-np.log(2) / span_ms)
            old_ewma = self.ewma_states[feature_name][span_name]
            if old_ewma is None:
                self.ewma_states[feature_name][span_name] = value
            else:
                self.ewma_states[feature_name][span_name] = alpha * value + (1 - alpha) * old_ewma

    def process_tick(self, tick: Dict) -> Optional[Dict]:
        """Processes a single tick and returns calculated features."""
        try:
            best_bid, best_ask = tick['best_bid_price'], tick['best_ask_price']
            best_bid_qty, best_ask_qty = tick['best_bid_qty'], tick['best_ask_qty']
            
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000
            microprice = ((best_bid * best_ask_qty) + (best_ask * best_bid_qty)) / (best_bid_qty + best_ask_qty)
            taker_flow = -tick['qty'] if tick['is_buyer_maker'] else tick['qty']
            
            ofi = 0
            if len(self.price_buffer) > 1:
                prev_bid, prev_ask = self.price_buffer[-2]['bid'], self.price_buffer[-2]['ask']
                bid_pressure = best_bid_qty if best_bid >= prev_bid else 0
                ask_pressure = best_ask_qty if best_ask <= prev_ask else 0
                ofi = bid_pressure - ask_pressure
            
            book_imbalance = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty + 1e-10)

            self.price_buffer.append({'bid': best_bid, 'ask': best_ask})
            if len(self.price_buffer) < 100: return None

            self._update_ewma('mid_price', mid_price)
            self._update_ewma('spread_bps', spread_bps)
            # ... update other EWMAs

            features = {'mid_price': mid_price, 'spread_bps': spread_bps, 'microprice': microprice, 'taker_flow': taker_flow, 'ofi': ofi, 'book_imbalance': book_imbalance}
            for feat_name, span_dict in self.ewma_states.items():
                for span_name, ewma_value in span_dict.items():
                    if ewma_value is not None:
                        features[f'{feat_name}_ewma_{span_name}'] = ewma_value
            # ... calculate other features like momentum, etc.
            
            return features
        except Exception as e:
            log.error(f"Error processing tick: {e}")
            return None

class OnlinePredictor:
    """Serves predictions from a trained model with confidence filtering."""
    def __init__(self, model_path: str, config: Dict):
        self.config = config
        self.model = self._load_model(model_path)
        self.feature_calculator = StreamingFeatureCalculator(config)
        self.min_confidence = config.get('min_confidence', 0.60)
        log.info(f"OnlinePredictor initialized with min_confidence={self.min_confidence}")

    def _load_model(self, model_path: str):
        log.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, tick: Dict) -> Optional[Dict]:
        """Generates a prediction from a streaming tick."""
        start_time = datetime.now()
        features = self.feature_calculator.process_tick(tick)
        if features is None: return None

        X = pd.DataFrame([features])
        for col in self.model.feature_name_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.model.feature_name_]
        
        try:
            pred_proba = self.model.predict_proba(X)[0]
            pred_class = np.argmax(pred_proba)
            confidence = pred_proba[pred_class]

            if confidence < self.min_confidence: return None

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            return {'prediction': int(pred_class), 'confidence': float(confidence), 'latency_ms': latency_ms}
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return None
  
