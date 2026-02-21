#!/usr/bin/env python3
"""
Advanced Fraud Detection MCP Server
Sophisticated fraud detection using cutting-edge 2024-2025 algorithms
"""

import os

# Prevent OMP segfaults when PyTorch and sklearn coexist in the same process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import joblib

# FastMCP for high-performance MCP server
from fastmcp import FastMCP

# Core ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from collections import deque

# Graph analysis
import networkx as nx

from config import get_config
from models_validation import TransactionData, PaymentMethod
from feature_engineering import FeatureEngineer
from async_inference import LRUCache

# Monitoring (graceful degradation if deps unavailable)
try:
    from monitoring import MonitoringManager, track_api_call
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    MonitoringManager = None
    track_api_call = None

# Training pipeline (graceful degradation if deps unavailable)
try:
    from training_pipeline import ModelTrainer
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    ModelTrainer = None

# Autoencoder ensemble (graceful degradation if unavailable)
try:
    from models.autoencoder import AutoencoderFraudDetector
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False
    AutoencoderFraudDetector = None

# Explainability module (graceful degradation if unavailable)
try:
    from explainability import FraudExplainer, SHAP_AVAILABLE
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    SHAP_AVAILABLE = False
    FraudExplainer = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Advanced Fraud Detection")


# =============================================================================
# Input Validation Functions
# =============================================================================

def validate_transaction_data(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate transaction data structure and values."""
    if not isinstance(data, dict):
        return False, "transaction_data must be a dictionary"

    # Validate amount if present
    if "amount" in data:
        amount = data["amount"]
        if isinstance(amount, bool):
            return False, "amount must be numeric, not boolean"
        if not isinstance(amount, (int, float)):
            return False, "amount must be numeric"
        if math.isnan(amount) or math.isinf(amount):
            return False, "amount must be a finite number"
        if amount < 0:
            return False, "amount cannot be negative"
        if amount > 1_000_000_000:  # 1 billion limit
            return False, "amount exceeds maximum allowed value"

    # Validate timestamp if present
    if "timestamp" in data:
        ts = data["timestamp"]
        if isinstance(ts, str):
            try:
                datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except ValueError:
                return False, "invalid timestamp format"

    return True, "valid"


def validate_behavioral_data(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate behavioral biometrics data."""
    if not isinstance(data, dict):
        return False, "behavioral_data must be a dictionary"

    # Validate keystroke dynamics
    if "keystroke_dynamics" in data:
        keystroke = data["keystroke_dynamics"]
        if not isinstance(keystroke, list):
            return False, "keystroke_dynamics must be a list"
        for item in keystroke[:100]:  # Check first 100 items
            if not isinstance(item, dict):
                return False, "keystroke_dynamics items must be dictionaries"
            # Validate timing values are reasonable (0-10 seconds)
            for key in ["dwell_time", "flight_time"]:
                if key in item:
                    val = item[key]
                    if not isinstance(val, (int, float)) or val < 0 or val > 10000:
                        return False, f"invalid {key} value"

    # Validate mouse movements
    if "mouse_movements" in data:
        mouse = data["mouse_movements"]
        if not isinstance(mouse, list):
            return False, "mouse_movements must be a list"
        for item in mouse[:1000]:  # Check first 1000 items
            if not isinstance(item, dict):
                return False, "mouse_movements items must be dictionaries"

    return True, "valid"


class BehavioralBiometrics:
    """Behavioral biometrics analysis for fraud detection"""

    def __init__(self):
        self.keystroke_model = None
        self.mouse_model = None
        self.touch_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize behavioral analysis models"""
        # Isolation Forest for keystroke dynamics
        self.keystroke_model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )

        # Fit with dummy training data (10 features: 5 dwell + 5 flight)
        dummy_keystroke_data = np.random.randn(100, 10) * 50 + 100
        self.keystroke_model.fit(dummy_keystroke_data)

        # One-Class SVM for mouse patterns
        self.mouse_model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )

        # Fit with dummy mouse data
        dummy_mouse_data = np.random.randn(100, 5) * 20 + 50
        self.mouse_model.fit(dummy_mouse_data)

        # Local Outlier Factor for touch patterns
        self.touch_model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )

        # Fit with dummy touch data
        dummy_touch_data = np.random.randn(100, 5) * 10 + 30
        self.touch_model.fit(dummy_touch_data)

    def analyze_keystroke_dynamics(self, keystroke_data: List[Dict]) -> Dict[str, Any]:
        """Analyze keystroke dynamics for behavioral anomalies"""
        try:
            if not keystroke_data:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "no_data"}

            if not isinstance(keystroke_data, list):
                return {"risk_score": 0.0, "confidence": 0.0, "status": "error",
                        "error": f"keystroke_data must be a list, got {type(keystroke_data).__name__}"}

            # Extract features from keystroke data
            features = self._extract_keystroke_features(keystroke_data)

            if features is None:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "error",
                        "error": "could not extract valid features from keystroke data"}

            # Predict anomaly
            anomaly_score = self.keystroke_model.decision_function([features])[0]
            is_anomaly = self.keystroke_model.predict([features])[0] == -1

            # Convert to risk score (0-1)
            risk_score = max(0, min(1, (0.5 - anomaly_score) * 2))

            return {
                "risk_score": float(risk_score),
                "is_anomaly": bool(is_anomaly),
                "confidence": 0.85,
                "analysis_type": "keystroke_dynamics",
                "features_analyzed": len(features) if hasattr(features, '__len__') else 1
            }

        except Exception as e:
            logger.error(f"Keystroke analysis error: {e}")
            return {"risk_score": 0.0, "confidence": 0.0, "status": "error", "error": str(e)}

    def _extract_keystroke_features(self, keystroke_data: List[Dict]) -> Optional[List[float]]:
        """Extract numerical features from keystroke timing data"""
        try:
            if len(keystroke_data) < 2:
                return None

            # Calculate dwell times (key press duration)
            dwell_times = []
            # Calculate flight times (time between key releases and next key presses)
            flight_times = []

            for i, keystroke in enumerate(keystroke_data):
                # Dwell time
                if 'press_time' in keystroke and 'release_time' in keystroke:
                    try:
                        dwell = float(keystroke['release_time']) - float(keystroke['press_time'])
                        dwell_times.append(dwell)
                    except (TypeError, ValueError):
                        pass  # Skip non-numeric timing values

                # Flight time
                if i > 0:
                    prev_keystroke = keystroke_data[i-1]
                    if 'release_time' in prev_keystroke and 'press_time' in keystroke:
                        try:
                            flight = float(keystroke['press_time']) - float(prev_keystroke['release_time'])
                            flight_times.append(flight)
                        except (TypeError, ValueError):
                            pass  # Skip non-numeric timing values

            if not dwell_times and not flight_times:
                return [0.0] * 10

            # Statistical features
            features = []

            if dwell_times:
                features.extend([
                    np.mean(dwell_times),
                    np.std(dwell_times),
                    np.median(dwell_times),
                    np.max(dwell_times),
                    np.min(dwell_times)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            if flight_times:
                features.extend([
                    np.mean(flight_times),
                    np.std(flight_times),
                    np.median(flight_times),
                    np.max(flight_times),
                    np.min(flight_times)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

class TransactionAnalyzer:
    """Advanced transaction pattern analysis"""

    DEFAULT_MODEL_DIR = Path("models/saved")

    def __init__(self, model_dir: Optional[Path] = None):
        self._model_source = "none"
        self.autoencoder = None
        self._ensemble_weights = {"isolation_forest": 0.6, "autoencoder": 0.4}
        self._model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.feature_engineer = FeatureEngineer()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200
        )
        # Try to load saved models first; fall back to synthetic training
        if not self.load_models():
            self._initialize_models()

    def _initialize_models(self):
        """Initialize models with synthetic training data"""
        from datetime import timedelta
        rng = np.random.RandomState(42)
        n = 200
        payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'crypto', 'paypal']
        locations = ['United States', 'United Kingdom', 'Canada', 'Germany', 'Japan',
                     'France', 'Australia', 'Brazil', 'India', 'Singapore']
        merchants = ['Amazon', 'Walmart', 'Target', 'BestBuy', 'Costco',
                     'Starbucks', 'McDonalds', 'Apple', 'Google', 'Netflix']

        synthetic_transactions = []
        for i in range(n):
            amount = round(max(0.01, rng.exponential(500)), 2)
            txn = TransactionData(
                transaction_id=f'train-{i:04d}',
                user_id=f'user-{i % 50:03d}',
                amount=amount,
                merchant=merchants[i % len(merchants)],
                location=locations[i % len(locations)],
                timestamp=datetime.now() - timedelta(days=int(rng.randint(1, 364))),
                payment_method=payment_methods[i % len(payment_methods)],
            )
            synthetic_transactions.append(txn)

        # Fit feature engineer and isolation forest on 46-feature space
        feature_matrix, _ = self.feature_engineer.fit_transform(synthetic_transactions)
        self.isolation_forest.fit(feature_matrix)

        # Train autoencoder on same feature matrix
        if AUTOENCODER_AVAILABLE and AutoencoderFraudDetector is not None:
            try:
                self.autoencoder = AutoencoderFraudDetector(
                    contamination=0.1,
                    epochs=20,
                    batch_size=32,
                )
                self.autoencoder.fit(feature_matrix)
                logger.info("Autoencoder trained on synthetic data")
            except Exception as e:
                logger.warning(f"Autoencoder training failed: {e}")
                self.autoencoder = None

        self._model_source = "synthetic"

    def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive transaction fraud analysis with ensemble scoring"""
        try:
            # Extract features
            features = self._extract_transaction_features(transaction_data)

            # Anomaly detection with Isolation Forest
            anomaly_score = self.isolation_forest.decision_function([features])[0]
            is_anomaly = self.isolation_forest.predict([features])[0] == -1

            # Calculate IF risk score (original formula preserved)
            if_risk = max(0, min(1, 0.5 - anomaly_score))

            # Calculate risk factors
            risk_factors = self._identify_risk_factors(transaction_data, features)

            # Ensemble scoring
            model_scores = {
                "isolation_forest": float(if_risk),
            }

            if self.autoencoder is not None:
                try:
                    ae_scores = self.autoencoder.decision_function(np.array([features]))
                    # Normalize autoencoder score to 0-1 range
                    if self.autoencoder.threshold is not None and self.autoencoder.threshold > 0:
                        ae_risk = float(np.clip(ae_scores[0] / (self.autoencoder.threshold * 3), 0, 1))
                    else:
                        ae_risk = 0.0
                    model_scores["autoencoder"] = ae_risk

                    # Weighted ensemble
                    w_if = self._ensemble_weights["isolation_forest"]
                    w_ae = self._ensemble_weights["autoencoder"]
                    base_risk = w_if * if_risk + w_ae * ae_risk
                except Exception as e:
                    logger.warning(f"Autoencoder scoring failed: {e}")
                    base_risk = if_risk
            else:
                base_risk = if_risk

            # Apply risk factor multiplier
            risk_multiplier = 1 + len(risk_factors) * 0.1
            final_risk = min(1.0, base_risk * risk_multiplier)
            model_scores["ensemble"] = float(final_risk)

            return {
                "risk_score": float(final_risk),
                "is_anomaly": bool(is_anomaly),
                "risk_factors": risk_factors,
                "confidence": 0.88,
                "analysis_type": "transaction_pattern",
                "anomaly_score": float(anomaly_score),
                "model_scores": model_scores,
            }

        except Exception as e:
            logger.error(f"Transaction analysis error: {e}")
            return {"risk_score": 0.0, "confidence": 0.0, "status": "error", "error": str(e)}

    def _extract_transaction_features(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """Extract features using FeatureEngineer (46 features)"""
        txn = _dict_to_transaction_data(transaction_data)
        return self.feature_engineer.transform(txn)

    def _identify_risk_factors(self, transaction: Dict[str, Any], features: List[float]) -> List[str]:
        """Identify specific risk factors in the transaction"""
        risk_factors = []

        amount = float(transaction.get('amount', 0))

        # High amount risk
        if amount > 10000:
            risk_factors.append("high_amount_transaction")

        # Unusual time risk
        timestamp = transaction.get('timestamp')
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.hour < 6 or dt.hour > 23:
                risk_factors.append("unusual_time_pattern")

        # Crypto payment risk
        if transaction.get('payment_method') == 'crypto':
            risk_factors.append("high_risk_payment_method")

        # Geographic risk - config-driven exact match
        location = transaction.get('location', '').lower().strip()
        app_config = get_config()
        if location in app_config.HIGH_RISK_LOCATIONS:
            risk_factors.append("high_risk_geographic_location")

        return risk_factors

    def save_models(self, model_dir: Optional[Path] = None) -> Dict[str, str]:
        """Save trained models and feature engineer to disk via joblib.

        Args:
            model_dir: Directory to save models to. Defaults to self._model_dir.

        Returns:
            Dict mapping model name to saved file path.
        """
        save_dir = Path(model_dir) if model_dir else self._model_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        iso_path = save_dir / "isolation_forest.joblib"
        joblib.dump(self.isolation_forest, iso_path)
        paths["isolation_forest"] = str(iso_path)

        fe_path = save_dir / "feature_engineer.joblib"
        joblib.dump(self.feature_engineer, fe_path)
        paths["feature_engineer"] = str(fe_path)

        if self.autoencoder is not None:
            ae_path = save_dir / "autoencoder.pt"
            try:
                self.autoencoder.save(str(ae_path))
                paths["autoencoder"] = str(ae_path)
            except Exception as e:
                logger.warning(f"Failed to save autoencoder: {e}")

        logger.info(f"Models saved to {save_dir}")
        return paths

    def load_models(self, model_dir: Optional[Path] = None) -> bool:
        """Load previously saved models from disk.

        Args:
            model_dir: Directory to load models from. Defaults to self._model_dir.

        Returns:
            True if models were loaded successfully, False otherwise.
        """
        load_dir = Path(model_dir) if model_dir else self._model_dir
        iso_path = load_dir / "isolation_forest.joblib"
        fe_path = load_dir / "feature_engineer.joblib"

        if not iso_path.exists() or not fe_path.exists():
            return False

        try:
            self.isolation_forest = joblib.load(iso_path)
            self.feature_engineer = joblib.load(fe_path)
            self._model_source = "saved"

            # Try to load autoencoder
            if AUTOENCODER_AVAILABLE and AutoencoderFraudDetector is not None:
                ae_path = load_dir / "autoencoder.pt"
                if ae_path.exists():
                    try:
                        self.autoencoder = AutoencoderFraudDetector(contamination=0.1)
                        self.autoencoder.load(str(ae_path))
                        logger.info("Autoencoder loaded from disk")
                    except Exception as e:
                        logger.warning(f"Failed to load autoencoder: {e}")
                        self.autoencoder = None

            logger.info(f"Models loaded from {load_dir}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load saved models: {e}")
            return False


class NetworkAnalyzer:
    """Graph-based network analysis for fraud ring detection"""

    MAX_GRAPH_NODES = 10000

    def __init__(self):
        self.transaction_graph = nx.Graph()
        self._node_order = deque()

    def analyze_network_risk(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network patterns for fraud ring detection"""
        try:
            entity_id = entity_data.get('entity_id')
            connections = entity_data.get('connections', [])

            if not entity_id:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "no_entity_id"}

            # Add entity and connections to graph
            self._update_graph(entity_id, connections)

            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(entity_id)

            # Detect suspicious patterns
            risk_patterns = self._detect_risk_patterns(entity_id, network_metrics)

            # Calculate network risk score
            risk_score = self._calculate_network_risk_score(network_metrics, risk_patterns)

            return {
                "risk_score": float(risk_score),
                "network_metrics": network_metrics,
                "risk_patterns": risk_patterns,
                "confidence": 0.82,
                "analysis_type": "network_analysis"
            }

        except Exception as e:
            logger.error(f"Network analysis error: {e}")
            return {"risk_score": 0.0, "confidence": 0.0, "status": "error", "error": str(e)}

    def _update_graph(self, entity_id: str, connections: List[Dict]):
        """Update the transaction graph with new entity and connections"""
        if entity_id not in self.transaction_graph:
            self._node_order.append(entity_id)
        self.transaction_graph.add_node(entity_id)

        for connection in connections:
            connected_entity = connection.get('entity_id')
            if connected_entity:
                if connected_entity not in self.transaction_graph:
                    self._node_order.append(connected_entity)
                self.transaction_graph.add_edge(
                    entity_id,
                    connected_entity,
                    weight=connection.get('strength', 1.0),
                    transaction_count=connection.get('transaction_count', 1)
                )

        # Evict oldest nodes if over cap
        while len(self.transaction_graph.nodes) > self.MAX_GRAPH_NODES:
            oldest = self._node_order.popleft()
            if oldest in self.transaction_graph:
                self.transaction_graph.remove_node(oldest)

    def _calculate_network_metrics(self, entity_id: str) -> Dict[str, float]:
        """Calculate network centrality and connectivity metrics"""
        if entity_id not in self.transaction_graph:
            return {}

        # Basic metrics
        degree = self.transaction_graph.degree(entity_id)
        clustering = nx.clustering(self.transaction_graph, entity_id)

        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(self.transaction_graph).get(entity_id, 0)
            closeness = nx.closeness_centrality(self.transaction_graph).get(entity_id, 0)
        except Exception as e:
            logger.error(f"Centrality calculation error: {e}")
            betweenness = 0
            closeness = 0

        return {
            "degree": float(degree),
            "clustering_coefficient": float(clustering),
            "betweenness_centrality": float(betweenness),
            "closeness_centrality": float(closeness)
        }

    def _detect_risk_patterns(self, entity_id: str, metrics: Dict[str, float]) -> List[str]:
        """Detect suspicious network patterns"""
        patterns = []

        # High connectivity risk
        if metrics.get("degree", 0) > 50:
            patterns.append("unusually_high_connectivity")

        # Hub behavior (high betweenness centrality)
        if metrics.get("betweenness_centrality", 0) > 0.1:
            patterns.append("potential_fraud_hub")

        # Tight clustering (potential fraud ring)
        if metrics.get("clustering_coefficient", 0) > 0.8:
            patterns.append("tight_clustering_pattern")

        return patterns

    def _calculate_network_risk_score(self, metrics: Dict[str, float], patterns: List[str]) -> float:
        """Calculate overall network risk score"""
        base_score = 0.0

        # Degree risk
        degree = metrics.get("degree", 0)
        base_score += min(0.3, degree / 100)

        # Centrality risk
        betweenness = metrics.get("betweenness_centrality", 0)
        base_score += min(0.4, betweenness * 2)

        # Pattern risk
        pattern_score = len(patterns) * 0.2

        return min(1.0, base_score + pattern_score)

# Initialize analyzers
behavioral_analyzer = BehavioralBiometrics()
transaction_analyzer = TransactionAnalyzer()
network_analyzer = NetworkAnalyzer()

# Payment method mapping for dict-to-Pydantic conversion
_PAYMENT_METHOD_MAP = {
    'credit_card': 'credit_card',
    'debit_card': 'debit_card',
    'bank_transfer': 'bank_transfer',
    'crypto': 'crypto',
    'paypal': 'paypal',
    'wire_transfer': 'wire_transfer',
    'check': 'check',
    'cash': 'cash',
    'unknown': 'other',
}


def _dict_to_transaction_data(data: Dict[str, Any]) -> TransactionData:
    """Convert a validated transaction dict to TransactionData for FeatureEngineer."""
    # Parse timestamp
    ts = data.get('timestamp')
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    elif not isinstance(ts, datetime):
        ts = datetime.now()

    # Map payment method to enum value
    pm = data.get('payment_method', 'other')
    pm = _PAYMENT_METHOD_MAP.get(pm, 'other')

    return TransactionData(
        transaction_id=data.get('transaction_id', f'txn-{uuid.uuid4().hex[:12]}'),
        user_id=data.get('user_id', 'anonymous'),
        amount=max(0.01, float(data.get('amount', 0.01))),
        merchant=data.get('merchant') or 'unknown',
        location=data.get('location') or 'unknown',
        timestamp=ts,
        payment_method=pm,
    )


# Initialize explainer with transaction analyzer's trained model
if EXPLAINABILITY_AVAILABLE and FraudExplainer is not None:
    try:
        fraud_explainer = FraudExplainer(
            model=transaction_analyzer.isolation_forest,
            feature_names=transaction_analyzer.feature_engineer.feature_names
        )
        logger.info("FraudExplainer initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize FraudExplainer: {e}")
        fraud_explainer = None
else:
    fraud_explainer = None

# Initialize prediction cache and inference statistics
prediction_cache = LRUCache(capacity=1000)
_inference_stats = {
    "total_predictions": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_time_ms": 0.0,
    "batch_predictions": 0,
}

# Initialize monitoring
if MONITORING_AVAILABLE:
    monitor = MonitoringManager(app_name="fraud-detection-mcp", version="2.1.0")
else:
    monitor = None


def _monitored(endpoint: str, method: str = "TOOL"):
    """Apply @track_api_call only when monitoring is available."""
    if MONITORING_AVAILABLE and track_api_call is not None:
        return track_api_call(endpoint=endpoint, method=method)
    return lambda fn: fn  # no-op decorator


def _generate_cache_key(transaction_data: Dict[str, Any]) -> str:
    """Generate a deterministic cache key from transaction data."""
    key_fields = {
        "amount": transaction_data.get("amount"),
        "merchant": transaction_data.get("merchant"),
        "location": transaction_data.get("location"),
        "timestamp": transaction_data.get("timestamp"),
        "payment_method": transaction_data.get("payment_method"),
        "user_id": transaction_data.get("user_id"),
    }
    import json
    key_str = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


# =============================================================================
# Implementation Functions (testable, import these in tests)
# =============================================================================

def analyze_transaction_impl(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """Implementation of comprehensive transaction fraud analysis"""
    import time as _time
    _start = _time.monotonic()
    try:
        # Validate inputs
        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {"error": f"Invalid transaction data: {msg}", "status": "validation_failed"}

        if behavioral_data:
            valid, msg = validate_behavioral_data(behavioral_data)
            if not valid:
                return {"error": f"Invalid behavioral data: {msg}", "status": "validation_failed"}

        # Check prediction cache (only for pure transaction analysis, not behavioral)
        cache_key = None
        if use_cache and not include_behavioral:
            cache_key = _generate_cache_key(transaction_data)
            cached = prediction_cache.get(cache_key)
            if cached is not None:
                _inference_stats["cache_hits"] += 1
                _inference_stats["total_predictions"] += 1
                elapsed = (_time.monotonic() - _start) * 1000
                _inference_stats["total_time_ms"] += elapsed
                if monitor is not None:
                    monitor.record_cache_hit(cache_type="prediction")
                result = dict(cached)
                result["cache_hit"] = True
                return result

        _inference_stats["cache_misses"] += 1

        # Primary transaction analysis
        transaction_result = transaction_analyzer.analyze_transaction(transaction_data)

        # Generate feature-level explanation
        feature_explanation = None
        if fraud_explainer is not None:
            try:
                features = transaction_analyzer._extract_transaction_features(transaction_data)
                feature_explanation = fraud_explainer.explain_prediction(
                    features, transaction_result.get("risk_score", 0.0)
                )
            except Exception as e:
                logger.warning(f"Feature explanation failed: {e}")

        results = {
            "transaction_analysis": transaction_result,
            "overall_risk_score": transaction_result.get("risk_score", 0.0),
            "risk_level": "LOW",
            "detected_anomalies": [],
            "explanations": [],
            "recommended_actions": []
        }

        if feature_explanation:
            results["feature_explanation"] = feature_explanation

        # Add transaction risk factors
        risk_factors = transaction_result.get("risk_factors", [])
        results["detected_anomalies"].extend(risk_factors)

        # Behavioral analysis if requested
        if include_behavioral and behavioral_data:
            behavioral_result = {}

            if "keystroke_dynamics" in behavioral_data:
                keystroke_result = behavioral_analyzer.analyze_keystroke_dynamics(
                    behavioral_data["keystroke_dynamics"]
                )
                behavioral_result["keystroke"] = keystroke_result

                if keystroke_result.get("is_anomaly"):
                    results["detected_anomalies"].append("abnormal_keystroke_dynamics")
                    results["overall_risk_score"] = min(1.0, results["overall_risk_score"] + 0.2)

            results["behavioral_analysis"] = behavioral_result

        # Determine risk level
        risk_score = results["overall_risk_score"]
        if risk_score >= 0.8:
            results["risk_level"] = "CRITICAL"
            results["recommended_actions"] = ["block_transaction", "require_manual_review"]
        elif risk_score >= 0.6:
            results["risk_level"] = "HIGH"
            results["recommended_actions"] = ["require_additional_verification", "flag_for_review"]
        elif risk_score >= 0.4:
            results["risk_level"] = "MEDIUM"
            results["recommended_actions"] = ["monitor_closely", "collect_additional_data"]
        else:
            results["risk_level"] = "LOW"
            results["recommended_actions"] = ["allow_transaction"]

        # Generate explanation
        if results["detected_anomalies"]:
            explanation = f"Transaction flagged due to: {', '.join(results['detected_anomalies'])}"
        else:
            explanation = "Transaction appears normal with no significant risk factors detected"

        results["explanation"] = explanation
        results["analysis_timestamp"] = datetime.now().isoformat()
        results["model_version"] = "v2.1.0"

        # Record monitoring metrics
        if monitor is not None:
            elapsed_s = (_time.monotonic() - _start)
            monitor.record_prediction(
                model_type="isolation_forest",
                feature_count=46,
                duration=elapsed_s,
                risk_score=risk_score,
                transaction_id=transaction_data.get("transaction_id", "unknown"),
            )
            monitor.record_fraud_transaction(
                risk_level=results["risk_level"].lower(),
                status="processed",
            )

        results["cache_hit"] = False

        # Store in prediction cache
        if cache_key is not None:
            prediction_cache.put(cache_key, results)

        # Update inference stats
        _inference_stats["total_predictions"] += 1
        elapsed = (_time.monotonic() - _start) * 1000
        _inference_stats["total_time_ms"] += elapsed

        return results

    except Exception as e:
        logger.error(f"Transaction analysis failed: {e}")
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }


def detect_behavioral_anomaly_impl(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of behavioral biometrics anomaly detection"""
    try:
        valid, msg = validate_behavioral_data(behavioral_data)
        if not valid:
            return {"error": f"Invalid behavioral data: {msg}", "status": "validation_failed"}

        results = {
            "overall_anomaly_score": 0.0,
            "behavioral_analyses": {},
            "detected_anomalies": [],
            "confidence": 0.0
        }

        total_confidence = 0.0
        analysis_count = 0

        # Keystroke dynamics analysis
        if "keystroke_dynamics" in behavioral_data:
            keystroke_result = behavioral_analyzer.analyze_keystroke_dynamics(
                behavioral_data["keystroke_dynamics"]
            )
            results["behavioral_analyses"]["keystroke"] = keystroke_result

            if keystroke_result.get("is_anomaly"):
                results["detected_anomalies"].append("keystroke_anomaly")

            results["overall_anomaly_score"] = max(
                results["overall_anomaly_score"],
                keystroke_result.get("risk_score", 0.0)
            )

            total_confidence += keystroke_result.get("confidence", 0.0)
            analysis_count += 1

        # Calculate average confidence
        if analysis_count > 0:
            results["confidence"] = total_confidence / analysis_count

        results["analysis_timestamp"] = datetime.now().isoformat()

        return results

    except Exception as e:
        logger.error(f"Behavioral analysis failed: {e}")
        return {
            "error": str(e),
            "overall_anomaly_score": 0.0,
            "status": "analysis_failed"
        }


def assess_network_risk_impl(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of network-based risk assessment"""
    return network_analyzer.analyze_network_risk(entity_data)


def generate_risk_score_impl(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Implementation of comprehensive risk score generation"""
    try:
        # Validate transaction data
        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {"error": f"Invalid transaction data: {msg}", "status": "validation_failed",
                    "overall_risk_score": 0.0, "risk_level": "UNKNOWN"}

        # Perform all analyses
        transaction_analysis = transaction_analyzer.analyze_transaction(transaction_data)

        # Initialize comprehensive results
        comprehensive_result = {
            "overall_risk_score": 0.0,
            "component_scores": {
                "transaction": transaction_analysis.get("risk_score", 0.0)
            },
            "risk_level": "LOW",
            "confidence": 0.0,
            "detected_anomalies": [],
            "comprehensive_explanation": "",
            "recommended_actions": []
        }

        scores = [transaction_analysis.get("risk_score", 0.0)]
        confidences = [transaction_analysis.get("confidence", 0.0)]

        # Add transaction anomalies
        comprehensive_result["detected_anomalies"].extend(
            transaction_analysis.get("risk_factors", [])
        )

        # Behavioral analysis
        if behavioral_data:
            behavioral_analysis = behavioral_analyzer.analyze_keystroke_dynamics(
                behavioral_data.get("keystroke_dynamics", [])
            )
            behavioral_score = behavioral_analysis.get("risk_score", 0.0)
            comprehensive_result["component_scores"]["behavioral"] = behavioral_score
            scores.append(behavioral_score)
            confidences.append(behavioral_analysis.get("confidence", 0.0))

            if behavioral_analysis.get("is_anomaly"):
                comprehensive_result["detected_anomalies"].append("behavioral_anomaly")

        # Network analysis
        if network_data:
            network_analysis = network_analyzer.analyze_network_risk(network_data)
            network_score = network_analysis.get("risk_score", 0.0)
            comprehensive_result["component_scores"]["network"] = network_score
            scores.append(network_score)
            confidences.append(network_analysis.get("confidence", 0.0))

            comprehensive_result["detected_anomalies"].extend(
                network_analysis.get("risk_patterns", [])
            )

        # Calculate weighted overall score
        if len(scores) == 1:
            overall_score = scores[0]
        elif len(scores) == 2:
            overall_score = (scores[0] * 0.6 + scores[1] * 0.4)
        else:
            overall_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)

        comprehensive_result["overall_risk_score"] = float(overall_score)
        comprehensive_result["confidence"] = float(np.mean(confidences))

        # Determine risk level and actions
        if overall_score >= 0.8:
            comprehensive_result["risk_level"] = "CRITICAL"
            comprehensive_result["recommended_actions"] = [
                "block_transaction",
                "require_manual_review",
                "investigate_account"
            ]
        elif overall_score >= 0.6:
            comprehensive_result["risk_level"] = "HIGH"
            comprehensive_result["recommended_actions"] = [
                "require_additional_verification",
                "flag_for_review",
                "monitor_account"
            ]
        elif overall_score >= 0.4:
            comprehensive_result["risk_level"] = "MEDIUM"
            comprehensive_result["recommended_actions"] = [
                "monitor_closely",
                "collect_additional_data"
            ]
        else:
            comprehensive_result["risk_level"] = "LOW"
            comprehensive_result["recommended_actions"] = ["allow_transaction"]

        # Generate comprehensive explanation
        if comprehensive_result["detected_anomalies"]:
            explanation = (
                f"Risk assessment detected {len(comprehensive_result['detected_anomalies'])} "
                f"anomalies: {', '.join(comprehensive_result['detected_anomalies'])}. "
                f"Combined analysis suggests {comprehensive_result['risk_level']} risk level."
            )
        else:
            explanation = (
                f"Comprehensive analysis found no significant anomalies. "
                f"Risk level assessed as {comprehensive_result['risk_level']}."
            )

        comprehensive_result["comprehensive_explanation"] = explanation
        comprehensive_result["analysis_timestamp"] = datetime.now().isoformat()
        comprehensive_result["analysis_components"] = list(comprehensive_result["component_scores"].keys())

        return comprehensive_result

    except Exception as e:
        logger.error(f"Comprehensive risk assessment failed: {e}")
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }


def explain_decision_impl(
    analysis_result: Dict[str, Any],
    transaction_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Implementation of explainable AI reasoning for fraud decisions.

    Args:
        analysis_result: Previous analysis result to explain.
        transaction_data: Optional original transaction data. When provided and
            the explainability module is available, SHAP-based feature-level
            explanations are generated on-demand.

    Returns:
        Detailed explanation of the decision-making process.
    """
    try:
        explanation = {
            "decision_summary": "",
            "key_factors": [],
            "algorithm_contributions": {},
            "confidence_breakdown": {},
            "alternative_scenarios": [],
            "explainability_method": "rule_based",
            "explanation_timestamp": datetime.now().isoformat()
        }

        risk_score = analysis_result.get("overall_risk_score", 0.0)
        risk_level = analysis_result.get("risk_level", "UNKNOWN")
        detected_anomalies = analysis_result.get("detected_anomalies", [])

        # Decision summary
        explanation["decision_summary"] = (
            f"The fraud detection system assessed this case as {risk_level} risk "
            f"with a confidence score of {risk_score:.2f}. "
            f"This decision was based on analysis of {len(detected_anomalies)} risk factors."
        )

        # Key contributing factors
        if detected_anomalies:
            explanation["key_factors"] = [
                {
                    "factor": anomaly,
                    "impact": "high" if "high" in anomaly else "medium",
                    "description": f"Detected pattern: {anomaly.replace('_', ' ')}"
                }
                for anomaly in detected_anomalies
            ]

        # Algorithm contributions
        component_scores = analysis_result.get("component_scores", {})
        if component_scores:
            for component, score in component_scores.items():
                if component == "transaction":
                    weight = 0.5 if len(component_scores) == 3 else (0.6 if len(component_scores) == 2 else 1.0)
                elif component == "behavioral":
                    weight = 0.3 if len(component_scores) == 3 else 0.4
                else:  # network
                    weight = 0.2 if len(component_scores) == 3 else 0.4

                explanation["algorithm_contributions"][component] = {
                    "score": float(score),
                    "weight": float(weight),
                    "contribution": f"{weight * 100:.1f}% of final decision"
                }

        # Confidence breakdown
        explanation["confidence_breakdown"] = {
            "model_confidence": "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low",
            "data_quality": "Good" if len(detected_anomalies) > 0 else "Limited",
            "recommendation_strength": "Strong" if risk_score > 0.8 or risk_score < 0.2 else "Moderate"
        }

        # Alternative scenarios
        if risk_score > 0.5:
            explanation["alternative_scenarios"].append(
                "If behavioral patterns were more consistent with user profile, "
                "risk score could be reduced by 0.2-0.3 points"
            )

        if "high_amount_transaction" in detected_anomalies:
            explanation["alternative_scenarios"].append(
                "For smaller transaction amounts, this would likely be classified as low risk"
            )

        # Include feature-level analysis if available in input
        if "feature_explanation" in analysis_result:
            explanation["feature_analysis"] = analysis_result["feature_explanation"]
            explanation["explainability_method"] = (
                analysis_result["feature_explanation"].get("method", "rule_based")
            )

        # Generate SHAP-based feature explanation on-demand when transaction_data
        # is provided and the explainability module is available
        if (
            transaction_data is not None
            and fraud_explainer is not None
            and "feature_analysis" not in explanation
        ):
            try:
                valid, msg = validate_transaction_data(transaction_data)
                if valid:
                    features = transaction_analyzer._extract_transaction_features(
                        transaction_data
                    )
                    feature_explanation = fraud_explainer.explain_prediction(
                        features, risk_score
                    )
                    explanation["feature_analysis"] = feature_explanation
                    explanation["explainability_method"] = feature_explanation.get(
                        "method", "Feature Importance"
                    )

                    # Generate human-readable summary
                    summary = fraud_explainer.generate_summary(feature_explanation)
                    explanation["human_readable_summary"] = summary
            except Exception as e:
                logger.warning(f"On-demand feature explanation failed: {e}")

        # Generate human-readable summary from pre-existing feature_analysis
        if (
            "feature_analysis" in explanation
            and "human_readable_summary" not in explanation
            and fraud_explainer is not None
        ):
            try:
                summary = fraud_explainer.generate_summary(explanation["feature_analysis"])
                explanation["human_readable_summary"] = summary
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")

        return explanation

    except Exception as e:
        logger.error(f"Decision explanation failed: {e}")
        return {
            "error": str(e),
            "decision_summary": "Unable to generate explanation",
            "status": "explanation_failed"
        }


def analyze_batch_impl(
    transactions: List[Dict[str, Any]],
    use_cache: bool = True
) -> Dict[str, Any]:
    """Analyze a batch of transactions and return aggregated results."""
    import time as _time
    _start = _time.monotonic()

    if not isinstance(transactions, list):
        return {"error": "transactions must be a list", "status": "validation_failed"}

    if len(transactions) == 0:
        return {"error": "transactions list is empty", "status": "validation_failed"}

    if len(transactions) > 1000:
        return {"error": "batch size exceeds maximum of 1000", "status": "validation_failed"}

    results = []
    risk_scores = []
    cache_hits = 0

    for txn in transactions:
        result = analyze_transaction_impl(txn, use_cache=use_cache)
        results.append(result)
        score = result.get("overall_risk_score", 0.0)
        risk_scores.append(score)
        if result.get("cache_hit", False):
            cache_hits += 1

    elapsed = (_time.monotonic() - _start) * 1000
    _inference_stats["batch_predictions"] += 1

    # Count risk levels
    risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in results:
        level = r.get("risk_level", "LOW")
        if level in risk_distribution:
            risk_distribution[level] += 1

    return {
        "batch_size": len(transactions),
        "results": results,
        "summary": {
            "total_analyzed": len(results),
            "average_risk_score": float(np.mean(risk_scores)) if risk_scores else 0.0,
            "max_risk_score": float(np.max(risk_scores)) if risk_scores else 0.0,
            "min_risk_score": float(np.min(risk_scores)) if risk_scores else 0.0,
            "risk_distribution": risk_distribution,
            "cache_hits": cache_hits,
            "processing_time_ms": round(elapsed, 2),
        },
        "analysis_timestamp": datetime.now().isoformat(),
    }


def get_inference_stats_impl() -> Dict[str, Any]:
    """Return inference statistics including cache performance."""
    total = _inference_stats["total_predictions"]
    hits = _inference_stats["cache_hits"]
    misses = _inference_stats["cache_misses"]
    total_ms = _inference_stats["total_time_ms"]

    return {
        "total_predictions": total,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        "cache_size": prediction_cache.size(),
        "cache_capacity": prediction_cache.capacity,
        "average_prediction_time_ms": round(total_ms / total, 2) if total > 0 else 0.0,
        "total_time_ms": round(total_ms, 2),
        "batch_predictions": _inference_stats["batch_predictions"],
    }


def health_check_impl() -> Dict[str, Any]:
    """Implementation of system health check."""
    result = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "models": {
            "isolation_forest": transaction_analyzer.isolation_forest is not None,
            "feature_engineer": transaction_analyzer.feature_engineer is not None,
            "autoencoder": transaction_analyzer.autoencoder is not None,
            "explainer": fraud_explainer is not None,
            "feature_count": len(transaction_analyzer.feature_engineer.feature_names),
            "model_source": transaction_analyzer._model_source,
        },
        "cache": {
            "size": prediction_cache.size(),
            "capacity": prediction_cache.capacity,
            "hit_rate": (
                _inference_stats["cache_hits"] / _inference_stats["total_predictions"]
                if _inference_stats["total_predictions"] > 0 else 0.0
            ),
        },
        "inference": {
            "total_predictions": _inference_stats["total_predictions"],
            "batch_predictions": _inference_stats["batch_predictions"],
        },
    }

    # Add system metrics if monitoring available
    if monitor is not None:
        try:
            system_health = monitor.health_check()
            result["system"] = system_health.get("system", {})
            result["checks"] = system_health.get("checks", {})
            if system_health.get("status") == "degraded":
                result["status"] = "degraded"
        except Exception as e:
            logger.warning(f"System health check failed: {e}")
            result["system"] = {"error": str(e)}

    return result


def train_models_impl(
    data_path: str,
    test_size: float = 0.2,
    use_smote: bool = True,
    optimize_hyperparams: bool = False,
) -> Dict[str, Any]:
    """Train fraud detection models using the training pipeline.

    Args:
        data_path: Path to training data CSV or JSON file.
        test_size: Proportion of data to use for testing (0.0-1.0).
        use_smote: Whether to apply SMOTE for class balancing.
        optimize_hyperparams: Whether to use Optuna for hyperparameter tuning.

    Returns:
        Training results with metrics, or error if training deps unavailable.
    """
    if not TRAINING_AVAILABLE:
        return {
            "error": "Training dependencies not available. Install imbalanced-learn, "
                     "xgboost, and optuna to enable model training.",
            "status": "unavailable",
            "training_available": False,
        }

    try:
        data_file = Path(data_path)
        if not data_file.exists():
            return {
                "error": f"Data file not found: {data_path}",
                "status": "file_not_found",
            }

        trainer = ModelTrainer(
            model_dir=transaction_analyzer._model_dir,
            enable_mlflow=False,
        )
        results = trainer.train_all_models(
            data_path=data_path,
            test_size=test_size,
            use_smote=use_smote,
            optimize_hyperparams=optimize_hyperparams,
            train_autoencoder=False,
            train_gnn=False,
        )

        # Reload the newly trained models into the active analyzer
        if transaction_analyzer.load_models():
            results["hot_reload"] = True
            results["model_source"] = transaction_analyzer._model_source
        else:
            results["hot_reload"] = False

        return results

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {
            "error": str(e),
            "status": "training_failed",
        }


def get_model_status_impl() -> Dict[str, Any]:
    """Return the current status of fraud detection models.

    Returns:
        Dict with model_source, training availability, model details, and paths.
    """
    model_dir = transaction_analyzer._model_dir

    # Check for saved model files
    iso_path = model_dir / "isolation_forest.joblib"
    fe_path = model_dir / "feature_engineer.joblib"

    return {
        "model_source": transaction_analyzer._model_source,
        "training_available": TRAINING_AVAILABLE,
        "models": {
            "isolation_forest": {
                "loaded": transaction_analyzer.isolation_forest is not None,
                "n_estimators": getattr(
                    transaction_analyzer.isolation_forest, "n_estimators", None
                ),
                "contamination": getattr(
                    transaction_analyzer.isolation_forest, "contamination", None
                ),
            },
            "feature_engineer": {
                "loaded": transaction_analyzer.feature_engineer is not None,
                "feature_count": len(
                    transaction_analyzer.feature_engineer.feature_names
                ) if transaction_analyzer.feature_engineer else 0,
                "feature_names": (
                    transaction_analyzer.feature_engineer.feature_names
                    if transaction_analyzer.feature_engineer else []
                ),
            },
            "autoencoder": {
                "loaded": transaction_analyzer.autoencoder is not None,
                "available": AUTOENCODER_AVAILABLE,
                "fallback_mode": (
                    getattr(transaction_analyzer.autoencoder, "fallback_mode", None)
                    if transaction_analyzer.autoencoder else None
                ),
                "contamination": (
                    getattr(transaction_analyzer.autoencoder, "contamination", None)
                    if transaction_analyzer.autoencoder else None
                ),
            },
        },
        "ensemble_weights": transaction_analyzer._ensemble_weights,
        "saved_models": {
            "isolation_forest": str(iso_path) if iso_path.exists() else None,
            "feature_engineer": str(fe_path) if fe_path.exists() else None,
        },
        "model_dir": str(model_dir),
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# MCP Tool Wrappers (thin delegates to _impl functions)
# =============================================================================

@_monitored("/analyze_transaction", "TOOL")
@mcp.tool()
def analyze_transaction(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive transaction fraud analysis with optional behavioral biometrics.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp, etc.)
        include_behavioral: Whether to include behavioral analysis
        behavioral_data: Behavioral biometrics data (keystroke dynamics, mouse movements)

    Returns:
        Fraud analysis results with risk score, level, anomalies, and recommendations
    """
    return analyze_transaction_impl(transaction_data, include_behavioral, behavioral_data)


@_monitored("/detect_behavioral_anomaly", "TOOL")
@mcp.tool()
def detect_behavioral_anomaly(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze behavioral biometrics for anomaly detection.

    Args:
        behavioral_data: Behavioral patterns (keystroke dynamics, mouse movements, etc.)

    Returns:
        Behavioral anomaly analysis results
    """
    return detect_behavioral_anomaly_impl(behavioral_data)


@_monitored("/assess_network_risk", "TOOL")
@mcp.tool()
def assess_network_risk(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze network patterns for fraud ring detection.

    Args:
        entity_data: Entity information and network connections

    Returns:
        Network-based risk assessment
    """
    return assess_network_risk_impl(entity_data)


@_monitored("/generate_risk_score", "TOOL")
@mcp.tool()
def generate_risk_score(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive risk score combining all analysis methods.

    Args:
        transaction_data: Transaction details
        behavioral_data: Behavioral biometrics data
        network_data: Network connection data

    Returns:
        Comprehensive risk assessment with detailed scoring
    """
    return generate_risk_score_impl(transaction_data, behavioral_data, network_data)


@_monitored("/explain_decision", "TOOL")
@mcp.tool()
def explain_decision(
    analysis_result: Dict[str, Any],
    transaction_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Provide explainable AI reasoning for fraud detection decisions.

    When transaction_data is provided, generates SHAP-based feature-level
    explanations showing which features contributed most to the risk score.

    Args:
        analysis_result: Previous analysis result to explain
        transaction_data: Optional original transaction data for feature-level explanation

    Returns:
        Detailed explanation of the decision-making process
    """
    return explain_decision_impl(analysis_result, transaction_data)


@_monitored("/analyze_batch", "TOOL")
@mcp.tool()
def analyze_batch(
    transactions: List[Dict[str, Any]],
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Analyze a batch of transactions for fraud detection.

    Args:
        transactions: List of transaction data dicts to analyze
        use_cache: Whether to use prediction cache (default: True)

    Returns:
        Batch analysis results with per-transaction results and aggregated summary
    """
    return analyze_batch_impl(transactions, use_cache)


@_monitored("/get_inference_stats", "TOOL")
@mcp.tool()
def get_inference_stats() -> Dict[str, Any]:
    """
    Get inference engine statistics including cache performance metrics.

    Returns:
        Statistics dict with total_predictions, cache_hit_rate, average_prediction_time_ms, etc.
    """
    return get_inference_stats_impl()


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """
    System health check with model status, cache stats, and system metrics.

    Returns:
        Health status including models loaded, cache performance, and system resource usage
    """
    return health_check_impl()


@mcp.tool()
def get_model_status() -> Dict[str, Any]:
    """
    Get current fraud detection model status and configuration.

    Returns:
        Model source (synthetic/saved/none), training availability,
        model details (feature count, estimators), and saved model paths
    """
    return get_model_status_impl()


@_monitored("/train_models", "TOOL")
@mcp.tool()
def train_models(
    data_path: str,
    test_size: float = 0.2,
    use_smote: bool = True,
    optimize_hyperparams: bool = False,
) -> Dict[str, Any]:
    """
    Train fraud detection models using the ML training pipeline.

    Requires training dependencies (imbalanced-learn, xgboost, optuna).
    After training, models are saved to disk and hot-reloaded into the server.

    Args:
        data_path: Path to training data CSV or JSON file with 'is_fraud' column
        test_size: Proportion of data for testing (default: 0.2)
        use_smote: Apply SMOTE for class balancing (default: True)
        optimize_hyperparams: Use Optuna for hyperparameter tuning (default: False)

    Returns:
        Training results with model metrics, or error if dependencies unavailable
    """
    return train_models_impl(data_path, test_size, use_smote, optimize_hyperparams)


if __name__ == "__main__":
    mcp.run()