#!/usr/bin/env python3
"""
Advanced Fraud Detection MCP Server
Sophisticated fraud detection using cutting-edge 2024-2025 algorithms
"""

import os

# Prevent OMP segfaults when PyTorch and sklearn coexist in the same process
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
import logging
import math
import threading
import uuid
from datetime import datetime, timedelta
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
from models_validation import TransactionData
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

# Synthetic data integration (graceful degradation if unavailable)
try:
    from integration import SyntheticDataIntegration

    SYNTHETIC_DATA_AVAILABLE = True
except ImportError:
    SYNTHETIC_DATA_AVAILABLE = False
    SyntheticDataIntegration = None

# Security utilities (graceful degradation if unavailable)
try:
    from security_utils import InputSanitizer, InMemoryRateLimiter

    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    SECURITY_UTILS_AVAILABLE = False
    InputSanitizer = None  # type: ignore[assignment,misc]
    InMemoryRateLimiter = None  # type: ignore[assignment,misc]

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
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
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
            contamination=0.1, random_state=42, n_estimators=100
        )

        # Fit with dummy training data (10 features: 5 dwell + 5 flight)
        dummy_keystroke_data = np.random.randn(100, 10) * 50 + 100
        self.keystroke_model.fit(dummy_keystroke_data)

        # One-Class SVM for mouse patterns
        self.mouse_model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)

        # Fit with dummy mouse data
        dummy_mouse_data = np.random.randn(100, 5) * 20 + 50
        self.mouse_model.fit(dummy_mouse_data)

        # Local Outlier Factor for touch patterns
        self.touch_model = LocalOutlierFactor(
            n_neighbors=20, contamination=0.1, novelty=True
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
                return {
                    "risk_score": 0.0,
                    "confidence": 0.0,
                    "status": "error",
                    "error": f"keystroke_data must be a list, got {type(keystroke_data).__name__}",
                }

            # Extract features from keystroke data
            features = self._extract_keystroke_features(keystroke_data)

            if features is None:
                return {
                    "risk_score": 0.0,
                    "confidence": 0.0,
                    "status": "error",
                    "error": "could not extract valid features from keystroke data",
                }

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
                "features_analyzed": len(features)
                if hasattr(features, "__len__")
                else 1,
            }

        except Exception as e:
            logger.error(f"Keystroke analysis error: {e}")
            return {
                "risk_score": 0.0,
                "confidence": 0.0,
                "status": "error",
                "error": str(e),
            }

    def _extract_keystroke_features(
        self, keystroke_data: List[Dict]
    ) -> Optional[List[float]]:
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
                if "press_time" in keystroke and "release_time" in keystroke:
                    try:
                        dwell = float(keystroke["release_time"]) - float(
                            keystroke["press_time"]
                        )
                        dwell_times.append(dwell)
                    except (TypeError, ValueError):
                        continue  # Skip non-numeric timing values

                # Flight time
                if i > 0:
                    prev_keystroke = keystroke_data[i - 1]
                    if "release_time" in prev_keystroke and "press_time" in keystroke:
                        try:
                            flight = float(keystroke["press_time"]) - float(
                                prev_keystroke["release_time"]
                            )
                            flight_times.append(flight)
                        except (TypeError, ValueError):
                            continue  # Skip non-numeric timing values

            if not dwell_times and not flight_times:
                return [0.0] * 10

            # Statistical features
            features = []

            if dwell_times:
                features.extend(
                    [
                        np.mean(dwell_times),
                        np.std(dwell_times),
                        np.median(dwell_times),
                        np.max(dwell_times),
                        np.min(dwell_times),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            if flight_times:
                features.extend(
                    [
                        np.mean(flight_times),
                        np.std(flight_times),
                        np.median(flight_times),
                        np.max(flight_times),
                        np.min(flight_times),
                    ]
                )
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
            contamination=0.1, random_state=42, n_estimators=200
        )
        # Try to load saved models first; fall back to synthetic training
        if not self.load_models():
            self._initialize_models()

    def _initialize_models(self):
        """Initialize models with synthetic training data"""
        from datetime import timedelta

        rng = np.random.RandomState(42)
        n = 200
        payment_methods = [
            "credit_card",
            "debit_card",
            "bank_transfer",
            "crypto",
            "paypal",
        ]
        locations = [
            "United States",
            "United Kingdom",
            "Canada",
            "Germany",
            "Japan",
            "France",
            "Australia",
            "Brazil",
            "India",
            "Singapore",
        ]
        merchants = [
            "Amazon",
            "Walmart",
            "Target",
            "BestBuy",
            "Costco",
            "Starbucks",
            "McDonalds",
            "Apple",
            "Google",
            "Netflix",
        ]

        synthetic_transactions = []
        for i in range(n):
            amount = round(max(0.01, rng.exponential(500)), 2)
            txn = TransactionData(
                transaction_id=f"train-{i:04d}",
                user_id=f"user-{i % 50:03d}",
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
                    if (
                        self.autoencoder.threshold is not None
                        and self.autoencoder.threshold > 0
                    ):
                        ae_risk = float(
                            np.clip(
                                ae_scores[0] / (self.autoencoder.threshold * 3), 0, 1
                            )
                        )
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
            return {
                "risk_score": 0.0,
                "confidence": 0.0,
                "status": "error",
                "error": str(e),
            }

    def _extract_transaction_features(
        self, transaction_data: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features using FeatureEngineer (46 features)"""
        txn = _dict_to_transaction_data(transaction_data)
        return self.feature_engineer.transform(txn)

    def _identify_risk_factors(
        self, transaction: Dict[str, Any], features: List[float]
    ) -> List[str]:
        """Identify specific risk factors in the transaction"""
        risk_factors = []

        amount = float(transaction.get("amount", 0))

        # High amount risk
        if amount > 10000:
            risk_factors.append("high_amount_transaction")

        # Unusual time risk
        timestamp = transaction.get("timestamp")
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.hour < 6 or dt.hour > 23:
                risk_factors.append("unusual_time_pattern")

        # Crypto payment risk
        if transaction.get("payment_method") == "crypto":
            risk_factors.append("high_risk_payment_method")

        # Geographic risk - config-driven exact match
        location = transaction.get("location", "").lower().strip()
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


class UserTransactionHistory:
    """In-memory per-user transaction history for velocity analysis.

    Thread-safe, bounded deque per user. No external dependencies.
    """

    def __init__(self, max_history: int = 100, max_users: int = 10000):
        self.max_history = max_history
        self.max_users = max_users
        self._history: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record(self, user_id: str, transaction: Dict[str, Any]) -> None:
        """Record a transaction for a user."""
        import time as _time

        entry = {
            "amount": float(transaction.get("amount", 0)),
            "merchant": str(transaction.get("merchant", "")),
            "location": str(transaction.get("location", "")),
            "timestamp": transaction.get("timestamp", ""),
            "recorded_at": _time.monotonic(),
        }
        with self._lock:
            if user_id not in self._history:
                # Evict oldest user if at capacity
                if len(self._history) >= self.max_users:
                    oldest_key = next(iter(self._history))
                    del self._history[oldest_key]
                self._history[user_id] = deque(maxlen=self.max_history)
            self._history[user_id].append(entry)

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get transaction history for a user."""
        with self._lock:
            if user_id in self._history:
                return list(self._history[user_id])
            return []

    def check_velocity(
        self, user_id: str, window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Check transaction velocity (count in time window).

        Args:
            user_id: User identifier.
            window_seconds: Lookback window in seconds (default: 1 hour).

        Returns:
            Dict with count, window, and is_suspicious flag.
        """
        import time as _time

        cutoff = _time.monotonic() - window_seconds
        history = self.get_history(user_id)
        recent = [h for h in history if h["recorded_at"] > cutoff]
        count = len(recent)
        return {
            "transaction_count": count,
            "window_seconds": window_seconds,
            "is_suspicious": count >= 10,
        }

    def check_amount_deviation(
        self, user_id: str, current_amount: float
    ) -> Dict[str, Any]:
        """Check if current amount deviates from user's historical pattern.

        Returns:
            Dict with mean, std, z_score, and is_suspicious flag.
        """
        history = self.get_history(user_id)
        amounts = [h["amount"] for h in history]
        if len(amounts) < 3:
            return {
                "mean": 0.0,
                "std": 0.0,
                "z_score": 0.0,
                "is_suspicious": False,
                "insufficient_history": True,
            }
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts))
        if std_amt < 1e-6:
            z_score = 0.0
        else:
            z_score = (current_amount - mean_amt) / std_amt
        return {
            "mean": round(mean_amt, 2),
            "std": round(std_amt, 2),
            "z_score": round(z_score, 2),
            "is_suspicious": abs(z_score) > 3.0,
            "insufficient_history": False,
        }

    def check_geographic_velocity(self, user_id: str) -> Dict[str, Any]:
        """Detect impossible travel (different locations in rapid succession).

        Returns:
            Dict with location_changes, time_between, and is_suspicious flag.
        """
        history = self.get_history(user_id)
        if len(history) < 2:
            return {
                "location_changes": 0,
                "is_suspicious": False,
                "insufficient_history": True,
            }
        last = history[-1]
        prev = history[-2]
        same_location = (
            last["location"].lower().strip() == prev["location"].lower().strip()
        )
        time_between = last["recorded_at"] - prev["recorded_at"]
        return {
            "location_changes": 0 if same_location else 1,
            "time_between_seconds": round(time_between, 2),
            "is_suspicious": not same_location and time_between < 300,
            "insufficient_history": False,
        }

    def check_merchant_diversity(
        self, user_id: str, window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Check merchant diversity in time window (card testing signal).

        Returns:
            Dict with unique_merchants, total, and is_suspicious flag.
        """
        import time as _time

        cutoff = _time.monotonic() - window_seconds
        history = self.get_history(user_id)
        recent = [h for h in history if h["recorded_at"] > cutoff]
        merchants = set(h["merchant"] for h in recent if h["merchant"])
        return {
            "unique_merchants": len(merchants),
            "total_transactions": len(recent),
            "window_seconds": window_seconds,
            "is_suspicious": len(merchants) >= 5 and len(recent) >= 5,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get history tracker statistics."""
        with self._lock:
            total_entries = sum(len(d) for d in self._history.values())
            return {
                "tracked_users": len(self._history),
                "max_users": self.max_users,
                "total_entries": total_entries,
                "max_history_per_user": self.max_history,
            }

    def reset(self, user_id: Optional[str] = None) -> None:
        """Reset history for a user or all users."""
        with self._lock:
            if user_id is not None:
                self._history.pop(user_id, None)
            else:
                self._history.clear()


class NetworkAnalyzer:
    """Graph-based network analysis for fraud ring detection"""

    MAX_GRAPH_NODES = 10000

    def __init__(self):
        self.transaction_graph = nx.Graph()
        self._node_order = deque()

    def analyze_network_risk(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network patterns for fraud ring detection"""
        try:
            entity_id = entity_data.get("entity_id")
            connections = entity_data.get("connections", [])

            if not entity_id:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "no_entity_id"}

            # Add entity and connections to graph
            self._update_graph(entity_id, connections)

            # Calculate network metrics
            network_metrics = self._calculate_network_metrics(entity_id)

            # Detect suspicious patterns
            risk_patterns = self._detect_risk_patterns(entity_id, network_metrics)

            # Calculate network risk score
            risk_score = self._calculate_network_risk_score(
                network_metrics, risk_patterns
            )

            return {
                "risk_score": float(risk_score),
                "network_metrics": network_metrics,
                "risk_patterns": risk_patterns,
                "confidence": 0.82,
                "analysis_type": "network_analysis",
            }

        except Exception as e:
            logger.error(f"Network analysis error: {e}")
            return {
                "risk_score": 0.0,
                "confidence": 0.0,
                "status": "error",
                "error": str(e),
            }

    def _update_graph(self, entity_id: str, connections: List[Dict]):
        """Update the transaction graph with new entity and connections"""
        if entity_id not in self.transaction_graph:
            self._node_order.append(entity_id)
        self.transaction_graph.add_node(entity_id)

        for connection in connections:
            connected_entity = connection.get("entity_id")
            if connected_entity:
                if connected_entity not in self.transaction_graph:
                    self._node_order.append(connected_entity)
                self.transaction_graph.add_edge(
                    entity_id,
                    connected_entity,
                    weight=connection.get("strength", 1.0),
                    transaction_count=connection.get("transaction_count", 1),
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
            betweenness = nx.betweenness_centrality(self.transaction_graph).get(
                entity_id, 0
            )
            closeness = nx.closeness_centrality(self.transaction_graph).get(
                entity_id, 0
            )
        except Exception as e:
            logger.error(f"Centrality calculation error: {e}")
            betweenness = 0
            closeness = 0

        return {
            "degree": float(degree),
            "clustering_coefficient": float(clustering),
            "betweenness_centrality": float(betweenness),
            "closeness_centrality": float(closeness),
        }

    def _detect_risk_patterns(
        self, entity_id: str, metrics: Dict[str, float]
    ) -> List[str]:
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

    def _calculate_network_risk_score(
        self, metrics: Dict[str, float], patterns: List[str]
    ) -> float:
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
    "credit_card": "credit_card",
    "debit_card": "debit_card",
    "bank_transfer": "bank_transfer",
    "crypto": "crypto",
    "paypal": "paypal",
    "wire_transfer": "wire_transfer",
    "check": "check",
    "cash": "cash",
    "unknown": "other",
}


def _dict_to_transaction_data(data: Dict[str, Any]) -> TransactionData:
    """Convert a validated transaction dict to TransactionData for FeatureEngineer."""
    # Parse timestamp
    ts = data.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    elif not isinstance(ts, datetime):
        ts = datetime.now()

    # Map payment method to enum value
    pm = data.get("payment_method", "other")
    pm = _PAYMENT_METHOD_MAP.get(pm, "other")

    return TransactionData(
        transaction_id=data.get("transaction_id", f"txn-{uuid.uuid4().hex[:12]}"),
        user_id=data.get("user_id", "anonymous"),
        amount=max(0.01, float(data.get("amount", 0.01))),
        merchant=data.get("merchant") or "unknown",
        location=data.get("location") or "unknown",
        timestamp=ts,
        payment_method=pm,
    )


# Initialize explainer with transaction analyzer's trained model
if EXPLAINABILITY_AVAILABLE and FraudExplainer is not None:
    try:
        fraud_explainer = FraudExplainer(
            model=transaction_analyzer.isolation_forest,
            feature_names=transaction_analyzer.feature_engineer.feature_names,
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

# Initialize synthetic data integration
if SYNTHETIC_DATA_AVAILABLE and SyntheticDataIntegration is not None:
    try:
        synthetic_data_integration = SyntheticDataIntegration()
        logger.info("SyntheticDataIntegration initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize SyntheticDataIntegration: {e}")
        synthetic_data_integration = None
else:
    synthetic_data_integration = None

# Initialize security utilities
if SECURITY_UTILS_AVAILABLE and InMemoryRateLimiter is not None:
    rate_limiter = InMemoryRateLimiter(max_requests=100, window_seconds=60.0)
    sanitizer = InputSanitizer
    logger.info("Security utilities initialized (sanitizer + rate limiter)")
else:
    rate_limiter = None
    sanitizer = None

# Initialize monitoring
if MONITORING_AVAILABLE:
    monitor = MonitoringManager(app_name="fraud-detection-mcp", version="2.3.0")
else:
    monitor = None

# Initialize user transaction history tracker
user_history = UserTransactionHistory(max_history=100, max_users=10000)

# =============================================================================
# Traffic Source Classifier
# =============================================================================

# Known AI agent User-Agent patterns
AGENT_USER_AGENT_PATTERNS = {
    "stripe_acp": ["stripe-acp", "stripe acp"],
    "visa_tap": ["visa-tap", "visa tap"],
    "mastercard_agent": ["mastercard-agent", "mastercard agent"],
    "openai": ["openai-operator", "openai operator", "openai-agent"],
    "anthropic": ["anthropic-agent", "anthropic agent", "claude-agent"],
    "google_ap2": ["google-ap2", "google ap2"],
    "paypal": ["paypal-agent", "paypal agent"],
    "x402": ["x402-client", "x402 client"],
    "coinbase": ["coinbase-agent", "coinbase agent", "agentkit"],
}

# Browser User-Agent patterns indicating human traffic
BROWSER_USER_AGENT_PATTERNS = [
    "mozilla/",
    "chrome/",
    "safari/",
    "firefox/",
    "edge/",
    "opera/",
]


class TrafficClassifier:
    """Classifies transaction traffic as human, agent, or unknown.

    Uses heuristic signals: explicit flags, user_agent patterns, agent
    identifiers, and absence/presence of behavioral data.
    """

    def classify(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a transaction's traffic source.

        Args:
            metadata: Transaction metadata including optional fields:
                is_agent, agent_identifier, user_agent, behavioral_data.

        Returns:
            Dict with source, confidence, agent_type, and signals.
        """
        signals = []
        agent_score = 0.0  # positive = agent, negative = human
        agent_type = None

        # Signal 1: Explicit is_agent flag (strongest signal)
        is_agent = metadata.get("is_agent")
        if is_agent is True:
            agent_score += 0.8
            signals.append("explicit_flag")
        elif is_agent is False:
            agent_score -= 0.8
            signals.append("explicit_flag")

        # Signal 2: Agent identifier present
        agent_id = metadata.get("agent_identifier")
        if agent_id and isinstance(agent_id, str) and len(agent_id.strip()) > 0:
            agent_score += 0.6
            signals.append("agent_identifier_present")
            # Try to extract agent type from identifier
            if not agent_type:
                agent_id_lower = agent_id.lower()
                for atype, patterns in AGENT_USER_AGENT_PATTERNS.items():
                    if any(p in agent_id_lower for p in patterns):
                        agent_type = atype
                        break

        # Signal 3: User-Agent string analysis
        user_agent = metadata.get("user_agent")
        if user_agent and isinstance(user_agent, str):
            ua_lower = user_agent.lower()

            # Check for known agent patterns
            for atype, patterns in AGENT_USER_AGENT_PATTERNS.items():
                if any(p in ua_lower for p in patterns):
                    agent_score += 0.7
                    signals.append("user_agent_match")
                    if not agent_type:
                        agent_type = atype
                    break
            else:
                # Check for browser patterns (human signal)
                if any(p in ua_lower for p in BROWSER_USER_AGENT_PATTERNS):
                    agent_score -= 0.5
                    signals.append("user_agent_match")

        # Clamp confidence to [0, 1]
        raw_confidence = min(abs(agent_score), 1.0)

        # Determine classification
        if agent_score > 0.3:
            source = "agent"
            confidence = raw_confidence
        elif agent_score < -0.3:
            source = "human"
            confidence = raw_confidence
        else:
            source = "unknown"
            confidence = raw_confidence  # low confidence in the unknown case

        return {
            "source": source,
            "confidence": float(confidence),
            "agent_type": agent_type,
            "signals": signals,
        }


traffic_classifier = TrafficClassifier()


# =============================================================================
# Agent Identity Registry
# =============================================================================


class AgentIdentityRegistry:
    """Thread-safe JSON-backed registry of known AI agent identities.

    Tracks agent identifiers, types, trust scores, and transaction history.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._path = registry_path or Path("data/agent_registry.json")
        self._lock = threading.Lock()
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load registry from disk if it exists."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._agents = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._agents = {}

    def _save(self):
        """Persist registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._agents, f, indent=2, default=str)

    def register(
        self, agent_id: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new agent or return existing entry."""
        with self._lock:
            if agent_id in self._agents:
                return self._agents[agent_id]
            entry = {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "transaction_count": 0,
                "trust_score": 0.5,
            }
            self._agents[agent_id] = entry
            self._save()
            return entry

    def lookup(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Look up an agent by identifier."""
        with self._lock:
            return self._agents.get(agent_id)

    def record_transaction(self, agent_id: str):
        """Record a transaction for an agent, incrementing count."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]["transaction_count"] += 1
                self._agents[agent_id]["last_seen"] = datetime.now().isoformat()
                self._save()

    def update_trust(self, agent_id: str, trust_score: float):
        """Update an agent's trust score (clamped to [0, 1])."""
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id]["trust_score"] = max(0.0, min(1.0, trust_score))
                self._save()

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered agents."""
        with self._lock:
            return dict(self._agents)


agent_registry = AgentIdentityRegistry()


class AgentIdentityVerifier:
    """Validates agent credentials and computes trust scores.

    Checks API key format, JWT token expiry, and registry membership.
    """

    # Minimum API key length for basic format validation
    MIN_KEY_LENGTH = 16

    def __init__(self, registry: AgentIdentityRegistry):
        self._registry = registry

    def verify(
        self,
        agent_identifier: Optional[str] = None,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify agent identity using available credentials.

        Args:
            agent_identifier: Agent identifier string.
            api_key: API key credential.
            token: JWT-style bearer token.

        Returns:
            Dict with verified, identity, trust_score, warnings.
        """
        warnings: List[str] = []
        trust_signals: List[float] = []
        identity: Dict[str, Any] = {}

        # Check for identifier
        if not agent_identifier:
            return {
                "verified": False,
                "identity": {},
                "trust_score": 0.0,
                "warnings": ["no_identifier"],
            }

        identity["agent_id"] = agent_identifier

        # Signal 1: Registry lookup
        registry_entry = self._registry.lookup(agent_identifier)
        if registry_entry:
            trust_signals.append(registry_entry["trust_score"])
            identity.update(registry_entry)
        else:
            warnings.append("not_in_registry")
            # Auto-register with low initial trust
            self._registry.register(agent_identifier)
            self._registry.update_trust(agent_identifier, 0.3)
            trust_signals.append(0.3)

        # Signal 2: API key format validation
        if api_key:
            if isinstance(api_key, str) and len(api_key) >= self.MIN_KEY_LENGTH:
                trust_signals.append(0.6)  # key present and reasonable format
            else:
                warnings.append("invalid_key_format")
                trust_signals.append(0.1)

        # Signal 3: JWT token validation (expiry check only)
        if token:
            token_trust = self._validate_token(token, warnings)
            trust_signals.append(token_trust)

        # Compute final trust score
        if trust_signals:
            trust_score = float(sum(trust_signals) / len(trust_signals))
        else:
            trust_score = 0.0

        # Verified if trust >= 0.5 and no critical warnings
        critical_warnings = {"no_identifier", "token_expired"}
        has_critical = bool(critical_warnings & set(warnings))
        verified = trust_score >= 0.5 and not has_critical

        return {
            "verified": verified,
            "identity": identity,
            "trust_score": trust_score,
            "warnings": warnings,
        }

    def _validate_token(self, token: str, warnings: List[str]) -> float:
        """Validate JWT token expiry. Returns trust signal."""
        import base64 as _b64
        import time as _time

        try:
            parts = token.split(".")
            if len(parts) != 3:
                warnings.append("token_parse_error")
                return 0.1

            # Decode payload (second part)
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_bytes = _b64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_bytes)

            # Check expiry
            exp = payload.get("exp")
            if exp and isinstance(exp, (int, float)):
                if exp < _time.time():
                    warnings.append("token_expired")
                    return 0.1
                else:
                    return 0.7  # valid expiry
            return 0.5  # no expiry claim, neutral

        except Exception:
            warnings.append("token_parse_error")
            return 0.1


agent_verifier = AgentIdentityVerifier(agent_registry)


# =============================================================================
# Agent Behavioral Fingerprinting
# =============================================================================


class AgentBehavioralFingerprint:
    """Behavioral fingerprinting for AI agent transactions.

    Tracks per-agent behavioral baselines: API call timing patterns,
    decision consistency, and request structure fingerprints.  Uses
    Isolation Forest to detect deviations from established patterns.

    Replaces BehavioralBiometrics for agent traffic -- agents have no
    keystroke/mouse signals but *do* exhibit measurable behavioral
    consistency that changes when compromised or impersonated.
    """

    # Maximum observations to keep per agent (bounded memory)
    MAX_HISTORY = 1000
    # Minimum observations before training the anomaly model
    MIN_OBSERVATIONS_FOR_MODEL = 10

    def __init__(self):
        self._lock = threading.Lock()
        # Per-agent observation history: agent_id -> deque of feature dicts
        self._history: Dict[str, deque] = {}
        # Per-agent trained Isolation Forest models
        self._models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(
        self,
        agent_id: str,
        api_timing_ms: float = 0.0,
        decision_pattern: Optional[str] = None,
        request_structure_hash: Optional[str] = None,
    ) -> None:
        """Record a single behavioral observation for an agent.

        Args:
            agent_id: Unique agent identifier.
            api_timing_ms: API response time in milliseconds.
            decision_pattern: Categorical decision (e.g. "approve", "reject").
            request_structure_hash: Hash of the request structure/shape.
        """
        obs = {
            "api_timing_ms": float(api_timing_ms),
            "decision_pattern": decision_pattern or "",
            "request_structure_hash": request_structure_hash or "",
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock:
            if agent_id not in self._history:
                self._history[agent_id] = deque(maxlen=self.MAX_HISTORY)
            self._history[agent_id].append(obs)

    # ------------------------------------------------------------------
    # Get baseline
    # ------------------------------------------------------------------

    def get_baseline(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return summary statistics for an agent's behavioral baseline.

        Returns None if the agent has no recorded observations.
        """
        with self._lock:
            history = self._history.get(agent_id)
            if not history:
                return None
            timings = [o["api_timing_ms"] for o in history]
            patterns = set(
                o["decision_pattern"] for o in history if o["decision_pattern"]
            )
            structures = set(
                o["request_structure_hash"]
                for o in history
                if o["request_structure_hash"]
            )
            return {
                "agent_id": agent_id,
                "observation_count": len(history),
                "timing_mean": float(np.mean(timings)) if timings else 0.0,
                "timing_std": float(np.std(timings)) if len(timings) > 1 else 0.0,
                "unique_decision_patterns": list(patterns),
                "unique_request_structures": list(structures),
            }

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        api_timing_ms: float,
        decision_pattern: Optional[str],
        request_structure_hash: Optional[str],
        baseline: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Extract a numeric feature vector for anomaly detection.

        Features (6 dimensions):
            0: api_timing_ms (raw)
            1: timing z-score vs baseline (0 if no baseline)
            2: log(1 + api_timing_ms)
            3: decision_pattern_novel (1 if unseen, 0 otherwise)
            4: request_structure_novel (1 if unseen, 0 otherwise)
            5: timing_ratio (current / baseline_mean, 1.0 if no baseline)
        """
        timing = max(0.0, api_timing_ms)  # clamp negatives

        # Z-score
        if baseline and baseline["timing_std"] > 0:
            z_score = (timing - baseline["timing_mean"]) / baseline["timing_std"]
        else:
            z_score = 0.0

        # Log timing
        log_timing = float(np.log1p(timing))

        # Novelty flags
        dp_novel = 0.0
        if decision_pattern and baseline:
            if decision_pattern not in baseline["unique_decision_patterns"]:
                dp_novel = 1.0

        rs_novel = 0.0
        if request_structure_hash and baseline:
            if request_structure_hash not in baseline["unique_request_structures"]:
                rs_novel = 1.0

        # Timing ratio
        if baseline and baseline["timing_mean"] > 0:
            timing_ratio = timing / baseline["timing_mean"]
        else:
            timing_ratio = 1.0

        return np.array(
            [[timing, z_score, log_timing, dp_novel, rs_novel, timing_ratio]]
        )

    # ------------------------------------------------------------------
    # Model training (per-agent)
    # ------------------------------------------------------------------

    def _train_model_unlocked(self, agent_id: str, baseline: Dict[str, Any]) -> None:
        """Train (or re-train) an Isolation Forest on the agent's history.

        MUST be called while self._lock is already held.  Accepts a
        pre-computed *baseline* dict so it never re-acquires the lock.
        """
        history = self._history.get(agent_id)
        if not history or len(history) < self.MIN_OBSERVATIONS_FOR_MODEL:
            return

        rows = []
        for obs in history:
            feat = self._extract_features(
                obs["api_timing_ms"],
                obs["decision_pattern"],
                obs["request_structure_hash"],
                baseline,
            )
            rows.append(feat[0])

        X = np.array(rows)
        model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        model.fit(X)
        self._models[agent_id] = model

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(
        self,
        agent_id: str,
        api_timing_ms: float = 0.0,
        decision_pattern: Optional[str] = None,
        request_structure_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze a single agent action against its behavioral baseline.

        Args:
            agent_id: Unique agent identifier.
            api_timing_ms: API response time in milliseconds.
            decision_pattern: Categorical decision label.
            request_structure_hash: Hash of the request structure.

        Returns:
            Dict with risk_score, confidence, is_anomaly, and details.
        """
        with self._lock:
            baseline = None
            if agent_id in self._history and self._history[agent_id]:
                # Compute baseline inside lock
                history = self._history[agent_id]
                timings = [o["api_timing_ms"] for o in history]
                patterns = set(
                    o["decision_pattern"] for o in history if o["decision_pattern"]
                )
                structures = set(
                    o["request_structure_hash"]
                    for o in history
                    if o["request_structure_hash"]
                )
                baseline = {
                    "agent_id": agent_id,
                    "observation_count": len(history),
                    "timing_mean": float(np.mean(timings)) if timings else 0.0,
                    "timing_std": float(np.std(timings)) if len(timings) > 1 else 0.0,
                    "unique_decision_patterns": list(patterns),
                    "unique_request_structures": list(structures),
                }

            detail_flags: List[str] = []
            risk_score = 0.5  # default: moderate when no info
            confidence = 0.3  # default: low confidence

            # No baseline at all -> elevated risk
            if baseline is None:
                detail_flags.append("no_baseline")
                risk_score = 0.5
                confidence = 0.2
            else:
                n_obs = baseline["observation_count"]
                # Confidence scales with observation count
                confidence = min(
                    0.9, 0.3 + (n_obs / self.MIN_OBSERVATIONS_FOR_MODEL) * 0.3
                )

                # --- Heuristic signals ---
                # Timing deviation
                if baseline["timing_std"] > 0:
                    z = (
                        abs(api_timing_ms - baseline["timing_mean"])
                        / baseline["timing_std"]
                    )
                    if z > 3:
                        detail_flags.append("timing_deviation_extreme")
                    elif z > 2:
                        detail_flags.append("timing_deviation_high")

                # Novel decision pattern
                dp = decision_pattern or ""
                if dp and dp not in baseline["unique_decision_patterns"]:
                    detail_flags.append("decision_pattern_novel")

                # Novel request structure
                rs = request_structure_hash or ""
                if rs and rs not in baseline["unique_request_structures"]:
                    detail_flags.append("request_structure_novel")

                # --- ML scoring (Isolation Forest) ---
                features = self._extract_features(
                    api_timing_ms, decision_pattern, request_structure_hash, baseline
                )

                # Train model if enough data and no model yet (or retrain periodically)
                if n_obs >= self.MIN_OBSERVATIONS_FOR_MODEL:
                    if agent_id not in self._models:
                        self._train_model_unlocked(agent_id, baseline)

                    model = self._models.get(agent_id)
                    if model is not None:
                        try:
                            anomaly_score = model.decision_function(features)[0]
                            # Convert: lower anomaly_score -> higher risk
                            risk_score = float(
                                max(0.0, min(1.0, (0.5 - anomaly_score) * 2))
                            )
                        except Exception:
                            # Fallback: heuristic scoring
                            risk_score = 0.3 + 0.15 * len(detail_flags)
                    else:
                        risk_score = 0.3 + 0.15 * len(detail_flags)
                else:
                    # Not enough data for ML, use heuristics
                    risk_score = 0.3 + 0.15 * len(detail_flags)

            # Clamp risk score
            risk_score = float(max(0.0, min(1.0, risk_score)))
            is_anomaly = risk_score >= 0.6

            # Record this observation for future baseline
            # (must be done outside the lock context above since record acquires lock)
            pass  # will record after releasing

        # Record outside lock (record acquires its own lock)
        self.record(
            agent_id=agent_id,
            api_timing_ms=api_timing_ms,
            decision_pattern=decision_pattern,
            request_structure_hash=request_structure_hash,
        )

        return {
            "risk_score": risk_score,
            "confidence": float(confidence),
            "is_anomaly": is_anomaly,
            "details": detail_flags,
            "agent_id": agent_id,
        }


agent_fingerprinter = AgentBehavioralFingerprint()


# =============================================================================
# Mandate Verifier
# =============================================================================


class MandateVerifier:
    """Stateless mandate compliance checker for agent transactions.

    Verifies whether a transaction falls within an agent's authorized scope.
    Mandates define constraints: spending limits, merchant whitelists/blacklists,
    time windows, and geographic restrictions. The mandate is passed per-call
    by the orchestrating agent.
    """

    def verify(
        self, transaction: Dict[str, Any], mandate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check transaction against mandate constraints.

        Args:
            transaction: Transaction data with amount, merchant, location, timestamp.
            mandate: Constraint dict with optional keys: max_amount, daily_limit,
                allowed_merchants, blocked_merchants, allowed_locations,
                time_window (start/end HH:MM).

        Returns:
            Dict with compliant (bool), violations (list), drift_score (0-1),
            mandate_utilization (dict), and checks_performed (int).
        """
        violations: List[str] = []
        checks = 0
        utilization: Dict[str, float] = {}

        amount = float(transaction.get("amount", 0.0))
        merchant = str(transaction.get("merchant", "")).lower()
        location = str(transaction.get("location", "")).lower()

        # --- Amount check ---
        max_amount = mandate.get("max_amount")
        if max_amount is not None:
            checks += 1
            max_amount = float(max_amount)
            utilization["amount_pct"] = amount / max_amount if max_amount > 0 else 0.0
            if amount > max_amount:
                violations.append(f"amount_exceeded: {amount} > {max_amount}")

        # --- Daily limit check ---
        daily_limit = mandate.get("daily_limit")
        if daily_limit is not None:
            checks += 1
            daily_limit = float(daily_limit)
            utilization["daily_pct"] = amount / daily_limit if daily_limit > 0 else 0.0
            if amount > daily_limit:
                violations.append(f"daily_limit_exceeded: {amount} > {daily_limit}")

        # --- Blocked merchants ---
        blocked = mandate.get("blocked_merchants")
        if blocked is not None:
            checks += 1
            blocked_lower = [m.lower() for m in blocked]
            if merchant in blocked_lower:
                violations.append(f"blocked_merchant: {merchant}")

        # --- Allowed merchants ---
        allowed_merchants = mandate.get("allowed_merchants")
        if allowed_merchants is not None:
            checks += 1
            allowed_lower = [m.lower() for m in allowed_merchants]
            if merchant and merchant != "unknown" and merchant not in allowed_lower:
                violations.append(f"merchant_not_allowed: {merchant}")

        # --- Allowed locations ---
        allowed_locations = mandate.get("allowed_locations")
        if allowed_locations is not None:
            checks += 1
            allowed_loc_lower = [loc.lower() for loc in allowed_locations]
            if location and location != "unknown" and location not in allowed_loc_lower:
                violations.append(f"location_not_allowed: {location}")

        # --- Time window ---
        time_window = mandate.get("time_window")
        if time_window is not None and "start" in time_window and "end" in time_window:
            checks += 1
            try:
                ts = transaction.get("timestamp", "")
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, datetime):
                    dt = ts
                else:
                    dt = datetime.now()

                txn_time = dt.strftime("%H:%M")
                start = time_window["start"]
                end = time_window["end"]

                if start <= end:
                    if not (start <= txn_time <= end):
                        violations.append(
                            f"outside_time_window: {txn_time} not in {start}-{end}"
                        )
                else:
                    # Overnight window (e.g., 22:00-06:00)
                    if end < txn_time < start:
                        violations.append(
                            f"outside_time_window: {txn_time} not in {start}-{end}"
                        )
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping time window check: {e}")

        drift_score = len(violations) / checks if checks > 0 else 0.0

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "drift_score": float(drift_score),
            "mandate_utilization": utilization,
            "checks_performed": checks,
        }


mandate_verifier = MandateVerifier()


# =============================================================================
# Collusion Detector
# =============================================================================


class CollusionDetector:
    """Graph-based detection of coordinated agent behavior.

    Maintains a directed graph of agent-to-agent transaction flows.
    Detects: circular money flows, temporal clustering (burst of agents
    hitting same target), and volume anomalies (coordinated spikes).
    """

    def __init__(self, max_nodes: int = 5000):
        self.graph = nx.DiGraph()
        self._node_order: deque = deque()
        self.max_nodes = max_nodes
        self._interactions: List[Dict[str, Any]] = []
        self._max_interactions = 50000

    def record_interaction(
        self,
        source: str,
        target: str,
        amount: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record an agent-to-agent or agent-to-merchant interaction.

        Args:
            source: Source agent ID.
            target: Target agent or merchant ID.
            amount: Transaction amount.
            timestamp: When the interaction occurred (defaults to now).
        """
        ts = timestamp or datetime.now()

        # Track node insertion order for eviction
        for node in (source, target):
            if node not in self.graph:
                self._node_order.append(node)
                self.graph.add_node(node)

        if self.graph.has_edge(source, target):
            edge = self.graph[source][target]
            edge["transaction_count"] += 1
            edge["total_amount"] += amount
            edge["timestamps"].append(ts)
        else:
            self.graph.add_edge(
                source,
                target,
                transaction_count=1,
                total_amount=amount,
                timestamps=[ts],
            )

        self._interactions.append(
            {"source": source, "target": target, "amount": amount, "timestamp": ts}
        )

        # Bound interaction history
        if len(self._interactions) > self._max_interactions:
            self._interactions = self._interactions[-self._max_interactions :]

        # Evict oldest nodes if over cap
        while len(self.graph.nodes) > self.max_nodes * 2:
            oldest = self._node_order.popleft()
            if oldest in self.graph:
                self.graph.remove_node(oldest)

    def detect(
        self, agent_ids: List[str], window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Detect collusion patterns among a set of agents.

        Args:
            agent_ids: Agent identifiers to analyze.
            window_seconds: Time window in seconds for temporal analysis.

        Returns:
            Dict with collusion_score (0-1), suspected_ring, evidence, graph_metrics.
        """
        if not agent_ids:
            return {
                "collusion_score": 0.0,
                "suspected_ring": [],
                "evidence": [],
                "graph_metrics": self._graph_metrics(),
            }

        evidence: List[str] = []
        suspected: set = set()
        score_components: List[float] = []

        # --- Circular flow detection ---
        subgraph_nodes = [a for a in agent_ids if a in self.graph]
        if len(subgraph_nodes) >= 2:
            subgraph = self.graph.subgraph(subgraph_nodes)
            try:
                cycles = list(nx.simple_cycles(subgraph))
                # Filter to cycles of length >= 3 (A->B->C->A)
                real_cycles = [c for c in cycles if len(c) >= 3]
                if real_cycles:
                    for cycle in real_cycles[:5]:  # Cap at 5 reported cycles
                        evidence.append(
                            f"circular_flow: {' -> '.join(cycle)} -> {cycle[0]}"
                        )
                        suspected.update(cycle)
                    score_components.append(min(1.0, len(real_cycles) * 0.3))
            except Exception as e:
                logger.debug(f"Cycle detection skipped: {e}")

        # --- Temporal clustering ---
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [
            i
            for i in self._interactions
            if i["timestamp"] >= cutoff and i["source"] in agent_ids
        ]

        # Group by target
        target_hits: Dict[str, List[str]] = {}
        for interaction in recent:
            t = interaction["target"]
            s = interaction["source"]
            if t not in target_hits:
                target_hits[t] = []
            if s not in target_hits[t]:
                target_hits[t].append(s)

        for target, sources in target_hits.items():
            if len(sources) >= 3:
                evidence.append(
                    f"temporal_cluster: {len(sources)} agents targeted {target} "
                    f"within {window_seconds}s"
                )
                suspected.update(sources)
                score_components.append(min(1.0, len(sources) * 0.15))

        # --- Volume anomaly ---
        for interaction in recent:
            src, tgt = interaction["source"], interaction["target"]
            if self.graph.has_edge(src, tgt):
                edge = self.graph[src][tgt]
                if edge["transaction_count"] >= 10:
                    recent_ts = [t for t in edge["timestamps"] if t >= cutoff]
                    if len(recent_ts) >= 10:
                        evidence.append(
                            f"volume_anomaly: {src} -> {tgt} had "
                            f"{len(recent_ts)} transactions in window"
                        )
                        suspected.add(src)
                        suspected.add(tgt)
                        score_components.append(min(1.0, len(recent_ts) * 0.05))
                        break  # One volume anomaly per detect call

        collusion_score = 0.0
        if score_components:
            collusion_score = float(min(1.0, max(score_components)))

        return {
            "collusion_score": collusion_score,
            "suspected_ring": sorted(suspected & set(agent_ids)),
            "evidence": evidence,
            "graph_metrics": self._graph_metrics(),
        }

    def _graph_metrics(self) -> Dict[str, Any]:
        """Return basic graph metrics."""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "interaction_count": len(self._interactions),
        }


collusion_detector = CollusionDetector()


# =============================================================================
# Agent Reputation Scorer
# =============================================================================


class AgentReputationScorer:
    """Longitudinal reputation scoring for AI agents.

    Computes a composite reputation from:
    - Trust score from identity registry (40%)
    - Transaction history length (25%)
    - Behavioral consistency from fingerprinter (25%)
    - Collusion safety from collusion detector (10%)
    """

    TRUST_WEIGHT = 0.4
    HISTORY_WEIGHT = 0.25
    CONSISTENCY_WEIGHT = 0.25
    COLLUSION_WEIGHT = 0.1
    HISTORY_CAP = 100  # transactions for full history credit

    def __init__(
        self,
        registry: Optional["AgentIdentityRegistry"] = None,
        fingerprinter: Optional["AgentBehavioralFingerprint"] = None,
        detector: Optional["CollusionDetector"] = None,
    ):
        self._registry = registry
        self._fingerprinter = fingerprinter
        self._detector = detector

    def score(self, agent_id: str) -> Dict[str, Any]:
        """Compute reputation score for an agent.

        Args:
            agent_id: Agent identifier to score.

        Returns:
            Dict with reputation_score (0-1), history_length, transaction_count,
            trust_score, behavioral_consistency, and components breakdown.
        """
        registry = self._registry or agent_registry
        fingerprinter = self._fingerprinter or agent_fingerprinter
        detector = self._detector or collusion_detector

        # --- Trust score from registry ---
        entry = registry.lookup(agent_id)
        if entry:
            trust = float(entry.get("trust_score", 0.5))
            txn_count = int(entry.get("transaction_count", 0))
            first_seen = entry.get("first_seen", "")
            last_seen = entry.get("last_seen", "")
        else:
            trust = 0.0
            txn_count = 0
            first_seen = ""
            last_seen = ""

        # --- History factor ---
        history_factor = (
            min(1.0, txn_count / self.HISTORY_CAP) if self.HISTORY_CAP > 0 else 0.0
        )

        # History length in days
        history_days = 0
        if first_seen and last_seen:
            try:
                fs = datetime.fromisoformat(first_seen)
                ls = datetime.fromisoformat(last_seen)
                history_days = max(0, (ls - fs).days)
            except (ValueError, TypeError) as e:
                logger.debug(f"History date parse failed: {e}")

        # --- Behavioral consistency ---
        consistency = 0.0
        baseline = fingerprinter.get_baseline(agent_id)
        if baseline and baseline.get("observation_count", 0) >= 10:
            # More observations = more consistent agent
            obs = baseline["observation_count"]
            # Low timing std relative to mean = consistent
            timing_std = baseline.get("timing_std", 0.0)
            timing_mean = baseline.get("timing_mean", 1.0)
            if timing_mean > 0:
                cv = timing_std / timing_mean  # coefficient of variation
                # CV < 0.3 = very consistent, CV > 1.0 = inconsistent
                consistency = max(0.0, min(1.0, 1.0 - cv))
            else:
                consistency = 0.5
            # Bonus for having many observations
            consistency = min(1.0, consistency * min(1.0, obs / 20))

        # --- Collusion safety ---
        collusion_safety = 1.0
        try:
            col_result = detector.detect([agent_id], window_seconds=86400)
            collusion_score = col_result.get("collusion_score", 0.0)
            collusion_safety = 1.0 - collusion_score
        except Exception as e:
            logger.warning(f"Collusion check failed for {agent_id}: {e}")

        # --- Weighted composite ---
        reputation = (
            self.TRUST_WEIGHT * trust
            + self.HISTORY_WEIGHT * history_factor
            + self.CONSISTENCY_WEIGHT * consistency
            + self.COLLUSION_WEIGHT * collusion_safety
        )
        reputation = float(max(0.0, min(1.0, reputation)))

        return {
            "reputation_score": reputation,
            "history_length": history_days,
            "transaction_count": txn_count,
            "trust_score": trust,
            "behavioral_consistency": consistency,
            "components": {
                "trust_score": trust,
                "history_factor": history_factor,
                "behavioral_consistency": consistency,
                "collusion_safety": collusion_safety,
            },
        }


reputation_scorer = AgentReputationScorer()


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
    key_str = json.dumps(key_fields, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


# =============================================================================
# Implementation Functions (testable, import these in tests)
# =============================================================================


def analyze_transaction_impl(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Implementation of comprehensive transaction fraud analysis"""
    import time as _time

    _start = _time.monotonic()
    try:
        # --- Security: sanitise inputs & enforce rate limit ---
        if sanitizer is not None and isinstance(transaction_data, dict):
            transaction_data = sanitizer.sanitize_dict(transaction_data)
            if isinstance(behavioral_data, dict):
                behavioral_data = sanitizer.sanitize_dict(behavioral_data)
        if rate_limiter is not None and isinstance(transaction_data, dict):
            user_key = str(transaction_data.get("user_id", "anonymous"))
            rl = rate_limiter.check_rate_limit(user_key)
            if not rl["allowed"]:
                return {
                    "error": "Rate limit exceeded",
                    "retry_after": rl["retry_after"],
                    "status": "rate_limited",
                }

        # Validate inputs
        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {
                "error": f"Invalid transaction data: {msg}",
                "status": "validation_failed",
            }

        if behavioral_data:
            valid, msg = validate_behavioral_data(behavioral_data)
            if not valid:
                return {
                    "error": f"Invalid behavioral data: {msg}",
                    "status": "validation_failed",
                }

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

        # Record transaction in user history for velocity analysis
        user_id = str(transaction_data.get("user_id", "anonymous"))
        user_history.record(user_id, transaction_data)

        # Velocity-based risk factors
        velocity_info = user_history.check_velocity(user_id)
        if velocity_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "high_transaction_velocity"
            )

        amount_info = user_history.check_amount_deviation(
            user_id, float(transaction_data.get("amount", 0))
        )
        if amount_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "unusual_amount_deviation"
            )

        geo_info = user_history.check_geographic_velocity(user_id)
        if geo_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "impossible_travel_detected"
            )

        merchant_info = user_history.check_merchant_diversity(user_id)
        if merchant_info["is_suspicious"]:
            transaction_result.setdefault("risk_factors", []).append(
                "high_merchant_diversity"
            )

        # Attach velocity details to results
        velocity_analysis = {
            "velocity": velocity_info,
            "amount_deviation": amount_info,
            "geographic": geo_info,
            "merchant_diversity": merchant_info,
        }

        # Generate feature-level explanation
        feature_explanation = None
        if fraud_explainer is not None:
            try:
                features = transaction_analyzer._extract_transaction_features(
                    transaction_data
                )
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
            "recommended_actions": [],
        }

        if feature_explanation:
            results["feature_explanation"] = feature_explanation

        # Add transaction risk factors
        risk_factors = transaction_result.get("risk_factors", [])
        results["detected_anomalies"].extend(risk_factors)

        results["velocity_analysis"] = velocity_analysis

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
                    results["overall_risk_score"] = min(
                        1.0, results["overall_risk_score"] + 0.2
                    )

            results["behavioral_analysis"] = behavioral_result

        # Determine risk level
        risk_score = results["overall_risk_score"]
        if risk_score >= 0.8:
            results["risk_level"] = "CRITICAL"
            results["recommended_actions"] = [
                "block_transaction",
                "require_manual_review",
            ]
        elif risk_score >= 0.6:
            results["risk_level"] = "HIGH"
            results["recommended_actions"] = [
                "require_additional_verification",
                "flag_for_review",
            ]
        elif risk_score >= 0.4:
            results["risk_level"] = "MEDIUM"
            results["recommended_actions"] = [
                "monitor_closely",
                "collect_additional_data",
            ]
        else:
            results["risk_level"] = "LOW"
            results["recommended_actions"] = ["allow_transaction"]

        # Generate explanation
        if results["detected_anomalies"]:
            explanation = f"Transaction flagged due to: {', '.join(results['detected_anomalies'])}"
        else:
            explanation = (
                "Transaction appears normal with no significant risk factors detected"
            )

        results["explanation"] = explanation
        results["analysis_timestamp"] = datetime.now().isoformat()
        results["model_version"] = "v2.3.0"

        # Record monitoring metrics
        if monitor is not None:
            elapsed_s = _time.monotonic() - _start
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
            "status": "analysis_failed",
        }


def detect_behavioral_anomaly_impl(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of behavioral biometrics anomaly detection"""
    try:
        # --- Security: sanitise inputs ---
        if sanitizer is not None and isinstance(behavioral_data, dict):
            behavioral_data = sanitizer.sanitize_dict(behavioral_data)

        valid, msg = validate_behavioral_data(behavioral_data)
        if not valid:
            return {
                "error": f"Invalid behavioral data: {msg}",
                "status": "validation_failed",
            }

        results = {
            "overall_anomaly_score": 0.0,
            "behavioral_analyses": {},
            "detected_anomalies": [],
            "confidence": 0.0,
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
                keystroke_result.get("risk_score", 0.0),
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
            "status": "analysis_failed",
        }


def assess_network_risk_impl(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of network-based risk assessment"""
    return network_analyzer.analyze_network_risk(entity_data)


def generate_risk_score_impl(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None,
    agent_behavior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Implementation of comprehensive risk score generation"""
    try:
        # Validate transaction data
        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {
                "error": f"Invalid transaction data: {msg}",
                "status": "validation_failed",
                "overall_risk_score": 0.0,
                "risk_level": "UNKNOWN",
            }

        # Classify traffic source
        classification = traffic_classifier.classify(transaction_data)
        traffic_source = classification["source"]
        is_agent_traffic = traffic_source == "agent"

        # Agent identity verification (only for agent traffic with identifier)
        identity_verification = None
        if is_agent_traffic and transaction_data.get("agent_identifier"):
            identity_verification = agent_verifier.verify(
                agent_identifier=str(transaction_data["agent_identifier"]),
                api_key=str(transaction_data.get("api_key", "")) or None,
                token=str(transaction_data.get("token", "")) or None,
            )

        # Perform all analyses
        transaction_analysis = transaction_analyzer.analyze_transaction(
            transaction_data
        )

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
            "recommended_actions": [],
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

        # Identity analysis (agent traffic only)
        if identity_verification:
            # Convert trust_score to risk_score: high trust = low risk
            id_trust = identity_verification.get("trust_score", 0.5)
            identity_risk_score = 1.0 - id_trust
            comprehensive_result["component_scores"]["identity"] = identity_risk_score
            scores.append(identity_risk_score)
            confidences.append(0.7 if identity_verification.get("verified") else 0.4)

            if not identity_verification.get("verified"):
                comprehensive_result["detected_anomalies"].append(
                    "unverified_agent_identity"
                )

            # Record transaction in registry
            agent_id = transaction_data.get("agent_identifier")
            if agent_id:
                agent_registry.record_transaction(str(agent_id))

        # Agent behavioral fingerprint (agent traffic only)
        if is_agent_traffic and transaction_data.get("agent_identifier"):
            behavior = agent_behavior or {}
            fp_result = agent_fingerprinter.analyze(
                agent_id=str(transaction_data["agent_identifier"]),
                api_timing_ms=float(behavior.get("api_timing_ms", 0.0)),
                decision_pattern=behavior.get("decision_pattern"),
                request_structure_hash=behavior.get("request_structure_hash"),
            )
            fp_score = fp_result.get("risk_score", 0.5)
            comprehensive_result["component_scores"]["behavioral_fingerprint"] = (
                fp_score
            )
            scores.append(fp_score)
            confidences.append(fp_result.get("confidence", 0.3))

            if fp_result.get("is_anomaly"):
                comprehensive_result["detected_anomalies"].append(
                    "agent_behavioral_fingerprint_anomaly"
                )

        # Calculate weighted overall score
        # Agent traffic: equal weighting across all available components
        # Human traffic: standard weights (transaction 50%, behavioral 30%, network 20%)
        if is_agent_traffic:
            n = len(scores)
            if n == 1:
                overall_score = scores[0]
            else:
                # For agent traffic, use equal weighting across all available components
                # This naturally adapts as more components (identity, network) are added
                overall_score = sum(scores) / n
        else:
            if len(scores) == 1:
                overall_score = scores[0]
            elif len(scores) == 2:
                overall_score = scores[0] * 0.6 + scores[1] * 0.4
            else:
                overall_score = scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2

        comprehensive_result["overall_risk_score"] = float(overall_score)
        comprehensive_result["confidence"] = float(np.mean(confidences))

        # Determine risk level and actions
        if overall_score >= 0.8:
            comprehensive_result["risk_level"] = "CRITICAL"
            comprehensive_result["recommended_actions"] = [
                "block_transaction",
                "require_manual_review",
                "investigate_account",
            ]
        elif overall_score >= 0.6:
            comprehensive_result["risk_level"] = "HIGH"
            comprehensive_result["recommended_actions"] = [
                "require_additional_verification",
                "flag_for_review",
                "monitor_account",
            ]
        elif overall_score >= 0.4:
            comprehensive_result["risk_level"] = "MEDIUM"
            comprehensive_result["recommended_actions"] = [
                "monitor_closely",
                "collect_additional_data",
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
        comprehensive_result["analysis_components"] = list(
            comprehensive_result["component_scores"].keys()
        )

        comprehensive_result["traffic_source"] = traffic_source
        comprehensive_result["agent_classification"] = classification

        return comprehensive_result

    except Exception as e:
        logger.error(f"Comprehensive risk assessment failed: {e}")
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed",
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
            "explanation_timestamp": datetime.now().isoformat(),
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

        # Agent-specific factor descriptions
        _AGENT_FACTOR_DESCRIPTIONS = {
            "unverified_agent_identity": (
                "Agent identity could not be verified through API key "
                "or JWT token validation"
            ),
            "behavioral_fingerprint_anomaly": (
                "Agent behavior deviates significantly from established "
                "baseline patterns"
            ),
            "agent_behavioral_fingerprint_anomaly": (
                "Agent behavioral fingerprint does not match historical patterns"
            ),
            "mandate_violation": (
                "Transaction violates the agent's authorized scope or spending mandate"
            ),
            "mandate_amount_exceeded": (
                "Transaction amount exceeds the agent's authorized spending limit"
            ),
            "mandate_blocked_merchant": (
                "Transaction targets a merchant on the agent's blocked list"
            ),
            "mandate_merchant_not_allowed": (
                "Transaction targets a merchant outside the agent's allowed list"
            ),
            "mandate_location_not_allowed": (
                "Transaction originates from a location outside the agent's "
                "authorized regions"
            ),
            "mandate_outside_time_window": (
                "Transaction occurred outside the agent's authorized operating hours"
            ),
            "mandate_daily_limit_exceeded": (
                "Transaction would push the agent past its daily spending limit"
            ),
            "agent_collusion_detected": (
                "Graph analysis detected coordinated behavior with other agents"
            ),
            "missing_agent_identifier": (
                "Agent transaction lacks an identifier for verification"
            ),
        }

        # Key contributing factors
        if detected_anomalies:
            explanation["key_factors"] = [
                {
                    "factor": anomaly,
                    "impact": "high"
                    if any(
                        k in anomaly
                        for k in ("unverified", "collusion", "mandate_violation")
                    )
                    else ("high" if "high" in anomaly else "medium"),
                    "description": _AGENT_FACTOR_DESCRIPTIONS.get(
                        anomaly,
                        f"Detected pattern: {anomaly.replace('_', ' ')}",
                    ),
                }
                for anomaly in detected_anomalies
            ]

        # Algorithm contributions
        component_scores = analysis_result.get("component_scores", {})
        traffic_source = analysis_result.get("traffic_source")

        # Agent-specific component weight map
        _AGENT_COMPONENT_WEIGHTS = {
            "transaction": 0.20,
            "identity": 0.25,
            "behavioral_fingerprint": 0.25,
            "mandate_compliance": 0.15,
            "collusion": 0.15,
            "reputation": 0.10,
            "behavioral": 0.0,  # not used for agent traffic
            "network": 0.15,
        }
        _HUMAN_COMPONENT_WEIGHTS = {
            "transaction": 0.50,
            "behavioral": 0.30,
            "network": 0.20,
        }

        if component_scores:
            is_agent = traffic_source == "agent"
            weight_map = (
                _AGENT_COMPONENT_WEIGHTS if is_agent else _HUMAN_COMPONENT_WEIGHTS
            )
            for component, score in component_scores.items():
                if not is_agent:
                    # Legacy weight logic for human traffic
                    if component == "transaction":
                        weight = (
                            0.5
                            if len(component_scores) == 3
                            else (0.6 if len(component_scores) == 2 else 1.0)
                        )
                    elif component == "behavioral":
                        weight = 0.3 if len(component_scores) == 3 else 0.4
                    else:  # network or other
                        weight = weight_map.get(component, 0.2)
                        if component == "network":
                            weight = 0.2 if len(component_scores) == 3 else 0.4
                else:
                    weight = weight_map.get(component, 0.1)

                explanation["algorithm_contributions"][component] = {
                    "score": float(score),
                    "weight": float(weight),
                    "contribution": f"{weight * 100:.1f}% of final decision",
                }

        # Include traffic source in explanation
        if traffic_source:
            explanation["traffic_source"] = traffic_source

        # Confidence breakdown
        explanation["confidence_breakdown"] = {
            "model_confidence": "High"
            if risk_score > 0.7
            else "Medium"
            if risk_score > 0.3
            else "Low",
            "data_quality": "Good" if len(detected_anomalies) > 0 else "Limited",
            "recommendation_strength": "Strong"
            if risk_score > 0.8 or risk_score < 0.2
            else "Moderate",
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
            explanation["explainability_method"] = analysis_result[
                "feature_explanation"
            ].get("method", "rule_based")

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
                summary = fraud_explainer.generate_summary(
                    explanation["feature_analysis"]
                )
                explanation["human_readable_summary"] = summary
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")

        return explanation

    except Exception as e:
        logger.error(f"Decision explanation failed: {e}")
        return {
            "error": str(e),
            "decision_summary": "Unable to generate explanation",
            "status": "explanation_failed",
        }


def classify_traffic_source_impl(
    transaction_data: Dict[str, Any], request_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Implementation of traffic source classification.

    Determines whether a transaction originates from a human user,
    an AI agent, or an unknown source.

    Args:
        transaction_data: Transaction data (may contain agent fields).
        request_metadata: Optional additional metadata (user_agent, is_agent, etc.).

    Returns:
        Classification result with source, confidence, agent_type, and signals.
    """
    try:
        if not isinstance(transaction_data, dict):
            return {
                "error": "transaction_data must be a dictionary",
                "status": "validation_failed",
                "source": "unknown",
                "confidence": 0.0,
            }

        # Merge transaction_data agent fields with request_metadata
        merged = {}
        for key in ("is_agent", "agent_identifier", "user_agent"):
            val = None
            if request_metadata and isinstance(request_metadata, dict):
                val = request_metadata.get(key)
            if val is None:
                val = transaction_data.get(key)
            if val is not None:
                merged[key] = val

        result = traffic_classifier.classify(merged)
        result["classification_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Traffic classification failed: {e}")
        return {
            "error": str(e),
            "source": "unknown",
            "confidence": 0.0,
            "status": "classification_failed",
        }


def verify_agent_identity_impl(
    agent_identifier: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Implementation of agent identity verification.

    Validates agent credentials against the identity registry and
    checks API key format and JWT token expiry.

    Args:
        agent_identifier: Agent identifier string.
        api_key: API key credential.
        token: JWT-style bearer token.

    Returns:
        Verification result with verified status, identity, trust_score, warnings.
    """
    try:
        result = agent_verifier.verify(
            agent_identifier=str(agent_identifier)
            if agent_identifier is not None
            else None,
            api_key=str(api_key) if api_key is not None else None,
            token=str(token) if token is not None else None,
        )
        result["verification_timestamp"] = datetime.now().isoformat()
        return result
    except Exception as e:
        logger.error(f"Agent identity verification failed: {e}")
        return {
            "error": str(e),
            "verified": False,
            "trust_score": 0.0,
            "warnings": ["verification_error"],
            "status": "verification_failed",
        }


def analyze_agent_transaction_impl(
    transaction_data: Dict[str, Any],
    agent_behavior: Optional[Dict[str, Any]] = None,
    mandate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Specialized transaction analysis for agent-initiated transactions.

    Combines traffic classification, agent identity verification, behavioral
    fingerprinting, mandate verification, and standard transaction analysis
    into a single pipeline optimized for AI agent traffic.

    Args:
        transaction_data: Transaction details. Should include ``is_agent`` and
            ``agent_identifier`` fields.  May also include ``api_key`` and
            ``token`` for identity verification.
        agent_behavior: Optional behavioral observation data with keys:
            ``api_timing_ms``, ``decision_pattern``, ``request_structure_hash``.
        mandate: Optional mandate constraints dict with keys: max_amount,
            daily_limit, allowed_merchants, blocked_merchants, allowed_locations,
            time_window (start/end HH:MM).

    Returns:
        Dict with risk_score, anomalies, fingerprint_match,
        mandate_compliance, identity_verified, traffic_source, and
        analysis_timestamp.
    """
    try:
        # --- Input validation ---
        if not isinstance(transaction_data, dict):
            return {
                "error": "transaction_data must be a dictionary",
                "status": "validation_failed",
            }

        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {
                "error": f"Invalid transaction data: {msg}",
                "status": "validation_failed",
            }

        anomalies: List[str] = []

        # --- Traffic classification ---
        classification = traffic_classifier.classify(transaction_data)
        traffic_source = classification["source"]

        # --- Identity verification ---
        agent_id = transaction_data.get("agent_identifier")
        identity_verified = False
        identity_trust = 0.0

        if agent_id:
            id_result = agent_verifier.verify(
                agent_identifier=str(agent_id),
                api_key=str(transaction_data.get("api_key", "")) or None,
                token=str(transaction_data.get("token", "")) or None,
            )
            identity_verified = id_result.get("verified", False)
            identity_trust = id_result.get("trust_score", 0.0)
            if not identity_verified:
                anomalies.append("unverified_agent_identity")
        else:
            anomalies.append("missing_agent_identifier")

        # --- Behavioral fingerprint ---
        fingerprint_score = 0.5  # neutral default
        fingerprint_confidence = 0.0
        behavior = agent_behavior or {}
        api_timing = float(behavior.get("api_timing_ms", 0.0))
        decision_pattern = behavior.get("decision_pattern")
        request_hash = behavior.get("request_structure_hash")

        if agent_id:
            fp_result = agent_fingerprinter.analyze(
                agent_id=str(agent_id),
                api_timing_ms=api_timing,
                decision_pattern=decision_pattern,
                request_structure_hash=request_hash,
            )
            fingerprint_score = fp_result.get("risk_score", 0.5)
            fingerprint_confidence = fp_result.get("confidence", 0.0)
            if fp_result.get("is_anomaly"):
                anomalies.append("behavioral_fingerprint_anomaly")
            anomalies.extend(f"fingerprint_{d}" for d in fp_result.get("details", []))

        # --- Mandate verification ---
        mandate_compliance = 1.0  # default: no mandate = fully compliant
        if mandate:
            mandate_result = mandate_verifier.verify(transaction_data, mandate)
            mandate_compliance = 1.0 - mandate_result.get("drift_score", 0.0)
            if not mandate_result.get("compliant", True):
                anomalies.append("mandate_violation")
                anomalies.extend(
                    f"mandate_{v.split(':')[0]}"
                    for v in mandate_result.get("violations", [])
                )

        # --- Standard transaction analysis ---
        txn_result = transaction_analyzer.analyze_transaction(transaction_data)
        txn_risk = txn_result.get("risk_score", 0.0)
        anomalies.extend(txn_result.get("risk_factors", []))

        # --- Composite risk score ---
        # Weights: transaction 35%, identity 25%, fingerprint 25%, base 15%
        identity_risk = 1.0 - identity_trust  # high trust -> low risk
        base_risk = 0.2 if traffic_source == "agent" else 0.0

        risk_score = (
            txn_risk * 0.35
            + identity_risk * 0.25
            + fingerprint_score * 0.25
            + base_risk * 0.15
        )
        risk_score = float(max(0.0, min(1.0, risk_score)))

        # Fingerprint match = 1 - fingerprint_score (high match = low risk)
        fingerprint_match = float(max(0.0, min(1.0, 1.0 - fingerprint_score)))

        return {
            "risk_score": risk_score,
            "anomalies": anomalies,
            "fingerprint_match": fingerprint_match,
            "fingerprint_confidence": fingerprint_confidence,
            "mandate_compliance": mandate_compliance,
            "identity_verified": identity_verified,
            "identity_trust_score": identity_trust,
            "traffic_source": traffic_source,
            "component_scores": {
                "transaction": txn_risk,
                "identity": identity_risk,
                "fingerprint": fingerprint_score,
            },
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Agent transaction analysis failed: {e}")
        return {
            "error": str(e),
            "risk_score": 0.0,
            "anomalies": [],
            "fingerprint_match": 0.0,
            "mandate_compliance": 0.0,
            "status": "analysis_failed",
        }


def verify_transaction_mandate_impl(
    transaction_data: Dict[str, Any],
    mandate: Dict[str, Any],
) -> Dict[str, Any]:
    """Check whether a transaction falls within an agent's authorized scope.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp).
        mandate: Constraint dict with optional keys: max_amount, daily_limit,
            allowed_merchants, blocked_merchants, allowed_locations,
            time_window (start/end HH:MM).

    Returns:
        Dict with compliant, violations, drift_score, mandate_utilization, and status.
    """
    try:
        if not isinstance(transaction_data, dict):
            return {
                "error": "transaction_data must be a dictionary",
                "status": "validation_failed",
            }

        if not isinstance(mandate, dict):
            return {
                "error": "mandate must be a dictionary",
                "status": "validation_failed",
            }

        result = mandate_verifier.verify(transaction_data, mandate)
        result["status"] = "verified"
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Mandate verification failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "compliant": False,
            "violations": [],
            "drift_score": 0.0,
        }


def detect_agent_collusion_impl(
    agent_ids: Any,
    window_seconds: int = 3600,
    transactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Detect coordinated agent behavior using graph analysis.

    Args:
        agent_ids: List of agent identifiers to analyze.
        window_seconds: Time window in seconds for temporal analysis.
        transactions: Optional list of transaction dicts to record before analysis.
            Each dict should have source, target, amount, and optional timestamp.

    Returns:
        Dict with collusion_score, suspected_ring, evidence, graph_metrics, and status.
    """
    try:
        if not isinstance(agent_ids, list):
            return {
                "error": "agent_ids must be a list",
                "status": "validation_failed",
            }

        # Record any provided transactions first
        if transactions:
            for txn in transactions:
                if isinstance(txn, dict):
                    ts = txn.get("timestamp")
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            ts = None
                    collusion_detector.record_interaction(
                        source=str(txn.get("source", "")),
                        target=str(txn.get("target", "")),
                        amount=float(txn.get("amount", 0.0)),
                        timestamp=ts,
                    )

        result = collusion_detector.detect(
            agent_ids=[str(a) for a in agent_ids],
            window_seconds=int(window_seconds),
        )
        result["status"] = "analyzed"
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Collusion detection failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "collusion_score": 0.0,
            "suspected_ring": [],
            "evidence": [],
        }


def score_agent_reputation_impl(
    agent_id: Any,
    time_window_days: int = 30,
) -> Dict[str, Any]:
    """Compute longitudinal reputation score for an AI agent.

    Args:
        agent_id: Agent identifier to score.
        time_window_days: Time window in days for reputation analysis (default 30).

    Returns:
        Dict with reputation_score, history_length, transaction_count,
        trust_score, behavioral_consistency, components, and status.
    """
    try:
        if not agent_id or not isinstance(agent_id, str):
            return {
                "error": "agent_id must be a non-empty string",
                "status": "validation_failed",
            }

        result = reputation_scorer.score(str(agent_id))
        result["status"] = "scored"
        result["agent_id"] = str(agent_id)
        result["analysis_timestamp"] = datetime.now().isoformat()
        return result

    except Exception as e:
        logger.error(f"Reputation scoring failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "reputation_score": 0.0,
        }


def analyze_batch_impl(
    transactions: List[Dict[str, Any]], use_cache: bool = True
) -> Dict[str, Any]:
    """Analyze a batch of transactions and return aggregated results."""
    import time as _time

    _start = _time.monotonic()

    if not isinstance(transactions, list):
        return {"error": "transactions must be a list", "status": "validation_failed"}

    if len(transactions) == 0:
        return {"error": "transactions list is empty", "status": "validation_failed"}

    if len(transactions) > 1000:
        return {
            "error": "batch size exceeds maximum of 1000",
            "status": "validation_failed",
        }

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
        "version": "2.3.0",
        "models": {
            "isolation_forest": transaction_analyzer.isolation_forest is not None,
            "feature_engineer": transaction_analyzer.feature_engineer is not None,
            "autoencoder": transaction_analyzer.autoencoder is not None,
            "explainer": fraud_explainer is not None,
            "feature_count": len(transaction_analyzer.feature_engineer.feature_names),
            "model_source": transaction_analyzer._model_source,
        },
        "explainability": {
            "available": EXPLAINABILITY_AVAILABLE,
            "shap_available": SHAP_AVAILABLE,
            "explainer_loaded": fraud_explainer is not None,
            "fallback_mode": (
                getattr(fraud_explainer, "fallback_mode", None)
                if fraud_explainer is not None
                else None
            ),
        },
        "cache": {
            "size": prediction_cache.size(),
            "capacity": prediction_cache.capacity,
            "hit_rate": (
                _inference_stats["cache_hits"] / _inference_stats["total_predictions"]
                if _inference_stats["total_predictions"] > 0
                else 0.0
            ),
        },
        "inference": {
            "total_predictions": _inference_stats["total_predictions"],
            "batch_predictions": _inference_stats["batch_predictions"],
        },
        "synthetic_data": {
            "available": SYNTHETIC_DATA_AVAILABLE,
            "integration_loaded": synthetic_data_integration is not None,
            "output_dir": (
                str(synthetic_data_integration.output_dir)
                if synthetic_data_integration is not None
                else None
            ),
        },
        "security_utils": {
            "available": SECURITY_UTILS_AVAILABLE,
            "sanitizer_loaded": sanitizer is not None,
            "rate_limiter": (
                rate_limiter.get_status() if rate_limiter is not None else None
            ),
        },
        "user_history": user_history.get_stats(),
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
                )
                if transaction_analyzer.feature_engineer
                else 0,
                "feature_names": (
                    transaction_analyzer.feature_engineer.feature_names
                    if transaction_analyzer.feature_engineer
                    else []
                ),
            },
            "autoencoder": {
                "loaded": transaction_analyzer.autoencoder is not None,
                "available": AUTOENCODER_AVAILABLE,
                "fallback_mode": (
                    getattr(transaction_analyzer.autoencoder, "fallback_mode", None)
                    if transaction_analyzer.autoencoder
                    else None
                ),
                "contamination": (
                    getattr(transaction_analyzer.autoencoder, "contamination", None)
                    if transaction_analyzer.autoencoder
                    else None
                ),
            },
            "explainer": {
                "loaded": fraud_explainer is not None,
                "available": EXPLAINABILITY_AVAILABLE,
                "shap_available": SHAP_AVAILABLE,
                "fallback_mode": (
                    getattr(fraud_explainer, "fallback_mode", None)
                    if fraud_explainer is not None
                    else None
                ),
                "method": (
                    "SHAP"
                    if (
                        fraud_explainer is not None
                        and not getattr(fraud_explainer, "fallback_mode", True)
                    )
                    else "Feature Importance"
                    if fraud_explainer is not None
                    else "unavailable"
                ),
            },
        },
        "ensemble_weights": transaction_analyzer._ensemble_weights,
        "synthetic_data": {
            "available": SYNTHETIC_DATA_AVAILABLE,
            "integration_loaded": synthetic_data_integration is not None,
            "output_dir": (
                str(synthetic_data_integration.output_dir)
                if synthetic_data_integration is not None
                else None
            ),
        },
        "saved_models": {
            "isolation_forest": str(iso_path) if iso_path.exists() else None,
            "feature_engineer": str(fe_path) if fe_path.exists() else None,
        },
        "model_dir": str(model_dir),
        "timestamp": datetime.now().isoformat(),
    }


def generate_synthetic_dataset_impl(
    num_transactions: int = 10000,
    fraud_percentage: float = 5.0,
    include_behavioral: bool = True,
    include_network: bool = True,
    output_format: str = "csv",
) -> Dict[str, Any]:
    """Generate a synthetic fraud detection dataset for testing and evaluation.

    Args:
        num_transactions: Total number of transactions to generate.
        fraud_percentage: Percentage of transactions that should be fraudulent (0-100).
        include_behavioral: Include behavioral biometrics data in the dataset.
        include_network: Include network relationship data in the dataset.
        output_format: Output file format, either 'csv' or 'json'.

    Returns:
        Dataset generation results with file paths, fraud distribution, and schema compliance.
    """
    if not SYNTHETIC_DATA_AVAILABLE or synthetic_data_integration is None:
        return {
            "error": "Synthetic data integration not available. "
            "Install pandas and numpy to enable synthetic data generation.",
            "status": "unavailable",
            "synthetic_data_available": False,
        }

    # Validate inputs
    if num_transactions < 1:
        return {
            "error": "num_transactions must be at least 1",
            "status": "validation_failed",
        }
    if num_transactions > 1_000_000:
        return {
            "error": "num_transactions exceeds maximum of 1,000,000",
            "status": "validation_failed",
        }
    if fraud_percentage < 0 or fraud_percentage > 100:
        return {
            "error": "fraud_percentage must be between 0 and 100",
            "status": "validation_failed",
        }
    if output_format not in ("csv", "json"):
        return {
            "error": "output_format must be 'csv' or 'json'",
            "status": "validation_failed",
        }

    try:
        result = synthetic_data_integration.generate_comprehensive_test_dataset(
            num_transactions=num_transactions,
            fraud_percentage=fraud_percentage,
            include_behavioral=include_behavioral,
            include_network=include_network,
            output_format=output_format,
        )
        return result
    except Exception as e:
        logger.error(f"Synthetic dataset generation failed: {e}")
        return {
            "error": str(e),
            "status": "generation_failed",
        }


def analyze_dataset_impl(
    dataset_path: str,
    fraud_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Analyze a stored dataset (CSV or JSON) for fraud patterns.

    Reads each transaction from the file and runs it through the active
    TransactionAnalyzer, aggregating risk distribution and flagging
    high-risk transactions.

    Args:
        dataset_path: Path to the dataset file (CSV or JSON).
        fraud_threshold: Risk score threshold for flagging transactions (0.0-1.0).

    Returns:
        Analysis results with risk distribution, flagged transactions,
        and optional performance metrics when ground truth labels exist.
    """
    import pandas as pd

    # Validate inputs
    if not dataset_path:
        return {"error": "dataset_path is required", "status": "validation_failed"}
    if fraud_threshold < 0.0 or fraud_threshold > 1.0:
        return {
            "error": "fraud_threshold must be between 0.0 and 1.0",
            "status": "validation_failed",
        }

    data_file = Path(dataset_path)
    if not data_file.exists():
        return {
            "error": f"Dataset file not found: {dataset_path}",
            "status": "file_not_found",
        }

    try:
        # Load dataset
        suffix = data_file.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".json":
            df = pd.read_json(dataset_path)
        else:
            return {
                "error": "Unsupported file format. Use CSV or JSON.",
                "status": "unsupported_format",
            }

        if df.empty:
            return {
                "error": "Dataset is empty",
                "status": "empty_dataset",
            }

        total_transactions = len(df)
        flagged_transactions: List[Dict[str, Any]] = []
        risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        # Analyze each transaction
        for idx, row in df.iterrows():
            txn_data = row.to_dict()

            result = transaction_analyzer.analyze_transaction(txn_data)
            risk_score = result.get("risk_score", 0.0)

            # Categorize risk
            if risk_score >= 0.8:
                risk_level = "critical"
            elif risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

            risk_distribution[risk_level] += 1

            # Flag high-risk transactions
            if risk_score >= fraud_threshold:
                flagged_transactions.append(
                    {
                        "transaction_id": txn_data.get("transaction_id", f"txn_{idx}"),
                        "risk_score": float(risk_score),
                        "risk_level": risk_level,
                        "risk_factors": result.get("risk_factors", []),
                        "actual_fraud": txn_data.get("is_fraud", None),
                    }
                )

        # Calculate performance metrics if ground truth is available
        performance_metrics = None
        if "is_fraud" in df.columns and "transaction_id" in df.columns:
            performance_metrics = _calculate_performance_metrics(
                df,
                flagged_transactions,
            )

        return {
            "dataset_info": {
                "file_path": dataset_path,
                "total_transactions": total_transactions,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "fraud_analysis": {
                "flagged_transactions": len(flagged_transactions),
                "fraud_rate_percent": (
                    round(len(flagged_transactions) / total_transactions * 100, 2)
                    if total_transactions > 0
                    else 0.0
                ),
                "fraud_threshold": fraud_threshold,
            },
            "risk_distribution": risk_distribution,
            "flagged_transactions": flagged_transactions,
            "performance_metrics": performance_metrics,
            "analysis_status": "success",
        }

    except Exception as e:
        logger.error(f"Dataset analysis failed: {e}")
        return {
            "error": str(e),
            "status": "analysis_failed",
            "dataset_path": dataset_path,
        }


def _calculate_performance_metrics(
    df: Any,
    flagged_transactions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate precision, recall, F1, and accuracy when ground truth labels exist."""
    flagged_ids = set(t["transaction_id"] for t in flagged_transactions)

    predictions = df["transaction_id"].isin(flagged_ids)
    actual = df["is_fraud"].astype(bool)

    tp = int(((predictions) & (actual)).sum())
    fp = int(((predictions) & (~actual)).sum())
    tn = int(((~predictions) & (~actual)).sum())
    fn = int(((~predictions) & (actual)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def run_benchmark_impl(
    num_transactions: int = 100,
    fraud_percentage: float = 10.0,
    include_latency_percentiles: bool = True,
) -> Dict[str, Any]:
    """Run a performance benchmark of the fraud detection pipeline.

    Generates synthetic transactions, runs each through the full analysis
    pipeline (Isolation Forest + Autoencoder + SHAP), and reports throughput,
    latency, and accuracy metrics.

    Args:
        num_transactions: Number of transactions to benchmark (10-5000).
        fraud_percentage: Percentage of fraudulent transactions (0-100).
        include_latency_percentiles: Include p50/p95/p99 latency stats.

    Returns:
        Dict with throughput, latency, accuracy, and pipeline configuration.
    """
    import time as _time

    # Validate inputs
    if not 10 <= num_transactions <= 5000:
        return {
            "error": "num_transactions must be between 10 and 5000",
            "status": "validation_failed",
        }
    if not 0 <= fraud_percentage <= 100:
        return {
            "error": "fraud_percentage must be between 0 and 100",
            "status": "validation_failed",
        }

    if not SYNTHETIC_DATA_AVAILABLE or synthetic_data_integration is None:
        return {
            "error": "Synthetic data generation not available (pandas required)",
            "status": "unavailable",
        }

    try:
        # Generate synthetic transactions in-memory
        integration = SyntheticDataIntegration()
        fraud_count = max(0, int(num_transactions * fraud_percentage / 100))
        legit_count = num_transactions - fraud_count

        transactions = []
        ground_truth = []

        for i in range(legit_count):
            txn = integration._generate_legitimate_transaction(i)
            transactions.append(txn)
            ground_truth.append(False)

        for i in range(fraud_count):
            fraud_patterns = integration.generate_fraud_patterns()
            fraud_types = list(fraud_patterns["transaction_fraud"].keys())
            fraud_type = fraud_types[i % len(fraud_types)]
            pattern = fraud_patterns["transaction_fraud"][fraud_type]
            txn = integration._generate_fraudulent_transaction(
                i + legit_count, fraud_type, pattern
            )
            transactions.append(txn)
            ground_truth.append(True)

        # Benchmark the pipeline
        latencies_ms = []
        risk_scores = []
        risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        flagged = []

        total_start = _time.monotonic()

        for idx, txn in enumerate(transactions):
            t0 = _time.monotonic()
            result = transaction_analyzer.analyze_transaction(txn)
            t1 = _time.monotonic()

            elapsed_ms = (t1 - t0) * 1000
            latencies_ms.append(elapsed_ms)

            score = result.get("risk_score", 0.0)
            risk_scores.append(score)

            if score >= 0.8:
                level = "CRITICAL"
            elif score >= 0.6:
                level = "HIGH"
            elif score >= 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"
            risk_distribution[level] += 1

            if score >= 0.6:
                flagged.append(idx)

        total_elapsed = (_time.monotonic() - total_start) * 1000

        # Throughput
        throughput_tps = (
            num_transactions / (total_elapsed / 1000) if total_elapsed > 0 else 0.0
        )

        # Latency stats
        latency_array = np.array(latencies_ms)
        latency_stats = {
            "avg_ms": round(float(np.mean(latency_array)), 3),
            "min_ms": round(float(np.min(latency_array)), 3),
            "max_ms": round(float(np.max(latency_array)), 3),
        }
        if include_latency_percentiles:
            latency_stats["p50_ms"] = round(float(np.percentile(latency_array, 50)), 3)
            latency_stats["p95_ms"] = round(float(np.percentile(latency_array, 95)), 3)
            latency_stats["p99_ms"] = round(float(np.percentile(latency_array, 99)), 3)

        # Accuracy metrics (ground truth available)
        flagged_set = set(flagged)
        tp = sum(
            1 for i in range(num_transactions) if i in flagged_set and ground_truth[i]
        )
        fp = sum(
            1
            for i in range(num_transactions)
            if i in flagged_set and not ground_truth[i]
        )
        tn = sum(
            1
            for i in range(num_transactions)
            if i not in flagged_set and not ground_truth[i]
        )
        fn = sum(
            1
            for i in range(num_transactions)
            if i not in flagged_set and ground_truth[i]
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Pipeline config
        pipeline_config = {
            "isolation_forest": transaction_analyzer.isolation_forest is not None,
            "autoencoder": transaction_analyzer.autoencoder is not None,
            "explainer": EXPLAINABILITY_AVAILABLE and fraud_explainer is not None,
            "model_source": transaction_analyzer._model_source,
            "ensemble_weights": transaction_analyzer._ensemble_weights,
        }

        return {
            "benchmark_config": {
                "num_transactions": num_transactions,
                "fraud_percentage": fraud_percentage,
                "actual_fraud_count": fraud_count,
                "actual_legit_count": legit_count,
            },
            "throughput": {
                "transactions_per_second": round(throughput_tps, 1),
                "total_time_ms": round(total_elapsed, 1),
            },
            "latency": latency_stats,
            "accuracy": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "accuracy": round((tp + tn) / num_transactions, 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "risk_distribution": risk_distribution,
            "pipeline": pipeline_config,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e), "status": "benchmark_failed"}


# =============================================================================
# MCP Tool Wrappers (thin delegates to _impl functions)
# =============================================================================


@_monitored("/analyze_transaction", "TOOL")
@mcp.tool()
def analyze_transaction(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None,
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
    return analyze_transaction_impl(
        transaction_data, include_behavioral, behavioral_data
    )


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
    network_data: Optional[Dict[str, Any]] = None,
    agent_behavior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive risk score combining all analysis methods.

    For human traffic: combines transaction, behavioral, and network analysis.
    For agent traffic: adds identity verification and behavioral fingerprinting.

    Args:
        transaction_data: Transaction details
        behavioral_data: Behavioral biometrics data (human traffic)
        network_data: Network connection data
        agent_behavior: Agent behavioral data with api_timing_ms, decision_pattern,
            request_structure_hash (agent traffic)

    Returns:
        Comprehensive risk assessment with detailed scoring
    """
    return generate_risk_score_impl(
        transaction_data, behavioral_data, network_data, agent_behavior
    )


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


@_monitored("/classify_traffic_source", "TOOL")
@mcp.tool()
def classify_traffic_source(
    transaction_data: Dict[str, Any], request_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify whether a transaction originates from a human, AI agent, or unknown source.

    Analyzes transaction metadata, User-Agent strings, and explicit agent flags
    to determine traffic source. Recognizes major agent commerce protocols:
    Stripe ACP, Visa TAP, Mastercard Agent Pay, Google AP2, PayPal Agent Ready,
    Coinbase x402, OpenAI Operator, and Anthropic Claude agents.

    Args:
        transaction_data: Transaction details (may include is_agent, agent_identifier, user_agent fields)
        request_metadata: Optional request metadata (user_agent, is_agent flag, agent_identifier)

    Returns:
        Classification with source (human/agent/unknown), confidence, agent_type, and signals
    """
    return classify_traffic_source_impl(transaction_data, request_metadata)


@_monitored("/verify_agent_identity", "TOOL")
@mcp.tool()
def verify_agent_identity(
    agent_identifier: Optional[str] = None,
    api_key: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify an AI agent's identity using available credentials.

    Validates agent credentials against the identity registry, checks API key
    format, and verifies JWT token expiry. Supports Stripe ACP, Visa TAP,
    Mastercard Agent Pay, Google AP2, and other agent commerce protocols.

    Args:
        agent_identifier: Agent identifier (e.g., 'stripe-acp:agent-123')
        api_key: API key credential for format validation
        token: JWT-style bearer token for expiry verification

    Returns:
        Verification result with verified status, identity details, trust score, and warnings
    """
    return verify_agent_identity_impl(agent_identifier, api_key, token)


@_monitored("/analyze_agent_transaction", "TOOL")
@mcp.tool()
def analyze_agent_transaction(
    transaction_data: Dict[str, Any],
    agent_behavior: Optional[Dict[str, Any]] = None,
    mandate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze an AI-agent-initiated transaction for fraud.

    Specialized analysis pipeline for agent transactions that combines traffic
    classification, agent identity verification (API key / JWT), behavioral
    fingerprinting (API timing consistency, decision patterns, request structure),
    and mandate compliance verification.
    Replaces human behavioral biometrics with agent-specific signals.

    Args:
        transaction_data: Transaction details including amount, merchant, location,
            timestamp, payment_method. Should include is_agent=True and
            agent_identifier for best results.  May also include api_key and token.
        agent_behavior: Optional agent behavioral data with api_timing_ms,
            decision_pattern, and request_structure_hash fields.
        mandate: Optional mandate constraints dict with keys: max_amount,
            daily_limit, allowed_merchants, blocked_merchants, allowed_locations,
            time_window (with start/end in HH:MM format).

    Returns:
        Analysis result with risk_score (0-1), anomalies list,
        fingerprint_match (0-1, higher is more consistent), mandate_compliance,
        identity_verified status, and per-component scores
    """
    return analyze_agent_transaction_impl(transaction_data, agent_behavior, mandate)


@_monitored("/verify_transaction_mandate", "TOOL")
@mcp.tool()
def verify_transaction_mandate(
    transaction_data: Dict[str, Any],
    mandate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Check whether a transaction falls within an agent's authorized scope.

    Validates transaction against mandate constraints including spending limits,
    merchant whitelists/blacklists, time windows, and geographic restrictions.
    Returns compliance status, violations list, and drift score.

    Args:
        transaction_data: Transaction details with amount, merchant, location, timestamp.
        mandate: Constraint dict with optional keys: max_amount, daily_limit,
            allowed_merchants, blocked_merchants, allowed_locations,
            time_window (with start/end in HH:MM format).

    Returns:
        Compliance result with compliant (bool), violations (list),
        drift_score (0-1, higher means more violations), and mandate_utilization
    """
    return verify_transaction_mandate_impl(transaction_data, mandate)


@_monitored("/detect_agent_collusion", "TOOL")
@mcp.tool()
def detect_agent_collusion(
    agent_ids: List[str],
    window_seconds: int = 3600,
    transactions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Detect coordinated agent behavior using graph analysis.

    Analyzes agent-to-agent transaction flows to detect collusion patterns:
    circular money flows (A->B->C->A), temporal clustering (multiple agents
    hitting same target in burst), and volume anomalies (sudden coordinated spikes).

    Args:
        agent_ids: List of agent identifiers to analyze for collusion.
        window_seconds: Time window in seconds for temporal analysis (default 3600).
        transactions: Optional list of transaction dicts to record before analysis.
            Each dict should have source, target, amount, and optional timestamp fields.

    Returns:
        Detection result with collusion_score (0-1), suspected_ring (list of agent IDs),
        evidence (list of findings), and graph_metrics
    """
    return detect_agent_collusion_impl(agent_ids, window_seconds, transactions)


@_monitored("/score_agent_reputation", "TOOL")
@mcp.tool()
def score_agent_reputation(
    agent_id: str,
    time_window_days: int = 30,
) -> Dict[str, Any]:
    """
    Compute longitudinal reputation score for an AI agent.

    Aggregates trust score from identity verification, transaction history length,
    behavioral consistency from fingerprinting, and collusion safety into a single
    reputation score. Higher scores indicate more trustworthy agents.

    Args:
        agent_id: Agent identifier to score (e.g., 'stripe-acp:agent-123').
        time_window_days: Time window in days for analysis (default 30).

    Returns:
        Reputation result with reputation_score (0-1, higher is better),
        history_length (days), transaction_count, trust_score,
        behavioral_consistency, and per-component breakdown
    """
    return score_agent_reputation_impl(agent_id, time_window_days)


@_monitored("/analyze_batch", "TOOL")
@mcp.tool()
def analyze_batch(
    transactions: List[Dict[str, Any]], use_cache: bool = True
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


@_monitored("/generate_synthetic_dataset", "TOOL")
@mcp.tool()
def generate_synthetic_dataset(
    num_transactions: int = 10000,
    fraud_percentage: float = 5.0,
    include_behavioral: bool = True,
    include_network: bool = True,
    output_format: str = "csv",
) -> Dict[str, Any]:
    """
    Generate a synthetic fraud detection dataset for testing and evaluation.

    Creates realistic transaction, behavioral, and network data with configurable
    fraud percentages and pattern distributions.

    Args:
        num_transactions: Total number of transactions to generate (1 - 1,000,000)
        fraud_percentage: Percentage of fraudulent transactions (0-100)
        include_behavioral: Include behavioral biometrics data
        include_network: Include network relationship data
        output_format: File format, 'csv' or 'json'

    Returns:
        Generation results with file paths, fraud distribution, and schema compliance
    """
    return generate_synthetic_dataset_impl(
        num_transactions,
        fraud_percentage,
        include_behavioral,
        include_network,
        output_format,
    )


@_monitored("/analyze_dataset", "TOOL")
@mcp.tool()
def analyze_dataset(
    dataset_path: str,
    fraud_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Analyze a stored dataset (CSV or JSON) for fraud patterns.

    Reads each transaction and runs it through the fraud detection models,
    aggregating risk distribution and flagging high-risk transactions.
    When ground truth labels exist, calculates precision, recall, F1, and accuracy.

    Args:
        dataset_path: Path to the dataset file (CSV or JSON)
        fraud_threshold: Risk score threshold for flagging transactions (0.0-1.0)

    Returns:
        Analysis results with risk distribution, flagged transactions,
        and optional performance metrics
    """
    return analyze_dataset_impl(dataset_path, fraud_threshold)


@_monitored("/run_benchmark", "TOOL")
@mcp.tool()
def run_benchmark(
    num_transactions: int = 100,
    fraud_percentage: float = 10.0,
    include_latency_percentiles: bool = True,
) -> Dict[str, Any]:
    """
    Run a performance benchmark of the fraud detection pipeline.

    Generates synthetic transactions and runs each through the full analysis
    pipeline (Isolation Forest + Autoencoder + SHAP explainability), measuring
    throughput, latency percentiles, and accuracy metrics.

    Args:
        num_transactions: Number of transactions to benchmark (10-5000, default 100)
        fraud_percentage: Percentage of fraudulent transactions (0-100, default 10.0)
        include_latency_percentiles: Include p50/p95/p99 latency stats (default True)

    Returns:
        Benchmark results with throughput (txn/sec), latency (avg/p50/p95/p99),
        accuracy (precision/recall/F1), risk distribution, and pipeline config
    """
    return run_benchmark_impl(
        num_transactions, fraud_percentage, include_latency_percentiles
    )


# =============================================================================
# Defense Compliance Modules (EO 13587, NITTF, SEAD 4/6, NIST 800-53)
# =============================================================================

# Graceful degradation for compliance modules
try:
    from compliance.insider_threat import InsiderThreatAssessor
    from compliance.siem_integration import SIEMIntegration
    from compliance.cleared_personnel import ClearedPersonnelAnalyzer
    from compliance.dashboard_metrics import ComplianceDashboard

    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    InsiderThreatAssessor = None  # type: ignore[assignment,misc]
    SIEMIntegration = None  # type: ignore[assignment,misc]
    ClearedPersonnelAnalyzer = None  # type: ignore[assignment,misc]
    ComplianceDashboard = None  # type: ignore[assignment,misc]

# Initialize compliance singletons (if available)
_insider_threat_assessor = InsiderThreatAssessor() if COMPLIANCE_AVAILABLE else None
_siem_integration = SIEMIntegration() if COMPLIANCE_AVAILABLE else None
_cleared_personnel_analyzer = ClearedPersonnelAnalyzer() if COMPLIANCE_AVAILABLE else None
_compliance_dashboard = ComplianceDashboard() if COMPLIANCE_AVAILABLE else None


def _compliance_not_available() -> Dict[str, Any]:
    """Return a standardized error when compliance modules are not loaded."""
    return {
        "error": "Defense compliance modules are not available",
        "detail": (
            "Install compliance module dependencies. Ensure the compliance/ "
            "package is on the Python path."
        ),
        "available": False,
    }


# --- Insider Threat Assessment Tool ---


def assess_insider_threat_impl(
    user_id: str,
    activity_data: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Implementation for assess_insider_threat tool."""
    if not COMPLIANCE_AVAILABLE or _insider_threat_assessor is None:
        return _compliance_not_available()

    try:
        # Update profile if provided
        if user_profile:
            _insider_threat_assessor.update_profile(
                user_id,
                role=user_profile.get("role"),
                department=user_profile.get("department"),
                clearance_level=user_profile.get("clearance_level"),
                authorized_resources=user_profile.get("authorized_resources"),
                work_hours=tuple(user_profile["work_hours"])
                if "work_hours" in user_profile else None,
            )

        # Run assessment
        result = _insider_threat_assessor.assess_user(user_id, activity_data)
        result["compliance_module"] = "insider_threat"
        result["available"] = True
        return result

    except Exception as e:
        logger.error("Insider threat assessment failed for %s: %s", user_id, e)
        return {
            "error": f"Assessment failed: {str(e)}",
            "user_id": user_id,
            "available": True,
        }


@_monitored("/assess_insider_threat", "TOOL")
@mcp.tool()
def assess_insider_threat(
    user_id: str,
    activity_data: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run an insider threat assessment on a user's activity data per EO 13587
    and NITTF guidance.

    Evaluates 28 behavioral indicators aligned with CNSSD 504 UAM requirements,
    calculates a weighted risk score (0-100), determines DHS NTAS-aligned threat
    level, and maps findings to NIST 800-53 controls and MITRE ATT&CK techniques.

    Args:
        user_id: Unique identifier for the user being assessed
        activity_data: Dictionary of current activity observations. Supported keys:
            - login_hour (int): Hour of login (0-23)
            - source_ip (str): Source IP address
            - login_success (bool): Whether login succeeded
            - data_volume_bytes (int): Data transfer volume
            - resource_id (str): Resource being accessed
            - accessed_classification (str): Classification level of accessed data
            - removable_media_type (str): Type of removable media detected
            - device_id (str): Device identifier
            - approved_devices (list): List of approved device IDs
            - security_bypass_actions (list): Detected bypass actions
            - travel_destination (str): Foreign travel destination
            - foreign_contacts_detected (list): Detected foreign contacts
            - reported_foreign_contacts (list): Previously reported contacts
            - financial_indicators (dict): Financial stress flags
            - disgruntlement_score (float): 0-1 disgruntlement score
            - hr_incidents (int): Number of HR incidents
            - ci_indicators (list): Counter-intelligence indicators
            - badge_anomalies (list): Physical access anomalies
            - personal_email_forwards (int): Count of personal email forwards
            - sensitive_attachments_forwarded (int): Sensitive attachment count
            - print_hour (int): Hour of print job
            - print_classification (str): Classification of printed document
            - vpn_location (str): VPN connection location
            - privilege_escalation_attempts (list): Escalation attempts
            - failed_login_count (int): Current failed login count
            - days_since_last_login (int): Days since last login
            - unauthorized_cloud_uploads (list): Cloud upload events
            - screen_capture_count (int): Screen capture count
            - unauthorized_encryption_tools (list): Unauthorized encryption
            - unauthorized_software (list): Unauthorized software
            - network_scanning_detected (bool): Network scanning flag
            - data_staging_detected (bool): Data staging flag
            - employment_status (str): Current employment status
            - concurrent_session_ips (list): IPs of concurrent sessions
            - accessed_compartments (list): Compartments accessed
            - authorized_compartments (list): Authorized compartments
            - covert_channel_indicators (list): Covert channel indicators
        user_profile: Optional profile data to set before assessment:
            - role (str): User's job role
            - department (str): User's department
            - clearance_level (str): Clearance level
            - authorized_resources (list): Authorized resource IDs
            - work_hours (list): [start_hour, end_hour]

    Returns:
        Assessment result with risk_score, threat_level, triggered_indicators,
        nist_control_violations, and recommended_actions
    """
    return assess_insider_threat_impl(user_id, activity_data, user_profile)


# --- SIEM Event Generation Tool ---


def generate_siem_events_impl(
    assessment_result: Dict[str, Any],
    output_formats: Optional[List[str]] = None,
    batch_export: bool = False,
    export_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Implementation for generate_siem_events tool."""
    if not COMPLIANCE_AVAILABLE or _siem_integration is None:
        return _compliance_not_available()

    try:
        if batch_export:
            filters = export_filters or {}
            result = _siem_integration.batch_export(
                start_time=filters.get("start_time"),
                end_time=filters.get("end_time"),
                user_id=filters.get("user_id"),
                min_severity=filters.get("min_severity"),
                output_format=filters.get("output_format", "json"),
            )
            result["compliance_module"] = "siem_integration"
            result["available"] = True
            return result

        result = _siem_integration.generate_events(
            assessment_result,
            output_formats=output_formats,
        )
        result["compliance_module"] = "siem_integration"
        result["available"] = True
        result["siem_stats"] = _siem_integration.get_stats()
        return result

    except Exception as e:
        logger.error("SIEM event generation failed: %s", e)
        return {
            "error": f"SIEM event generation failed: {str(e)}",
            "available": True,
        }


@_monitored("/generate_siem_events", "TOOL")
@mcp.tool()
def generate_siem_events(
    assessment_result: Dict[str, Any],
    output_formats: Optional[List[str]] = None,
    batch_export: bool = False,
    export_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Export fraud and insider threat events in defense-grade SIEM formats.

    Generates events in Common Event Format (CEF) for ArcSight, Log Event
    Extended Format (LEEF) for IBM QRadar, and Syslog RFC 5424 with structured
    data. Events are enriched with MITRE ATT&CK technique mappings, classified
    per DoD 8570/8140 incident categories, and run through correlation rules
    to detect multi-indicator attack patterns.

    Args:
        assessment_result: Result from assess_insider_threat or similar
            assessment containing risk_score, threat_level, triggered_indicators.
            Required keys: user_id, risk_score, threat_level, triggered_indicators.
        output_formats: List of formats to generate. Options: "cef", "leef", "syslog".
            Defaults to all three if not specified.
        batch_export: If True, export buffered events instead of generating new ones.
            Use export_filters to control the export.
        export_filters: Filters for batch export mode:
            - start_time (str): ISO format start time
            - end_time (str): ISO format end time
            - user_id (str): Filter by user
            - min_severity (str): Minimum severity (INFORMATIONAL/LOW/MEDIUM/HIGH/CRITICAL)
            - output_format (str): "json" or "csv"

    Returns:
        Generated events in requested formats with MITRE enrichment,
        DoD incident category, correlation alerts, and SIEM statistics
    """
    return generate_siem_events_impl(
        assessment_result, output_formats, batch_export, export_filters
    )


# --- Cleared Personnel Evaluation Tool ---


def evaluate_cleared_personnel_impl(
    person_id: str,
    activity_data: Dict[str, Any],
    clearance_info: Optional[Dict[str, Any]] = None,
    polygraph_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Implementation for evaluate_cleared_personnel tool."""
    if not COMPLIANCE_AVAILABLE or _cleared_personnel_analyzer is None:
        return _compliance_not_available()

    try:
        # Set clearance info if provided
        if clearance_info:
            _cleared_personnel_analyzer.set_clearance(
                person_id,
                level=clearance_info.get("level", "UNCLASSIFIED"),
                status=clearance_info.get("status", "FINAL"),
                compartments=clearance_info.get("compartments"),
                sap_accesses=clearance_info.get("sap_accesses"),
                date_granted=clearance_info.get("date_granted"),
                date_expires=clearance_info.get("date_expires"),
                sponsoring_agency=clearance_info.get("sponsoring_agency", ""),
                investigation_type=clearance_info.get("investigation_type", ""),
            )

        # Record polygraph if provided
        if polygraph_info:
            _cleared_personnel_analyzer.record_polygraph(
                person_id,
                polygraph_type=polygraph_info.get("type", "CI"),
                date=polygraph_info.get("date", ""),
                result=polygraph_info.get("result", ""),
                next_due=polygraph_info.get("next_due"),
            )

        # Run evaluation
        result = _cleared_personnel_analyzer.evaluate_cleared_personnel(
            person_id, activity_data
        )
        result["compliance_module"] = "cleared_personnel"
        result["available"] = True
        return result

    except Exception as e:
        logger.error(
            "Cleared personnel evaluation failed for %s: %s", person_id, e
        )
        return {
            "error": f"Evaluation failed: {str(e)}",
            "person_id": person_id,
            "available": True,
        }


@_monitored("/evaluate_cleared_personnel", "TOOL")
@mcp.tool()
def evaluate_cleared_personnel(
    person_id: str,
    activity_data: Dict[str, Any],
    clearance_info: Optional[Dict[str, Any]] = None,
    polygraph_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run cleared personnel analytics per SEAD 4/6 for users with
    security clearances.

    Performs Continuous Evaluation (CE) per SEAD 6, Whole Person Assessment
    using the 13 adjudicative guidelines from SEAD 4, need-to-know
    verification per NIST 800-53 AC-3/AC-25, SF-86 consistency checks,
    reporting compliance validation, and polygraph compliance tracking.

    Args:
        person_id: Unique identifier for the cleared person
        activity_data: Dictionary with evaluation data:
            - accessed_classification (str): Classification of accessed data
                (UNCLASSIFIED, CUI, CONFIDENTIAL, SECRET, TOP SECRET, SCI)
            - accessed_compartments (list): SCI compartments accessed
            - accessed_resource (str): Resource identifier
            - justification (str): Need-to-know justification
            - foreign_travel (list): Travel entries with destination, travel_date,
                reported_date fields
            - foreign_contacts (list): Contact entries with name, country,
                reported fields
            - financial_changes (list): Financial events with type, detail,
                reported fields
            - criminal_events (list): Criminal events with type, detail
            - public_records_flags (list): Public record flags with finding,
                detail, severity
            - guideline_data (dict): Data for SEAD 4 adjudicative guidelines
                keyed by letter (A-M) with boolean indicator fields
            - sf86_current (dict): Current SF-86 data for consistency check
        clearance_info: Optional clearance data to set/update:
            - level (str): Clearance level
            - status (str): PENDING/INTERIM/FINAL/SUSPENDED/REVOKED/EXPIRED
            - compartments (list): SCI compartments
            - sap_accesses (list): SAP accesses
            - date_granted (str): ISO date granted
            - date_expires (str): ISO date expires
            - sponsoring_agency (str): Sponsoring agency
            - investigation_type (str): Investigation type (T3, T5, SSBI)
        polygraph_info: Optional polygraph data:
            - type (str): CI, FS, or LIFESTYLE
            - date (str): Examination date
            - result (str): Examination result
            - next_due (str): Next examination due date

    Returns:
        Evaluation result with clearance_summary, overall_risk_score,
        findings, need_to_know_verification, continuous_evaluation,
        whole_person_assessment, and recommended_actions
    """
    return evaluate_cleared_personnel_impl(
        person_id, activity_data, clearance_info, polygraph_info
    )


# --- Compliance Dashboard Tool ---


def get_compliance_dashboard_impl(
    include_maturity: bool = True,
    include_kris: bool = True,
    include_compliance_posture: bool = True,
    include_model_drift: bool = True,
    export_format: Optional[str] = None,
    include_history: bool = False,
) -> Dict[str, Any]:
    """Implementation for get_compliance_dashboard tool."""
    if not COMPLIANCE_AVAILABLE:
        return _compliance_not_available()

    if _compliance_dashboard is None:
        return _compliance_not_available()

    try:
        # If export requested, use the export method
        if export_format:
            result = _compliance_dashboard.export_metrics(
                output_format=export_format,
                include_history=include_history,
            )
            result["compliance_module"] = "dashboard_metrics"
            result["available"] = True
            return result

        # Build dashboard response
        dashboard: Dict[str, Any] = {
            "compliance_module": "dashboard_metrics",
            "available": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if include_maturity:
            dashboard["program_maturity"] = _compliance_dashboard.calculate_maturity_score()

        if include_kris:
            dashboard["key_risk_indicators"] = _compliance_dashboard.calculate_kris()

        if include_compliance_posture:
            dashboard["compliance_posture"] = _compliance_dashboard.calculate_compliance_posture()

        if include_model_drift:
            dashboard["model_drift"] = _compliance_dashboard.detect_model_drift()

        # Include component stats if available
        component_stats = {}
        if _insider_threat_assessor:
            component_stats["insider_threat"] = _insider_threat_assessor.get_assessment_stats()
        if _siem_integration:
            component_stats["siem"] = _siem_integration.get_stats()
        if _cleared_personnel_analyzer:
            component_stats["cleared_personnel"] = _cleared_personnel_analyzer.get_stats()
        component_stats["dashboard"] = _compliance_dashboard.get_stats()
        dashboard["component_stats"] = component_stats

        # Generate executive summary if all components requested
        if all([include_maturity, include_kris, include_compliance_posture]):
            dashboard["executive_summary"] = _compliance_dashboard.generate_executive_summary(
                insider_threat_stats=component_stats.get("insider_threat"),
                siem_stats=component_stats.get("siem"),
                personnel_stats=component_stats.get("cleared_personnel"),
            )

        return dashboard

    except Exception as e:
        logger.error("Compliance dashboard generation failed: %s", e)
        return {
            "error": f"Dashboard generation failed: {str(e)}",
            "available": True,
        }


@_monitored("/get_compliance_dashboard", "TOOL")
@mcp.tool()
def get_compliance_dashboard(
    include_maturity: bool = True,
    include_kris: bool = True,
    include_compliance_posture: bool = True,
    include_model_drift: bool = True,
    export_format: Optional[str] = None,
    include_history: bool = False,
) -> Dict[str, Any]:
    """
    Get defense compliance metrics and dashboard data.

    Provides insider threat program maturity scoring per NITTF framework
    (Initial through Optimizing), Key Risk Indicators with trend analysis,
    NIST 800-53 PS/PE/AC compliance posture scoring, MTTD/MTTR tracking,
    false positive rate monitoring, model drift detection, and executive
    summary reports suitable for CSO/CISO briefings.

    Args:
        include_maturity: Include NITTF maturity model scoring (default True)
        include_kris: Include Key Risk Indicators with trends (default True)
        include_compliance_posture: Include NIST 800-53 control scoring (default True)
        include_model_drift: Include model drift detection (default True)
        export_format: If set, export metrics in this format ("json" or "csv")
            instead of generating dashboard
        include_history: Include historical KRI data in export (default False)

    Returns:
        Dashboard data with program_maturity, key_risk_indicators,
        compliance_posture, model_drift, component_stats, and
        executive_summary (when all sections included)
    """
    return get_compliance_dashboard_impl(
        include_maturity, include_kris, include_compliance_posture,
        include_model_drift, export_format, include_history,
    )


# --- Threat Referral Generation Tool ---


def generate_threat_referral_impl(
    user_id: str,
    referral_type: str = "insider_threat",
    assessment_id: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Implementation for generate_threat_referral tool."""
    if not COMPLIANCE_AVAILABLE:
        return _compliance_not_available()

    try:
        if referral_type == "insider_threat":
            if _insider_threat_assessor is None:
                return _compliance_not_available()
            result = _insider_threat_assessor.generate_case_referral(
                user_id,
                assessment_id=assessment_id,
                additional_context=additional_context,
            )
            result["compliance_module"] = "insider_threat"
            result["available"] = True
            return result

        elif referral_type == "personnel_security":
            if _cleared_personnel_analyzer is None:
                return _compliance_not_available()
            result = _cleared_personnel_analyzer.generate_personnel_security_action_report(
                user_id,
                action_type="REVIEW",
                narrative=additional_context,
            )
            result["compliance_module"] = "cleared_personnel"
            result["available"] = True
            return result

        else:
            return {
                "error": f"Unknown referral type: {referral_type}",
                "supported_types": ["insider_threat", "personnel_security"],
                "available": True,
            }

    except Exception as e:
        logger.error("Threat referral generation failed for %s: %s", user_id, e)
        return {
            "error": f"Referral generation failed: {str(e)}",
            "user_id": user_id,
            "available": True,
        }


@_monitored("/generate_threat_referral", "TOOL")
@mcp.tool()
def generate_threat_referral(
    user_id: str,
    referral_type: str = "insider_threat",
    assessment_id: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a formal insider threat case referral or personnel security
    action report.

    For insider_threat type: Generates a case referral per EO 13587
    with executive summary, risk timeline, NIST control impacts, MITRE ATT&CK
    technique mapping, and recommended actions. Marked FOR OFFICIAL USE ONLY
    and includes legal notices per CNSSD 504.

    For personnel_security type: Generates a Personnel Security Action report
    with Whole Person Assessment summary, SF-86 discrepancies, reporting
    violations, and appeal rights per EO 12968.

    Args:
        user_id: User identifier for the referral subject
        referral_type: Type of referral to generate:
            - "insider_threat": Full insider threat case referral
            - "personnel_security": Personnel security action report
        assessment_id: Optional specific assessment ID to reference
        additional_context: Optional free-text narrative from the analyst
            or security officer to include in the referral

    Returns:
        Structured referral/report with referral_id, executive_summary,
        risk_summary, indicator_timeline, recommended_actions, and
        legal_notice
    """
    return generate_threat_referral_impl(
        user_id, referral_type, assessment_id, additional_context
    )


if __name__ == "__main__":
    mcp.run()
