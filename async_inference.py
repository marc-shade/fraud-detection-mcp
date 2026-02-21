#!/usr/bin/env python3
"""
Async Inference Engine for Fraud Detection
High-performance async prediction with caching and batch support
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
import logging
from collections import OrderedDict

from models_validation import TransactionData, BehavioralData, NetworkData, RiskLevel
from feature_engineering import FeatureEngineer
from explainability import FraudExplainer

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for predictions"""

    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self):
        """Clear cache"""
        self.cache.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class AsyncInferenceEngine:
    """
    Async inference engine for fraud detection
    Supports single and batch predictions with caching
    """

    def __init__(
        self,
        model,
        feature_engineer: FeatureEngineer,
        explainer: Optional[FraudExplainer] = None,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize inference engine

        Args:
            model: Trained fraud detection model
            feature_engineer: Feature extraction pipeline
            explainer: Optional explainer for predictions
            cache_size: Maximum cache size
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.explainer = explainer
        self.cache = LRUCache(capacity=cache_size)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.cache_metadata = {}  # Stores timestamp for each cache entry

        # Performance metrics
        self.total_predictions = 0
        self.cache_hits = 0
        self.total_time_ms = 0.0

        logger.info(f"AsyncInferenceEngine initialized with cache_size={cache_size}")

    async def predict_single(
        self,
        transaction: TransactionData,
        behavioral: Optional[BehavioralData] = None,
        network: Optional[NetworkData] = None,
        use_cache: bool = True,
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict fraud risk for single transaction

        Args:
            transaction: Transaction data
            behavioral: Optional behavioral data
            network: Optional network data
            use_cache: Whether to use cache
            include_explanation: Whether to include explanation

        Returns:
            AnalysisResponse dictionary
        """
        start_time = datetime.now()

        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(transaction, behavioral, network)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                logger.debug(f"Cache hit for transaction {transaction.transaction_id}")
                return cached_result

        # Extract features
        features = await self._extract_features_async(transaction, behavioral, network)

        # Run prediction
        prediction = await self._predict_async(features)

        # Generate explanation if requested
        explanation_data = None
        if include_explanation and self.explainer is not None:
            explanation_data = await self._explain_async(features, prediction)

        # Build response
        response = self._build_response(
            transaction, prediction, explanation_data, start_time
        )

        # Cache result
        if use_cache:
            self._put_in_cache(cache_key, response)

        # Update metrics
        self.total_predictions += 1
        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.total_time_ms += elapsed_ms

        return response

    async def predict_batch(
        self,
        transactions: List[TransactionData],
        behavioral_data: Optional[List[BehavioralData]] = None,
        network_data: Optional[List[NetworkData]] = None,
        use_cache: bool = True,
        include_explanation: bool = True,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Predict fraud risk for batch of transactions

        Args:
            transactions: List of transactions
            behavioral_data: Optional list of behavioral data
            network_data: Optional list of network data
            use_cache: Whether to use cache
            include_explanation: Whether to include explanations
            batch_size: Batch size for parallel processing

        Returns:
            List of AnalysisResponse dictionaries
        """
        logger.info(f"Starting batch prediction for {len(transactions)} transactions")

        # Pad optional data
        behavioral_data = behavioral_data or [None] * len(transactions)
        network_data = network_data or [None] * len(transactions)

        # Process in batches to avoid overwhelming resources
        results = []
        for i in range(0, len(transactions), batch_size):
            batch_end = min(i + batch_size, len(transactions))
            batch_txns = transactions[i:batch_end]
            batch_behav = behavioral_data[i:batch_end]
            batch_net = network_data[i:batch_end]

            # Create tasks for parallel processing
            tasks = [
                self.predict_single(
                    txn,
                    behav,
                    net,
                    use_cache=use_cache,
                    include_explanation=include_explanation,
                )
                for txn, behav, net in zip(batch_txns, batch_behav, batch_net)
            ]

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle errors
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Prediction failed for transaction {batch_txns[j].transaction_id}: {result}"
                    )
                    # Create error response
                    results.append(
                        self._create_error_response(batch_txns[j], str(result))
                    )
                else:
                    results.append(result)

            logger.debug(
                f"Completed batch {i // batch_size + 1} ({batch_end}/{len(transactions)})"
            )

        logger.info(f"Batch prediction complete. Processed {len(results)} transactions")
        return results

    async def _extract_features_async(
        self,
        transaction: TransactionData,
        behavioral: Optional[BehavioralData],
        network: Optional[NetworkData],
    ) -> np.ndarray:
        """Extract features asynchronously"""
        # Feature extraction is CPU-bound, but wrapped for async
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            None, self.feature_engineer.transform, transaction, behavioral, network
        )
        return features

    async def _predict_async(self, features: np.ndarray) -> float:
        """Run model prediction asynchronously"""
        # Model inference is CPU-bound, wrapped for async
        loop = asyncio.get_event_loop()

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Run prediction
        if hasattr(self.model, "predict_proba"):
            prediction = await loop.run_in_executor(
                None, lambda: self.model.predict_proba(features)[0][1]
            )
        else:
            prediction = await loop.run_in_executor(
                None, lambda: self.model.predict(features)[0]
            )

        return float(prediction)

    async def _explain_async(
        self, features: np.ndarray, prediction: float
    ) -> Optional[Dict[str, Any]]:
        """Generate explanation asynchronously"""
        if self.explainer is None:
            return None

        loop = asyncio.get_event_loop()
        explanation = await loop.run_in_executor(
            None,
            self.explainer.explain_prediction,
            features,
            prediction,
            10,  # top_n features
        )
        return explanation

    def _build_response(
        self,
        transaction: TransactionData,
        prediction: float,
        explanation: Optional[Dict[str, Any]],
        start_time: datetime,
    ) -> Dict[str, Any]:
        """Build AnalysisResponse dictionary"""
        # Determine risk level
        risk_level = self._calculate_risk_level(prediction)

        # Generate anomalies and recommendations
        detected_anomalies = self._detect_anomalies(
            transaction, prediction, explanation
        )
        recommended_actions = self._generate_recommendations(
            risk_level, detected_anomalies
        )

        # Calculate processing time
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Build explanation text
        explanation_text = None
        if explanation:
            if self.explainer:
                explanation_text = self.explainer.generate_summary(explanation)
            else:
                explanation_text = "Explanation not available"

        response = {
            "transaction_id": transaction.transaction_id,
            "risk_score": float(prediction),
            "risk_level": risk_level.value,
            "confidence": self._calculate_confidence(prediction),
            "detected_anomalies": detected_anomalies,
            "recommended_actions": recommended_actions,
            "explanation": explanation_text,
            "model_version": "2.0.0",
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_time_ms": processing_time_ms,
            "transaction_risk_score": float(prediction),
            "behavioral_risk_score": None,  # Could be separated in future
            "network_risk_score": None,
        }

        return response

    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Calculate risk level from score"""
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.85:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _calculate_confidence(self, risk_score: float) -> float:
        """Calculate confidence in prediction"""
        # Distance from decision boundary (0.5)
        distance = abs(risk_score - 0.5)
        confidence = min(0.5 + distance, 1.0)
        return float(confidence)

    def _detect_anomalies(
        self,
        transaction: TransactionData,
        risk_score: float,
        explanation: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Detect specific anomalies"""
        anomalies = []

        # High amount
        if transaction.amount > 10000:
            anomalies.append(f"High transaction amount: ${transaction.amount:,.2f}")

        # Night time transaction
        hour = transaction.timestamp.hour
        if hour < 6 or hour >= 22:
            anomalies.append(f"Transaction at unusual hour: {hour:02d}:00")

        # Weekend transaction
        if transaction.timestamp.weekday() >= 5:
            anomalies.append("Weekend transaction")

        # Use explanation data if available
        if explanation and "risk_factors" in explanation:
            for factor in explanation["risk_factors"][:3]:
                feature = factor.get("feature", "")
                if "crypto" in feature and factor.get("value", 0) > 0.5:
                    anomalies.append("Cryptocurrency payment method")

        return anomalies

    def _generate_recommendations(
        self, risk_level: RiskLevel, anomalies: List[str]
    ) -> List[str]:
        """Generate recommended actions"""
        recommendations = []

        if risk_level == RiskLevel.LOW:
            recommendations.append("Approve transaction")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Request additional verification")
            recommendations.append("Monitor for related suspicious activity")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Hold transaction for manual review")
            recommendations.append("Contact customer for verification")
            recommendations.append("Check for similar recent transactions")
        else:  # CRITICAL
            recommendations.append("Block transaction immediately")
            recommendations.append("Notify customer of suspicious activity")
            recommendations.append("Flag account for investigation")
            recommendations.append("Review recent transaction history")

        return recommendations

    def _create_error_response(
        self, transaction: TransactionData, error_message: str
    ) -> Dict[str, Any]:
        """Create error response"""
        return {
            "transaction_id": transaction.transaction_id,
            "risk_score": 0.5,
            "risk_level": RiskLevel.UNKNOWN.value,
            "confidence": 0.0,
            "detected_anomalies": ["Prediction error occurred"],
            "recommended_actions": ["Manual review required"],
            "explanation": f"Error during prediction: {error_message}",
            "model_version": "2.0.0",
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_time_ms": 0.0,
        }

    def _generate_cache_key(
        self,
        transaction: TransactionData,
        behavioral: Optional[BehavioralData],
        network: Optional[NetworkData],
    ) -> str:
        """Generate cache key from transaction data"""
        # Create deterministic hash
        key_data = {
            "transaction_id": transaction.transaction_id,
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "merchant": transaction.merchant,
            "location": transaction.location,
            "timestamp": transaction.timestamp.isoformat(),
            "payment_method": transaction.payment_method,
            "has_behavioral": behavioral is not None,
            "has_network": network is not None,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache if not expired"""
        result = self.cache.get(cache_key)
        if result is None:
            return None

        # Check if expired
        metadata = self.cache_metadata.get(cache_key)
        if metadata:
            cached_time = metadata.get("timestamp")
            if cached_time and (datetime.now() - cached_time) > self.cache_ttl:
                # Expired, remove from cache
                self.cache.cache.pop(cache_key, None)
                self.cache_metadata.pop(cache_key, None)
                return None

        return result

    def _put_in_cache(self, cache_key: str, result: Dict[str, Any]):
        """Put prediction in cache"""
        self.cache.put(cache_key, result)
        self.cache_metadata[cache_key] = {"timestamp": datetime.now()}

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        cache_hit_rate = (
            self.cache_hits / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        avg_time_ms = (
            self.total_time_ms / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        return {
            "total_predictions": self.total_predictions,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": self.cache.size(),
            "average_time_ms": avg_time_ms,
        }

    def clear_cache(self):
        """Clear prediction cache"""
        self.cache.clear()
        self.cache_metadata.clear()
        logger.info("Cache cleared")


class AsyncFraudDetector:
    """
    High-level async fraud detector integrating all models
    Uses ensemble of traditional ML and advanced deep learning models
    """

    def __init__(
        self,
        fraud_pipeline,
        feature_engineer: FeatureEngineer,
        use_advanced_models: bool = True,
    ):
        """
        Initialize async fraud detector

        Args:
            fraud_pipeline: Trained FraudDetectionPipeline (ModelTrainer)
            feature_engineer: Feature extraction pipeline
            use_advanced_models: Use autoencoder/GNN if available
        """
        self.pipeline = fraud_pipeline
        self.feature_engineer = feature_engineer
        self.use_advanced_models = use_advanced_models

        # Check which models are available
        self.has_autoencoder = "autoencoder" in fraud_pipeline.models
        self.has_gnn = "gnn" in fraud_pipeline.models
        self.has_isolation_forest = "isolation_forest" in fraud_pipeline.models
        self.has_xgboost = "xgboost" in fraud_pipeline.models

        logger.info("AsyncFraudDetector initialized:")
        logger.info(f"  - Isolation Forest: {self.has_isolation_forest}")
        logger.info(f"  - XGBoost: {self.has_xgboost}")
        logger.info(f"  - Autoencoder: {self.has_autoencoder}")
        logger.info(f"  - GNN: {self.has_gnn}")

    async def analyze_transaction_async(self, analysis_request) -> Dict[str, Any]:
        """
        Analyze transaction asynchronously using ensemble of models

        Args:
            analysis_request: AnalysisRequest object

        Returns:
            Analysis result dictionary
        """
        transaction = analysis_request.transaction_data

        # Extract features
        features, feature_names = self.feature_engineer.extract_features(transaction)
        features_array = np.array(features).reshape(1, -1)

        # Scale features if scaler available
        if "transaction" in self.pipeline.scalers:
            features_array = self.pipeline.scalers["transaction"].transform(
                features_array
            )

        # Collect predictions from available models
        predictions = {}
        probabilities = {}

        # Traditional models
        if self.has_isolation_forest:
            iso_score = self.pipeline.models["isolation_forest"].score_samples(
                features_array
            )[0]
            iso_proba = self._anomaly_score_to_proba(iso_score)
            predictions["isolation_forest"] = 1 if iso_proba > 0.5 else 0
            probabilities["isolation_forest"] = iso_proba

        if self.has_xgboost:
            xgb_proba = self.pipeline.models["xgboost"].predict_proba(features_array)[
                0, 1
            ]
            predictions["xgboost"] = 1 if xgb_proba > 0.5 else 0
            probabilities["xgboost"] = xgb_proba

        # Advanced models
        if self.use_advanced_models and self.has_autoencoder:
            auto_proba = self.pipeline.models["autoencoder"].predict_proba(
                features_array
            )[0, 1]
            predictions["autoencoder"] = 1 if auto_proba > 0.5 else 0
            probabilities["autoencoder"] = auto_proba

        if self.use_advanced_models and self.has_gnn:
            gnn_proba = self.pipeline.models["gnn"].predict_proba(features_array)[0, 1]
            predictions["gnn"] = 1 if gnn_proba > 0.5 else 0
            probabilities["gnn"] = gnn_proba

        # Ensemble prediction (weighted average)
        weights = self._get_model_weights()
        ensemble_proba = sum(
            probabilities[name] * weights[name] for name in probabilities.keys()
        ) / sum(weights[name] for name in probabilities.keys())

        # Determine risk level
        if ensemble_proba < 0.3:
            risk_level = "LOW"
        elif ensemble_proba < 0.6:
            risk_level = "MEDIUM"
        elif ensemble_proba < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "transaction_id": transaction.transaction_id,
            "risk_score": float(ensemble_proba),
            "risk_level": risk_level,
            "is_fraud": bool(ensemble_proba > 0.5),
            "model_predictions": predictions,
            "model_probabilities": probabilities,
            "confidence": float(1.0 - abs(ensemble_proba - 0.5) * 2),
            "features_analyzed": len(features),
        }

    def _anomaly_score_to_proba(self, score: float) -> float:
        """Convert anomaly score to probability"""
        # Isolation forest scores are negative, more negative = more anomalous
        # Normalize to [0, 1] range
        return 1 / (1 + np.exp(score * 5))  # Sigmoid transformation

    def _get_model_weights(self) -> Dict[str, float]:
        """
        Get model weights for ensemble
        Advanced models get higher weights if available
        """
        weights = {}

        if self.has_isolation_forest:
            weights["isolation_forest"] = 0.2

        if self.has_xgboost:
            weights["xgboost"] = 0.4

        if self.use_advanced_models and self.has_autoencoder:
            weights["autoencoder"] = 0.2

        if self.use_advanced_models and self.has_gnn:
            weights["gnn"] = 0.3

        return weights


# Factory function
def create_inference_engine(
    model,
    feature_engineer: FeatureEngineer,
    explainer: Optional[FraudExplainer] = None,
    cache_size: int = 1000,
) -> AsyncInferenceEngine:
    """
    Create async inference engine

    Args:
        model: Trained model
        feature_engineer: Feature engineer
        explainer: Optional explainer
        cache_size: Cache size

    Returns:
        AsyncInferenceEngine instance
    """
    return AsyncInferenceEngine(
        model=model,
        feature_engineer=feature_engineer,
        explainer=explainer,
        cache_size=cache_size,
    )


__all__ = [
    "AsyncInferenceEngine",
    "AsyncFraudDetector",
    "create_inference_engine",
    "LRUCache",
]
