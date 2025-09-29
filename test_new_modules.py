#!/usr/bin/env python3
"""
Quick test for new fraud detection modules
Verifies basic functionality and integration
"""

import numpy as np
from datetime import datetime
import sys

from models_validation import TransactionData, BehavioralData, NetworkData, PaymentMethod
from feature_engineering import FeatureEngineer
from explainability import FraudExplainer, SHAP_AVAILABLE
from async_inference import AsyncInferenceEngine
from models.autoencoder import AutoencoderFraudDetector, PYTORCH_AVAILABLE as AE_PYTORCH
from models.gnn_fraud_detector import GNNFraudDetector, TORCH_GEOMETRIC_AVAILABLE


def test_feature_engineering():
    """Test feature engineering"""
    print("\n=== Testing Feature Engineering ===")

    # Create sample transactions
    transactions = [
        TransactionData(
            transaction_id=f"txn_{i}",
            user_id=f"user_{i % 5}",
            amount=float(100 * (i + 1)),
            merchant=f"merchant_{i % 3}",
            location=f"location_{i % 2}",
            timestamp=datetime.now(),
            payment_method=PaymentMethod.CREDIT_CARD
        )
        for i in range(10)
    ]

    # Initialize and fit feature engineer
    fe = FeatureEngineer()
    fe.fit(transactions)

    print(f"Feature engineer fitted")
    print(f"Number of features: {len(fe.feature_names)}")
    print(f"Feature names: {fe.feature_names[:10]}...")

    # Transform single transaction
    features = fe.transform(transactions[0])
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector (first 10): {features[:10]}")

    # Batch transform
    feature_matrix, names = fe.fit_transform(transactions)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    print("Feature engineering: PASSED")
    return fe, transactions


def test_explainability(fe, transactions):
    """Test explainability module"""
    print("\n=== Testing Explainability ===")

    # Create dummy model
    from sklearn.ensemble import RandomForestClassifier

    # Generate dummy data for training
    X = np.random.randn(100, len(fe.feature_names))
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Create explainer
    explainer = FraudExplainer(model, fe.feature_names)

    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"Explainer fallback mode: {explainer.fallback_mode}")

    # Test single explanation
    test_features = fe.transform(transactions[0])
    prediction = model.predict_proba(test_features.reshape(1, -1))[0][1]

    explanation = explainer.explain_prediction(test_features, prediction, top_n=5)

    print(f"Explanation method: {explanation['method']}")
    print(f"Risk factors: {len(explanation['risk_factors'])}")
    print(f"Protective factors: {len(explanation['protective_factors'])}")

    # Generate summary
    summary = explainer.generate_summary(explanation)
    print(f"Summary:\n{summary[:200]}...")

    print("Explainability: PASSED")
    return model, explainer


def test_async_inference(model, fe, explainer, transactions):
    """Test async inference engine"""
    print("\n=== Testing Async Inference ===")
    import asyncio

    # Create inference engine
    engine = AsyncInferenceEngine(
        model=model,
        feature_engineer=fe,
        explainer=explainer,
        cache_size=100
    )

    # Test single prediction
    async def test_single():
        result = await engine.predict_single(
            transactions[0],
            use_cache=True,
            include_explanation=True
        )
        return result

    result = asyncio.run(test_single())

    print(f"Transaction ID: {result['transaction_id']}")
    print(f"Risk score: {result['risk_score']:.3f}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Processing time: {result['processing_time_ms']:.2f}ms")
    print(f"Anomalies detected: {len(result['detected_anomalies'])}")

    # Test batch prediction
    async def test_batch():
        results = await engine.predict_batch(
            transactions[:5],
            use_cache=True,
            include_explanation=False,
            batch_size=5
        )
        return results

    batch_results = asyncio.run(test_batch())
    print(f"\nBatch prediction results: {len(batch_results)} transactions")

    # Check statistics
    stats = engine.get_statistics()
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Average time: {stats['average_time_ms']:.2f}ms")

    print("Async inference: PASSED")
    return engine


def test_autoencoder():
    """Test autoencoder model"""
    print("\n=== Testing Autoencoder ===")

    print(f"PyTorch available: {AE_PYTORCH}")

    # Create dummy data
    X = np.random.randn(100, 46)  # 46 features from feature engineering
    y = np.concatenate([np.zeros(90), np.ones(10)])  # 10% fraud

    # Create and train autoencoder
    ae = AutoencoderFraudDetector(
        contamination=0.1,
        epochs=5,  # Quick test
        batch_size=32
    )

    print(f"Fallback mode: {ae.fallback_mode}")

    ae.fit(X, y)
    print("Autoencoder trained")

    # Test prediction
    predictions = ae.predict(X[:10])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # Test probabilities
    probas = ae.predict_proba(X[:10])
    print(f"Probabilities shape: {probas.shape}")
    print(f"First probability: {probas[0]}")

    print("Autoencoder: PASSED")


def test_gnn():
    """Test GNN model"""
    print("\n=== Testing GNN ===")

    print(f"PyTorch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")

    # Create dummy data
    X = np.random.randn(50, 46)
    y = np.random.randint(0, 2, 50)

    # Create and train GNN
    gnn = GNNFraudDetector(
        hidden_dim=32,
        num_layers=2,
        epochs=5,  # Quick test
        batch_size=32
    )

    print(f"Fallback mode: {gnn.fallback_mode}")

    gnn.fit(X, y)
    print("GNN trained")

    # Test prediction
    predictions = gnn.predict(X[:10])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # Test probabilities
    probas = gnn.predict_proba(X[:10])
    print(f"Probabilities shape: {probas.shape}")
    print(f"First probability: {probas[0]}")

    print("GNN: PASSED")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing New Fraud Detection Modules")
    print("=" * 60)

    try:
        # Test feature engineering
        fe, transactions = test_feature_engineering()

        # Test explainability
        model, explainer = test_explainability(fe, transactions)

        # Test async inference
        engine = test_async_inference(model, fe, explainer, transactions)

        # Test autoencoder
        test_autoencoder()

        # Test GNN
        test_gnn()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())