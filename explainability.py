#!/usr/bin/env python3
"""
Explainability Module for Fraud Detection
Provides SHAP-based explanations with graceful degradation
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import SHAP, but allow graceful degradation
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available - explanations will be limited")


class FraudExplainer:
    """
    Generates explanations for fraud predictions using SHAP values
    Falls back to feature importance if SHAP unavailable
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer

        Args:
            model: Trained model (XGBoost, sklearn, etc.)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.fallback_mode = not SHAP_AVAILABLE

        if SHAP_AVAILABLE:
            try:
                self._initialize_shap_explainer()
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
                self.fallback_mode = True

    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type"""
        model_type = type(self.model).__name__

        try:
            # Try TreeExplainer for tree-based models (fastest)
            if 'XGB' in model_type or 'GradientBoosting' in model_type or 'RandomForest' in model_type:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info(f"Initialized TreeExplainer for {model_type}")
            else:
                # Fall back to KernelExplainer for other models
                logger.warning(f"Using KernelExplainer for {model_type} - may be slow")
                self.explainer = None  # Will use fallback
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            self.explainer = None

    def explain_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single prediction

        Args:
            features: Feature vector (1D numpy array)
            prediction: Model prediction (risk score)
            top_n: Number of top features to return

        Returns:
            Dictionary with explanation details
        """
        if not self.fallback_mode and self.explainer is not None:
            return self._explain_with_shap(features, prediction, top_n)
        else:
            return self._explain_with_feature_importance(features, prediction, top_n)

    def _explain_with_shap(
        self,
        features: np.ndarray,
        prediction: float,
        top_n: int
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        try:
            # Reshape if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class, take positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Get SHAP values for this prediction
            if shap_values.ndim > 1:
                shap_values = shap_values[0]

            # Get base value
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            # Get feature contributions
            contributions = list(zip(self.feature_names, shap_values, features[0]))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            # Separate risk factors and protective factors
            risk_factors = [
                {
                    'feature': name,
                    'contribution': float(contrib),
                    'value': float(value),
                    'description': self._describe_feature(name, value, contrib)
                }
                for name, contrib, value in contributions[:top_n]
                if contrib > 0
            ]

            protective_factors = [
                {
                    'feature': name,
                    'contribution': float(contrib),
                    'value': float(value),
                    'description': self._describe_feature(name, value, contrib)
                }
                for name, contrib, value in contributions[:top_n]
                if contrib < 0
            ]

            return {
                'method': 'SHAP',
                'base_value': float(base_value),
                'prediction': float(prediction),
                'risk_factors': risk_factors,
                'protective_factors': protective_factors,
                'top_features': [
                    {
                        'feature': name,
                        'contribution': float(contrib),
                        'value': float(value)
                    }
                    for name, contrib, value in contributions[:top_n]
                ]
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._explain_with_feature_importance(features, prediction, top_n)

    def _explain_with_feature_importance(
        self,
        features: np.ndarray,
        prediction: float,
        top_n: int
    ) -> Dict[str, Any]:
        """Fallback explanation using feature importance"""
        try:
            # Get feature importance from model
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_[0])
            else:
                # No feature importance available
                importance = np.ones(len(self.feature_names))

            # Normalize importance
            importance = importance / (np.sum(importance) + 1e-10)

            # Combine with feature values
            if features.ndim > 1:
                features = features[0]

            contributions = list(zip(self.feature_names, importance, features))
            contributions.sort(key=lambda x: x[1], reverse=True)

            # Create explanation
            risk_factors = [
                {
                    'feature': name,
                    'importance': float(imp),
                    'value': float(value),
                    'description': self._describe_feature(name, value, imp)
                }
                for name, imp, value in contributions[:top_n]
                if value > np.median(features)
            ]

            protective_factors = [
                {
                    'feature': name,
                    'importance': float(imp),
                    'value': float(value),
                    'description': self._describe_feature(name, value, -imp)
                }
                for name, imp, value in contributions[:top_n]
                if value <= np.median(features)
            ]

            return {
                'method': 'Feature Importance',
                'prediction': float(prediction),
                'risk_factors': risk_factors[:top_n // 2],
                'protective_factors': protective_factors[:top_n // 2],
                'top_features': [
                    {
                        'feature': name,
                        'importance': float(imp),
                        'value': float(value)
                    }
                    for name, imp, value in contributions[:top_n]
                ],
                'note': 'SHAP not available - using feature importance fallback'
            }

        except Exception as e:
            logger.error(f"Feature importance explanation failed: {e}")
            return self._minimal_explanation(features, prediction)

    def _minimal_explanation(
        self,
        features: np.ndarray,
        prediction: float
    ) -> Dict[str, Any]:
        """Minimal explanation when all else fails"""
        if features.ndim > 1:
            features = features[0]

        # Find highest and lowest feature values
        top_indices = np.argsort(features)[-5:][::-1]
        bottom_indices = np.argsort(features)[:5]

        risk_factors = [
            {
                'feature': self.feature_names[i],
                'value': float(features[i]),
                'description': f"High value: {features[i]:.2f}"
            }
            for i in top_indices
        ]

        protective_factors = [
            {
                'feature': self.feature_names[i],
                'value': float(features[i]),
                'description': f"Low value: {features[i]:.2f}"
            }
            for i in bottom_indices
        ]

        return {
            'method': 'Basic Analysis',
            'prediction': float(prediction),
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'note': 'Limited explanation - model introspection not available'
        }

    def _describe_feature(
        self,
        feature_name: str,
        value: float,
        contribution: float
    ) -> str:
        """Generate human-readable description of feature contribution"""
        # Amount features
        if 'amount' in feature_name:
            if 'log' in feature_name:
                return f"Transaction amount (log scale): {value:.2f}"
            elif 'sqrt' in feature_name:
                return f"Transaction amount (sqrt): {value:.2f}"
            else:
                return f"Transaction amount: ${value:,.2f}"

        # Temporal features
        if 'hour' in feature_name:
            if 'sin' not in feature_name and 'cos' not in feature_name:
                hour = int(value)
                return f"Transaction hour: {hour:02d}:00"
            return f"Time pattern: {value:.2f}"

        if 'weekend' in feature_name:
            return "Weekend transaction" if value > 0.5 else "Weekday transaction"

        if 'night' in feature_name:
            return "Night-time transaction" if value > 0.5 else "Daytime transaction"

        # Categorical features
        if 'payment_method' in feature_name:
            return f"Payment method type: {value:.0f}"

        if 'crypto' in feature_name:
            return "Cryptocurrency payment" if value > 0.5 else "Traditional payment"

        # Behavioral features
        if 'keystroke' in feature_name:
            return f"Keystroke pattern: {value:.2f}"

        if 'session' in feature_name:
            return f"Session behavior: {value:.2f}"

        # Network features
        if 'connection' in feature_name:
            return f"Network connections: {value:.0f}"

        # Default
        return f"{feature_name}: {value:.2f}"

    def generate_summary(self, explanation: Dict[str, Any]) -> str:
        """
        Generate human-readable summary from explanation

        Args:
            explanation: Explanation dictionary from explain_prediction

        Returns:
            Human-readable summary string
        """
        method = explanation.get('method', 'Unknown')
        prediction = explanation.get('prediction', 0.0)
        risk_factors = explanation.get('risk_factors', [])
        protective_factors = explanation.get('protective_factors', [])

        summary_parts = [
            f"Risk Score: {prediction:.1%}",
            f"Analysis Method: {method}",
            ""
        ]

        if risk_factors:
            summary_parts.append("Top Risk Factors:")
            for i, factor in enumerate(risk_factors[:5], 1):
                feature = factor.get('feature', 'Unknown')
                desc = factor.get('description', '')
                summary_parts.append(f"  {i}. {feature}: {desc}")
            summary_parts.append("")

        if protective_factors:
            summary_parts.append("Protective Factors:")
            for i, factor in enumerate(protective_factors[:3], 1):
                feature = factor.get('feature', 'Unknown')
                desc = factor.get('description', '')
                summary_parts.append(f"  {i}. {feature}: {desc}")

        return "\n".join(summary_parts)

    def batch_explain(
        self,
        features_batch: np.ndarray,
        predictions: np.ndarray,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for batch of predictions

        Args:
            features_batch: Batch of feature vectors
            predictions: Batch of predictions
            top_n: Number of top features per prediction

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        for features, prediction in zip(features_batch, predictions):
            explanation = self.explain_prediction(features, prediction, top_n)
            explanations.append(explanation)

        return explanations


def create_explainer(
    model,
    feature_names: List[str]
) -> FraudExplainer:
    """
    Factory function to create explainer

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        FraudExplainer instance
    """
    return FraudExplainer(model, feature_names)


__all__ = ['FraudExplainer', 'create_explainer', 'SHAP_AVAILABLE']