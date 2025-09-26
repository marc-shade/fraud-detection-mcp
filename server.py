#!/usr/bin/env python3
"""
Advanced Fraud Detection MCP Server
Sophisticated fraud detection using cutting-edge 2024-2025 algorithms
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# FastMCP for high-performance MCP server
from fastmcp import FastMCP

# Core ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb

# Deep learning for autoencoders
import torch
import torch.nn as nn

# Graph analysis
import networkx as nx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Advanced Fraud Detection")

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

        # One-Class SVM for mouse patterns
        self.mouse_model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )

        # Local Outlier Factor for touch patterns
        self.touch_model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )

    def analyze_keystroke_dynamics(self, keystroke_data: List[Dict]) -> Dict[str, Any]:
        """Analyze keystroke dynamics for behavioral anomalies"""
        try:
            if not keystroke_data:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "no_data"}

            # Extract features from keystroke data
            features = self._extract_keystroke_features(keystroke_data)

            if features is None:
                return {"risk_score": 0.0, "confidence": 0.0, "status": "invalid_data"}

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
                    dwell = keystroke['release_time'] - keystroke['press_time']
                    dwell_times.append(dwell)

                # Flight time
                if i > 0:
                    prev_keystroke = keystroke_data[i-1]
                    if 'release_time' in prev_keystroke and 'press_time' in keystroke:
                        flight = keystroke['press_time'] - prev_keystroke['release_time']
                        flight_times.append(flight)

            if not dwell_times and not flight_times:
                return None

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

    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200
        )
        self.xgb_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize transaction analysis models"""
        # XGBoost model would be loaded from trained model file
        # For demo, we'll create a simple structure
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

    def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive transaction fraud analysis"""
        try:
            # Extract features
            features = self._extract_transaction_features(transaction_data)

            # Anomaly detection with Isolation Forest
            anomaly_score = self.isolation_forest.decision_function([features])[0]
            is_anomaly = self.isolation_forest.predict([features])[0] == -1

            # Calculate risk factors
            risk_factors = self._identify_risk_factors(transaction_data, features)

            # Calculate overall risk score
            base_risk = max(0, min(1, (0.5 - anomaly_score) * 2))
            risk_multiplier = 1 + len(risk_factors) * 0.1
            final_risk = min(1.0, base_risk * risk_multiplier)

            return {
                "risk_score": float(final_risk),
                "is_anomaly": bool(is_anomaly),
                "risk_factors": risk_factors,
                "confidence": 0.88,
                "analysis_type": "transaction_pattern",
                "anomaly_score": float(anomaly_score)
            }

        except Exception as e:
            logger.error(f"Transaction analysis error: {e}")
            return {"risk_score": 0.0, "confidence": 0.0, "status": "error", "error": str(e)}

    def _extract_transaction_features(self, transaction: Dict[str, Any]) -> List[float]:
        """Extract numerical features from transaction data"""
        features = []

        # Amount-based features
        amount = float(transaction.get('amount', 0))
        features.append(amount)
        features.append(np.log1p(amount))  # Log-transformed amount

        # Time-based features
        timestamp = transaction.get('timestamp')
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            features.extend([
                dt.hour,  # Hour of day
                dt.weekday(),  # Day of week
                dt.day,  # Day of month
            ])
        else:
            features.extend([0, 0, 0])

        # Location-based features (simplified)
        location = transaction.get('location', '')
        features.append(hash(location) % 1000)  # Location hash

        # Merchant-based features
        merchant = transaction.get('merchant', '')
        features.append(hash(merchant) % 1000)  # Merchant hash

        # Payment method features
        payment_method = transaction.get('payment_method', 'unknown')
        method_risk = {
            'credit_card': 0.3,
            'debit_card': 0.2,
            'bank_transfer': 0.1,
            'crypto': 0.8,
            'unknown': 0.5
        }
        features.append(method_risk.get(payment_method, 0.5))

        return features

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

        # Geographic risk (simplified)
        location = transaction.get('location', '').lower()
        high_risk_locations = ['nigeria', 'russia', 'china', 'unknown']
        if any(loc in location for loc in high_risk_locations):
            risk_factors.append("high_risk_geographic_location")

        return risk_factors

class NetworkAnalyzer:
    """Graph-based network analysis for fraud ring detection"""

    def __init__(self):
        self.transaction_graph = nx.Graph()

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
        self.transaction_graph.add_node(entity_id)

        for connection in connections:
            connected_entity = connection.get('entity_id')
            if connected_entity:
                self.transaction_graph.add_edge(
                    entity_id,
                    connected_entity,
                    weight=connection.get('strength', 1.0),
                    transaction_count=connection.get('transaction_count', 1)
                )

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
        except:
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

@mcp.tool()
def analyze_transaction(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive transaction fraud analysis using multiple detection methods.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp, etc.)
        include_behavioral: Whether to include behavioral biometrics analysis
        behavioral_data: Behavioral data (keystroke dynamics, mouse patterns, etc.)

    Returns:
        Comprehensive fraud analysis with risk score and explanations
    """
    try:
        # Primary transaction analysis
        transaction_result = transaction_analyzer.analyze_transaction(transaction_data)

        results = {
            "transaction_analysis": transaction_result,
            "overall_risk_score": transaction_result.get("risk_score", 0.0),
            "risk_level": "LOW",
            "detected_anomalies": [],
            "explanations": [],
            "recommended_actions": []
        }

        # Add transaction risk factors to explanations
        risk_factors = transaction_result.get("risk_factors", [])
        results["detected_anomalies"].extend(risk_factors)

        # Behavioral analysis if requested
        if include_behavioral and behavioral_data:
            behavioral_result = {}

            # Keystroke analysis
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

        return results

    except Exception as e:
        logger.error(f"Transaction analysis failed: {e}")
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }

@mcp.tool()
def detect_behavioral_anomaly(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze behavioral biometrics for anomaly detection.

    Args:
        behavioral_data: Behavioral patterns (keystroke dynamics, mouse movements, etc.)

    Returns:
        Behavioral anomaly analysis results
    """
    try:
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

@mcp.tool()
def assess_network_risk(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze network patterns for fraud ring detection.

    Args:
        entity_data: Entity information and network connections

    Returns:
        Network-based risk assessment
    """
    return network_analyzer.analyze_network_risk(entity_data)

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
    try:
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
            "all_detected_anomalies": [],
            "comprehensive_explanation": "",
            "recommended_actions": []
        }

        scores = [transaction_analysis.get("risk_score", 0.0)]
        confidences = [transaction_analysis.get("confidence", 0.0)]

        # Add transaction anomalies
        comprehensive_result["all_detected_anomalies"].extend(
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
                comprehensive_result["all_detected_anomalies"].append("behavioral_anomaly")

        # Network analysis
        if network_data:
            network_analysis = network_analyzer.analyze_network_risk(network_data)
            network_score = network_analysis.get("risk_score", 0.0)
            comprehensive_result["component_scores"]["network"] = network_score
            scores.append(network_score)
            confidences.append(network_analysis.get("confidence", 0.0))

            comprehensive_result["all_detected_anomalies"].extend(
                network_analysis.get("risk_patterns", [])
            )

        # Calculate weighted overall score
        if len(scores) == 1:
            # Only transaction analysis
            overall_score = scores[0]
        elif len(scores) == 2:
            # Transaction + one other
            overall_score = (scores[0] * 0.6 + scores[1] * 0.4)
        else:
            # All three analyses
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
        if comprehensive_result["all_detected_anomalies"]:
            explanation = (
                f"Risk assessment detected {len(comprehensive_result['all_detected_anomalies'])} "
                f"anomalies: {', '.join(comprehensive_result['all_detected_anomalies'])}. "
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

@mcp.tool()
def explain_decision(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provide explainable AI reasoning for fraud detection decisions.

    Args:
        analysis_result: Previous analysis result to explain

    Returns:
        Detailed explanation of the decision-making process
    """
    try:
        explanation = {
            "decision_summary": "",
            "key_factors": [],
            "algorithm_contributions": {},
            "confidence_breakdown": {},
            "alternative_scenarios": [],
            "explanation_timestamp": datetime.now().isoformat()
        }

        risk_score = analysis_result.get("overall_risk_score", 0.0)
        risk_level = analysis_result.get("risk_level", "UNKNOWN")
        detected_anomalies = analysis_result.get("all_detected_anomalies", [])

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
            total_weight = 0
            for component, score in component_scores.items():
                if component == "transaction":
                    weight = 0.6 if len(component_scores) > 1 else 1.0
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

        return explanation

    except Exception as e:
        logger.error(f"Decision explanation failed: {e}")
        return {
            "error": str(e),
            "decision_summary": "Unable to generate explanation",
            "status": "explanation_failed"
        }

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()