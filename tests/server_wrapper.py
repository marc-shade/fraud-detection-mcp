"""
Wrapper module to expose MCP tool implementations for testing
Since FastMCP decorators create FunctionTool objects that aren't directly callable,
this module extracts the underlying implementations.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the analyzers and validation functions
from server import (
    behavioral_analyzer,
    transaction_analyzer,
    network_analyzer,
    validate_transaction_data,
    validate_behavioral_data,
    BehavioralBiometrics,
    TransactionAnalyzer,
    NetworkAnalyzer
)


def analyze_transaction(
    transaction_data: Dict[str, Any],
    include_behavioral: bool = False,
    behavioral_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Testable implementation of analyze_transaction"""
    try:
        # Validate inputs
        valid, msg = validate_transaction_data(transaction_data)
        if not valid:
            return {
                "error": f"Invalid transaction data: {msg}",
                "status": "validation_failed",
                "overall_risk_score": 0.0,
                "risk_level": "UNKNOWN"
            }

        if behavioral_data:
            valid, msg = validate_behavioral_data(behavioral_data)
            if not valid:
                return {
                    "error": f"Invalid behavioral data: {msg}",
                    "status": "validation_failed",
                    "overall_risk_score": 0.0,
                    "risk_level": "UNKNOWN"
                }

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
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }


def detect_behavioral_anomaly(behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
    """Testable implementation of detect_behavioral_anomaly"""
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
        return {
            "error": str(e),
            "overall_anomaly_score": 0.0,
            "status": "analysis_failed"
        }


def assess_network_risk(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    """Testable implementation of assess_network_risk"""
    return network_analyzer.analyze_network_risk(entity_data)


def generate_risk_score(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Testable implementation of generate_risk_score"""
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
            overall_score = scores[0]
        elif len(scores) == 2:
            overall_score = (scores[0] * 0.6 + scores[1] * 0.4)
        else:
            overall_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)

        comprehensive_result["overall_risk_score"] = float(overall_score)

        import numpy as np
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
        return {
            "error": str(e),
            "overall_risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed"
        }


def explain_decision(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Testable implementation of explain_decision"""
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
        return {
            "error": str(e),
            "decision_summary": "Unable to generate explanation",
            "status": "explanation_failed"
        }


# Export all the testable functions
__all__ = [
    'analyze_transaction',
    'detect_behavioral_anomaly',
    'assess_network_risk',
    'generate_risk_score',
    'explain_decision',
    'BehavioralBiometrics',
    'TransactionAnalyzer',
    'NetworkAnalyzer',
    'validate_transaction_data',
    'validate_behavioral_data'
]
