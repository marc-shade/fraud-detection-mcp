#!/usr/bin/env python3
"""
Production Fraud Detection MCP Server v2.0
Integrates all advanced components with proper configuration and security
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# FastMCP for high-performance MCP server
from fastmcp import FastMCP

# Local imports - Configuration
from config import get_config

# Input validation
from models_validation import (
    TransactionData,
    BehavioralData,
    NetworkData,
    AnalysisRequest
)

# ML Components
from training_pipeline import FraudDetectionPipeline
from feature_engineering import FeatureEngineer

# Explainability
from explainability import ExplainabilityEngine

# Security
from security import AuthManager, RateLimiter

# Async inference
from async_inference import AsyncFraudDetector

# Monitoring
from monitoring import MetricsCollector, HealthMonitor

# Setup logging
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Production Fraud Detection v2.0")

# Global instances
feature_engineer: Optional[FeatureEngineer] = None
fraud_pipeline: Optional[FraudDetectionPipeline] = None
async_detector: Optional[AsyncFraudDetector] = None
explainability_engine: Optional[ExplainabilityEngine] = None
auth_manager: Optional[AuthManager] = None
rate_limiter: Optional[RateLimiter] = None
metrics_collector: Optional[MetricsCollector] = None
health_monitor: Optional[HealthMonitor] = None


async def initialize_system():
    """Initialize all system components"""
    global feature_engineer, fraud_pipeline, async_detector
    global explainability_engine, auth_manager, rate_limiter
    global metrics_collector, health_monitor

    logger.info("Initializing Fraud Detection System v2.0...")

    try:
        # 1. Configuration
        logger.info(f"Environment: {config.ENVIRONMENT}")
        logger.info(f"Model Directory: {config.MODEL_DIR}")

        # 2. Feature Engineering
        logger.info("Initializing feature engineer...")
        feature_engineer = FeatureEngineer()

        # 3. ML Pipeline - Load trained models
        logger.info("Loading trained ML models...")
        fraud_pipeline = FraudDetectionPipeline()

        # Check if models exist
        model_files = list(config.MODEL_DIR.glob("*.joblib"))
        if model_files:
            logger.info(f"Found {len(model_files)} trained model files")
            fraud_pipeline.load_models()
        else:
            logger.warning(
                "No trained models found. Run training_pipeline.py first or "
                "models will operate in demo mode."
            )

        # 4. Async Inference Engine
        logger.info("Initializing async inference engine...")
        async_detector = AsyncFraudDetector(
            fraud_pipeline=fraud_pipeline,
            feature_engineer=feature_engineer
        )

        # 5. Explainability Engine
        logger.info("Initializing explainability engine...")
        explainability_engine = ExplainabilityEngine(fraud_pipeline)

        # 6. Security Components
        if config.ENVIRONMENT == "production":
            logger.info("Initializing security layer...")
            auth_manager = AuthManager()
            rate_limiter = RateLimiter()
            logger.info("Security layer initialized")
        else:
            logger.info("Running in development mode - security layer disabled")

        # 7. Monitoring
        if config.ENABLE_METRICS:
            logger.info("Initializing monitoring...")
            metrics_collector = MetricsCollector()
            health_monitor = HealthMonitor()
            logger.info(f"Metrics endpoint: http://localhost:{config.METRICS_PORT}/metrics")

        logger.info("=" * 60)
        logger.info("✅ Fraud Detection System v2.0 initialized successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize system: {e}", exc_info=True)
        raise


@mcp.tool()
async def analyze_transaction_v2(
    transaction_data: Dict[str, Any],
    behavioral_data: Optional[Dict[str, Any]] = None,
    network_data: Optional[Dict[str, Any]] = None,
    include_explanation: bool = True,
    user_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Advanced transaction fraud analysis with all features integrated.

    Args:
        transaction_data: Transaction details (amount, merchant, location, timestamp, etc.)
        behavioral_data: Optional behavioral biometrics data
        network_data: Optional network connection data
        include_explanation: Whether to include SHAP explanations
        user_id: User ID for authentication (production only)
        api_key: API key for authentication (production only)

    Returns:
        Comprehensive fraud analysis with risk score and explanations
    """
    analysis_start_time = datetime.now()

    try:
        # Security - Rate limiting (production only)
        if rate_limiter and api_key:
            # Check rate limit
            user_tier = "free"  # Would be looked up from database
            await rate_limiter.check_rate_limit(
                identifier=api_key,
                tier=user_tier
            )

        # 1. Validate input data
        try:
            trans = TransactionData(**transaction_data)
        except Exception as e:
            logger.error(f"Transaction validation failed: {e}")
            return {
                "error": "Invalid transaction data",
                "details": str(e),
                "status": "validation_failed"
            }

        # Validate optional data
        behavioral_obj = None
        if behavioral_data:
            try:
                behavioral_obj = BehavioralData(**behavioral_data)
            except Exception as e:
                logger.warning(f"Behavioral data validation failed: {e}")

        network_obj = None
        if network_data:
            try:
                network_obj = NetworkData(**network_data)
            except Exception as e:
                logger.warning(f"Network data validation failed: {e}")

        # 2. Create analysis request
        analysis_request = AnalysisRequest(
            transaction_data=trans,
            behavioral_data=behavioral_obj,
            network_data=network_obj,
            include_explanation=include_explanation
        )

        # 3. Perform async inference
        result = await async_detector.analyze_transaction_async(analysis_request)

        # 4. Add explainability if requested
        if include_explanation and result.get("risk_score", 0) > 0.3:
            try:
                # Extract features for explanation
                features, feature_names = feature_engineer.extract_features(trans)

                # Get SHAP explanation
                explanation = explainability_engine.explain_prediction(
                    features=features,
                    feature_names=feature_names,
                    transaction_id=trans.transaction_id
                )

                result["shap_explanation"] = {
                    "top_risk_factors": explanation.top_risk_factors,
                    "top_protective_factors": explanation.top_protective_factors,
                    "explanation_text": explanation.explanation_text,
                    "waterfall_plot_path": str(explanation.waterfall_plot_path) if explanation.waterfall_plot_path else None
                }
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
                result["shap_explanation"] = None

        # 5. Record metrics
        if metrics_collector:
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            risk_level = result.get("risk_level", "UNKNOWN")

            metrics_collector.record_transaction(
                risk_level=risk_level,
                processing_time=analysis_time
            )

        # 6. Add metadata
        result["analysis_timestamp"] = datetime.now().isoformat()
        result["model_version"] = "2.0.0"
        result["system_info"] = {
            "environment": config.ENVIRONMENT,
            "security_enabled": auth_manager is not None,
            "explainability_available": explainability_engine is not None
        }

        logger.info(
            f"Transaction {trans.transaction_id} analyzed - "
            f"Risk: {result.get('risk_level', 'UNKNOWN')} "
            f"(Score: {result.get('risk_score', 0):.3f})"
        )

        return result

    except Exception as e:
        logger.error(f"Transaction analysis failed: {e}", exc_info=True)

        # Record error metric
        if metrics_collector:
            metrics_collector.record_error("analysis_error")

        return {
            "error": str(e),
            "risk_score": 0.0,
            "risk_level": "UNKNOWN",
            "status": "analysis_failed",
            "analysis_timestamp": datetime.now().isoformat()
        }


@mcp.tool()
async def batch_analyze_transactions(
    transactions: List[Dict[str, Any]],
    parallel: bool = True,
    include_explanation: bool = False
) -> Dict[str, Any]:
    """
    Batch analysis of multiple transactions with parallel processing.

    Args:
        transactions: List of transaction data dictionaries
        parallel: Whether to process in parallel
        include_explanation: Whether to include explanations

    Returns:
        Batch analysis results with summary statistics
    """
    try:
        logger.info(f"Starting batch analysis of {len(transactions)} transactions...")

        # Validate all transactions
        validated_transactions = []
        validation_errors = []

        for i, trans_data in enumerate(transactions):
            try:
                trans = TransactionData(**trans_data)
                validated_transactions.append(trans)
            except Exception as e:
                validation_errors.append({
                    "index": i,
                    "error": str(e),
                    "transaction_id": trans_data.get("transaction_id", "unknown")
                })

        logger.info(
            f"Validated {len(validated_transactions)} transactions, "
            f"{len(validation_errors)} errors"
        )

        # Analyze batch
        results = await async_detector.analyze_batch_async(
            transactions=validated_transactions,
            include_explanation=include_explanation
        )

        # Calculate summary statistics
        risk_scores = [r.get("risk_score", 0) for r in results]
        risk_levels = [r.get("risk_level", "UNKNOWN") for r in results]

        summary = {
            "total_analyzed": len(results),
            "validation_errors": len(validation_errors),
            "risk_distribution": {
                "CRITICAL": risk_levels.count("CRITICAL"),
                "HIGH": risk_levels.count("HIGH"),
                "MEDIUM": risk_levels.count("MEDIUM"),
                "LOW": risk_levels.count("LOW"),
                "UNKNOWN": risk_levels.count("UNKNOWN")
            },
            "statistics": {
                "mean_risk_score": float(np.mean(risk_scores)) if risk_scores else 0,
                "median_risk_score": float(np.median(risk_scores)) if risk_scores else 0,
                "max_risk_score": float(np.max(risk_scores)) if risk_scores else 0,
                "min_risk_score": float(np.min(risk_scores)) if risk_scores else 0
            }
        }

        return {
            "summary": summary,
            "results": results,
            "validation_errors": validation_errors,
            "batch_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "batch_analysis_failed"
        }


@mcp.tool()
def get_model_performance() -> Dict[str, Any]:
    """
    Get current model performance metrics and statistics.

    Returns:
        Model performance data
    """
    try:
        if not fraud_pipeline:
            return {"error": "Model pipeline not initialized"}

        # Get model info
        model_info = {
            "isolation_forest": {
                "loaded": fraud_pipeline.isolation_forest is not None,
                "type": "unsupervised_anomaly_detection"
            },
            "xgboost": {
                "loaded": fraud_pipeline.xgb_model is not None,
                "type": "supervised_classification"
            },
            "autoencoder": {
                "loaded": fraud_pipeline.autoencoder is not None,
                "type": "deep_learning_anomaly"
            },
            "gnn": {
                "loaded": fraud_pipeline.gnn_model is not None,
                "type": "graph_neural_network"
            }
        }

        # Get metrics from monitoring
        performance_metrics = {}
        if metrics_collector:
            performance_metrics = metrics_collector.get_summary()

        return {
            "models": model_info,
            "performance_metrics": performance_metrics,
            "model_version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        return {"error": str(e)}


@mcp.tool()
async def get_system_health() -> Dict[str, Any]:
    """
    Get system health status and diagnostics.

    Returns:
        System health information
    """
    try:
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }

        # Check each component
        health_status["components"]["feature_engineer"] = {
            "status": "ok" if feature_engineer else "not_initialized"
        }

        health_status["components"]["ml_pipeline"] = {
            "status": "ok" if fraud_pipeline else "not_initialized",
            "models_loaded": bool(fraud_pipeline and fraud_pipeline.isolation_forest)
        }

        health_status["components"]["async_detector"] = {
            "status": "ok" if async_detector else "not_initialized"
        }

        health_status["components"]["explainability"] = {
            "status": "ok" if explainability_engine else "not_initialized"
        }

        health_status["components"]["security"] = {
            "status": "ok" if auth_manager else "disabled",
            "environment": config.ENVIRONMENT
        }

        health_status["components"]["monitoring"] = {
            "status": "ok" if metrics_collector else "disabled"
        }

        # Get system metrics if available
        if health_monitor:
            system_metrics = health_monitor.get_system_metrics()
            health_status["system_metrics"] = system_metrics

        # Determine overall status
        component_statuses = [
            comp.get("status") for comp in health_status["components"].values()
        ]

        if any(status == "error" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif any(status == "not_initialized" for status in component_statuses):
            health_status["status"] = "degraded"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@mcp.tool()
def explain_fraud_decision(
    transaction_id: str,
    analysis_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get detailed explanation for a fraud detection decision using SHAP.

    Args:
        transaction_id: ID of the transaction to explain
        analysis_result: Optional previous analysis result

    Returns:
        Detailed explanation with SHAP values
    """
    try:
        if not explainability_engine:
            return {"error": "Explainability engine not initialized"}

        if not analysis_result:
            return {
                "error": "No analysis result provided. Run analyze_transaction_v2 first."
            }

        # Generate explanation
        explanation = {
            "transaction_id": transaction_id,
            "risk_level": analysis_result.get("risk_level", "UNKNOWN"),
            "risk_score": analysis_result.get("risk_score", 0),
            "explanation_timestamp": datetime.now().isoformat()
        }

        # Add SHAP explanation if available
        if "shap_explanation" in analysis_result:
            explanation["shap_analysis"] = analysis_result["shap_explanation"]

        # Add detected anomalies
        if "detected_anomalies" in analysis_result:
            explanation["detected_anomalies"] = analysis_result["detected_anomalies"]

        # Add recommended actions
        if "recommended_actions" in analysis_result:
            explanation["recommended_actions"] = analysis_result["recommended_actions"]

        return explanation

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return {"error": str(e)}


@mcp.tool()
async def train_models_tool(
    training_data_path: str,
    test_size: float = 0.2,
    use_smote: bool = True,
    optimize_hyperparams: bool = True
) -> Dict[str, Any]:
    """
    Train or retrain fraud detection models.

    Args:
        training_data_path: Path to training data CSV
        test_size: Proportion of data for testing
        use_smote: Whether to use SMOTE for class balancing
        optimize_hyperparams: Whether to optimize hyperparameters with Optuna

    Returns:
        Training results and performance metrics
    """
    try:
        logger.info(f"Starting model training with data: {training_data_path}")

        from training_pipeline import ModelTrainer

        trainer = ModelTrainer()

        # Train models
        training_results = trainer.train_all_models(
            data_path=training_data_path,
            test_size=test_size,
            use_smote=use_smote,
            optimize_hyperparams=optimize_hyperparams
        )

        logger.info("Model training completed successfully")

        # Reload models in the pipeline
        if fraud_pipeline:
            fraud_pipeline.load_models()
            logger.info("Models reloaded in pipeline")

        return {
            "status": "training_completed",
            "results": training_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        return {
            "status": "training_failed",
            "error": str(e)
        }


@mcp.tool()
def get_configuration() -> Dict[str, Any]:
    """
    Get current system configuration.

    Returns:
        System configuration details
    """
    try:
        return {
            "environment": config.ENVIRONMENT,
            "debug": config.DEBUG,
            "model_directory": str(config.MODEL_DIR),
            "data_directory": str(config.DATA_DIR),
            "log_level": config.LOG_LEVEL,
            "metrics_enabled": config.ENABLE_METRICS,
            "security": {
                "enabled": config.ENVIRONMENT == "production",
                "rate_limit_free": config.RATE_LIMIT_FREE_TIER,
                "rate_limit_paid": config.RATE_LIMIT_PAID_TIER,
                "rate_limit_enterprise": config.RATE_LIMIT_ENTERPRISE
            },
            "model_settings": {
                "isolation_forest_contamination": config.ISOLATION_FOREST_CONTAMINATION,
                "xgboost_n_estimators": config.XGBOOST_N_ESTIMATORS,
                "threshold_high_risk": config.THRESHOLD_HIGH_RISK,
                "threshold_critical_risk": config.THRESHOLD_CRITICAL_RISK
            },
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        return {"error": str(e)}


# Startup handler
@mcp.on_startup
async def startup():
    """MCP server startup handler"""
    await initialize_system()


if __name__ == "__main__":
    # Run the MCP server
    logger.info("Starting Fraud Detection MCP Server v2.0...")
    mcp.run()