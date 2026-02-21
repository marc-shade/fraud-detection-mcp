#!/usr/bin/env python3
"""
Configuration management for fraud detection MCP
Handles environment variables, paths, and system settings
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class AppConfig(BaseSettings):
    """Application configuration with environment variable support"""

    # Application settings
    APP_NAME: str = "fraud-detection-mcp"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path.cwd())
    MODEL_DIR: Path = Field(default_factory=lambda: Path.cwd() / "models")
    DATA_DIR: Path = Field(default_factory=lambda: Path.cwd() / "data")
    TEST_DATA_DIR: Path = Field(default_factory=lambda: Path.cwd() / "test_data")
    LOG_DIR: Path = Field(default_factory=lambda: Path.cwd() / "logs")
    CACHE_DIR: Path = Field(default_factory=lambda: Path.cwd() / "cache")

    # Model settings
    ISOLATION_FOREST_CONTAMINATION: float = 0.1
    ISOLATION_FOREST_N_ESTIMATORS: int = 200
    XGBOOST_N_ESTIMATORS: int = 200
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1

    # Risk thresholds
    THRESHOLD_HIGH_AMOUNT: float = 10000.0
    THRESHOLD_CRITICAL_RISK: float = 0.8
    THRESHOLD_HIGH_RISK: float = 0.6
    THRESHOLD_MEDIUM_RISK: float = 0.4

    # High-risk indicators
    HIGH_RISK_LOCATIONS: list = Field(
        default_factory=lambda: ["unknown"]
    )
    HIGH_RISK_PAYMENT_METHODS: list = Field(
        default_factory=lambda: ["crypto", "prepaid_card", "money_order"]
    )
    UNUSUAL_HOURS_START: int = 0
    UNUSUAL_HOURS_END: int = 6

    # Network analysis settings
    NETWORK_HIGH_CONNECTIVITY_THRESHOLD: int = 50
    NETWORK_HIGH_BETWEENNESS_THRESHOLD: float = 0.1
    NETWORK_HIGH_CLUSTERING_THRESHOLD: float = 0.8

    # Performance settings
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    CACHE_TTL_SECONDS: int = 3600

    # Database settings (optional)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[str] = Field(default="redis://localhost:6379", env="REDIS_URL")

    # Security settings
    API_KEY_HEADER: str = "X-API-Key"
    JWT_SECRET_KEY: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Rate limiting
    RATE_LIMIT_FREE_TIER: str = "10/minute"
    RATE_LIMIT_PAID_TIER: str = "1000/minute"
    RATE_LIMIT_ENTERPRISE: str = "10000/minute"

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # MLflow settings
    MLFLOW_TRACKING_URI: Optional[str] = Field(
        default=None, env="MLFLOW_TRACKING_URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = "fraud-detection"

    @field_validator("MODEL_DIR", "DATA_DIR", "TEST_DATA_DIR", "LOG_DIR", "CACHE_DIR")
    @classmethod
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret(cls, v, info):
        """Generate JWT secret if not provided in production"""
        if v is None and info.data.get("ENVIRONMENT") == "production":
            import secrets
            return secrets.token_urlsafe(32)
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class ModelConfig:
    """Configuration for ML models"""

    def __init__(self, config: AppConfig):
        self.config = config

    def get_isolation_forest_params(self) -> Dict[str, Any]:
        """Get Isolation Forest hyperparameters"""
        return {
            "contamination": self.config.ISOLATION_FOREST_CONTAMINATION,
            "n_estimators": self.config.ISOLATION_FOREST_N_ESTIMATORS,
            "random_state": 42,
            "n_jobs": -1,
            "max_samples": 256
        }

    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost hyperparameters"""
        return {
            "n_estimators": self.config.XGBOOST_N_ESTIMATORS,
            "max_depth": self.config.XGBOOST_MAX_DEPTH,
            "learning_rate": self.config.XGBOOST_LEARNING_RATE,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "aucpr"
        }

    def get_autoencoder_params(self) -> Dict[str, Any]:
        """Get Autoencoder hyperparameters"""
        return {
            "hidden_dims": [64, 32, 16, 32, 64],
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 50,
            "dropout_rate": 0.2
        }

    def get_gnn_params(self) -> Dict[str, Any]:
        """Get Graph Neural Network hyperparameters"""
        return {
            "hidden_channels": 64,
            "num_layers": 3,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100,
            "dropout": 0.5
        }


# Global configuration instance
config = AppConfig()
model_config = ModelConfig(config)


def get_config() -> AppConfig:
    """Get global configuration instance"""
    return config


def get_model_config() -> ModelConfig:
    """Get model configuration instance"""
    return model_config


def update_config(**kwargs):
    """Update configuration with new values"""
    global config, model_config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    model_config = ModelConfig(config)


if __name__ == "__main__":
    # Print configuration for debugging
    print("Fraud Detection MCP Configuration")
    print("=" * 60)
    print(f"Environment: {config.ENVIRONMENT}")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Model Directory: {config.MODEL_DIR}")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Log Level: {config.LOG_LEVEL}")
    print(f"Metrics Enabled: {config.ENABLE_METRICS}")
    print("=" * 60)