#!/usr/bin/env python3
"""
Pydantic validation models for fraud detection inputs
Ensures all inputs are validated and sanitized
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum


class PaymentMethod(str, Enum):
    """Valid payment methods"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"
    PAYPAL = "paypal"
    WIRE_TRANSFER = "wire_transfer"
    CHECK = "check"
    CASH = "cash"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class KeystrokeEvent(BaseModel):
    """Single keystroke event with timing"""
    key: str = Field(..., max_length=10)
    press_time: int = Field(..., ge=0, le=10**15)  # Unix timestamp in ms
    release_time: int = Field(..., ge=0, le=10**15)

    @field_validator('release_time')
    @classmethod
    def validate_release_after_press(cls, v, info):
        """Ensure release happens after press"""
        if 'press_time' in info.data and v < info.data['press_time']:
            raise ValueError('Release time must be after press time')
        return v

    @field_validator('key')
    @classmethod
    def sanitize_key(cls, v):
        """Remove potentially dangerous characters"""
        if any(char in v for char in ['<', '>', '&', '"', "'"]):
            raise ValueError('Invalid characters in key field')
        return v


class TransactionData(BaseModel):
    """Validated transaction data"""

    transaction_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Unique transaction identifier"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="User identifier"
    )
    amount: float = Field(
        ...,
        gt=0.0,
        le=10_000_000.0,  # $10M max
        description="Transaction amount in USD"
    )
    merchant: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Merchant name"
    )
    merchant_category: Optional[str] = Field(
        None,
        max_length=50,
        description="Merchant category code"
    )
    location: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Transaction location"
    )
    timestamp: datetime = Field(
        ...,
        description="Transaction timestamp"
    )
    payment_method: PaymentMethod = Field(
        ...,
        description="Payment method used"
    )

    # Optional fields
    device_id: Optional[str] = Field(
        None,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$'
    )
    ip_address: Optional[str] = Field(
        None,
        pattern=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'  # IPv4
    )
    card_type: Optional[str] = Field(None, max_length=50)
    currency: Optional[str] = Field("USD", max_length=3, pattern=r'^[A-Z]{3}$')
    merchant_id: Optional[str] = Field(None, max_length=100)

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Ensure timestamp is reasonable"""
        now = datetime.now()

        # Not in future
        if v > now + timedelta(hours=1):  # Allow 1 hour clock skew
            raise ValueError('Timestamp cannot be in the future')

        # Not too old (reject >1 year old transactions)
        one_year_ago = now - timedelta(days=365)
        if v < one_year_ago:
            raise ValueError('Timestamp too old (>1 year)')

        return v

    @field_validator('merchant', 'location', 'merchant_category')
    @classmethod
    def sanitize_text_fields(cls, v):
        """Remove potentially dangerous characters"""
        if v is None:
            return v

        dangerous = [
            '<script>', 'javascript:', '--', ';--',
            'DROP', 'DELETE', 'INSERT', 'UPDATE',
            '<', '>', 'onclick', 'onerror'
        ]
        v_upper = v.upper()

        for dangerous_str in dangerous:
            if dangerous_str in v_upper:
                raise ValueError(f'Invalid characters detected')

        return v.strip()

    @field_validator('amount')
    @classmethod
    def validate_amount_precision(cls, v):
        """Ensure amount has reasonable precision"""
        # Round to 2 decimal places
        return round(v, 2)

    @model_validator(mode='after')
    def validate_transaction_logic(self):
        """Cross-field validation"""
        amount = self.amount
        payment_method = self.payment_method

        # Large crypto transactions require additional validation
        if payment_method == PaymentMethod.CRYPTO and amount and amount > 50000:
            # In production, would check for additional KYC fields
            pass

        return self

    class Config:
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BehavioralData(BaseModel):
    """Validated behavioral biometrics data"""

    keystroke_dynamics: Optional[List[KeystrokeEvent]] = Field(
        None,
        max_items=10000,  # Prevent memory exhaustion
        description="Keystroke timing events"
    )
    mouse_patterns: Optional[List[Dict[str, Any]]] = Field(
        None,
        max_items=10000,
        description="Mouse movement patterns"
    )
    session_duration: Optional[int] = Field(
        None,
        ge=0,
        le=86400,  # Max 24 hours
        description="Session duration in seconds"
    )
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Session identifier"
    )

    @field_validator('keystroke_dynamics')
    @classmethod
    def validate_keystroke_data(cls, v):
        """Validate keystroke event sequence"""
        if v is None or len(v) == 0:
            return v

        # Check for reasonable timing
        for i in range(len(v) - 1):
            time_gap = v[i+1].press_time - v[i].release_time

            # Unrealistic if gap > 60 seconds between keys
            if time_gap > 60000:
                raise ValueError('Unrealistic timing between keystrokes')

            # Unrealistic if negative (time travel)
            if time_gap < 0:
                raise ValueError('Keystroke events out of order')

        return v


class NetworkData(BaseModel):
    """Validated network connection data"""

    entity_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Entity identifier"
    )
    entity_type: Optional[str] = Field(
        "user",
        max_length=50,
        description="Type of entity (user, merchant, device)"
    )
    connections: List[Dict[str, Any]] = Field(
        default_factory=list,
        max_items=10000,
        description="List of connected entities"
    )

    @field_validator('connections')
    @classmethod
    def validate_connections(cls, v):
        """Validate connection structure"""
        for conn in v:
            if 'entity_id' not in conn:
                raise ValueError('Connection missing entity_id')

            # Validate entity_id format
            entity_id = str(conn['entity_id'])
            if not entity_id or len(entity_id) > 100:
                raise ValueError('Invalid entity_id in connection')

            # Validate optional fields
            if 'strength' in conn:
                strength = float(conn['strength'])
                if not (0.0 <= strength <= 1.0):
                    raise ValueError('Connection strength must be between 0 and 1')

            if 'transaction_count' in conn:
                count = int(conn['transaction_count'])
                if count < 0:
                    raise ValueError('Transaction count cannot be negative')

        return v


class AnalysisRequest(BaseModel):
    """Complete analysis request with all data types"""

    transaction_data: TransactionData
    behavioral_data: Optional[BehavioralData] = None
    network_data: Optional[NetworkData] = None
    include_explanation: bool = Field(
        default=True,
        description="Include explainable AI reasoning"
    )

    class Config:
        validate_assignment = True


class AnalysisResponse(BaseModel):
    """Standardized analysis response"""

    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_anomalies: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    model_version: str = Field(default="2.0.0")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    # Component scores
    transaction_risk_score: Optional[float] = None
    behavioral_risk_score: Optional[float] = None
    network_risk_score: Optional[float] = None

    # Performance metrics
    processing_time_ms: Optional[float] = None

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingRequest(BaseModel):
    """Request to train fraud detection models"""

    training_data_path: str = Field(
        ...,
        description="Path to training data CSV/JSON"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Proportion of data for testing"
    )
    validation_size: float = Field(
        default=0.1,
        ge=0.0,
        le=0.3,
        description="Proportion of data for validation"
    )
    use_smote: bool = Field(
        default=True,
        description="Use SMOTE for class balancing"
    )
    enable_mlflow: bool = Field(
        default=True,
        description="Log to MLflow"
    )

    @model_validator(mode='after')
    def validate_splits(self):
        """Ensure train/test/val splits sum to <= 1"""
        test = self.test_size
        val = self.validation_size

        if test + val >= 1.0:
            raise ValueError('Test + validation size must be < 1.0')

        return self


class BatchAnalysisRequest(BaseModel):
    """Request for batch transaction analysis"""

    transactions: List[TransactionData] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="Batch of transactions to analyze"
    )
    parallel: bool = Field(
        default=True,
        description="Process transactions in parallel"
    )

    @field_validator('transactions')
    @classmethod
    def validate_unique_ids(cls, v):
        """Ensure transaction IDs are unique"""
        ids = [t.transaction_id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError('Transaction IDs must be unique')
        return v


# Export all validation models
__all__ = [
    'PaymentMethod',
    'RiskLevel',
    'KeystrokeEvent',
    'TransactionData',
    'BehavioralData',
    'NetworkData',
    'AnalysisRequest',
    'AnalysisResponse',
    'TrainingRequest',
    'BatchAnalysisRequest'
]