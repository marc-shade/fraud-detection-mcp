#!/usr/bin/env python3
"""
Feature Engineering for Fraud Detection
Extracts 40+ features from transaction data with proper encoding
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime
from models_validation import TransactionData, BehavioralData, NetworkData
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Extracts comprehensive features from transaction data
    Handles cyclical encoding, categorical variables, and derived features
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self._payment_method_mapping: Dict[str, int] = {}
        self._location_mapping: Dict[str, int] = {}
        self._merchant_mapping: Dict[str, int] = {}
        self._initialized = False

    def fit(self, transactions: List[TransactionData]) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data to learn encodings

        Args:
            transactions: List of TransactionData objects

        Returns:
            self for method chaining
        """
        # Build mappings for categorical variables
        payment_methods = set()
        locations = set()
        merchants = set()

        for txn in transactions:
            payment_methods.add(txn.payment_method)
            locations.add(txn.location)
            merchants.add(txn.merchant)

        # Create mappings with 0 reserved for unknown
        self._payment_method_mapping = {pm: idx + 1 for idx, pm in enumerate(sorted(payment_methods))}
        self._location_mapping = {loc: idx + 1 for idx, loc in enumerate(sorted(locations))}
        self._merchant_mapping = {merch: idx + 1 for idx, merch in enumerate(sorted(merchants))}

        # Build feature names
        self._build_feature_names()
        self._initialized = True

        logger.info(f"FeatureEngineer fitted with {len(self.feature_names)} features")
        return self

    def transform(
        self,
        transaction: TransactionData,
        behavioral: BehavioralData = None,
        network: NetworkData = None
    ) -> np.ndarray:
        """
        Transform a single transaction into feature vector

        Args:
            transaction: TransactionData object
            behavioral: Optional behavioral data
            network: Optional network data

        Returns:
            numpy array of features
        """
        if not self._initialized:
            raise RuntimeError("FeatureEngineer must be fitted before transform")

        features = []

        # 1. Amount features (3 features)
        features.extend(self._extract_amount_features(transaction))

        # 2. Temporal features (12 features)
        features.extend(self._extract_temporal_features(transaction))

        # 3. Categorical features (6 features)
        features.extend(self._extract_categorical_features(transaction))

        # 4. Location features (2 features)
        features.extend(self._extract_location_features(transaction))

        # 5. Merchant features (2 features)
        features.extend(self._extract_merchant_features(transaction))

        # 6. Behavioral features (10 features)
        features.extend(self._extract_behavioral_features(behavioral))

        # 7. Network features (8 features)
        features.extend(self._extract_network_features(network))

        # 8. Derived features (3 features)
        features.extend(self._extract_derived_features(transaction))

        return np.array(features, dtype=np.float32)

    def fit_transform(
        self,
        transactions: List[TransactionData],
        behavioral_data: List[BehavioralData] = None,
        network_data: List[NetworkData] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit and transform in one step

        Args:
            transactions: List of transactions
            behavioral_data: Optional list of behavioral data
            network_data: Optional list of network data

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.fit(transactions)

        behavioral_data = behavioral_data or [None] * len(transactions)
        network_data = network_data or [None] * len(transactions)

        features = []
        for txn, behav, net in zip(transactions, behavioral_data, network_data):
            features.append(self.transform(txn, behav, net))

        return np.vstack(features), self.feature_names

    def _extract_amount_features(self, txn: TransactionData) -> List[float]:
        """Extract amount-based features"""
        amount = float(txn.amount)

        return [
            amount,  # Raw amount
            np.log1p(amount),  # Log transform (handles 0)
            np.sqrt(amount) if amount >= 0 else 0.0,  # Square root
        ]

    def _extract_temporal_features(self, txn: TransactionData) -> List[float]:
        """Extract time-based features with cyclical encoding"""
        dt = txn.timestamp

        # Hour (0-23) - cyclical encoding
        hour = dt.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Day of week (0-6) - cyclical encoding
        day_of_week = dt.weekday()
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        # Day of month (1-31) - cyclical encoding
        day_of_month = dt.day
        dom_sin = np.sin(2 * np.pi * day_of_month / 31)
        dom_cos = np.cos(2 * np.pi * day_of_month / 31)

        # Month (1-12) - cyclical encoding
        month = dt.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Binary features
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        is_night = 1.0 if hour < 6 or hour >= 22 else 0.0

        return [
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            dom_sin, dom_cos,
            month_sin, month_cos,
            is_weekend,
            is_night,
            float(hour),  # Raw hour for some models
            float(day_of_week)  # Raw day for some models
        ]

    def _extract_categorical_features(self, txn: TransactionData) -> List[float]:
        """Extract categorical features with learned encoding"""
        # Payment method encoding
        payment_method_encoded = float(
            self._payment_method_mapping.get(txn.payment_method, 0)
        )

        # One-hot-like encoding for common payment methods
        is_credit = 1.0 if 'credit' in str(txn.payment_method).lower() else 0.0
        is_debit = 1.0 if 'debit' in str(txn.payment_method).lower() else 0.0
        is_crypto = 1.0 if 'crypto' in str(txn.payment_method).lower() else 0.0
        is_wire = 1.0 if 'wire' in str(txn.payment_method).lower() else 0.0
        is_online = 1.0 if str(txn.payment_method).lower() in ['paypal', 'crypto'] else 0.0

        return [
            payment_method_encoded,
            is_credit,
            is_debit,
            is_crypto,
            is_wire,
            is_online
        ]

    def _extract_location_features(self, txn: TransactionData) -> List[float]:
        """Extract location features"""
        location_encoded = float(
            self._location_mapping.get(txn.location, 0)
        )

        # Location string length as proxy for specificity
        location_length = len(txn.location) if txn.location else 0.0

        return [
            location_encoded,
            float(location_length)
        ]

    def _extract_merchant_features(self, txn: TransactionData) -> List[float]:
        """Extract merchant features"""
        merchant_encoded = float(
            self._merchant_mapping.get(txn.merchant, 0)
        )

        # Merchant string length
        merchant_length = len(txn.merchant) if txn.merchant else 0.0

        return [
            merchant_encoded,
            float(merchant_length)
        ]

    def _extract_behavioral_features(self, behavioral: BehavioralData) -> List[float]:
        """Extract behavioral biometrics features"""
        if behavioral is None:
            return [0.0] * 10

        features = []

        # Session duration
        session_duration = float(behavioral.session_duration or 0.0)
        features.append(session_duration)

        # Keystroke dynamics features
        if behavioral.keystroke_dynamics and len(behavioral.keystroke_dynamics) > 0:
            events = behavioral.keystroke_dynamics

            # Calculate timing statistics
            hold_times = [e.release_time - e.press_time for e in events]
            flight_times = [
                events[i+1].press_time - events[i].release_time
                for i in range(len(events) - 1)
            ]

            features.extend([
                np.mean(hold_times) if hold_times else 0.0,
                np.std(hold_times) if len(hold_times) > 1 else 0.0,
                np.mean(flight_times) if flight_times else 0.0,
                np.std(flight_times) if len(flight_times) > 1 else 0.0,
                float(len(events))
            ])
        else:
            features.extend([0.0] * 5)

        # Mouse patterns
        if behavioral.mouse_patterns and len(behavioral.mouse_patterns) > 0:
            features.extend([
                float(len(behavioral.mouse_patterns)),
                1.0  # Has mouse data
            ])
        else:
            features.extend([0.0, 0.0])

        # Session indicator
        has_session = 1.0 if behavioral.session_id else 0.0
        features.append(has_session)

        return features

    def _extract_network_features(self, network: NetworkData) -> List[float]:
        """Extract network graph features"""
        if network is None:
            return [0.0] * 8

        features = []

        # Connection count
        conn_count = float(len(network.connections))
        features.append(conn_count)

        # Connection statistics
        if network.connections:
            # Strength statistics
            strengths = [
                float(c.get('strength', 0.5))
                for c in network.connections
            ]
            features.extend([
                np.mean(strengths),
                np.std(strengths) if len(strengths) > 1 else 0.0,
                np.max(strengths) if strengths else 0.0,
                np.min(strengths) if strengths else 0.0
            ])

            # Transaction count statistics
            tx_counts = [
                float(c.get('transaction_count', 0))
                for c in network.connections
            ]
            features.extend([
                np.mean(tx_counts),
                np.max(tx_counts) if tx_counts else 0.0
            ])
        else:
            features.extend([0.0] * 6)

        # Entity type indicator
        is_user = 1.0 if network.entity_type == 'user' else 0.0
        features.append(is_user)

        return features

    def _extract_derived_features(self, txn: TransactionData) -> List[float]:
        """Extract derived/interaction features"""
        amount = float(txn.amount)
        hour = txn.timestamp.hour
        is_weekend = 1.0 if txn.timestamp.weekday() >= 5 else 0.0

        # Amount * night time interaction
        is_night = 1.0 if hour < 6 or hour >= 22 else 0.0
        amount_night = amount * is_night

        # Amount * weekend interaction
        amount_weekend = amount * is_weekend

        # High amount indicator
        high_amount = 1.0 if amount > 10000.0 else 0.0

        return [
            amount_night,
            amount_weekend,
            high_amount
        ]

    def _build_feature_names(self):
        """Build comprehensive list of feature names"""
        self.feature_names = [
            # Amount features (3)
            'amount',
            'amount_log',
            'amount_sqrt',

            # Temporal features (12)
            'hour_sin',
            'hour_cos',
            'day_of_week_sin',
            'day_of_week_cos',
            'day_of_month_sin',
            'day_of_month_cos',
            'month_sin',
            'month_cos',
            'is_weekend',
            'is_night',
            'hour_raw',
            'day_of_week_raw',

            # Categorical features (6)
            'payment_method_encoded',
            'is_credit',
            'is_debit',
            'is_crypto',
            'is_wire',
            'is_online_payment',

            # Location features (2)
            'location_encoded',
            'location_length',

            # Merchant features (2)
            'merchant_encoded',
            'merchant_length',

            # Behavioral features (10)
            'session_duration',
            'keystroke_hold_mean',
            'keystroke_hold_std',
            'keystroke_flight_mean',
            'keystroke_flight_std',
            'keystroke_count',
            'mouse_pattern_count',
            'has_mouse_data',
            'has_session_id',
            'behavioral_available',

            # Network features (8)
            'connection_count',
            'connection_strength_mean',
            'connection_strength_std',
            'connection_strength_max',
            'connection_strength_min',
            'connection_tx_count_mean',
            'connection_tx_count_max',
            'is_user_entity',

            # Derived features (3)
            'amount_night_interaction',
            'amount_weekend_interaction',
            'high_amount_flag'
        ]

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()

    def get_feature_importance_map(self) -> Dict[str, str]:
        """Get mapping of features to their categories"""
        return {
            'amount': 'Amount Features',
            'temporal': 'Temporal Features',
            'categorical': 'Categorical Features',
            'behavioral': 'Behavioral Features',
            'network': 'Network Features',
            'derived': 'Derived Features'
        }


# Utility function for batch processing
def extract_features_batch(
    transactions: List[TransactionData],
    behavioral_data: List[BehavioralData] = None,
    network_data: List[NetworkData] = None,
    feature_engineer: FeatureEngineer = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from batch of transactions

    Args:
        transactions: List of transactions
        behavioral_data: Optional behavioral data
        network_data: Optional network data
        feature_engineer: Pre-fitted FeatureEngineer (will create new if None)

    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(transactions)

    return feature_engineer.fit_transform(
        transactions,
        behavioral_data,
        network_data
    )


__all__ = ['FeatureEngineer', 'extract_features_batch']