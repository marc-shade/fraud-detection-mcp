#!/usr/bin/env python3
"""
Integration module for Fraud Detection MCP with Synthetic Data MCP
Provides seamless data generation and analysis pipeline for stored data fraud detection
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataIntegration:
    """Integration with synthetic-data-mcp for fraud detection testing"""

    def __init__(self):
        import os

        base_dir = os.environ.get(
            "FRAUD_DETECTION_DATA_DIR", str(Path(__file__).parent / "test_data")
        )
        self.output_dir = Path(base_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_fraud_patterns(self) -> Dict[str, Any]:
        """Generate comprehensive fraud patterns for synthetic data creation"""

        fraud_patterns = {
            "transaction_fraud": {
                "high_amount_patterns": {
                    "description": "Unusually high transaction amounts",
                    "amount_ranges": [(5000, 50000), (50000, 100000), (100000, 500000)],
                    "frequency": 0.02,  # 2% of transactions
                    "merchants": [
                        "Electronics Warehouse",
                        "Jewelry Store",
                        "Cash Advance",
                    ],
                    "risk_indicators": ["amount_anomaly", "merchant_risk"],
                },
                "velocity_fraud": {
                    "description": "Multiple transactions in short time periods",
                    "pattern": "burst",
                    "transaction_count": (5, 20),
                    "time_window": (60, 300),  # 1-5 minutes
                    "amount_range": (100, 2000),
                    "frequency": 0.01,
                    "risk_indicators": ["velocity_anomaly", "time_clustering"],
                },
                "geographic_anomaly": {
                    "description": "Transactions from unusual locations",
                    "high_risk_locations": [
                        "Unknown Location",
                        "High Risk Country A",
                        "Offshore Territory",
                        "Sanctioned Region",
                    ],
                    "frequency": 0.03,
                    "risk_indicators": ["geographic_risk", "location_anomaly"],
                },
                "temporal_anomaly": {
                    "description": "Transactions at unusual times",
                    "unusual_hours": [2, 3, 4, 5],  # 2 AM - 5 AM
                    "unusual_days": ["Sunday", "Holiday"],
                    "frequency": 0.015,
                    "risk_indicators": ["time_anomaly", "schedule_deviation"],
                },
                "payment_method_fraud": {
                    "description": "High-risk payment methods",
                    "high_risk_methods": ["crypto", "prepaid_card", "money_order"],
                    "frequency": 0.025,
                    "risk_indicators": ["payment_method_risk"],
                },
            },
            "behavioral_fraud": {
                "keystroke_anomalies": {
                    "description": "Abnormal typing patterns indicating account takeover",
                    "patterns": {
                        "too_slow": {
                            "dwell_time_multiplier": 3.0,
                            "flight_time_multiplier": 2.5,
                        },
                        "too_fast": {
                            "dwell_time_multiplier": 0.3,
                            "flight_time_multiplier": 0.4,
                        },
                        "irregular": {
                            "variation_multiplier": 5.0,
                            "consistency_factor": 0.2,
                        },
                    },
                    "frequency": 0.008,
                    "risk_indicators": ["keystroke_anomaly", "behavioral_deviation"],
                },
                "session_anomalies": {
                    "description": "Unusual session behavior patterns",
                    "patterns": {
                        "too_quick": {"session_duration": (5, 30)},  # Seconds
                        "unusual_navigation": {
                            "page_jumps": True,
                            "back_button_abuse": True,
                        },
                        "copy_paste_heavy": {"copy_paste_ratio": 0.8},
                    },
                    "frequency": 0.012,
                    "risk_indicators": ["session_anomaly", "navigation_risk"],
                },
            },
            "network_fraud": {
                "fraud_rings": {
                    "description": "Coordinated fraud networks",
                    "ring_sizes": [3, 5, 8, 12, 20],
                    "connection_patterns": ["star", "mesh", "chain"],
                    "transaction_patterns": {
                        "money_laundering": {
                            "circular_transfers": True,
                            "amount_structuring": True,
                        },
                        "account_farming": {
                            "new_accounts": True,
                            "similar_patterns": True,
                        },
                    },
                    "frequency": 0.005,
                    "risk_indicators": ["network_clustering", "coordinated_activity"],
                },
                "synthetic_identity": {
                    "description": "Fake identity creation and usage",
                    "identity_indicators": {
                        "new_account": True,
                        "minimal_history": True,
                        "inconsistent_data": True,
                    },
                    "frequency": 0.007,
                    "risk_indicators": ["identity_risk", "new_account_risk"],
                },
            },
        }

        return fraud_patterns

    def create_dataset_schema(self) -> Dict[str, Any]:
        """Create standardized schema for fraud detection datasets"""

        schema = {
            "transaction_data": {
                "required_fields": [
                    "transaction_id",
                    "user_id",
                    "amount",
                    "merchant",
                    "merchant_category",
                    "location",
                    "timestamp",
                    "payment_method",
                ],
                "optional_fields": [
                    "device_id",
                    "ip_address",
                    "user_agent",
                    "geolocation_lat",
                    "geolocation_lon",
                    "merchant_id",
                    "card_type",
                    "currency",
                ],
                "data_types": {
                    "transaction_id": "string",
                    "user_id": "string",
                    "amount": "float",
                    "merchant": "string",
                    "location": "string",
                    "timestamp": "datetime",
                    "payment_method": "categorical",
                },
            },
            "behavioral_data": {
                "keystroke_dynamics": {
                    "fields": [
                        "key",
                        "press_time",
                        "release_time",
                        "user_id",
                        "session_id",
                    ],
                    "data_types": {
                        "key": "string",
                        "press_time": "integer",
                        "release_time": "integer",
                        "user_id": "string",
                        "session_id": "string",
                    },
                },
                "session_data": {
                    "fields": [
                        "session_id",
                        "user_id",
                        "start_time",
                        "end_time",
                        "pages_visited",
                        "actions_taken",
                        "form_interactions",
                        "copy_paste_events",
                        "idle_time",
                    ]
                },
            },
            "network_data": {
                "fields": [
                    "entity_id",
                    "entity_type",
                    "connected_entities",
                    "relationship_type",
                    "relationship_strength",
                    "transaction_count",
                    "total_amount",
                ]
            },
            "labels": {
                "is_fraud": "boolean",
                "fraud_type": "categorical",
                "fraud_confidence": "float",
                "manual_review": "boolean",
            },
        }

        return schema

    def generate_comprehensive_test_dataset(
        self,
        num_transactions: int = 10000,
        fraud_percentage: float = 5.0,
        include_behavioral: bool = True,
        include_network: bool = True,
        output_format: str = "csv",
    ) -> Dict[str, Any]:
        """Generate comprehensive test dataset with all fraud patterns"""

        try:
            fraud_patterns = self.generate_fraud_patterns()
            schema = self.create_dataset_schema()

            # Calculate distribution
            num_fraud = int(num_transactions * fraud_percentage / 100)
            num_legitimate = num_transactions - num_fraud

            transactions = []
            behavioral_data = []
            network_data = []

            # Generate legitimate transactions
            for i in range(num_legitimate):
                transaction = self._generate_legitimate_transaction(i)
                transactions.append(transaction)

                if include_behavioral:
                    behavioral = self._generate_normal_behavioral_data(
                        transaction["user_id"]
                    )
                    behavioral_data.extend(behavioral)

            # Generate fraudulent transactions with specific patterns
            fraud_types = list(fraud_patterns["transaction_fraud"].keys())

            for i in range(num_fraud):
                fraud_type = np.random.choice(np.array(fraud_types))
                pattern = fraud_patterns["transaction_fraud"][fraud_type]

                transaction = self._generate_fraudulent_transaction(
                    i + num_legitimate, fraud_type, pattern
                )
                transactions.append(transaction)

                if include_behavioral:
                    behavioral = self._generate_anomalous_behavioral_data(
                        transaction["user_id"], fraud_type
                    )
                    behavioral_data.extend(behavioral)

            # Generate network connections if requested
            if include_network:
                network_data = self._generate_network_connections(
                    transactions, fraud_patterns
                )

            # Create DataFrames
            transactions_df = pd.DataFrame(transactions)

            # Save datasets
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"fraud_test_dataset_{timestamp}"

            # Save transaction data
            if output_format == "csv":
                transactions_path = (
                    self.output_dir / f"{base_filename}_transactions.csv"
                )
                transactions_df.to_csv(transactions_path, index=False)
            else:
                transactions_path = (
                    self.output_dir / f"{base_filename}_transactions.json"
                )
                transactions_df.to_json(transactions_path, orient="records", indent=2)

            # Save behavioral data if generated
            behavioral_path = None
            if include_behavioral and behavioral_data:
                behavioral_df = pd.DataFrame(behavioral_data)
                if output_format == "csv":
                    behavioral_path = (
                        self.output_dir / f"{base_filename}_behavioral.csv"
                    )
                    behavioral_df.to_csv(behavioral_path, index=False)
                else:
                    behavioral_path = (
                        self.output_dir / f"{base_filename}_behavioral.json"
                    )
                    behavioral_df.to_json(behavioral_path, orient="records", indent=2)

            # Save network data if generated
            network_path = None
            if include_network and network_data:
                network_df = pd.DataFrame(network_data)
                if output_format == "csv":
                    network_path = self.output_dir / f"{base_filename}_network.csv"
                    network_df.to_csv(network_path, index=False)
                else:
                    network_path = self.output_dir / f"{base_filename}_network.json"
                    network_df.to_json(network_path, orient="records", indent=2)

            # Generate statistics
            fraud_distribution = transactions_df["fraud_type"].value_counts().to_dict()

            result = {
                "generation_info": {
                    "total_transactions": num_transactions,
                    "legitimate_transactions": num_legitimate,
                    "fraudulent_transactions": num_fraud,
                    "fraud_percentage": fraud_percentage,
                    "generation_timestamp": datetime.now().isoformat(),
                    "includes_behavioral": include_behavioral,
                    "includes_network": include_network,
                },
                "dataset_paths": {
                    "transactions": str(transactions_path),
                    "behavioral": str(behavioral_path) if behavioral_path else None,
                    "network": str(network_path) if network_path else None,
                },
                "fraud_distribution": fraud_distribution,
                "schema_compliance": self._validate_schema_compliance(
                    transactions_df, schema
                ),
                "integration_status": "success",
                "ready_for_analysis": True,
            }

            return result

        except Exception as e:
            logger.error(f"Comprehensive dataset generation failed: {e}")
            return {
                "error": str(e),
                "status": "generation_failed",
                "integration_status": "error",
            }

    def _generate_legitimate_transaction(self, index: int) -> Dict[str, Any]:
        """Generate a legitimate transaction"""
        return {
            "transaction_id": f"legit_{index:08d}",
            "user_id": f"user_{np.random.randint(1000, 9999)}",
            "amount": round(
                np.random.lognormal(4.0, 1.0), 2
            ),  # Log-normal distribution
            "merchant": np.random.choice(
                [
                    "Grocery Store",
                    "Gas Station",
                    "Coffee Shop",
                    "Restaurant",
                    "Pharmacy",
                    "Department Store",
                    "Online Retailer",
                ]
            ),
            "merchant_category": "retail",
            "location": np.random.choice(
                [
                    "New York, NY",
                    "Los Angeles, CA",
                    "Chicago, IL",
                    "Houston, TX",
                    "Phoenix, AZ",
                    "Philadelphia, PA",
                    "San Antonio, TX",
                ]
            ),
            "timestamp": (
                datetime.now() - timedelta(days=np.random.randint(0, 365))
            ).isoformat(),
            "payment_method": np.random.choice(
                np.array(["credit_card", "debit_card", "bank_transfer"])
            ),
            "is_fraud": False,
            "fraud_type": "none",
            "fraud_confidence": 0.0,
        }

    def _generate_fraudulent_transaction(
        self, index: int, fraud_type: str, pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a fraudulent transaction with specific pattern"""

        base_transaction = {
            "transaction_id": f"fraud_{index:08d}",
            "user_id": f"user_{np.random.randint(1000, 9999)}",
            "amount": 100.0,  # Default, will be overridden
            "merchant": "Suspicious Merchant",
            "merchant_category": "high_risk",
            "location": "Unknown Location",
            "timestamp": datetime.now().isoformat(),
            "payment_method": "credit_card",
            "is_fraud": True,
            "fraud_type": fraud_type,
            "fraud_confidence": 0.9,
        }

        # Apply specific fraud pattern
        if fraud_type == "high_amount_patterns":
            amount_range = pattern["amount_ranges"][
                np.random.randint(0, len(pattern["amount_ranges"]))
            ]
            base_transaction["amount"] = round(np.random.uniform(*amount_range), 2)
            base_transaction["merchant"] = np.random.choice(
                np.array(pattern["merchants"])
            )

        elif fraud_type == "velocity_fraud":
            base_transaction["amount"] = round(
                np.random.uniform(*pattern["amount_range"]), 2
            )
            # Simulate burst timing
            base_transaction["timestamp"] = (
                datetime.now() - timedelta(seconds=np.random.randint(60, 300))
            ).isoformat()

        elif fraud_type == "geographic_anomaly":
            base_transaction["location"] = np.random.choice(
                np.array(pattern["high_risk_locations"])
            )

        elif fraud_type == "temporal_anomaly":
            unusual_hour = np.random.choice(np.array(pattern["unusual_hours"]))
            anomalous_time = datetime.now().replace(
                hour=unusual_hour, minute=np.random.randint(0, 59)
            )
            base_transaction["timestamp"] = anomalous_time.isoformat()

        elif fraud_type == "payment_method_fraud":
            base_transaction["payment_method"] = np.random.choice(
                pattern["high_risk_methods"]
            )

        return base_transaction

    def _generate_normal_behavioral_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate normal behavioral patterns"""
        behavioral_data = []

        # Normal keystroke patterns
        base_dwell = 80  # milliseconds

        for i, key in enumerate("password123"):
            press_time = 1000 + i * 150 + np.random.normal(0, 20)
            dwell_time = base_dwell + np.random.normal(0, 15)

            behavioral_data.append(
                {
                    "user_id": user_id,
                    "session_id": f"session_{user_id}",
                    "key": key,
                    "press_time": int(press_time),
                    "release_time": int(press_time + dwell_time),
                    "is_anomaly": False,
                }
            )

        return behavioral_data

    def _generate_anomalous_behavioral_data(
        self, user_id: str, fraud_type: str
    ) -> List[Dict[str, Any]]:
        """Generate anomalous behavioral patterns"""
        behavioral_data = []

        # Anomalous keystroke patterns (account takeover simulation)
        if fraud_type in ["high_amount_patterns", "velocity_fraud"]:
            # Simulate different user typing
            base_dwell = 150  # Much slower

            for i, key in enumerate("password123"):
                press_time = 1000 + i * 400 + np.random.normal(0, 100)  # More variation
                dwell_time = base_dwell + np.random.normal(0, 50)

                behavioral_data.append(
                    {
                        "user_id": user_id,
                        "session_id": f"session_{user_id}",
                        "key": key,
                        "press_time": int(press_time),
                        "release_time": int(press_time + dwell_time),
                        "is_anomaly": True,
                    }
                )

        return behavioral_data

    def _generate_network_connections(
        self, transactions: List[Dict[str, Any]], fraud_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate network connection data"""
        network_data = []

        # Create user networks
        users = list(set(t["user_id"] for t in transactions))
        fraud_users = [t["user_id"] for t in transactions if t["is_fraud"]]

        # Create fraud rings
        if len(fraud_users) >= 3:
            # Group fraud users into rings
            ring_size = min(5, len(fraud_users))
            fraud_ring = fraud_users[:ring_size]

            # Create connections within the ring
            for i, user1 in enumerate(fraud_ring):
                for user2 in fraud_ring[i + 1 :]:
                    network_data.append(
                        {
                            "entity_id": user1,
                            "connected_entity": user2,
                            "relationship_type": "frequent_interaction",
                            "relationship_strength": 0.8,
                            "transaction_count": np.random.randint(5, 20),
                            "is_suspicious": True,
                        }
                    )

        # Create normal connections
        normal_users = [u for u in users if u not in fraud_users]
        for user in normal_users[:20]:  # Limit for performance
            # Each user connects to 1-3 others normally
            num_connections = np.random.randint(1, 4)
            connected_users = np.random.choice(
                [u for u in normal_users if u != user],
                size=min(num_connections, len(normal_users) - 1),
                replace=False,
            )

            for connected_user in connected_users:
                network_data.append(
                    {
                        "entity_id": user,
                        "connected_entity": connected_user,
                        "relationship_type": "normal_interaction",
                        "relationship_strength": 0.3,
                        "transaction_count": np.random.randint(1, 5),
                        "is_suspicious": False,
                    }
                )

        return network_data

    def _validate_schema_compliance(
        self, df: pd.DataFrame, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate dataset compliance with schema"""

        required_fields = schema["transaction_data"]["required_fields"]
        optional_fields = schema["transaction_data"]["optional_fields"]

        compliance = {
            "has_required_fields": all(
                field in df.columns for field in required_fields
            ),
            "missing_required_fields": [
                field for field in required_fields if field not in df.columns
            ],
            "has_optional_fields": [
                field for field in optional_fields if field in df.columns
            ],
            "data_quality": {
                "null_values": df.isnull().sum().to_dict(),
                "duplicate_transactions": df["transaction_id"].duplicated().sum(),
                "data_types_correct": True,  # Simplified validation
            },
            "fraud_labels_present": "is_fraud" in df.columns,
            "schema_version": "1.0",
        }

        return compliance
