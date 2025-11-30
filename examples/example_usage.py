#!/usr/bin/env python3
"""
Example usage of the Advanced Fraud Detection MCP
Demonstrates all major capabilities and real-world scenarios
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Example MCP client (would be imported from your MCP client library)
class MockMCPClient:
    """Mock MCP client for demonstration purposes"""

    def __init__(self, server_url: str = "localhost:8080"):
        self.server_url = server_url

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool call - in real implementation, this would call the MCP server"""
        print(f"🔧 Calling tool: {tool_name}")
        print(f"📊 Parameters: {json.dumps(parameters, indent=2)}")

        # Mock response based on tool
        if tool_name == "analyze_transaction":
            return self._mock_transaction_analysis(parameters)
        elif tool_name == "detect_behavioral_anomaly":
            return self._mock_behavioral_analysis(parameters)
        elif tool_name == "generate_risk_score":
            return self._mock_comprehensive_analysis(parameters)
        else:
            return {"error": "Unknown tool", "status": "failed"}

    def _mock_transaction_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock transaction analysis response"""
        amount = params.get("transaction_data", {}).get("amount", 0)

        # Simulate risk based on amount
        if amount > 10000:
            risk_score = 0.85
            risk_level = "HIGH"
            anomalies = ["high_amount_transaction", "unusual_merchant_category"]
        elif amount > 1000:
            risk_score = 0.45
            risk_level = "MEDIUM"
            anomalies = ["above_average_amount"]
        else:
            risk_score = 0.15
            risk_level = "LOW"
            anomalies = []

        return {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "detected_anomalies": anomalies,
            "explanation": f"Transaction analysis complete. Amount: ${amount:,.2f}",
            "recommended_actions": ["monitor_closely"] if risk_level == "MEDIUM" else ["allow_transaction"],
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _mock_behavioral_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock behavioral analysis response"""
        keystroke_data = params.get("behavioral_data", {}).get("keystroke_dynamics", [])

        if len(keystroke_data) > 5:
            anomaly_score = 0.3
            detected_anomalies = []
        else:
            anomaly_score = 0.7
            detected_anomalies = ["insufficient_behavioral_data"]

        return {
            "overall_anomaly_score": anomaly_score,
            "detected_anomalies": detected_anomalies,
            "confidence": 0.88,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _mock_comprehensive_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock comprehensive analysis response"""
        amount = params.get("transaction_data", {}).get("amount", 0)
        has_behavioral = params.get("behavioral_data") is not None
        has_network = params.get("network_data") is not None

        # Calculate mock comprehensive score
        base_score = 0.3 if amount > 5000 else 0.1
        if has_behavioral:
            base_score += 0.1
        if has_network:
            base_score += 0.15

        overall_score = min(1.0, base_score)

        if overall_score >= 0.6:
            risk_level = "HIGH"
            actions = ["require_additional_verification", "flag_for_review"]
        elif overall_score >= 0.4:
            risk_level = "MEDIUM"
            actions = ["monitor_closely"]
        else:
            risk_level = "LOW"
            actions = ["allow_transaction"]

        return {
            "overall_risk_score": overall_score,
            "risk_level": risk_level,
            "component_scores": {
                "transaction": base_score - (0.1 if has_behavioral else 0) - (0.15 if has_network else 0),
                "behavioral": 0.1 if has_behavioral else None,
                "network": 0.15 if has_network else None
            },
            "recommended_actions": actions,
            "comprehensive_explanation": f"Multi-modal analysis complete with {2 + has_behavioral + has_network} components",
            "analysis_timestamp": datetime.now().isoformat()
        }

async def example_basic_transaction_analysis():
    """Example 1: Basic transaction fraud analysis"""
    print("🚀 Example 1: Basic Transaction Analysis")
    print("=" * 50)

    client = MockMCPClient()

    # Sample transaction data
    transaction_data = {
        "transaction_id": "txn_2025_001",
        "amount": 15000.00,
        "merchant": "Electronics Warehouse",
        "merchant_category": "electronics",
        "location": "Las Vegas, NV",
        "timestamp": datetime.now().isoformat(),
        "payment_method": "credit_card",
        "card_type": "visa",
        "user_id": "user_12345"
    }

    # Analyze transaction
    result = await client.call_tool("analyze_transaction", {
        "transaction_data": transaction_data,
        "include_behavioral": False
    })

    print("📊 Analysis Result:")
    print(f"   Risk Score: {result.get('overall_risk_score', 0):.2f}")
    print(f"   Risk Level: {result.get('risk_level', 'UNKNOWN')}")
    print(f"   Anomalies: {', '.join(result.get('detected_anomalies', []))}")
    print(f"   Recommendation: {', '.join(result.get('recommended_actions', []))}")
    print()

async def example_behavioral_biometrics_analysis():
    """Example 2: Behavioral biometrics fraud detection"""
    print("🧠 Example 2: Behavioral Biometrics Analysis")
    print("=" * 50)

    client = MockMCPClient()

    # Sample behavioral data (keystroke dynamics)
    behavioral_data = {
        "keystroke_dynamics": [
            {"key": "p", "press_time": 1000, "release_time": 1050},
            {"key": "a", "press_time": 1120, "release_time": 1160},
            {"key": "s", "press_time": 1200, "release_time": 1240},
            {"key": "s", "press_time": 1280, "release_time": 1320},
            {"key": "w", "press_time": 1380, "release_time": 1420},
            {"key": "o", "press_time": 1480, "release_time": 1520},
            {"key": "r", "press_time": 1580, "release_time": 1620},
            {"key": "d", "press_time": 1680, "release_time": 1720},
        ],
        "mouse_patterns": {
            "movements": [
                {"x": 100, "y": 200, "timestamp": 1000},
                {"x": 150, "y": 220, "timestamp": 1100},
                {"x": 200, "y": 240, "timestamp": 1200}
            ],
            "clicks": [
                {"x": 300, "y": 400, "timestamp": 1500, "button": "left"}
            ]
        }
    }

    # Analyze behavioral patterns
    result = await client.call_tool("detect_behavioral_anomaly", {
        "behavioral_data": behavioral_data
    })

    print("🔍 Behavioral Analysis Result:")
    print(f"   Anomaly Score: {result.get('overall_anomaly_score', 0):.2f}")
    print(f"   Confidence: {result.get('confidence', 0):.2f}")
    print(f"   Detected Anomalies: {', '.join(result.get('detected_anomalies', []))}")
    print()

async def example_comprehensive_fraud_detection():
    """Example 3: Comprehensive multi-modal fraud detection"""
    print("🎯 Example 3: Comprehensive Multi-Modal Analysis")
    print("=" * 50)

    client = MockMCPClient()

    # Comprehensive transaction scenario
    transaction_data = {
        "transaction_id": "txn_2025_002",
        "amount": 7500.00,
        "merchant": "Luxury Watch Store",
        "merchant_category": "jewelry",
        "location": "Miami, FL",
        "timestamp": datetime.now().isoformat(),
        "payment_method": "credit_card",
        "card_type": "amex",
        "user_id": "user_67890",
        "ip_address": "192.0.2.217",
        "device_fingerprint": "device_abc123"
    }

    # Behavioral data (more comprehensive)
    behavioral_data = {
        "keystroke_dynamics": [
            {"key": "1", "press_time": 1000, "release_time": 1080},
            {"key": "2", "press_time": 1150, "release_time": 1200},
            {"key": "3", "press_time": 1270, "release_time": 1320},
            {"key": "4", "press_time": 1400, "release_time": 1450},
        ],
        "session_data": {
            "session_duration": 300,  # 5 minutes
            "pages_visited": 8,
            "form_interactions": 3,
            "copy_paste_events": 1
        }
    }

    # Network/relationship data
    network_data = {
        "entity_id": "user_67890",
        "connections": [
            {"entity_id": "user_11111", "strength": 0.8, "transaction_count": 5},
            {"entity_id": "merchant_xyz", "strength": 0.3, "transaction_count": 1},
            {"entity_id": "device_abc123", "strength": 1.0, "transaction_count": 50}
        ]
    }

    # Perform comprehensive analysis
    result = await client.call_tool("generate_risk_score", {
        "transaction_data": transaction_data,
        "behavioral_data": behavioral_data,
        "network_data": network_data
    })

    print("🎯 Comprehensive Analysis Result:")
    print(f"   Overall Risk Score: {result.get('overall_risk_score', 0):.2f}")
    print(f"   Risk Level: {result.get('risk_level', 'UNKNOWN')}")
    print(f"   Confidence: {result.get('confidence', 0):.2f}")
    print("   Component Scores:")

    component_scores = result.get("component_scores", {})
    for component, score in component_scores.items():
        if score is not None:
            print(f"     {component.title()}: {score:.2f}")

    print(f"   Recommended Actions: {', '.join(result.get('recommended_actions', []))}")
    print(f"   Explanation: {result.get('comprehensive_explanation', 'N/A')}")
    print()

async def example_real_time_monitoring():
    """Example 4: Real-time transaction monitoring simulation"""
    print("⚡ Example 4: Real-Time Transaction Monitoring")
    print("=" * 50)

    client = MockMCPClient()

    # Simulate a stream of transactions
    transactions = [
        {"id": "txn_001", "amount": 50.00, "merchant": "Coffee Shop", "risk": "low"},
        {"id": "txn_002", "amount": 500.00, "merchant": "Gas Station", "risk": "medium"},
        {"id": "txn_003", "amount": 12000.00, "merchant": "Electronics Store", "risk": "high"},
        {"id": "txn_004", "amount": 25.00, "merchant": "Restaurant", "risk": "low"},
        {"id": "txn_005", "amount": 8000.00, "merchant": "Unknown Merchant", "risk": "high"},
    ]

    print("🔄 Processing transaction stream...")

    for i, txn in enumerate(transactions):
        print(f"\n📦 Transaction {i+1}/{len(transactions)}: {txn['id']}")

        transaction_data = {
            "transaction_id": txn["id"],
            "amount": txn["amount"],
            "merchant": txn["merchant"],
            "timestamp": (datetime.now() + timedelta(seconds=i*10)).isoformat(),
            "user_id": "user_stream_test"
        }

        # Quick analysis
        result = await client.call_tool("analyze_transaction", {
            "transaction_data": transaction_data
        })

        risk_score = result.get("overall_risk_score", 0)
        risk_level = result.get("risk_level", "UNKNOWN")

        # Color coding for terminal output
        if risk_level == "HIGH":
            status_icon = "🚨"
        elif risk_level == "MEDIUM":
            status_icon = "⚠️"
        else:
            status_icon = "✅"

        print(f"   {status_icon} Amount: ${txn['amount']:,.2f} | Risk: {risk_level} ({risk_score:.2f})")

        # Simulate processing delay
        await asyncio.sleep(0.1)

    print("\n✅ Real-time monitoring simulation complete")
    print()

async def example_fraud_investigation():
    """Example 5: Fraud investigation workflow"""
    print("🕵️ Example 5: Fraud Investigation Workflow")
    print("=" * 50)

    client = MockMCPClient()

    # Suspicious transaction scenario
    suspicious_transaction = {
        "transaction_id": "txn_investigation_001",
        "amount": 25000.00,
        "merchant": "Cash Advance Service",
        "location": "Unknown Location",
        "timestamp": "2025-09-26T03:30:00Z",  # 3:30 AM
        "payment_method": "debit_card",
        "user_id": "user_investigation"
    }

    # Potentially compromised behavioral data
    suspicious_behavioral = {
        "keystroke_dynamics": [
            {"key": "1", "press_time": 1000, "release_time": 1200},  # Unusually slow
            {"key": "2", "press_time": 1500, "release_time": 1600},  # Long gaps
            {"key": "3", "press_time": 2000, "release_time": 2100},
        ],
        "session_data": {
            "session_duration": 45,  # Very quick session
            "pages_visited": 2,
            "form_interactions": 1,
            "user_agent": "Unknown Browser",
            "geolocation_mismatch": True
        }
    }

    print("🔍 Step 1: Initial Transaction Analysis")
    basic_result = await client.call_tool("analyze_transaction", {
        "transaction_data": suspicious_transaction,
        "include_behavioral": True,
        "behavioral_data": suspicious_behavioral
    })

    print(f"   Initial Risk Assessment: {basic_result.get('risk_level', 'UNKNOWN')}")
    print(f"   Risk Score: {basic_result.get('overall_risk_score', 0):.2f}")

    print("\n🎯 Step 2: Comprehensive Multi-Modal Analysis")
    comprehensive_result = await client.call_tool("generate_risk_score", {
        "transaction_data": suspicious_transaction,
        "behavioral_data": suspicious_behavioral
    })

    print(f"   Comprehensive Risk Score: {comprehensive_result.get('overall_risk_score', 0):.2f}")
    print(f"   Final Risk Level: {comprehensive_result.get('risk_level', 'UNKNOWN')}")
    print(f"   Recommended Actions:")

    for action in comprehensive_result.get("recommended_actions", []):
        print(f"     • {action.replace('_', ' ').title()}")

    print("\n📝 Investigation Summary:")
    print("   This transaction exhibits multiple high-risk indicators:")
    print("   • Unusual amount for time of day (3:30 AM)")
    print("   • Atypical keystroke patterns suggesting different user")
    print("   • High-risk merchant category (cash advance)")
    print("   • Geolocation mismatch")
    print("   • Unusually quick session duration")
    print("\n   🚨 RECOMMENDATION: BLOCK TRANSACTION AND INVESTIGATE ACCOUNT")
    print()

async def main():
    """Run all fraud detection examples"""
    print("🛡️  Advanced Fraud Detection MCP - Example Usage")
    print("=" * 60)
    print()

    # Run all examples
    await example_basic_transaction_analysis()
    await example_behavioral_biometrics_analysis()
    await example_comprehensive_fraud_detection()
    await example_real_time_monitoring()
    await example_fraud_investigation()

    print("🎉 All examples completed successfully!")
    print()
    print("💡 Key Takeaways:")
    print("   • Multi-modal analysis provides better fraud detection")
    print("   • Behavioral biometrics add an extra security layer")
    print("   • Real-time processing enables immediate fraud prevention")
    print("   • Explainable AI helps with fraud investigation")
    print("   • Network analysis can detect fraud rings and patterns")
    print()
    print("🔗 Next Steps:")
    print("   • Integrate with your existing fraud detection pipeline")
    print("   • Train models on your specific transaction data")
    print("   • Customize risk thresholds for your business needs")
    print("   • Set up real-time monitoring and alerting")

if __name__ == "__main__":
    asyncio.run(main())