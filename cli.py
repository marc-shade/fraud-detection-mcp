#!/usr/bin/env python3
"""
Advanced Fraud Detection CLI
Command-line interface for fraud detection analysis on stored data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Import fraud detection components
from server import (
    transaction_analyzer
)
from integration import SyntheticDataIntegration

class FraudDetectionCLI:
    """Command-line interface for fraud detection operations"""

    def __init__(self):
        self.integration = SyntheticDataIntegration()

    def analyze_dataset(self, args) -> None:
        """Analyze a stored dataset for fraud patterns"""
        try:
            print(f"🔍 Analyzing dataset: {args.input}")
            print(f"📊 Fraud threshold: {args.threshold}")
            print(f"⚙️  Analysis type: {args.analysis_type}")
            print("=" * 60)

            # Load dataset
            if not Path(args.input).exists():
                print(f"❌ Error: Dataset file not found: {args.input}")
                sys.exit(1)

            if args.input.endswith('.csv'):
                df = pd.read_csv(args.input)
            elif args.input.endswith('.json'):
                df = pd.read_json(args.input)
            else:
                print("❌ Error: Unsupported file format. Use CSV or JSON.")
                sys.exit(1)

            total_transactions = len(df)
            print(f"📁 Dataset loaded: {total_transactions:,} transactions")

            # Analyze transactions
            flagged_transactions = []
            risk_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

            print("🚀 Starting fraud analysis...")

            # Progress tracking
            processed = 0
            for idx, row in df.iterrows():
                transaction_data = row.to_dict()

                # Perform fraud analysis
                result = transaction_analyzer.analyze_transaction(transaction_data)
                risk_score = result.get("risk_score", 0.0)

                # Categorize risk
                if risk_score >= 0.8:
                    risk_level = "critical"
                elif risk_score >= 0.6:
                    risk_level = "high"
                elif risk_score >= 0.4:
                    risk_level = "medium"
                else:
                    risk_level = "low"

                risk_distribution[risk_level] += 1

                # Flag high-risk transactions
                if risk_score >= args.threshold:
                    flagged_transactions.append({
                        "transaction_id": transaction_data.get("transaction_id", f"txn_{idx}"),
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "risk_factors": result.get("risk_factors", []),
                        "actual_fraud": transaction_data.get("is_fraud", None)
                    })

                processed += 1
                if processed % 1000 == 0:
                    print(f"  📊 Processed: {processed:,}/{total_transactions:,} transactions")

            # Display results
            print("\n✅ Analysis Complete!")
            print("=" * 60)

            fraud_rate = len(flagged_transactions) / total_transactions * 100
            print(f"🚨 Flagged Transactions: {len(flagged_transactions):,} ({fraud_rate:.2f}%)")
            print(f"🎯 Risk Threshold: {args.threshold}")

            print("\n📊 Risk Distribution:")
            for level, count in risk_distribution.items():
                percentage = count / total_transactions * 100
                print(f"  {level.upper():>8}: {count:>6,} ({percentage:>5.1f}%)")

            # Show flagged transactions
            if flagged_transactions and args.show_details:
                print(f"\n🚨 Top {min(10, len(flagged_transactions))} Flagged Transactions:")
                for i, txn in enumerate(flagged_transactions[:10], 1):
                    print(f"  {i:>2}. ID: {txn['transaction_id']:<15} "
                          f"Risk: {txn['risk_score']:.3f} "
                          f"Level: {txn['risk_level']:<8} "
                          f"Factors: {', '.join(txn['risk_factors'][:2])}")

            # Calculate performance metrics if ground truth available
            if "is_fraud" in df.columns:
                metrics = self._calculate_performance_metrics(df, flagged_transactions, args.threshold)
                print("\n📈 Performance Metrics:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")

            # Save results if output specified
            if args.output:
                self._save_results(flagged_transactions, risk_distribution, args.output)
                print(f"\n💾 Results saved to: {args.output}")

        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            sys.exit(1)

    def generate_test_data(self, args) -> None:
        """Generate synthetic test data for fraud detection"""
        try:
            print("🔧 Generating synthetic dataset...")
            print(f"📊 Transactions: {args.count:,}")
            print(f"🚨 Fraud percentage: {args.fraud_rate}%")
            print(f"🧠 Include behavioral: {args.behavioral}")
            print(f"🕸️  Include network: {args.network}")
            print("=" * 60)

            result = self.integration.generate_comprehensive_test_dataset(
                num_transactions=args.count,
                fraud_percentage=args.fraud_rate,
                include_behavioral=args.behavioral,
                include_network=args.network,
                output_format=args.format
            )

            if result.get("integration_status") == "success":
                print("✅ Dataset generation complete!")
                print(f"📁 Transactions: {result['dataset_paths']['transactions']}")

                if result['dataset_paths']['behavioral']:
                    print(f"🧠 Behavioral: {result['dataset_paths']['behavioral']}")

                if result['dataset_paths']['network']:
                    print(f"🕸️  Network: {result['dataset_paths']['network']}")

                print("\n📊 Generation Statistics:")
                info = result['generation_info']
                print(f"  Total transactions: {info['total_transactions']:,}")
                print(f"  Legitimate: {info['legitimate_transactions']:,}")
                print(f"  Fraudulent: {info['fraudulent_transactions']:,}")

                print("\n🚨 Fraud Distribution:")
                for fraud_type, count in result['fraud_distribution'].items():
                    if count > 0:
                        print(f"  {fraud_type}: {count}")

            else:
                print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error during generation: {e}")
            sys.exit(1)

    def interactive_analysis(self, args) -> None:
        """Interactive fraud investigation mode"""
        print("🕵️  Interactive Fraud Analysis Mode")
        print("Type 'help' for commands, 'quit' to exit")
        print("=" * 60)

        current_dataset = None

        while True:
            try:
                command = input("\nfraud-detect> ").strip().lower()

                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                elif command == 'help':
                    self._show_interactive_help()

                elif command.startswith('load '):
                    dataset_path = command[5:].strip()
                    current_dataset = self._load_dataset_interactive(dataset_path)

                elif command.startswith('analyze'):
                    if current_dataset is None:
                        print("❌ No dataset loaded. Use 'load <path>' first.")
                        continue

                    parts = command.split()
                    threshold = 0.6
                    if len(parts) > 1:
                        try:
                            threshold = float(parts[1])
                        except ValueError:
                            print("❌ Invalid threshold. Using default 0.6")

                    self._analyze_dataset_interactive(current_dataset, threshold)

                elif command.startswith('show '):
                    if current_dataset is None:
                        print("❌ No dataset loaded. Use 'load <path>' first.")
                        continue

                    self._show_dataset_info(current_dataset)

                elif command.startswith('generate'):
                    self._generate_interactive()

                elif command == 'status':
                    self._show_status(current_dataset)

                else:
                    print("❌ Unknown command. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _show_interactive_help(self) -> None:
        """Show interactive mode help"""
        print("\n📚 Available Commands:")
        print("  load <path>        - Load dataset from file")
        print("  analyze [thresh]   - Analyze loaded dataset (optional threshold)")
        print("  show info          - Show dataset information")
        print("  generate           - Generate synthetic test data")
        print("  status             - Show current status")
        print("  help               - Show this help")
        print("  quit/exit/q        - Exit interactive mode")

    def _load_dataset_interactive(self, dataset_path: str) -> Optional[pd.DataFrame]:
        """Load dataset in interactive mode"""
        try:
            if not Path(dataset_path).exists():
                print(f"❌ File not found: {dataset_path}")
                return None

            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            else:
                print("❌ Unsupported format. Use CSV or JSON.")
                return None

            print(f"✅ Loaded dataset: {len(df):,} transactions")
            return df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def _analyze_dataset_interactive(self, df: pd.DataFrame, threshold: float) -> None:
        """Analyze dataset in interactive mode"""
        print(f"🔍 Analyzing {len(df):,} transactions with threshold {threshold}...")

        flagged_count = 0
        risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for _, row in df.iterrows():
            transaction_data = row.to_dict()
            result = transaction_analyzer.analyze_transaction(transaction_data)
            risk_score = result.get("risk_score", 0.0)

            if risk_score >= 0.8:
                risk_level = "critical"
            elif risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

            risk_counts[risk_level] += 1

            if risk_score >= threshold:
                flagged_count += 1

        print(f"🚨 Flagged: {flagged_count} transactions ({flagged_count/len(df)*100:.1f}%)")
        print("📊 Risk distribution:")
        for level, count in risk_counts.items():
            print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")

    def _show_dataset_info(self, df: pd.DataFrame) -> None:
        """Show dataset information"""
        print("\n📊 Dataset Information:")
        print(f"  Transactions: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.get('timestamp', pd.Series()).min()} to {df.get('timestamp', pd.Series()).max()}")

        if 'is_fraud' in df.columns:
            fraud_count = df['is_fraud'].sum()
            print(f"  Known fraud: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")

    def _generate_interactive(self) -> None:
        """Generate data in interactive mode"""
        try:
            count = int(input("Number of transactions [10000]: ") or "10000")
            fraud_rate = float(input("Fraud percentage [5.0]: ") or "5.0")

            print("🔧 Generating synthetic dataset...")

            result = self.integration.generate_comprehensive_test_dataset(
                num_transactions=count,
                fraud_percentage=fraud_rate,
                include_behavioral=True,
                include_network=True,
                output_format="csv"
            )

            if result.get("integration_status") == "success":
                print(f"✅ Generated: {result['dataset_paths']['transactions']}")
            else:
                print(f"❌ Generation failed: {result.get('error')}")

        except (ValueError, KeyboardInterrupt):
            print("❌ Generation cancelled")

    def _show_status(self, current_dataset: Optional[pd.DataFrame]) -> None:
        """Show current status"""
        print("\n🔧 System Status:")
        print(f"  Dataset loaded: {'Yes' if current_dataset is not None else 'No'}")
        if current_dataset is not None:
            print(f"  Transactions: {len(current_dataset):,}")
        print("  Models initialized: Yes")
        print("  Integration ready: Yes")

    def _calculate_performance_metrics(
        self, df: pd.DataFrame, flagged_transactions: List[Dict[str, Any]], threshold: float
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        flagged_ids = set(t["transaction_id"] for t in flagged_transactions)
        predictions = df["transaction_id"].isin(flagged_ids)
        actual = df["is_fraud"]

        tp = (predictions & actual).sum()
        fp = (predictions & ~actual).sum()
        tn = (~predictions & ~actual).sum()
        fn = (~predictions & actual).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(df)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy
        }

    def _save_results(
        self, flagged_transactions: List[Dict[str, Any]],
        risk_distribution: Dict[str, int],
        output_path: str
    ) -> None:
        """Save analysis results"""
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "flagged_transactions": flagged_transactions,
            "risk_distribution": risk_distribution,
            "summary": {
                "total_flagged": len(flagged_transactions),
                "risk_levels": risk_distribution
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Fraud Detection CLI - Analyze stored data for fraud patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fraud-detect analyze transactions.csv
  fraud-detect analyze --threshold 0.7 --show-details data.json
  fraud-detect generate --count 50000 --fraud-rate 3.5
  fraud-detect interactive
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset for fraud patterns')
    analyze_parser.add_argument('input', help='Input dataset file (CSV or JSON)')
    analyze_parser.add_argument('--threshold', '-t', type=float, default=0.6,
                               help='Risk score threshold for flagging (default: 0.6)')
    analyze_parser.add_argument('--analysis-type', choices=['quick', 'comprehensive', 'network'],
                               default='comprehensive', help='Type of analysis')
    analyze_parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    analyze_parser.add_argument('--show-details', action='store_true',
                               help='Show detailed information about flagged transactions')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic test data')
    generate_parser.add_argument('--count', '-c', type=int, default=10000,
                                help='Number of transactions to generate (default: 10000)')
    generate_parser.add_argument('--fraud-rate', '-f', type=float, default=5.0,
                                help='Fraud percentage (default: 5.0)')
    generate_parser.add_argument('--behavioral', action='store_true',
                                help='Include behavioral biometrics data')
    generate_parser.add_argument('--network', action='store_true',
                                help='Include network relationship data')
    generate_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                                help='Output format (default: csv)')

    # Interactive command
    subparsers.add_parser('interactive', help='Interactive analysis mode')

    # Version command
    subparsers.add_parser('version', help='Show version information')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = FraudDetectionCLI()

    if args.command == 'analyze':
        cli.analyze_dataset(args)
    elif args.command == 'generate':
        cli.generate_test_data(args)
    elif args.command == 'interactive':
        cli.interactive_analysis(args)
    elif args.command == 'version':
        print("Advanced Fraud Detection CLI v1.0.0")
        print("Based on 2024-2025 fraud detection research")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()