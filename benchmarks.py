#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for fraud detection system
Validates performance claims with real datasets and empirical testing
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.metrics import (  # noqa: E402
    confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
    f1_score, accuracy_score, recall_score, precision_score
)
from sklearn.model_selection import train_test_split  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Local imports
from config import get_config  # noqa: E402
from training_pipeline import ModelTrainer  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from integration import FraudDataGenerator  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for fraud detection system
    Tests all models against real datasets and validates claims
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite

        Args:
            output_dir: Directory for benchmark results
        """
        self.config = get_config()
        self.output_dir = output_dir or self.config.BASE_DIR / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.results = {}

        logger.info(f"Benchmark output directory: {self.output_dir}")

    def download_real_datasets(self) -> Dict[str, Path]:
        """
        Download real fraud detection datasets

        Returns:
            Dictionary mapping dataset names to file paths
        """
        logger.info("Checking for real fraud datasets...")

        datasets = {}
        data_dir = self.config.DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)

        # Check for Kaggle Credit Card Fraud dataset
        kaggle_path = data_dir / "creditcard.csv"
        if kaggle_path.exists():
            logger.info(f"Found Kaggle dataset: {kaggle_path}")
            datasets['kaggle_cc'] = kaggle_path
        else:
            logger.warning(
                "Kaggle Credit Card Fraud dataset not found. "
                "Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )

        # Generate synthetic dataset as fallback
        synthetic_path = data_dir / "synthetic_fraud_data.csv"
        if not synthetic_path.exists():
            logger.info("Generating synthetic fraud dataset...")
            generator = FraudDataGenerator()
            transactions = generator.generate_transactions(n_transactions=50000)

            # Convert to DataFrame
            df = pd.DataFrame([t.__dict__ for t in transactions])
            df.to_csv(synthetic_path, index=False)
            logger.info(f"Synthetic dataset saved: {synthetic_path}")

        datasets['synthetic'] = synthetic_path

        return datasets

    def load_and_prepare_dataset(self, dataset_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare dataset for benchmarking

        Args:
            dataset_path: Path to dataset CSV

        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)

        # Detect dataset type and prepare accordingly
        if 'Class' in df.columns:
            # Kaggle format
            y = df['Class'].values
            X = df.drop('Class', axis=1).values
        elif 'is_fraud' in df.columns:
            # Synthetic format
            y = df['is_fraud'].astype(int).values

            # Extract features using feature engineer
            from models_validation import TransactionData
            transactions = []

            for _, row in df.iterrows():
                try:
                    trans = TransactionData(**row.to_dict())
                    transactions.append(trans)
                except Exception:
                    continue

            # Engineer features
            X_list = []
            for trans in transactions:
                features, _ = self.feature_engineer.extract_features(trans)
                X_list.append(features)

            X = np.array(X_list)
        else:
            raise ValueError(f"Unknown dataset format: {dataset_path}")

        logger.info(f"Dataset shape: {X.shape}, Fraud rate: {y.mean():.2%}")
        return X, y

    def benchmark_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Benchmark a single model

        Args:
            model: Trained model with predict/predict_proba methods
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Benchmarking {model_name}...")

        # Measure inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        # Get probability scores if available
        try:
            y_proba = model.predict_proba(X_test)
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]  # Probability of fraud class
        except (AttributeError, NotImplementedError):
            y_proba = y_pred.astype(float)

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            'avg_precision': average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            'inference_time_total': inference_time,
            'inference_time_per_sample': inference_time / len(X_test) * 1000,  # ms
            'throughput_per_second': len(X_test) / inference_time if inference_time > 0 else 0
        }

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, "
                   f"ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Path:
        """
        Plot and save confusion matrix

        Args:
            y_test: True labels
            y_pred: Predicted labels
            model_name: Model name for plot title

        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        output_path = self.output_dir / f'confusion_matrix_{model_name.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved: {output_path}")
        return output_path

    def plot_roc_curve(
        self,
        models_data: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> Path:
        """
        Plot ROC curves for multiple models

        Args:
            models_data: List of tuples (model_name, y_test, y_proba)

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 8))

        for model_name, y_test, y_proba in models_data:
            if len(np.unique(y_test)) > 1:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = roc_auc_score(y_test, y_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / 'roc_curves_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"ROC curves saved: {output_path}")
        return output_path

    def plot_precision_recall_curve(
        self,
        models_data: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> Path:
        """
        Plot precision-recall curves for multiple models

        Args:
            models_data: List of tuples (model_name, y_test, y_proba)

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(10, 8))

        for model_name, y_test, y_proba in models_data:
            if len(np.unique(y_test)) > 1:
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                ap_score = average_precision_score(y_test, y_proba)
                plt.plot(recall, precision,
                        label=f'{model_name} (AP = {ap_score:.4f})',
                        linewidth=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / 'pr_curves_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"PR curves saved: {output_path}")
        return output_path

    def run_full_benchmark(self, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run complete benchmark suite

        Args:
            dataset_path: Optional path to specific dataset, otherwise uses default

        Returns:
            Complete benchmark results
        """
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE FRAUD DETECTION BENCHMARK")
        logger.info("="*60)

        # Get datasets
        if dataset_path is None:
            datasets = self.download_real_datasets()
            # Use first available dataset
            dataset_path = list(datasets.values())[0]

        # Load and prepare data
        X, y = self.load_and_prepare_dataset(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Fraud rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

        # Train all models
        logger.info("\n" + "="*60)
        logger.info("TRAINING MODELS")
        logger.info("="*60)

        trainer = ModelTrainer()

        # Save training data temporarily
        temp_data_path = self.config.DATA_DIR / "temp_train_data.csv"
        train_df = pd.DataFrame(X_train)
        train_df['is_fraud'] = y_train
        train_df.to_csv(temp_data_path, index=False)

        # Train models
        models = trainer.train_all_models(
            data_path=str(temp_data_path),
            test_size=0.2,
            use_smote=True,
            optimize_hyperparams=False  # Skip for speed in benchmark
        )

        # Clean up temp file
        temp_data_path.unlink()

        # Benchmark each model
        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING MODELS")
        logger.info("="*60)

        results = {}
        models_for_plots = []

        # 1. Isolation Forest
        if 'isolation_forest' in models:
            metrics = self.benchmark_model(
                models['isolation_forest'],
                X_test,
                y_test,
                'Isolation Forest'
            )
            results['isolation_forest'] = metrics

            # Get predictions for plotting
            y_pred = models['isolation_forest'].predict(X_test)
            self.plot_confusion_matrix(y_test, y_pred, 'Isolation_Forest')

            try:
                y_proba = models['isolation_forest'].predict_proba(X_test)[:, 1]
                models_for_plots.append(('Isolation Forest', y_test, y_proba))
            except Exception:
                pass

        # 2. XGBoost
        if 'xgboost' in models:
            metrics = self.benchmark_model(
                models['xgboost'],
                X_test,
                y_test,
                'XGBoost'
            )
            results['xgboost'] = metrics

            y_pred = models['xgboost'].predict(X_test)
            self.plot_confusion_matrix(y_test, y_pred, 'XGBoost')

            try:
                y_proba = models['xgboost'].predict_proba(X_test)[:, 1]
                models_for_plots.append(('XGBoost', y_test, y_proba))
            except Exception:
                pass

        # 3. Ensemble
        if 'ensemble' in models:
            pipeline = models['ensemble']

            # Measure ensemble performance
            start_time = time.time()
            ensemble_result = pipeline.predict_ensemble(X_test)
            inference_time = time.time() - start_time

            y_pred = (ensemble_result['ensemble_score'] > 0.5).astype(int)
            y_proba = ensemble_result['ensemble_score']

            metrics = {
                'model_name': 'Ensemble',
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
                'avg_precision': average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
                'inference_time_total': inference_time,
                'inference_time_per_sample': inference_time / len(X_test) * 1000,
                'throughput_per_second': len(X_test) / inference_time if inference_time > 0 else 0
            }

            results['ensemble'] = metrics
            self.plot_confusion_matrix(y_test, y_pred, 'Ensemble')
            models_for_plots.append(('Ensemble', y_test, y_proba))

            logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, "
                       f"F1: {metrics['f1_score']:.4f}, "
                       f"ROC-AUC: {metrics['roc_auc']:.4f}")

        # Plot comparison curves
        if models_for_plots:
            self.plot_roc_curve(models_for_plots)
            self.plot_precision_recall_curve(models_for_plots)

        # Create summary report
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)

        summary = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'test_set_size': len(X_test),
            'fraud_rate': float(y_test.mean()),
            'models': results
        }

        # Save results
        results_path = self.output_dir / 'benchmark_results.json'
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nBenchmark results saved to: {results_path}")

        # Print summary table
        self._print_summary_table(results)

        # Validate claims
        self._validate_performance_claims(results)

        return summary

    def _print_summary_table(self, results: Dict[str, Dict[str, float]]):
        """Print formatted summary table of results"""

        print("\n" + "="*100)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*100)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} "
              f"{'F1':<10} {'ROC-AUC':<10} {'Latency(ms)':<12}")
        print("-"*100)

        for model_name, metrics in results.items():
            print(f"{metrics['model_name']:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['roc_auc']:<10.4f} "
                  f"{metrics['inference_time_per_sample']:<12.2f}")

        print("="*100)

    def _validate_performance_claims(self, results: Dict[str, Dict[str, float]]):
        """Validate performance against claimed metrics"""

        print("\n" + "="*60)
        print("PERFORMANCE CLAIMS VALIDATION")
        print("="*60)

        claims = {
            'detection_rate': 0.95,  # Claimed >95%
            'false_positive_rate': 0.02,  # Claimed <2%
            'latency_ms': 100,  # Claimed <100ms
            'throughput_tps': 10000  # Claimed 10,000+ TPS
        }

        # Check ensemble or best model
        best_model = None
        if 'ensemble' in results:
            best_model = results['ensemble']
        else:
            # Find model with highest F1 score
            best_f1 = 0
            for model_metrics in results.values():
                if model_metrics.get('f1_score', 0) > best_f1:
                    best_f1 = model_metrics['f1_score']
                    best_model = model_metrics

        if best_model:
            actual_recall = best_model.get('recall', 0)
            actual_fpr = best_model.get('false_positive_rate', 0)
            actual_latency = best_model.get('inference_time_per_sample', 0)
            actual_throughput = best_model.get('throughput_per_second', 0)

            print(f"\nClaimed vs Actual Performance ({best_model['model_name']}):")
            print(f"  Detection Rate: {claims['detection_rate']:.1%} claimed, "
                  f"{actual_recall:.1%} actual - "
                  f"{'✓ PASS' if actual_recall >= claims['detection_rate'] else '✗ FAIL'}")

            print(f"  False Positive Rate: <{claims['false_positive_rate']:.1%} claimed, "
                  f"{actual_fpr:.1%} actual - "
                  f"{'✓ PASS' if actual_fpr <= claims['false_positive_rate'] else '✗ FAIL'}")

            print(f"  Latency: <{claims['latency_ms']}ms claimed, "
                  f"{actual_latency:.2f}ms actual - "
                  f"{'✓ PASS' if actual_latency < claims['latency_ms'] else '✗ FAIL'}")

            print(f"  Throughput: >{claims['throughput_tps']} TPS claimed, "
                  f"{actual_throughput:.0f} TPS actual - "
                  f"{'✓ PASS' if actual_throughput >= claims['throughput_tps'] else '✗ FAIL'}")

        print("="*60)


def main():
    """Run complete benchmark suite"""

    # Create benchmark instance
    benchmark = PerformanceBenchmark()

    # Run full benchmark
    benchmark.run_full_benchmark()

    print("\n✅ Benchmark complete! Check the benchmark_results directory for detailed reports.")
    print(f"📊 Results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()