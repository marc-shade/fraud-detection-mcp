#!/usr/bin/env python3
"""
Complete model training pipeline with MLOps best practices
Includes SMOTE, cross-validation, model persistence, and MLflow tracking
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Optional MLflow support
try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
from tqdm import tqdm
import logging

from config import config, model_config
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Complete ML training pipeline with MLOps integration"""

    def __init__(self, model_dir: Optional[Path] = None, enable_mlflow: bool = True):
        self.model_dir = model_dir or config.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metrics = {}
        self.feature_names = []

        # Setup MLflow
        if self.enable_mlflow and config.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        elif enable_mlflow and not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow requested but not available - proceeding without MLflow"
            )

    def train_all_models(
        self,
        data_path: str,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        use_smote: bool = True,
        optimize_hyperparams: bool = True,
        train_autoencoder: bool = False,
        train_gnn: bool = False,
    ) -> Dict[str, Any]:
        """
        Complete training pipeline for all fraud detection models

        Args:
            data_path: Path to training data CSV/JSON
            test_size: Test set proportion
            validation_size: Validation set proportion
            use_smote: Apply SMOTE for class balancing
            optimize_hyperparams: Use Optuna for hyperparameter tuning
            train_autoencoder: Train autoencoder for anomaly detection (requires PyTorch)
            train_gnn: Train Graph Neural Network (requires PyTorch Geometric)

        Returns:
            Training results with metrics
        """

        logger.info(f"Starting training pipeline: {data_path}")

        if self.enable_mlflow:
            run_name = f"fraud_training_{datetime.now():%Y%m%d_%H%M%S}"
            mlflow.start_run(run_name=run_name)

        try:
            # 1. Load and prepare data
            logger.info("Loading data...")
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(
                data_path, test_size, validation_size
            )

            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Validation samples: {len(X_val)}")
            logger.info(f"Test samples: {len(X_test)}")
            logger.info(f"Fraud rate: {y_train.mean():.2%}")

            # 2. Apply SMOTE if requested
            if use_smote and y_train.mean() < 0.05:
                logger.info("Applying SMOTE for class balancing...")
                X_train, y_train = self._apply_smote(X_train, y_train)
                logger.info(f"After SMOTE - Fraud rate: {y_train.mean():.2%}")

            # 3. Train Isolation Forest (unsupervised)
            logger.info("\n" + "=" * 60)
            logger.info("Training Isolation Forest...")
            logger.info("=" * 60)
            iso_metrics = self._train_isolation_forest(X_train, X_test, y_test)

            # 4. Train XGBoost (supervised)
            logger.info("\n" + "=" * 60)
            logger.info("Training XGBoost...")
            logger.info("=" * 60)

            if optimize_hyperparams:
                logger.info("Optimizing hyperparameters with Optuna...")
                best_params = self._optimize_xgboost(X_train, y_train, X_val, y_val)
            else:
                best_params = model_config.get_xgboost_params()

            xgb_metrics = self._train_xgboost(
                X_train, X_val, y_train, y_val, X_test, y_test, params=best_params
            )

            # 5. Feature importance analysis
            feature_importance = self._analyze_feature_importance()

            # 6. Ensemble evaluation
            logger.info("\n" + "=" * 60)
            logger.info("Evaluating Ensemble Model...")
            logger.info("=" * 60)
            ensemble_metrics = self._evaluate_ensemble(X_test, y_test)

            # 7. Train advanced models if requested
            autoencoder_metrics = None
            if train_autoencoder:
                logger.info("\n" + "=" * 60)
                logger.info("Training Autoencoder...")
                logger.info("=" * 60)
                try:
                    autoencoder_metrics = self._train_autoencoder(
                        X_train, X_test, y_test, contamination=y_train.mean()
                    )
                except Exception as e:
                    logger.error(f"Autoencoder training failed: {e}")

            gnn_metrics = None
            if train_gnn:
                logger.info("\n" + "=" * 60)
                logger.info("Training Graph Neural Network...")
                logger.info("=" * 60)
                try:
                    gnn_metrics = self._train_gnn(X_train, X_test, y_train, y_test)
                except Exception as e:
                    logger.error(f"GNN training failed: {e}")

            # 8. Find optimal threshold
            optimal_threshold = self._find_optimal_threshold(X_test, y_test)

            # 9. Save models
            self._save_models()

            # 9. Log to MLflow
            if self.enable_mlflow:
                self._log_to_mlflow(
                    iso_metrics,
                    xgb_metrics,
                    ensemble_metrics,
                    feature_importance,
                    optimal_threshold,
                    best_params,
                )

            # Compile results
            results = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data_info": {
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "fraud_rate": float(y_train.mean()),
                    "n_features": X_train.shape[1],
                },
                "isolation_forest": iso_metrics,
                "xgboost": xgb_metrics,
                "ensemble": ensemble_metrics,
                "optimal_threshold": float(optimal_threshold),
                "feature_importance": feature_importance[:20],  # Top 20
                "model_paths": {
                    "isolation_forest": str(
                        self.model_dir / "isolation_forest_latest.joblib"
                    ),
                    "xgboost": str(self.model_dir / "xgboost_latest.joblib"),
                    "scaler": str(self.model_dir / "scaler_latest.joblib"),
                },
            }

            # Add advanced model metrics if trained
            if autoencoder_metrics:
                results["autoencoder"] = autoencoder_metrics
                results["model_paths"]["autoencoder"] = str(
                    self.model_dir / "autoencoder_latest.pth"
                )

            if gnn_metrics:
                results["gnn"] = gnn_metrics
                results["model_paths"]["gnn"] = str(self.model_dir / "gnn_latest.pth")

            logger.info("\n" + "=" * 60)
            logger.info("Training Complete!")
            logger.info("=" * 60)
            logger.info(f"Ensemble ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
            logger.info(f"Ensemble PR-AUC: {ensemble_metrics['pr_auc']:.4f}")
            logger.info(f"Ensemble F1-Score: {ensemble_metrics['f1_score']:.4f}")
            logger.info(f"False Positive Rate: {ensemble_metrics['fpr']:.4f}")

            if self.enable_mlflow:
                mlflow.end_run()

            return results

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            if self.enable_mlflow:
                mlflow.end_run(status="FAILED")
            raise

    def _prepare_data(
        self, data_path: str, test_size: float, validation_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare training data"""

        # Load data
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".json"):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Data must be CSV or JSON")

        # Verify required columns
        if "is_fraud" not in df.columns:
            raise ValueError("Data must have 'is_fraud' column")

        # Extract features
        logger.info("Engineering features...")
        feature_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
            transaction_data = row.to_dict()
            features, names = self.feature_engineer.extract_features(transaction_data)
            feature_list.append(features)

        self.feature_names = names
        X = np.array(feature_list)
        y = df["is_fraud"].values.astype(int)

        # Split data: train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp,
        )

        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        self.scalers["transaction"] = scaler

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for class balancing"""

        # Combine SMOTE (oversample minority) and undersampling (majority)
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)

        pipeline = ImbPipeline([("over", over), ("under", under)])

        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        return X_resampled, y_resampled

    def _train_isolation_forest(
        self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train Isolation Forest for anomaly detection"""

        params = model_config.get_isolation_forest_params()
        iso_forest = IsolationForest(**params)

        # Train (unsupervised - ignores labels)
        iso_forest.fit(X_train)
        self.models["isolation_forest"] = iso_forest

        # Evaluate
        pred = iso_forest.predict(X_test)
        pred_binary = (pred == -1).astype(int)  # -1 = anomaly
        scores = iso_forest.score_samples(X_test)

        # Convert scores to probabilities (0-1)
        pred_proba = self._anomaly_scores_to_proba(scores)

        metrics = self._calculate_metrics(
            y_test, pred_binary, pred_proba, "Isolation Forest"
        )

        return metrics

    def _optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters with Optuna"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
                "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
                "random_state": 42,
                "n_jobs": -1,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False,
            )

            pred_proba = model.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, pred_proba)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best PR-AUC: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train XGBoost classifier"""

        # Add fixed params
        params["random_state"] = 42
        params["n_jobs"] = -1
        params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()

        model = xgb.XGBClassifier(**params)

        # Train with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val), (X_test, y_test)],
            eval_metric=["aucpr", "logloss"],
            early_stopping_rounds=20,
            verbose=True,
        )

        self.models["xgboost"] = model

        # Evaluate
        pred_proba = model.predict_proba(X_test)[:, 1]
        pred = (pred_proba > 0.5).astype(int)

        metrics = self._calculate_metrics(y_test, pred, pred_proba, "XGBoost")

        return metrics

    def _evaluate_ensemble(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate ensemble of all models"""

        # Get predictions from each model
        iso_scores = self.models["isolation_forest"].score_samples(X_test)
        iso_proba = self._anomaly_scores_to_proba(iso_scores)

        xgb_proba = self.models["xgboost"].predict_proba(X_test)[:, 1]

        # Weighted ensemble (XGBoost weighted higher as supervised)
        ensemble_proba = 0.3 * iso_proba + 0.7 * xgb_proba
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        metrics = self._calculate_metrics(
            y_test, ensemble_pred, ensemble_proba, "Ensemble"
        )

        return metrics

    def _anomaly_scores_to_proba(self, scores: np.ndarray) -> np.ndarray:
        """Convert anomaly scores to probabilities"""
        # Normalize scores to [0, 1] range
        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score == 0:
            return np.full_like(scores, 0.5)

        normalized = (scores - min_score) / (max_score - min_score)
        # Invert: low score = high fraud probability
        proba = 1 - normalized

        return proba

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        model_name: str,
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            average_precision_score,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_proba))

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update(
            {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            }
        )

        # Print summary
        logger.info(f"\n{model_name} Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if y_pred_proba is not None:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            logger.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        logger.info(f"  FP Rate:   {metrics['fpr']:.4f}")

        return metrics

    def _analyze_feature_importance(self) -> List[Dict[str, Any]]:
        """Analyze feature importance from XGBoost"""

        if "xgboost" not in self.models:
            return []

        importances = self.models["xgboost"].feature_importances_

        feature_importance = [
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(self.feature_names, importances)
        ]

        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return feature_importance

    def _find_optimal_threshold(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Find optimal classification threshold"""

        # Get ensemble predictions
        iso_scores = self.models["isolation_forest"].score_samples(X_test)
        iso_proba = self._anomaly_scores_to_proba(iso_scores)
        xgb_proba = self.models["xgboost"].predict_proba(X_test)[:, 1]
        ensemble_proba = 0.3 * iso_proba + 0.7 * xgb_proba

        # Find threshold that maximizes F1 score
        precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_proba)

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        )

        logger.info(f"\nOptimal Threshold: {optimal_threshold:.4f}")
        logger.info(f"  F1-Score at threshold: {f1_scores[optimal_idx]:.4f}")
        logger.info(f"  Precision: {precisions[optimal_idx]:.4f}")
        logger.info(f"  Recall: {recalls[optimal_idx]:.4f}")

        return optimal_threshold

    def _train_autoencoder(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Train autoencoder for anomaly detection

        Args:
            X_train: Training features (normal transactions only)
            X_test: Test features
            y_test: Test labels
            contamination: Expected fraud rate for threshold setting

        Returns:
            Training metrics
        """
        from models.autoencoder import AutoencoderFraudDetector

        logger.info("Training Autoencoder...")

        # Filter training data to normal transactions only
        # (For unsupervised anomaly detection)
        X_train_normal = X_train  # In real scenario, filter out known fraud

        # Initialize and train
        autoencoder = AutoencoderFraudDetector(
            input_dim=X_train.shape[1], encoding_dim=16, contamination=contamination
        )

        autoencoder.fit(X_train_normal, epochs=50, batch_size=256, verbose=1)

        # Evaluate
        y_pred = autoencoder.predict(X_test)
        y_pred_proba = autoencoder.predict_proba(X_test)[:, 1]

        # Store model
        self.models["autoencoder"] = autoencoder

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, "Autoencoder")

        return metrics

    def _train_gnn(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train Graph Neural Network for fraud detection

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels

        Returns:
            Training metrics
        """
        from models.gnn_fraud_detector import GNNFraudDetector

        logger.info("Training Graph Neural Network...")

        # Initialize and train
        gnn = GNNFraudDetector(input_dim=X_train.shape[1], hidden_dim=64, num_layers=2)

        gnn.fit(X_train, y_train, epochs=50, batch_size=256, verbose=1)

        # Evaluate
        y_pred = gnn.predict(X_test)
        y_pred_proba = gnn.predict_proba(X_test)[:, 1]

        # Store model
        self.models["gnn"] = gnn

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, "GNN")

        return metrics

    def _save_models(self):
        """Save trained models and scalers (handles both sklearn and PyTorch models)"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models with timestamp and as latest
        for name, model in self.models.items():
            # Check if model has PyTorch save method (for Autoencoder/GNN)
            if hasattr(model, "save_model") and callable(model.save_model):
                # PyTorch model - use its own save method
                latest_path = self.model_dir / f"{name}_latest.pth"
                timestamped_path = self.model_dir / f"{name}_{timestamp}.pth"

                model.save_model(str(latest_path))
                model.save_model(str(timestamped_path))
                logger.info(f"Saved PyTorch {name} to {latest_path}")
            else:
                # Sklearn model - use joblib
                model_path = self.model_dir / f"{name}_{timestamp}.joblib"
                joblib.dump(model, model_path)

                # Latest version
                latest_path = self.model_dir / f"{name}_latest.joblib"
                joblib.dump(model, latest_path)

                logger.info(f"Saved {name} to {latest_path}")

        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = self.model_dir / f"scaler_{name}_latest.joblib"
            joblib.dump(scaler, scaler_path)

        # Save feature names
        feature_names_path = self.model_dir / "feature_names.joblib"
        joblib.dump(self.feature_names, feature_names_path)

    def _log_to_mlflow(
        self,
        iso_metrics: Dict,
        xgb_metrics: Dict,
        ensemble_metrics: Dict,
        feature_importance: List[Dict],
        optimal_threshold: float,
        xgb_params: Dict,
    ):
        """Log training results to MLflow"""

        # Log parameters
        mlflow.log_params(xgb_params)
        mlflow.log_param("optimal_threshold", optimal_threshold)

        # Log metrics for each model
        for prefix, metrics in [
            ("iso_", iso_metrics),
            ("xgb_", xgb_metrics),
            ("ensemble_", ensemble_metrics),
        ]:
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{prefix}{metric_name}", value)

        # Log feature importance
        importance_path = self.model_dir / "feature_importance.csv"
        pd.DataFrame(feature_importance).to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        # Log models
        mlflow.sklearn.log_model(self.models["xgboost"], "xgboost_model")

    def load_models(self, load_advanced: bool = True) -> bool:
        """
        Load trained models (both sklearn and PyTorch)

        Args:
            load_advanced: Whether to load advanced models (autoencoder, GNN)

        Returns:
            True if core models loaded successfully
        """

        try:
            # Load core sklearn models
            self.models["isolation_forest"] = joblib.load(
                self.model_dir / "isolation_forest_latest.joblib"
            )
            self.models["xgboost"] = joblib.load(
                self.model_dir / "xgboost_latest.joblib"
            )
            self.scalers["transaction"] = joblib.load(
                self.model_dir / "scaler_transaction_latest.joblib"
            )
            self.feature_names = joblib.load(self.model_dir / "feature_names.joblib")

            logger.info("Core models loaded successfully")

            # Optionally load advanced PyTorch models
            if load_advanced:
                # Try to load autoencoder
                autoencoder_path = self.model_dir / "autoencoder_latest.pth"
                if autoencoder_path.exists():
                    try:
                        from models.autoencoder import AutoencoderFraudDetector

                        input_dim = len(self.feature_names)
                        autoencoder = AutoencoderFraudDetector(input_dim=input_dim)
                        autoencoder.load_model(str(autoencoder_path))
                        self.models["autoencoder"] = autoencoder
                        logger.info("Autoencoder loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load autoencoder: {e}")

                # Try to load GNN
                gnn_path = self.model_dir / "gnn_latest.pth"
                if gnn_path.exists():
                    try:
                        from models.gnn_fraud_detector import GNNFraudDetector

                        input_dim = len(self.feature_names)
                        gnn = GNNFraudDetector(input_dim=input_dim)
                        gnn.load_model(str(gnn_path))
                        self.models["gnn"] = gnn
                        logger.info("GNN loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load GNN: {e}")

            return True

        except FileNotFoundError as e:
            logger.warning(f"Core models not found: {e}")
            return False


# Alias for backward compatibility
FraudDetectionPipeline = ModelTrainer


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()

    # Train on sample data
    results = trainer.train_all_models(
        data_path="test_data/fraud_dataset.csv",
        use_smote=True,
        optimize_hyperparams=True,
    )

    print("\nTraining Results:")
    print(f"Ensemble ROC-AUC: {results['ensemble']['roc_auc']:.4f}")
    print(f"Ensemble PR-AUC: {results['ensemble']['pr_auc']:.4f}")
