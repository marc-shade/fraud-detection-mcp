#!/usr/bin/env python3
"""
Autoencoder for Anomaly Detection in Fraud Detection
PyTorch-based autoencoder with sklearn-compatible API
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch, gracefully handle missing dependency
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch loaded successfully for Autoencoder")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available - Autoencoder will use fallback implementation")


if PYTORCH_AVAILABLE:
    class AutoencoderNetwork(nn.Module):
        """
        Symmetric autoencoder architecture
        input -> 64 -> 32 -> 16 -> 32 -> 64 -> output
        """

        def __init__(self, input_dim: int):
            super(AutoencoderNetwork, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.2),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, input_dim)
            )

        def forward(self, x):
            """Forward pass"""
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        def encode(self, x):
            """Get encoded representation"""
            return self.encoder(x)


class AutoencoderFraudDetector:
    """
    Autoencoder-based anomaly detector for fraud detection
    sklearn-compatible API with graceful degradation
    """

    def __init__(
        self,
        contamination: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize autoencoder

        Args:
            contamination: Expected proportion of fraud (for threshold)
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.contamination = contamination
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.threshold = None
        self.input_dim = None
        self.scaler_mean = None
        self.scaler_std = None

        # Determine device
        if not PYTORCH_AVAILABLE:
            self.device = 'cpu'
            self.fallback_mode = True
            logger.warning("Using fallback mode (no PyTorch)")
        else:
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            self.fallback_mode = False
            logger.info(f"AutoencoderFraudDetector initialized on device: {self.device}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AutoencoderFraudDetector':
        """
        Fit autoencoder on normal (non-fraud) transactions

        Args:
            X: Feature matrix
            y: Optional labels (0=normal, 1=fraud). If provided, trains only on normal data

        Returns:
            self for method chaining
        """
        if self.fallback_mode:
            return self._fit_fallback(X, y)

        # Filter to normal transactions if labels provided
        if y is not None:
            X_train = X[y == 0]
            logger.info(f"Training on {len(X_train)} normal transactions (filtered from {len(X)})")
        else:
            X_train = X
            logger.info(f"Training on {len(X_train)} transactions (assuming mostly normal)")

        # Normalize data
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-7
        X_normalized = (X_train - self.scaler_mean) / self.scaler_std

        # Store input dimension
        self.input_dim = X_train.shape[1]

        # Create model
        self.model = AutoencoderNetwork(self.input_dim).to(self.device)

        # Create data loader
        tensor_x = torch.FloatTensor(X_normalized)
        dataset = TensorDataset(tensor_x, tensor_x)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_target in dataloader:
                batch_x = batch_x.to(self.device)
                batch_target = batch_target.to(self.device)

                # Forward pass
                output = self.model(batch_x)
                loss = criterion(output, batch_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

        # Calculate threshold based on reconstruction error
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            errors = reconstruction_errors.cpu().numpy()

        # Set threshold at contamination percentile
        self.threshold = np.percentile(errors, (1 - self.contamination) * 100)
        logger.info(f"Threshold set at {self.threshold:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud (1) or normal (0)

        Args:
            X: Feature matrix

        Returns:
            Binary predictions
        """
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability matrix [normal_prob, fraud_prob]
        """
        scores = self.decision_function(X)

        # Normalize scores to probabilities
        fraud_proba = np.clip(scores / (self.threshold * 3), 0, 1)
        normal_proba = 1 - fraud_proba

        return np.column_stack([normal_proba, fraud_proba])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error (anomaly score)

        Args:
            X: Feature matrix

        Returns:
            Reconstruction errors
        """
        if self.fallback_mode:
            return self._decision_function_fallback(X)

        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Normalize data
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        # Calculate reconstruction error
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            errors = reconstruction_errors.cpu().numpy()

        return errors

    def _fit_fallback(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AutoencoderFraudDetector':
        """Fallback fitting using simple statistics"""
        logger.warning("Using fallback mode - limited functionality")

        # Filter to normal if labels provided
        if y is not None:
            X_train = X[y == 0]
        else:
            X_train = X

        # Store statistics
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-7
        self.input_dim = X_train.shape[1]

        # Calculate simple threshold
        X_normalized = (X_train - self.scaler_mean) / self.scaler_std
        distances = np.sum(X_normalized ** 2, axis=1)
        self.threshold = np.percentile(distances, (1 - self.contamination) * 100)

        logger.info("Fallback model fitted")
        return self

    def _decision_function_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback decision function using Euclidean distance"""
        if self.scaler_mean is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Normalize
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        # Calculate distance from origin (proxy for anomaly)
        distances = np.sum(X_normalized ** 2, axis=1)

        return distances

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get encoded (latent) representation of data

        Args:
            X: Feature matrix

        Returns:
            Latent representations
        """
        if self.fallback_mode:
            # Return normalized input as "latent"
            return (X - self.scaler_mean) / self.scaler_std

        if self.model is None:
            raise RuntimeError("Model must be fitted before encoding")

        # Normalize
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        # Get encoding
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            encoded = self.model.encode(X_tensor)
            return encoded.cpu().numpy()

    def save(self, path: str):
        """Save model to file"""
        if self.fallback_mode:
            # Save simple statistics
            np.savez(
                path,
                scaler_mean=self.scaler_mean,
                scaler_std=self.scaler_std,
                threshold=self.threshold,
                input_dim=self.input_dim,
                fallback_mode=True
            )
            logger.info(f"Fallback model saved to {path}")
        else:
            # Save PyTorch model
            torch.save({
                'model_state': self.model.state_dict(),
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std,
                'threshold': self.threshold,
                'input_dim': self.input_dim,
                'contamination': self.contamination
            }, path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        if self.fallback_mode or not PYTORCH_AVAILABLE:
            # Load simple statistics
            data = np.load(path)
            self.scaler_mean = data['scaler_mean']
            self.scaler_std = data['scaler_std']
            self.threshold = data['threshold']
            self.input_dim = data['input_dim']
            self.fallback_mode = True
            logger.info(f"Fallback model loaded from {path}")
        else:
            # Load PyTorch model
            checkpoint = torch.load(path, map_location=self.device)
            self.scaler_mean = checkpoint['scaler_mean']
            self.scaler_std = checkpoint['scaler_std']
            self.threshold = checkpoint['threshold']
            self.input_dim = checkpoint['input_dim']
            self.contamination = checkpoint.get('contamination', 0.1)

            self.model = AutoencoderNetwork(self.input_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            logger.info(f"Model loaded from {path}")


# Factory function
def create_autoencoder(
    contamination: float = 0.1,
    **kwargs
) -> AutoencoderFraudDetector:
    """
    Create autoencoder fraud detector

    Args:
        contamination: Expected fraud rate
        **kwargs: Additional arguments for AutoencoderFraudDetector

    Returns:
        AutoencoderFraudDetector instance
    """
    return AutoencoderFraudDetector(contamination=contamination, **kwargs)


__all__ = ['AutoencoderFraudDetector', 'create_autoencoder', 'PYTORCH_AVAILABLE']