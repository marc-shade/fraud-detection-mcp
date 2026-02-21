#!/usr/bin/env python3
"""
Graph Neural Network for Fraud Detection
Detects fraud patterns in transaction networks using GNN
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch Geometric, gracefully handle missing dependency
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch Geometric loaded successfully for GNN")
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    try:
        import torch
        PYTORCH_AVAILABLE = True
        logger.warning("PyTorch available but PyTorch Geometric missing - using fallback")
    except ImportError:
        PYTORCH_AVAILABLE = False
        logger.warning("PyTorch not available - GNN will use fallback implementation")


if TORCH_GEOMETRIC_AVAILABLE:
    class GNNNetwork(nn.Module):
        """
        Graph Neural Network for fraud detection
        Uses GraphSAGE convolutions
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2
        ):
            super(GNNNetwork, self).__init__()

            self.num_layers = num_layers

            # Graph convolution layers
            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

            # Batch normalization
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Dropout
            self.dropout = nn.Dropout(dropout)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)  # Binary classification
            )

        def forward(self, x, edge_index, batch=None):
            """
            Forward pass

            Args:
                x: Node features [num_nodes, input_dim]
                edge_index: Edge connectivity [2, num_edges]
                batch: Batch assignment [num_nodes]

            Returns:
                Logits for classification
            """
            # Graph convolutions
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)

            # Global pooling (if batch provided)
            if batch is not None:
                x = global_mean_pool(x, batch)

            # Classification
            out = self.classifier(x)
            return out


class GNNFraudDetector:
    """
    GNN-based fraud detector with sklearn-compatible API
    Analyzes transaction networks for fraud patterns
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize GNN fraud detector

        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            device: Device ('cuda', 'cpu', or None for auto)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.input_dim = None
        self.scaler_mean = None
        self.scaler_std = None

        # Determine device
        if not TORCH_GEOMETRIC_AVAILABLE:
            self.device = 'cpu'
            self.fallback_mode = True
            logger.warning("Using fallback mode (no PyTorch Geometric)")
        else:
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            self.fallback_mode = False
            logger.info(f"GNNFraudDetector initialized on device: {self.device}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        edge_index: Optional[np.ndarray] = None
    ) -> 'GNNFraudDetector':
        """
        Fit GNN on transaction network

        Args:
            X: Node features [num_nodes, num_features]
            y: Labels [num_nodes]
            edge_index: Edge connectivity [2, num_edges] or None

        Returns:
            self for method chaining
        """
        if self.fallback_mode:
            return self._fit_fallback(X, y)

        logger.info(f"Training GNN on {len(X)} nodes")

        # Normalize features
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-7
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        self.input_dim = X.shape[1]

        # Create edge index if not provided (fully connected)
        if edge_index is None:
            edge_index = self._create_knn_edges(X_normalized, k=5)

        # Create PyTorch Geometric data
        data = Data(
            x=torch.FloatTensor(X_normalized),
            edge_index=torch.LongTensor(edge_index),
            y=torch.LongTensor(y)
        ).to(self.device)

        # Create model
        self.model = GNNNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass
            out = self.model(data.x, data.edge_index)
            loss = criterion(out, data.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                # Calculate accuracy
                pred = out.argmax(dim=1)
                correct = (pred == data.y).sum().item()
                acc = correct / len(data.y)
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}, Acc: {acc:.4f}")

        logger.info("GNN training complete")
        return self

    def predict(
        self,
        X: np.ndarray,
        edge_index: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict fraud labels

        Args:
            X: Node features
            edge_index: Edge connectivity

        Returns:
            Binary predictions
        """
        if self.fallback_mode:
            return self._predict_fallback(X)

        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Normalize
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        # Create edge index if not provided
        if edge_index is None:
            edge_index = self._create_knn_edges(X_normalized, k=5)

        # Create data
        data = Data(
            x=torch.FloatTensor(X_normalized),
            edge_index=torch.LongTensor(edge_index)
        ).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            return pred.cpu().numpy()

    def predict_proba(
        self,
        X: np.ndarray,
        edge_index: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict fraud probabilities

        Args:
            X: Node features
            edge_index: Edge connectivity

        Returns:
            Probability matrix [normal_prob, fraud_prob]
        """
        if self.fallback_mode:
            return self._predict_proba_fallback(X)

        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Normalize
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        # Create edge index if not provided
        if edge_index is None:
            edge_index = self._create_knn_edges(X_normalized, k=5)

        # Create data
        data = Data(
            x=torch.FloatTensor(X_normalized),
            edge_index=torch.LongTensor(edge_index)
        ).to(self.device)

        # Predict probabilities
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            proba = F.softmax(out, dim=1)
            return proba.cpu().numpy()

    def _create_knn_edges(self, X: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Create k-nearest neighbor edges

        Args:
            X: Node features
            k: Number of neighbors

        Returns:
            Edge index [2, num_edges]
        """
        from sklearn.neighbors import kneighbors_graph

        # Create KNN graph
        A = kneighbors_graph(X, k, mode='connectivity', include_self=False)

        # Convert to edge list
        edge_index = np.array(A.nonzero())

        return edge_index

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> 'GNNFraudDetector':
        """Fallback fitting using simple classifier"""
        logger.warning("Using fallback mode - limited functionality")

        from sklearn.ensemble import RandomForestClassifier

        # Normalize
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-7
        X_normalized = (X - self.scaler_mean) / self.scaler_std

        # Train simple classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_normalized, y)

        logger.info("Fallback model fitted")
        return self

    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction"""
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X_normalized = (X - self.scaler_mean) / self.scaler_std
        return self.model.predict(X_normalized)

    def _predict_proba_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback probability prediction"""
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        X_normalized = (X - self.scaler_mean) / self.scaler_std
        return self.model.predict_proba(X_normalized)

    def save(self, path: str):
        """Save model to file"""
        if self.fallback_mode or not TORCH_GEOMETRIC_AVAILABLE:
            # Save using joblib for sklearn model
            import joblib
            joblib.dump({
                'model': self.model,
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std,
                'fallback_mode': True
            }, path)
            logger.info(f"Fallback model saved to {path}")
        else:
            # Save PyTorch model
            torch.save({
                'model_state': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std
            }, path)
            logger.info(f"GNN model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        if self.fallback_mode or not TORCH_GEOMETRIC_AVAILABLE:
            # Load using joblib
            import joblib
            data = joblib.load(path)
            self.model = data['model']
            self.scaler_mean = data['scaler_mean']
            self.scaler_std = data['scaler_std']
            self.fallback_mode = True
            logger.info(f"Fallback model loaded from {path}")
        else:
            # Load PyTorch model
            checkpoint = torch.load(path, map_location=self.device)
            self.input_dim = checkpoint['input_dim']
            self.hidden_dim = checkpoint['hidden_dim']
            self.num_layers = checkpoint['num_layers']
            self.scaler_mean = checkpoint['scaler_mean']
            self.scaler_std = checkpoint['scaler_std']

            self.model = GNNNetwork(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            logger.info(f"GNN model loaded from {path}")


# Factory function
def create_gnn_detector(**kwargs) -> GNNFraudDetector:
    """
    Create GNN fraud detector

    Args:
        **kwargs: Arguments for GNNFraudDetector

    Returns:
        GNNFraudDetector instance
    """
    return GNNFraudDetector(**kwargs)


__all__ = [
    'GNNFraudDetector',
    'create_gnn_detector',
    'TORCH_GEOMETRIC_AVAILABLE',
    'PYTORCH_AVAILABLE'
]