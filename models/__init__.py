#!/usr/bin/env python3
"""
Advanced fraud detection models
Includes autoencoder and GNN implementations
"""

from .autoencoder import AutoencoderFraudDetector, PYTORCH_AVAILABLE
from .gnn_fraud_detector import GNNFraudDetector, TORCH_GEOMETRIC_AVAILABLE

__all__ = [
    "AutoencoderFraudDetector",
    "GNNFraudDetector",
    "PYTORCH_AVAILABLE",
    "TORCH_GEOMETRIC_AVAILABLE",
]
