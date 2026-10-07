"""Classical image classifiers implemented during the course."""

from cs231n_practice.classifiers.cnn import SmallConvNet
from cs231n_practice.classifiers.knn import KNearestNeighbor
from cs231n_practice.classifiers.linear import (
    TrainingResult,
    classification_accuracy,
    linear_scores,
    predict_linear,
    softmax_loss_and_gradient,
    svm_loss_and_gradient,
    train_linear_classifier,
)
from cs231n_practice.classifiers.neural_net import TwoLayerNet
from cs231n_practice.classifiers.self_supervised import (
    FineTunedClassifier,
    MAEEncoderView,
    ProjectionHead,
    RotationPredictionModel,
    SmallEncoder,
    SmallSimCLR,
    TinyMaskedAutoencoder,
)
from cs231n_practice.classifiers.transformer import TransformerSequenceClassifier
from cs231n_practice.classifiers.video import (
    Small3DVideoClassifier,
    TwoStreamVideoClassifier,
)
from cs231n_practice.classifiers.vision_transformer import TinyVisionTransformer

__all__ = [
    "FineTunedClassifier",
    "KNearestNeighbor",
    "MAEEncoderView",
    "ProjectionHead",
    "RotationPredictionModel",
    "SmallConvNet",
    "SmallEncoder",
    "SmallSimCLR",
    "Small3DVideoClassifier",
    "TrainingResult",
    "TwoLayerNet",
    "TransformerSequenceClassifier",
    "TwoStreamVideoClassifier",
    "TinyMaskedAutoencoder",
    "TinyVisionTransformer",
    "classification_accuracy",
    "linear_scores",
    "predict_linear",
    "softmax_loss_and_gradient",
    "svm_loss_and_gradient",
    "train_linear_classifier",
]
