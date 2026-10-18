"""Reusable image models implemented during the course."""

from cs231n_practice.classifiers.autoregressive import (
    BinaryImageAutoregressiveModel,
)
from cs231n_practice.classifiers.autoencoder import (
    ConvolutionalAutoencoder,
    ConvolutionalDecoder,
    ConvolutionalVAE,
    TinyAutoencoder,
    VAEEncoderView,
)
from cs231n_practice.classifiers.cnn import SmallConvNet
from cs231n_practice.classifiers.gan import (
    ConvolutionalDiscriminator,
    ConvolutionalGenerator,
    MLPDiscriminator,
    MLPGenerator,
)
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
from cs231n_practice.classifiers.rectified_flow import (
    ConditionalVelocityMLP,
    SmallImageVelocityNet,
    VelocityMLP,
)
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
from cs231n_practice.classifiers.vae import TinyVAE

__all__ = [
    "BinaryImageAutoregressiveModel",
    "ConvolutionalAutoencoder",
    "ConvolutionalDecoder",
    "ConvolutionalDiscriminator",
    "ConvolutionalGenerator",
    "ConvolutionalVAE",
    "ConditionalVelocityMLP",
    "FineTunedClassifier",
    "KNearestNeighbor",
    "MAEEncoderView",
    "MLPDiscriminator",
    "MLPGenerator",
    "ProjectionHead",
    "RotationPredictionModel",
    "SmallConvNet",
    "SmallEncoder",
    "SmallImageVelocityNet",
    "SmallSimCLR",
    "Small3DVideoClassifier",
    "TrainingResult",
    "TwoLayerNet",
    "TransformerSequenceClassifier",
    "TwoStreamVideoClassifier",
    "TinyMaskedAutoencoder",
    "TinyAutoencoder",
    "TinyVisionTransformer",
    "TinyVAE",
    "VAEEncoderView",
    "VelocityMLP",
    "classification_accuracy",
    "linear_scores",
    "predict_linear",
    "softmax_loss_and_gradient",
    "svm_loss_and_gradient",
    "train_linear_classifier",
]
