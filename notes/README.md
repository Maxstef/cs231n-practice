# Notes

Concise conceptual references accompanying the executable notebooks. These
notes are written in original language from course study and experimentation;
they are not copies of Stanford lecture slides.

## Contents

### Python, NumPy, and PyTorch foundations

- [Reordering array and tensor axes](00-python-numpy/axis-reordering.md)
- [Reshaping and flattening](00-python-numpy/reshaping-and-flattening.md)
- [PyTorch shape operations and gather](00-python-numpy/pytorch-shapes-and-gather.md)
- [Broadcasting and new axes](00-python-numpy/broadcasting-and-new-axes.md)
- [Reductions and axes](00-python-numpy/reductions-and-axes.md)
- [Vector normalization and cosine similarity](00-python-numpy/vector-normalization-and-cosine-similarity.md)
- [Indexing and scatter-add](00-python-numpy/indexing-and-scatter-add.md)
- [PyTorch: detach, cpu, and clone](00-python-numpy/detach-cpu-and-clone.md)
- [PyTorch gradient methods and properties](00-python-numpy/pytorch-grad-basics.md)

### Course context

- [Computer vision and deep learning](01-course-context/computer-vision-and-deep-learning.md)

### Image classification

- [k-Nearest Neighbors](02-image-classification/knn.md)
- [Data splitting and model selection](02-image-classification/model-selection.md)

### Linear classifiers

- [Scores and geometry](03-linear-classifiers/scores-and-geometry.md)
- [Multiclass SVM loss](03-linear-classifiers/multiclass-svm-loss.md)
- [Regularization](03-linear-classifiers/regularization.md)
- [Softmax, cross-entropy, and log loss](03-linear-classifiers/softmax-cross-entropy-log-loss.md)
- [Stable softmax cross-entropy from logits](03-linear-classifiers/stable-softmax-cross-entropy.md)
- [Sigmoid, logistic, and softmax](03-linear-classifiers/sigmoid-logistic-softmax.md)

### Optimization

- [Gradient descent and stochastic gradient descent](04-optimization/gradient-descent-vs-sgd.md)
- [SGD optimizers](04-optimization/sgd-optimizers.md)
- [Hyperparameter tuning and learning rates](04-optimization/hyperparameter-tuning-and-learning-rates.md)
- [Reading training and validation curves](04-optimization/reading-learning-curves.md)

### Neural networks and backpropagation

- [Computational graphs](05-neural-networks/computational-graphs.md)
- [Backpropagation with vectors and matrices](05-neural-networks/backpropagation.md)
- [Activation functions](05-neural-networks/activation-functions.md)
- [Two-layer neural networks](05-neural-networks/two-layer-neural-network.md)
- [Weight initialization](05-neural-networks/weight-initialization.md)
- [Normalization layers](05-neural-networks/normalization.md)
- [Dropout](05-neural-networks/dropout.md)

### Convolutional neural networks

- [Convolution layers](06-convolutional-networks/convolution-layers.md)
- [Convolution versus cross-correlation](06-convolutional-networks/convolution-vs-cross-correlation.md)
- [Pooling and spatial downsampling](06-convolutional-networks/pooling-and-downsampling.md)
- [Receptive fields and small CNNs](06-convolutional-networks/receptive-fields-and-small-cnns.md)
- [Dilation and the parameters that affect receptive fields](06-convolutional-networks/dilation-and-receptive-fields.md)
- [From early CNNs to deeper architectures](06-convolutional-networks/cnn-architectures.md)
- [Residual connections](06-convolutional-networks/residual-connections.md)
- [Image data augmentation](06-convolutional-networks/data-augmentation.md)
- [Transfer learning with CNNs](06-convolutional-networks/transfer-learning.md)

### Sequence models

- [Vanilla recurrent neural networks](07-sequence-models/vanilla-rnns.md)
- [Backpropagation through time and gradient flow](07-sequence-models/bptt-and-gradient-flow.md)
- [LSTM and GRU gated recurrent cells](07-sequence-models/lstm-and-gru.md)
- [Sequence generation and image captioning](07-sequence-models/sequence-generation-and-captioning.md)

### Attention and Transformers

- [Attention, self-attention, and cross-attention](08-attention-transformers/attention-self-and-cross-attention.md)
- [RNN sequence-to-sequence models with attention](08-attention-transformers/rnn-sequence-to-sequence-attention.md)
- [Attention heads and multi-head attention](08-attention-transformers/attention-heads-and-multi-head-attention.md)
- [Transformer blocks and architecture](08-attention-transformers/transformer-blocks-and-architecture.md)
- [Vision Transformers](08-attention-transformers/vision-transformers.md)

### Object detection

- [The object-detection problem](09-detection/detection-problem-and-localization.md)
- [Two-stage and one-stage object detectors](09-detection/two-stage-and-one-stage-detectors.md)
- [Detection post-processing and evaluation](09-detection/detection-postprocessing-and-evaluation.md)
- [DETR and object detection as set prediction](09-detection/detr-and-set-prediction.md)

### Segmentation and model understanding

- [Semantic segmentation and dense prediction](09-segmentation/semantic-segmentation-and-dense-prediction.md)
- [Upsampling, encoders, and decoders](09-segmentation/upsampling-encoders-and-decoders.md)
- [Instance and panoptic segmentation](09-segmentation/instance-and-panoptic-segmentation.md)
- [Visualizing and understanding vision models](09-segmentation/visualizing-and-understanding-vision-models.md)

### Video understanding

- [Video representations, sampling, and fusion](10-video-understanding/video-representations-sampling-and-fusion.md)
- [3D convolution and video CNNs](10-video-understanding/3d-convolution-and-video-cnns.md)
- [Inflating 2D networks into I3D](10-video-understanding/inflating-2d-networks-to-i3d.md)
- [Measuring motion with optical flow](10-video-understanding/optical-flow.md)
- [Motion and two-stream networks](10-video-understanding/motion-and-two-stream-networks.md)
- [Long-range video modeling](10-video-understanding/long-range-video-modeling.md)
- [Video tasks and modern systems](10-video-understanding/video-tasks-and-modern-systems.md)
- [Multimodal and audio-visual video understanding](10-video-understanding/multimodal-audio-visual-understanding.md)

### Large-scale distributed training

- [GPU hardware and cluster communication](11-distributed-training/gpu-hardware-and-clusters.md)
- [GPU versus TPU](11-distributed-training/gpu-vs-tpu.md)
- [Data parallelism, FSDP, and HSDP](11-distributed-training/data-parallelism-and-sharding.md)
- [Context, pipeline, and tensor parallelism](11-distributed-training/model-parallelism.md)
- [Training memory and compute efficiency](11-distributed-training/memory-and-efficiency.md)

### Self-supervised learning

- [Self-supervised learning and evaluation](12-self-supervised/self-supervised-learning-and-evaluation.md)
- [Contrastive learning, SimCLR, and InfoNCE](12-self-supervised/contrastive-learning-simclr-and-infonce.md)
- [Masked image modeling and MAE](12-self-supervised/masked-image-modeling-and-mae.md)
- [MoCo, CPC, and DINO](12-self-supervised/moco-cpc-and-dino.md)

### Generative models

- [Generative models and likelihood](13-generative-models/generative-models-and-likelihood.md)
- [Autoregressive generative models](13-generative-models/autoregressive-generative-models.md)
- [Autoencoders and variational autoencoders](13-generative-models/autoencoders-and-variational-autoencoders.md)
- [VAE ELBO and reparameterization](13-generative-models/vae-elbo-and-reparameterization.md)

## Conventions

- Equations and diagrams should be recreated rather than copied from slides.
- Each note should distinguish established facts from experiment observations.
- Reusable implementation details belong in `cs231n_practice/`.
- Longer executable derivations and visualizations belong in `notebooks/`.
- Private lecture screenshots used for study stay in the ignored
  `_references/` directory and must not be committed.

## Primary course source

- [Stanford CS231n Spring 2025 schedule and materials](https://cs231n.stanford.edu/2025/schedule.html)
