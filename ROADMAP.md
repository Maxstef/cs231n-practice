# Learning Roadmap

The sequence follows CS231n Spring 2025 while adding explicit implementation,
verification, and reflection stages. A checked box should mean that the concept
can be explained and implemented—not merely that its lecture was watched.

## Phase 0: Foundations

- [x] Create a reproducible Python environment.
- [x] Review NumPy arrays, broadcasting, vectorization, and shape reasoning.
- [x] Establish notebook and testing conventions.
- [x] Load, inspect, and visualize CIFAR-10.

## Phase 1: Classical and linear classifiers

- [x] Understand the image-classification pipeline and train/validation/test
      splits.
- [x] Implement k-nearest neighbors with held-out validation.
- [x] Derive and implement the multiclass SVM loss.
- [x] Derive and implement the softmax loss.
- [x] Derive analytic gradients and verify them numerically.
- [x] Train a linear classifier with stochastic gradient descent.
- [x] Use k-fold cross-validation to select linear-classifier hyperparameters.

## Phase 2: Optimization and neural networks

- [x] Understand computational graphs, the chain rule, and gradient
      accumulation.
- [x] Implement reusable numerical gradient-checking utilities.
- [x] Implement reusable affine, ReLU, affine-ReLU, and softmax-loss layers.
- [x] Build and train a two-layer neural network.
- [x] Train, tune, and evaluate the two-layer network on CIFAR-10.
- [ ] Compare SGD, momentum, RMSProp, and Adam.
- [ ] Explore learning-rate schedules and regularization.
- [ ] Generalize it into a fully connected network.
- [x] Add batch normalization and dropout.

## Phase 3: Convolutional networks

- [x] Derive convolution and pooling output shapes.
- [x] Implement naive convolution and pooling forward/backward passes.
- [x] Build and gradient-check a small convolutional network.
- [x] Study AlexNet, VGG, and ResNet design choices.
- [x] Implement and gradient-check a reusable affine residual block.
- [ ] Use transfer learning and compare it with training from scratch.

## Phase 4: Sequences, attention, and transformers

- [x] Implement and gradient-check vanilla RNN sequence forward/backward passes
      and LSTM/GRU recurrent steps.
- [x] Implement reusable embeddings, masked recurrent sequences, temporal
      projections, and temporal softmax loss.
- [x] Study BPTT, truncated BPTT, vanishing/exploding gradients, and global-norm
      clipping.
- [x] Build and gradient-check a small recurrent image-captioning pipeline with
      greedy decoding.
- [ ] Derive scaled dot-product and multi-head attention.
- [ ] Implement transformer building blocks.
- [ ] Compare recurrent and transformer-based captioning.
- [ ] Study the Vision Transformer representation of images.

## Phase 5: Understanding and localizing images

- [ ] Understand semantic, instance, and panoptic segmentation.
- [ ] Compare single-stage and two-stage object detectors.
- [ ] Study region proposals, IoU, non-maximum suppression, and detection metrics.
- [ ] Visualize learned filters, saliency maps, and intermediate activations.
- [ ] Explore adversarial examples, feature inversion, and style transfer.

## Phase 6: Remaining Spring 2025 topics

- [ ] Video understanding.
- [ ] Large-scale distributed training.
- [ ] Self-supervised and contrastive learning.
- [ ] Variational autoencoders, GANs, and autoregressive models.
- [ ] Diffusion models.
- [ ] 3D vision and neural implicit representations.
- [ ] Vision-language models.
- [ ] Robot learning and human-centered AI.

## Topic completion checklist

A topic is complete when its work includes:

- [ ] a concise conceptual explanation;
- [ ] notation, equations, and tensor shapes;
- [ ] an implementation or focused practical exercise;
- [ ] a correctness check or test;
- [ ] at least one controlled experiment;
- [ ] interpretation of results and failure cases;
- [ ] a short set of review questions.
