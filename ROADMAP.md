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
- [x] Derive scaled dot-product and multi-head attention.
- [x] Implement transformer building blocks.
- [x] Compare recurrent and transformer-based captioning as conditional
      autoregressive generation, connecting it to exact likelihood, teacher
      forcing, causal masking, and sequential decoding.
- [x] Study the Vision Transformer representation of images and run a small
      controlled CIFAR-10 experiment.

## Phase 5: Understanding and localizing images

- [x] Understand semantic, instance, and panoptic segmentation.
- [x] Compare single-stage and two-stage object detectors.
- [x] Study region proposals, IoU, non-maximum suppression, and detection metrics.
- [x] Visualize learned filters, saliency maps, and intermediate activations;
      compare input saliency, occlusion, CAM, and Grad-CAM.
- [ ] Explore adversarial examples, feature inversion, and style transfer.

## Phase 6: Later Spring 2025 topics

- [x] Understand video tensors, clip sampling, and temporal fusion.
- [x] Implement and compare 3D convolution, motion representations, and
      two-stream video classifiers.
- [x] Explore long-range temporal modeling and multimodal video understanding.
- [x] Run a controlled video-classification comparison and extract reusable
      utilities.
- [x] Study large-scale distributed training conceptually: accelerator and
      cluster hardware, distributed parallelism, memory trade-offs, and
      utilization.
- [x] Study self-supervised pretext tasks, contrastive learning, linear probes,
      and masked image modeling.
- [x] Implement and compare autoregressive image models, autoencoders, and
      variational autoencoders.
- [x] Understand GAN generation, adversarial training, common architectures,
      and training difficulties through a small controlled experiment.
- [x] Understand diffusion as iterative noise-to-data generation, then practice
      a simple rectified-flow training and sampling process.
- [x] Study conditional guidance, latent diffusion, and diffusion Transformers
      at an intuition-first level without requiring full-scale training.
- [x] Compare small image GAN and rectified-flow models in terms of training
      behavior, diversity, generation speed, interpolation, and sensitivity.
- [x] Summarize how autoregressive models, VAEs, GANs, diffusion, flow, and
      conditional generation differ and how modern systems combine them.
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
