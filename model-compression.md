### Model Compression Glossary

### Pruning
Pruning involves removing unnecessary connections or nodes from the neural network. This can be done based on various criteria such as weight magnitude, activation values, or gradients. Pruning can significantly reduce the size of the model while preserving its accuracy.


### Quantization
Quantization involves reducing the precision of the model’s weights and activations from floating-point numbers to fixed-point numbers with fewer bits. This can reduce the memory requirements and computational complexity of the model, but may also result in a slight decrease in accuracy.

### Distillation
Knowledge distillation involves training a smaller model (the student) to mimic the predictions of a larger, more complex model (the teacher). This can transfer the knowledge and accuracy of the larger model to the smaller model, allowing it to achieve similar performance with fewer parameters.

### Low-rank factorization
Low-rank factorization involves decomposing the weight matrices of the neural network into smaller matrices with lower rank. This can reduce the number of parameters and computations required by the model, while preserving its accuracy.


### Knowledge Distillation
A smaller "student" model is trained to mimic a larger "teacher" model, achieving similar performance with a more compact architecture.

### Tensor Decomposition
It breaks down weights into tensors with fewer parameters, maintaining accuracy while improving efficiency.

### Weight Sharing
Reduces model size by using the same weights for multiple connections, effectively sharing parameters across the network.

### Transfer Learning 
Utilizes a pre-trained model on a new task, requiring only fine-tuning rather than training from scratch, saving resources.

### Binary and Ternary Networks
These networks use binary or ternary values for weights and activations, drastically reducing memory usage.

### Network Slimming
Identifies and removes less significant channels in a neural network, leading to a slimmer and faster model.

### Structured Pruning
Targets specific structures like convolutional filters or channels for removal, optimizing the network's architecture.

### Sparse Representations
Encourages sparsity in the network's parameters, leading to fewer connections and a lighter model.

### Parameter Sharing
Similar to weight sharing, it reuses parameters across different parts of the model to minimize redundancy.

### Dynamic Network Surgery
Combines pruning and splicing to dynamically adjust the network structure during training for optimal compression.

### Deep Compression 
A multi-stage pipeline that includes pruning, quantization, and Huffman coding to compress neural networks.

### Filter Pruning
Focuses on pruning filters from convolutional layers to reduce the number of computations and model size.

### Neural Architecture Search (NAS) 
Automatically searches for efficient neural network architectures that require fewer resources.

### Feature Map Reuse
Reduces computation by reusing intermediate feature maps in a neural network.

### Energy-Aware Pruning
Prunes the network with a focus on reducing energy consumption during inference.

### Data-Free Quantization
Quantizes a network without using original training data, relying on synthetic or distilled data.

### Layer Fusion 
Combines multiple layers into one to reduce the computational graph's complexity and improve efficiency.