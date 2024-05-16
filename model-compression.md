### Model Compression Glossary

### Pruning
Pruning involves removing unnecessary connections or nodes from the neural network. This can be done based on various criteria such as weight magnitude, activation values, or gradients. Pruning can significantly reduce the size of the model while preserving its accuracy.


### Quantization
Quantization involves reducing the precision of the model’s weights and activations from floating-point numbers to fixed-point numbers with fewer bits. This can reduce the memory requirements and computational complexity of the model, but may also result in a slight decrease in accuracy.

### Distillation
Knowledge distillation involves training a smaller model (the student) to mimic the predictions of a larger, more complex model (the teacher). This can transfer the knowledge and accuracy of the larger model to the smaller model, allowing it to achieve similar performance with fewer parameters.

### Low-rank factorization
Low-rank factorization involves decomposing the weight matrices of the neural network into smaller matrices with lower rank. This can reduce the number of parameters and computations required by the model, while preserving its accuracy.