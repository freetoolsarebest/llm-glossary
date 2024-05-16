### Model Structure and Training Glossary

### Transformers
The Transformer model is a neural network architecture that is particularly effective for natural language processing tasks.

### Encoder
The encoder consists of multiple layers of self-attention and feed-forward neural networks. In each layer, the input embeddings are first transformed using multi-head self-attention, which allows the model to selectively attend to different parts of the input sequence.

### Positional Encoder
Since the Transformer does not use recurrent connections, it needs a way to capture the order and position of the input sequence. This is achieved using positional encoding, which adds a fixed encoding to the input embeddings based on the position of the element in the sequence.

### Decoder
The decoder also consists of multiple layers of self-attention and feed-forward neural networks. In each layer, the decoder takes as input a combination of the output of the previous layer and the encoded input sequence. The decoder is trained to generate the output sequence, one element at a time, by predicting the next element in the sequence based on the previous elements and the encoded input.

### CLM
Causal Language Modeling, a pretraining task where the model reads the texts in order and has to predict the next word.

### MLM
Masked Language Modeling, a pretraining task where the model sees a corrupted version of the texts, usually done by masking some tokens randomly, and has to predict the original text.

### Vision Transformer
The Vision Transformer (ViT) model was proposed in “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”. It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures. A image is being cut into 16x16 grids and treat each grad as a token (word).


### Embeddings
A Mapping from a high-dimensional space (such as a one-hot encoded vector) to a lower-dimensional space (such as a dense vector with a fixed number of dimensions). Embeddings are commonly used to represent categorical or discrete variables (such as words, users, or products) as continuous vectors that can be used as input to a neural network.

### Pre-Training
A key aspect of large language modeling, which involves training a language model on a massive amount of unlabeled text data. The purpose of pre-training is to teach the model to understand the underlying structure of language and to learn useful patterns and relationships between words and phrases.

### Finetuning
Fine-tuning, aka Adaption Tuning or Domain Adaption is the process of adapting a pre-trained language model to a specific task by training it on a smaller, task-specific dataset. Fine-tuning is an approach to transfer learning in which the weights of a pre-trained model are trained on new data. Fine-tuning can be done on the entire neural network, or on only a subset of its layers, in which case the layers that are not being fine-tuned are “frozen” (not updated during the backpropagation step).

### Instruction Tuning
Instruction tuning is the approach to fine-tuning pre-trained LLMs on a collection of formatted instances in the form of natural language, which is highly related to supervised fine-tuning and multi-task prompted training. Instruction tuning is an emergent paradigm in NLP wherein natural language instructions are leveraged with language models to induce zero-shot performance on unseen tasks.

### PEFT
Parameter-Efficient Fine-Tuning methods enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model’s parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs.

