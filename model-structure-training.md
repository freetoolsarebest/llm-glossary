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

A general language model could be fine-tuned with academic papers and texts from a specific discipline to better understand and generate text relevant to that field. This process involves adjusting the model’s parameters slightly so that it better aligns with the nuances and terminologies of the target domain while retaining the broad knowledge it gained during initial training.

Fine-tuning offers a balance between the extensive learning of a large, general model and the specific expertise required for particular tasks.

### Instruction Tuning
Instruction tuning is the approach to fine-tuning pre-trained LLMs on a collection of formatted instances in the form of natural language, which is highly related to supervised fine-tuning and multi-task prompted training. Instruction tuning is an emergent paradigm in NLP wherein natural language instructions are leveraged with language models to induce zero-shot performance on unseen tasks.

### PEFT
Parameter-Efficient Fine-Tuning methods enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model’s parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs.

### Few-shot learning
Few-shot learning is a concept in machine learning where the model is designed to learn and make accurate predictions or decisions based on a very limited amount of training data. Traditional machine learning models typically require large datasets to learn effectively. However, few-shot learning techniques enable AI models to generalize from a small number of examples, often just a handful or even a single instance. This approach is especially valuable in situations where collecting large datasets is impractical or impossible, such as specialized academic fields or rare languages.

### Parameters
Parameters are the internal variables of an AI model that are learned from the training data. These parameters are the core components that define the behaviour of the model and determine how it processes input data to produce output. In a neural network, parameters typically include weights and biases associated with the neurons.

Each neuron in a neural network has a weight assigned to its input, which signifies the importance or influence of that input in the neuron’s overall calculation. The bias is an additional parameter that allows the neuron to adjust its output independently of its input. During the training process, the model adjusts these parameters to minimize the difference between its output and the actual data. The better these parameters are tuned, the more accurately the model can perform its intended task.

When looking at open source models you may see them prefixed with “7b” or “70b” i.e. llama2-70b. This often refers to the number of parameters, in this case 70 billion.

### Semantic Network
A semantic network, or frame network is a graphical representation of knowledge that interlinks concepts through their semantic relationships. In these networks, nodes represent concepts or entities, and the edges represent the relationships between these concepts, such as “is a type of,” “has a property of,” or “is part of.” This structure enables the representation of complex interrelationships and hierarchies within a given set of data or knowledge .

Semantic networks can enhance natural language processing capabilities by helping systems understand context and the relationships between different words or phrases.

### Tuning
Tuning describes the process of adjusting a pre-trained model to better suit a specific task or set of data. This involves modifying the model’s parameters so that it can more effectively process, understand, and generate information relevant to a particular application. Tuning is different from the initial training phase, where a model learns from a large, diverse dataset. Instead, it focuses on refining the model’s capabilities based on a more targeted dataset or specific performance objectives.
