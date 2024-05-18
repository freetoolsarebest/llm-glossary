### Training and Inference Arguments Glossary

### Memory
Memory is the concept of persisting state between calls of a chain/agent. LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.

### Indexes
Language models are often more powerful when combined with your own text data — this module covers best practices for doing exactly that.

### Chains
Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.

### Agents
Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done. LangChain provides a standard interface for agents, a selection of agents to choose from, and examples of end to end agents.

### Temperature
The LLM temperature is a hyperparameter that regulates the randomness, or creativity, of the AI’s responses. A higher temperature value typically makes the output more diverse and creative but might also increase its likelihood of straying from the context.

### Tokens
In a language model, the atomic unit that the model is training on and making predictions on.

### Text Span
The array index span associated with a specific subsection of a text string. For example, the word good in the Python string s="Be good now" occupies the text span from 3 to 6.

### Training Data
The large dataset consisting of text used to train a language model. This data helps the model learn language patterns, grammar, facts, and other linguistic information.

### Epoch
One complete pass through the entire training dataset. Multiple epochs are often used to train models, allowing them to learn from the data more effectively.

### Batch Size
The number of training examples utilized in one iteration of model training. Smaller batches require less memory and can make training faster, while larger batches can lead to more stable gradients.

### Gradient Descent
An optimization algorithm used to minimize the loss function by iteratively moving towards the minimum value by updating model parameters.

### Learning Rate
A hyperparameter that controls how much the model's parameters are adjusted with respect to the loss gradient. A smaller learning rate means the model is updated more slowly.

### Loss Function
A function that measures how well the model's predictions match the actual target values. Common loss functions include cross-entropy loss and mean squared error.

### Overfitting
A scenario where the model learns the training data too well, including its noise and outliers, resulting in poor generalization to new, unseen data.

### Regularization
Techniques used to prevent overfitting by adding a penalty to the loss function for large weights, such as L1 or L2 regularization.

### Backpropagation
The process of calculating the gradient of the loss function with respect to each weight by the chain rule, used to update the model's parameters.

### Latency
The time taken to produce an output from the model after receiving an input during inference. Lower latency is crucial for real-time applications.

### Throughput
The number of inferences a model can make per unit of time. Higher throughput indicates more efficient model performance.

### Fine-tuning
Adjusting a pre-trained model on a new, often smaller, dataset to improve performance on specific tasks without starting the training process from scratch.

### Prompt
The initial input or query given to a language model to generate a response. The quality and clarity of the prompt can significantly impact the model's output.

### Context Window
The length of input text the model can consider at once. Models with larger context windows can understand and generate more coherent responses over longer texts.

### Beam Search
A search algorithm used in sequence generation tasks to find the most likely sequence of words. It maintains multiple candidate sequences at each step to improve the quality of predictions.

### Greedy Search
A simpler decoding method where the model selects the most likely word at each step without considering future possibilities, which can lead to suboptimal results.

### Softmax
A function that converts the model's raw output scores (logits) into probabilities, used in the final layer of classification models to predict the likelihood of each class.

### Sampling
A method to generate text by randomly selecting the next word based on its predicted probability, allowing for more diverse outputs compared to deterministic methods like greedy search.

### Tokenization
The process of breaking down text into smaller units (tokens) that the model can process. Tokens can be words, subwords, or characters.

### Attention Mechanism
A component of models like Transformers that allows the model to weigh the importance of different words in the input when generating an output, improving the model's ability to capture relationships in the data.