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
