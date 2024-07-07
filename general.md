### General Glossary

### LM
Language Model, a language model is a probability distribution over sequences of words.

### LLM
Large Language Model, a language model consisting of a neural network with many parameters (typically billions of weights or more), trained on large quantities of unlabeled text using self-supervised learning.

### Generative AI
A generative artificial intelligence or generative AI / (GenAI) is a type of AI system capable of generating text, images, or other media in response to prompts. Generative AI systems use generative models such as large language models to produce data based on the training data set that was used to create them.

### Diffusion Models
Diffusion Models are a class of probabilistic generative models used in machine learning to simulate the dynamics of complex systems over time. Unlike traditional generative models such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), Diffusion Models simulate the dynamics of a stochastic process that evolves over time, rather than generating a fixed sample from a learned distribution.

### Hallucination
In artificial intelligence (AI), a hallucination or artificial hallucination (also occasionally called confabulation or delusion) is a confident response by an AI that does not seem to be justified by its training data.

### Prompt Engineering
A process of designing and constructing effective natural language prompts for use with large language models. Prompts are input patterns that are used to guide the behavior of LLMs and generate text that is relevant to a specific task or domain.

### Zero-shot Prompting/Learning
Zero-shot learning in NLP allows a pre-trained LLM to generate responses to tasks that it hasn’t been specifically trained for. In this technique, the model is provided with an input text and a prompt that describes the expected output from the model in natural language.

### Context Window / Context Length
Context Window is the number of tokens that are considered when predicting the next token.

### Matryoshka Embeddings
New state-of-the-art (text) embedding models started producing embeddings with increasingly higher output dimensions, i.e., every input text is represented using more values. Although this improves performance, it comes at the cost of efficiency of downstream tasks such as search or classification.
[Read More](https://huggingface.co/blog/matryoshka)

### AGI or Artificial General Intelligence
Artificial General Intelligence (AGI) represents a level of AI development where machines possess the ability to understand, learn, and apply intelligence across a broad range of tasks, mimicking the cognitive abilities of a human being. Unlike most current AI systems, which are designed for specific tasks (narrow AI), AGI can theoretically perform any intellectual task that a human can. It encompasses a wide array of cognitive skills, including reasoning, problem-solving, perception, language understanding, and general knowledge application.

### Annotation
Annotation is the process of labelling or tagging data, which is then used to train and fine-tune AI models. This data can be in various forms, such as text, images, or audio. In text-based generative AI, annotation might involve categorizing sentences, identifying parts of speech, or marking sentiment in text snippets. These annotated data-sets become the foundational building blocks that enable the AI to learn and understand patterns, contexts, and nuances of the data it is meant to generate or interpret.

### ASI or Artificial Super Intelligence
Artificial Super Intelligence (ASI) refers to a stage of artificial intelligence that surpasses human intelligence across all fields, including creativity, general wisdom, and problem-solving capabilities. Unlike Artificial General Intelligence (AGI), which aims to match human cognitive abilities, ASI represents an AI that is vastly more advanced than the best human brains in practically every field, including scientific understanding, general knowledge, and social skills.

### Bias (in Gen AI)
There are 2 distinct ways of using the term ‘bias’ with regards to Generative AI.

Firstly, bias can, somewhat more commonly known, refer to a systemic skew or prejudice in the AI model’s output, often reflecting inherent or learned prejudices in the data it was trained on. Bias in AI can manifest in various forms, such as cultural, gender, racial, political, or socioeconomic biases. These biases can lead to AI systems making decisions or generating content that is unfair, stereotypical, or discriminative in nature.

Secondly, in the technical construction of AI models, particularly neural networks, bias refers to a parameter that is used alongside “weights” to influence the output of a node in the network. While weights determine how much influence an input will have on a node, biases allow for an adjustment to the output independently of its inputs. The bias parameter is essential in tuning a model‘s behaviour, as it provides the flexibility needed for the model to accurately represent complex patterns in the data. Without biases, a neural network might be significantly less capable of fitting diverse and nuanced datasets, limiting its effectiveness and accuracy.

### Bot
In the context of Generative AI, a ‘bot’ (short for robot) typically refers to a software application that is programmed to perform automated tasks. These tasks can range from simple, repetitive activities to more complex functions involving decision-making and interactions with human users. They are often equipped with advanced capabilities such as understanding and generating language, responding to user queries, or creating content based on specific guidelines or prompts.

Certain tools such as ChatGPT or Poe allow you to create your own ‘bots’ (called GPTs for ChatGPT).

### Chat Bot
A chatbot is a software application designed to simulate conversation with human users, especially over the internet. It utilizes techniques from the field of natural language processing (NLP) and sometimes machine learning (ML) to understand and respond to user queries. Chatbots can range from simple, rule-based systems that respond to specific keywords or phrases with pre-defined responses, to more sophisticated AI-driven bots capable of handling complex, nuanced, and context-dependent conversations.

### Completions
Completions are the output produced by AI in response to a given input or prompt. When a user inputs a prompt, the AI model processes it and generates text that logically follows or completes the given input. These completions are based on the patterns, structures, and information the model has learned during its training phase on vast datasets.

### Conversational or Chat AI
Conversational AI or Chat AI refers to the branch of artificial intelligence focused on enabling machines to understand, process, and respond to human language in a natural and conversational manner. This technology underpins chat bots and virtual assistants, which are designed to simulate human-like conversations with users, providing responses that are contextually relevant and coherent. Conversational AI combines elements of natural language processing (NLP), machine learning (ML), and sometimes speech recognition to interpret and engage in dialogue.

### GPT or Generative Pre-Trained Transformers
Generative Pre-trained Transformers (GPT) are a type of advanced artificial intelligence model primarily used for natural language processing tasks. GPT models are based on the transformer architecture, which allows them to efficiently process and generate human-like text by learning from vast amounts of data. The “pre-trained” aspect refers to the initial extensive training these models undergo on large text corpora, allowing them to understand and predict language patterns. This pre-training equips the GPT models with a broad understanding of language, context, and aspects of world knowledge.

The Generative aspect is important to remember – these tools are designed to generate human-like responses rather than, for example, a Google search which regurgitates information.

### Hallucinations
Hallucinations are incorrect or misleading results that AI models generate. These errors can be caused by a variety of factors, including insufficient training data, incorrect assumptions made by the model, or biases in the data used to train the model. The concept of AI hallucinations underscores the need for critical evaluation and verification of AI-generated information, as relying solely on AI outputs without scrutiny could lead to the dissemination of misinformation or flawed analyses.

### Inference
Inference is the process where a trained AI model applies its learned knowledge to new, unseen data to make predictions, decisions, or generate content. It is essentially the phase where the AI model, after being trained on a large dataset, is now being used in real-world applications. Unlike the training phase, where the model is learning from examples, during inference, the model is utilizing its learned patterns to perform the specific tasks it was designed for.

For example a language model that has been trained on a vast corpus of text can perform inference by generating a new essay, answering a student’s query, or summarizing a research article.

### Model
Models are the computational structure and algorithms that enable Generative AI to process data, learn patterns, and perform tasks such as generating text, images, or making decisions. Essentially, it is the core framework that embodies an AI’s learned knowledge and capabilities. A model in AI is created through a process called training, where it is fed large amounts of data and learns to recognize patterns, make predictions, or generate outputs based on that data.

Each model has its specific architecture (such as neural networks) and parameters, which define its abilities and limitations. The quality, diversity, and size of the data used in training also significantly influence a model’s effectiveness and reliability in practical applications.

### NLP or Natural Language Programming
NLP is a field at the intersection of computer science, artificial intelligence, and linguistics, focused on enabling computers to understand, interpret, and generate human language in a way that is both meaningful and useful. It involves the development of algorithms and systems that can analyze, comprehend, and respond to text or voice data in a manner similar to how humans do.

### Prompt
A prompt is the input given to an AI model to initiate or guide its generation process. This input acts as a directive or a set of instructions that the AI uses to produce its output. Prompts are crucial in defining the nature, scope, and specificity of the output generated by the AI system. For instance, in a text-based Generative AI model like GPT (Generative Pre-trained Transformer), a prompt could be a sentence or a question that the model then completes or answers in a coherent and contextually appropriate manner.

### Reinforcement Learning
Reinforcement Learning (RL) is a type of learning algorithm where an agent learns to make decisions by performing actions in an environment to achieve a certain goal. The learning process is guided by feedback in the form of rewards or punishments — positive reinforcement for desired actions and negative reinforcement for undesired actions. The agent learns to maximize its cumulative reward through trial and error, gradually improving its strategy or policy over time.

### Retrieval Augmented Generation (RAG)
Retrieval Augmented Generation (RAG) is a technique that combines the strengths of both retrieval-based and generative models. In this approach, an AI system first retrieves information from a large dataset or knowledge base and then uses this retrieved data to generate a response or output. Essentially, the RAG model augments the generation process with additional context or information pulled from relevant sources.

For example, using an RAG pipeline, you might be able to provide private research data (without exposing it to 3rd-party tools) which you can then ask complex questions about, and ask for analyses on. Normally, this data wouldn’t be available to a general model, but with the RAG pipeline, you can provide custom, private data.

### Training
Training is the process by which a machine learning model, such as a neural network, learns to perform a specific task. This is achieved by exposing the model to a large set of data, known as the training dataset, and allowing it to iteratively adjust its internal parameters to minimize errors in its output.

During training, the model makes predictions or generates outputs based on its current state. These outputs are then compared to the desired results, and the difference (or error) is used to adjust the model’s parameters. This process is repeated numerous times, with the model gradually improving its accuracy and ability to perform the task. For example, a language model is trained on vast amounts of text so that it learns to understand and generate human-like language.

### Zero-Shot Learning
Zero-shot learning is a concept that describes a concept where an AI model learns to perform tasks that it has not explicitly been trained to do. Unlike traditional machine learning methods that require examples from each class or category they’re expected to handle, zero-shot learning enables the model to generalize from its training and make inferences about new, unseen categories.

This is achieved by training the model to understand and relate abstract concepts or attributes that can be applied broadly. For instance, a model trained in zero-shot learning could categorize animals it has never seen before based on learned attributes like size, habitat, or diet. It infers knowledge about these new categories by relying on its understanding of the relationships and similarities between different concepts.