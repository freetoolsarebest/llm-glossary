### Retrieval augmented generation (RAG) related Glossary


### RAG (Retrieval Augmented Generation)
A technique that makes AI writing smarter by looking up information before answering. It's like giving the AI a library to check facts before it speaks.

### Retriever
The part of RAG that searches for information. It's like a super-fast librarian that finds relevant books or articles when you ask a question.

### Generator
The part of RAG that writes the answer. It uses the information from the Retriever to create a response, like a writer using research to craft a story.

### Knowledge Base
A big collection of information that the Retriever searches through. Think of it as a massive digital library full of facts and data.

### Query
The question or request you give to the RAG system. It's like the topic you'd give a librarian to help you find the right books.

### Document
A piece of text in the knowledge base. It could be an article, a paragraph, or any chunk of information that might be useful for answering queries.

### Embedding
A way to turn words into numbers that computers can understand. It helps the computer find similar meanings, even if the words are different.

### Vector Database
A special kind of database that stores embeddings. It's designed to quickly find similar items, like a library catalog that can find books with similar themes super fast.

### Relevance Score
A number that shows how well a document matches the query. Higher scores mean the document is more likely to contain useful information for the answer.

### Context Window
The amount of text the AI can consider at one time. It's like how much you can fit on a desk while working - too much, and things get messy.

### Prompt Engineering
The art of asking AI the right questions to get good answers. It's like knowing how to phrase your request to a genie to get exactly what you want.

### Fine-tuning
Teaching an AI model to be better at specific tasks. It's like giving extra training to an athlete for their particular sport.

### Chunking
Breaking big texts into smaller pieces. This helps the AI digest information better, like cutting a large meal into bite-sized portions.

### Indexing
Organizing information so it's easy to find later. It's like creating a really good table of contents for a huge book.

### Latency
The time it takes for the RAG system to give you an answer. Lower latency means faster responses, like the difference between instant messaging and sending a letter.

### Semantic Search
A way of finding information based on meaning, not just exact words. It's like having a librarian who understands what you mean, even if you use different words than the author.

### Token
The smallest unit of text that an AI model processes. Tokens are like the individual Lego blocks that make up the whole structure of language.

### Dense Retrieval
A method of finding relevant information using the entire meaning of a query. It's like comparing whole concepts rather than just matching keywords.

### Sparse Retrieval
A method that finds information by matching specific keywords. It's like using a book's index to find pages that mention certain words.

### Hybrid Search
Combining dense and sparse retrieval for better results. It's like using both a detailed map and local knowledge to find the best route.

### Reranking
Sorting retrieved documents to put the most relevant ones first. It's like organizing search results so the most useful ones are at the top of the list.

### Zero-shot Learning
The ability of an AI to perform tasks it wasn't specifically trained for. It's like a chef creating a new dish using their general cooking knowledge, without following a specific recipe.

### Few-shot Learning
Teaching an AI to do new tasks with just a few examples. It's similar to showing someone how to do a simple task once or twice and then letting them try it themselves.

### Corpus
The entire collection of documents used in a RAG system. Think of it as the complete library that the AI can access for information.

### Cross-encoder
A model that compares a query and a document together for better matching. It's like having a judge who looks at both sides of an argument at the same time to make a fair decision.

### Bi-encoder
A model that creates separate embeddings for queries and documents. It's like having two translators, one for questions and one for answers, who then match their translations.

### Distillation
A technique to create smaller, faster models that mimic larger ones. It's like creating a pocket-sized expert that knows almost as much as the full-sized version.

### Attention Mechanism
A part of AI models that helps focus on important information. It's like having a spotlight that highlights the most relevant parts of a text.

### Transformer
A type of AI model architecture that's great at understanding context. It's like having a super-smart reader who can understand complex relationships between words and ideas.

### BERT
A popular language model that understands context in both directions. Think of it as a reader who can understand a sentence by looking at words before and after each word.

### GPT
A type of model that predicts what comes next in a sequence. It's like a writer who can continue a story based on what's been written so far.

### T5
A versatile model that can handle many language tasks. It's like a linguistic Swiss Army knife, able to tackle various language challenges.

### Cosine Similarity
A way to measure how similar two embeddings are. It's like comparing the directions two arrows are pointing to see how closely they align.

### Euclidean Distance
Another way to measure the difference between embeddings. It's like measuring the straight-line distance between two points on a map.

### FAISS
A library for efficient similarity search of embeddings. It's like having a super-fast sorting system for finding similar items in a huge collection.

### Sentence-BERT
A version of BERT optimized for creating sentence embeddings. It's like having a specialist who's really good at summarizing the meaning of whole sentences.

### BM25
A ranking function used in information retrieval. It's like a scoring system that helps decide which documents are most relevant to a query.

### TF-IDF
A method to evaluate how important a word is in a document. It's like weighing words based on how often they appear and how unique they are.

### LSH (Locality-Sensitive Hashing)
A technique that hashes similar items into the same "buckets" with high probability. It's like sorting people into rooms based on their interests, so similar people are likely to be in the same room.

### HNSW (Hierarchical Navigable Small World)
An ANN algorithm that creates a graph structure for efficient search. It's like creating a network of friends-of-friends to quickly find people with similar interests.

### IVFPQ (Inverted File Product Quantization)
A method that combines two techniques for fast and memory-efficient similarity search. It's like having a smart filing system that quickly narrows down options and uses compact codes to save space.

### IVF (Inverted File)
A way to organize data into clusters for faster searching. It's like sorting books into different sections of a library so you don't have to search the whole library for each book.

### PQ (Product Quantization)
A technique to compress vectors by splitting them into smaller parts and encoding each part. It's like summarizing a long description using a few key symbols, saving space but keeping the main ideas.

### Coarse Quantizer
The part of IVF that assigns vectors to clusters. It's like a librarian who quickly decides which section a book belongs in without reading the whole thing.

### Residual Vector
The difference between a vector and its closest centroid in IVF. It's like noting how a book differs from the typical book in its library section.

### Subvector
A part of a vector used in Product Quantization. It's like breaking down a long description into shorter phrases that are easier to summarize.

### Codebook
A collection of representative vectors used in quantization. It's like a dictionary of common phrases used to encode longer descriptions.

### Centroid
The center point of a cluster in IVF. It's like the "average" book that represents a whole section in the library.

### Nprobe
The number of closest clusters to search in IVF. It's like deciding how many library sections to check when looking for a specific book.

### Compression Ratio
The amount of space saved by using PQ compared to storing full vectors. It's like how much shelf space you save by using summaries instead of full books.

### Negative Sampling
A technique used in training to improve the model's ability to distinguish relevant from irrelevant information. It's like teaching by showing both good and bad examples.

### In-context Learning
The ability of a model to adapt to new tasks using examples in the prompt. It's like giving a smart student a few example problems before asking them to solve a new one.

### Prompt Template
A pre-designed structure for creating effective prompts. It's like having a fill-in-the-blank form that helps you ask the AI the right questions.

### Retrieval Depth
The number of documents retrieved for each query. It's like deciding how many books to pull off the shelf when researching a topic.

### Passage Ranking
Ordering retrieved text passages by relevance. It's similar to arranging puzzle pieces from most to least important for solving the puzzle.

### Query Expansion
Adding related terms to a query to improve search results. It's like broadening your question to catch more potentially relevant answers.

### Document Compression
Reducing the size of documents while keeping important information. It's like creating a concise summary of a long article.

### Ensemble Methods
Combining multiple models or techniques for better performance. It's like getting opinions from a group of experts instead of just one.

### Multi-hop Reasoning
The ability to connect information from multiple sources to answer complex questions. It's like solving a mystery by connecting clues from different places.

### Semantic Caching
Storing and reusing results based on the meaning of queries. It's like remembering answers to similar questions you've asked before.

### Contrastive Learning
A training method that helps models distinguish between similar and dissimilar items. It's like teaching by showing what things are alike and what things are different.

### Zero-shot Retrieval
Finding relevant information without any previous training on the specific task. It's like being able to find books on a new topic in a library you've never visited before.