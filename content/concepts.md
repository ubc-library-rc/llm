---
layout: default
title: Concepts in Large Language Models
nav_order: 5
---

# Concepts in Large Language Models

### Tokenization


1. **Types of Tokenization**:
   - **Word Tokenization**: Splits text into individual words. For instance, "Fine-tuning is fun" becomes `["Fine-tuning", "is", "fun"]`.
   - **Subword Tokenization**: Breaks down words into smaller units to handle out-of-vocabulary (OOV) words and morphological variations. For example, "fine-tuning" might be split into `["fine", "-", "tuning"]`.
   - **Character Tokenization**: Treats each character as a token, which can be useful for languages with complex word forms or to handle rare words.

2. **Unified Vocabulary**:
   - A model’s vocabulary is constructed from the training corpus and includes a list of all possible tokens the model might encounter. Each token is associated with a unique identifier. For example, in a vocabulary:
     - "Fine-tuning" -> ID `101`
     - "is" -> ID `102`
     - "fun" -> ID `103`

3. **Tokenization Example**:
   - **Sentence**: "Tokenization is essential."
   - **Tokenized**: `["Tokenization", "is", "essential", "."]`
   - **IDs**: `["201", "202", "203", "204"]`

4. **Handling OOV Tokens**:
   - Tokens not present in the vocabulary are managed using techniques like subword tokenization. For instance, "supercalifragilisticexpialidocious" might be broken into smaller, known subwords or characters to be processed.


![Tokenization Example](https://miro.medium.com/v2/resize:fit:786/format:webp/1*gWP5Whykah1101EpYy17qQ.png)  
_Image from: https://teetracker.medium.com/llm-fine-tuning-step-tokenizing-caebb280cfc2

In the context of large language models (LLMs), the vocabulary used for tokenization is generally fixed before the pre-training phase of the model. This fixed vocabulary is created using a process that involves analyzing a large corpus of text with techniques such as Byte-Pair Encoding (BPE), WordPiece, or SentencePiece. The vocabulary is determined based on the most efficient set of subwords or tokens and is then used consistently throughout both the pre-training and fine-tuning stages.

This vocabulary generation occurs during a setup phase, separate from the actual pre-training of the model. Once the vocabulary is established, it does not change during the model's training. This is distinct from the model learning to predict or understand sequences of these tokens, which happens during pre-training and fine-tuning.

For example, if you have a pre-trained model like GPT-3 or BERT and want to fine-tune it on a task such as sentiment analysis or named entity recognition, you would use the same tokenizer (with its fixed vocabulary) that was used during the pre-training of the model. This ensures consistency in how the text is tokenized and processed by the model.

This process is common in fine-tuning tasks because it ensures that the model's understanding of the tokenized text remains consistent across different phases, leading to better performance and easier integration.

## Embeddings

**Embeddings** are numerical representations of words or tokens in a continuous vector space. They are crucial in enabling models to understand and process language. Unlike one-hot encoding, which represents words as discrete, high-dimensional vectors, embeddings capture semantic relationships between words in a dense, lower-dimensional space.

### How Embeddings Work

1. **Word Embedding Basics**:
   - **Dense Vectors**: Each word is represented by a dense vector of fixed size, where each dimension captures a different aspect of the word's meaning.
   - **Semantic Similarity**: Words with similar meanings are represented by vectors that are close to each other in the embedding space.

2. **Example of Word Embeddings**:
   Let's consider the sentence "The cat sat on the mat."

   - **Original Words**: ["The", "cat", "sat", "on", "the", "mat"]
   - **Embedding Representation**:
     - "The": `[0.1, -0.2, 0.3, ...]`
     - "cat": `[0.2, 0.1, -0.3, ...]`
     - "sat": `[0.0, -0.1, 0.2, ...]`
     - "on": `[0.1, -0.3, 0.2, ...]`
     - "mat": `[0.3, -0.2, -0.1, ...]`

   In this example, each word is mapped to a vector in a high-dimensional space (e.g., 100-dimensional). The vectors capture semantic relationships: words like "cat" and "mat" might be closer to each other in the vector space compared to "cat" and "sat," reflecting their related meanings in context.

3. **Learning Embeddings**:
   - **Training Process**: Embeddings are learned during the training of models such as Word2Vec, GloVe, or BERT. The model adjusts the vectors based on context and co-occurrence patterns in the training data.
   - **Contextual Information**: Modern models like BERT produce contextual embeddings, where the representation of a word changes based on its surrounding words. For example, the word "bank" will have different embeddings in "river bank" versus "financial bank."

### Positional Encoding


**Positional Encoding** is a technique used in transformer models to provide information about the position of tokens in a sequence. Unlike recurrent neural networks (RNNs) or convolutional neural networks (CNNs), transformers do not have inherent mechanisms to understand token order. Positional encoding helps address this by encoding the position of each token in a way that the model can incorporate into its processing.  Positional encoding injects order information into the input embeddings to ensure that the model understands the sequence of tokens. This is crucial because the transformer architecture processes all tokens simultaneously and lacks the sequential processing of RNNs.

**Example**:
   - **Token Sequence**: Consider a sentence "The cat sat."
   - **Positions**: Each token is assigned a positional encoding vector:
     - "The": Positional encoding vector for position 1
     - "cat": Positional encoding vector for position 2
     - "sat": Positional encoding vector for position 3
   - **Combined Representation**: The final representation for each token is the sum of its word embedding and its positional encoding, allowing the model to use both the word's meaning and its position in the sentence.

![Positional EncodingExample](https://machinelearningmastery.com/wp-content/uploads/2022/01/PE1.png)
Image from: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

#### References
[Complete Guide to Subword Tokenization Methods in the Neural Era](https://blog.octanove.org/guide-to-subword-tokenization/)
[Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary)
[Word Embedding: Basics](https://medium.com/@hari4om/word-embedding-d816f643140)
