# ContextValve

ContextValve is a lightweight binary classifier that decides whether retrieved context is sufficient to answer a given query — sitting between the retriever and the LLM in a RAG pipeline.

**Problem:** In RAG pipelines, we retrieve k chunks after every query — sometimes they're enough, sometimes they aren't, and sometimes retrieval isn't needed at all. Passing all retrieved chunks to the LLM every time wastes tokens and adds noise. ContextValve sits after retrieval and decides how much context to actually pass on, saving tokens and improving response quality.


## Datasets Used

| Dataset              | Hugging Face                                                     |
| -------------------- | ---------------------------------------------------------------- |
| General-Knowledge    | https://huggingface.co/datasets/MuskumPillerum/General-Knowledge |
| SQuAD                | https://huggingface.co/datasets/rajpurkar/squad                  |
| MATH-500             | https://huggingface.co/datasets/HuggingFaceH4/MATH-500           |
| SQuAD v2             | https://huggingface.co/datasets/rajpurkar/squad_v2               |
| CodeQA               | https://huggingface.co/datasets/vm2825/CodeQA-dataset            |
| CodeAlpaca-20K       | https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K     |
| Databricks Dolly 15K | https://huggingface.co/datasets/databricks/databricks-dolly-15k  |


## Data Generation Pipeline

This dataset is built by combining multiple sources to cover a wide range of LLM use-cases, including general knowledge, reasoning, math, and code-based tasks. The generation process not only ensures broad domain coverage but also reduces multiple dataset biases that typically arise in naive merging strategies.

Positive data (label 1):
- General Knowledge (empty context queries)
- SQuAD, Dolly, CodeQA, CodeAlpaca
- MATH-500

Negative data (label 0):
- SQuAD v2 unanswerable questions

Class Imbalance Handling

Since positive samples heavily dominate, additional negative samples are generated from positive data to prevent model collapse toward majority class learning.

Negative Sample Generation

- Context removal (emptying relevant context from positive samples)
- Context swapping with unrelated samples
- Chunk-level corruption using FAISS-based hard negative mining and replacing most query-relevant chunks with retrieved hard negatives

Bias Reduction

- Prevents “empty context ⇒ positive label” shortcut by using context-removed negative samples.  
- Reduces “long context ⇒ always relevant” bias via context swapping and corruption.  
- Mitigates cosine-similarity overfitting using FAISS-based hard negatives and chunk replacement.

Output

Final dataset contains balanced positive and synthetic negative samples across general, code, math, and reasoning domains. It improves robustness for query–context relevance modeling under both easy and hard scenarios.


## XGBoost Approach

XGBoost is used for its strong performance on high dimensional feature spaces and its ability to effectively learn from complex, non linear feature interactions when sufficient data is available.

Feature Engineering:
- Query and context are encoded using a SentenceTransformer (Qwen/Qwen3-Embedding-0.6B)
- Features constructed per pair:
  - Absolute difference: |q_emb - c_emb|
  - Element-wise product: q_emb * c_emb
  - Cosine similarity between query and context embeddings

Model:
- XGBoost classifier trained on combined feature space
- Uses regularization (max_depth, min_child_weight, subsample, colsample_bytree) to reduce overfitting
- Early stopping based on validation loss

Training Setup:
- A Sample of 83k data points were used
- 80/20 stratified train-test split
- Evaluation using accuracy and classification report

Results:
- Training Accuracy: ~0.90
- Validation Accuracy: ~0.82

Some Outputs:
![alt text](images/XGBoost%20Outputs.png)