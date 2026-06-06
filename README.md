# Mingi Jeong

**ML / LLM Engineer** — RAG, LLM evaluation, information extraction, and serving.

7 years across NLP/LLM and data science. I build LLM systems and, just as
importantly, *measure whether they actually work* — RAG pipelines, extraction,
and quantitative LLM evaluation. I report limitations honestly (self-judge bias,
missing faithfulness metrics, negative results) rather than cherry-picking numbers.

Currently going deeper on **model serving (vLLM)** and **MLOps (MLflow, Airflow)**
to own systems end-to-end — from notebook to a deployed, monitored endpoint.

📍 Korea  ·  NLP/LLM @ 42Maru (5.4y), Data Scientist @ MetLife (1.5y)  ·  [LinkedIn](https://www.linkedin.com/in/mingi-jeong-8a9210180/)

### Open source
Most of these came out of building a Korean insurance RAG + evaluation stack and dogfooding the libraries it runs on — reproduced and fixed locally, not drive-by PRs:
- **run-llama/llama_index** [#21900](https://github.com/run-llama/llama_index/pull/21900) — fixed an infinite-recursion crash (`RecursionError`, then `IndexError`) in `TokenTextSplitter`/`SentenceSplitter` when a single CJK/emoji token is larger than `chunk_size`, with regression tests *(open)*.
- **explosion/spaCy** [#13974](https://github.com/explosion/spaCy/pull/13974) — fixed the Korean tokenizer collapsing runs of whitespace, so `doc.text` round-trips and character offsets stay aligned, with a regression test *(open)*.
- **huggingface/sentence-transformers** [#3800](https://github.com/huggingface/sentence-transformers/pull/3800) — fixed a bf16/fp16 training crash across six learning-to-rank losses (float32 cast + a regression test) *(open)*.
- **mlflow/mlflow** — triaged a MySQL migration crash-loop ([#23721](https://github.com/mlflow/mlflow/issues/23721), reproduced on MySQL 8.4) and a `genai.evaluate` regression ([#23746](https://github.com/mlflow/mlflow/issues/23746)); reported a silent empty-document bug in the RAG groundedness scorer ([#23817](https://github.com/mlflow/mlflow/issues/23817)) with a fix.
- Also reported, with reproductions: a silent `trust_remote_code` enable in **FlagEmbedding** and a `roc_curve` `drop_intermediate` edge case in **scikit-learn**.

### Focus
- **RAG** — retrieval quality, faithfulness / grounding, citation
- **LLM evaluation** — eval-harness design, judge reliability, benchmark construction
- **Extraction** — unstructured → structured, schema enforcement
- **Serving & MLOps** *(in progress)* — vLLM serving, MLflow / Airflow pipelines, observability

### Selected work
- [population-baseline-risk](https://github.com/Incheonkirin/population-baseline-risk) — XGBoost risk modeling on Korean KNHANES/NHIS public data, with calibration and an explicit self-critique of what the model can and cannot claim
- [insurance-bias-probe](https://github.com/Incheonkirin/insurance-bias-probe) — measuring age/gender bias in LLM responses across structured scenarios (10 scenarios × 6 demographic variants)
- **Dispatch** — a classify-then-route agent for insurance QA (SQL / clause search / compare / reason) that goes beyond plain RAG by sending each question to the tool that can actually answer it — *opening the repo shortly*

### Tech
Python · PyTorch · LLM eval (ragas) · RAG (BGE-M3, Chroma) · XGBoost · MLflow · Docker · FastAPI
