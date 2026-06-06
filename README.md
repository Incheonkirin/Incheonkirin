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
I work inside the tools my own projects run on — reproduced and verified locally, not drive-by PRs:
- **huggingface/sentence-transformers** [#3800](https://github.com/huggingface/sentence-transformers/pull/3800) — opened a PR fixing a bf16/fp16 training crash across six learning-to-rank losses: a float32 loss cast plus a regression test that fails on the old code and passes on the fix *(under review)*.
- **mlflow/mlflow** — triaged a MySQL migration crash-loop ([#23721](https://github.com/mlflow/mlflow/issues/23721), reproduced locally on MySQL 8.4) and a `genai.evaluate` regression ([#23746](https://github.com/mlflow/mlflow/issues/23746)).

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
