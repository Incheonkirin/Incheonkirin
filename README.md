<img src="banner.svg" width="100%" alt="Mingi Jeong" />

**ML/LLM Engineer @ MetLife** — search & RAG systems, LLM serving and evaluation

Previously 5.5 years on the search team at **42Maru** — Korean BM25/IR engines,
MRC (machine reading comprehension), and enterprise RAG QA systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

---

## 🔧 Upstream contributions

Found by dogfooding my own Korean RAG + evaluation stack,
[**search_system**](https://github.com/Incheonkirin/search_system) — a Korean
insurance-clause retrieval testbed (nori BM25 + BGE-M3 hybrid, qrels-based
eval, failure-class catalog). Most of these share one shape:

> **Data that is valid on one side of a representation boundary silently
> breaks the other** — NFD Hangul vs. the analyzer, stop strings vs.
> byte-fragment tokens, a literal `</tool_call>` vs. the tool-call parser,
> bf16 logits vs. a float32 loss. Korean hits these boundaries constantly;
> English-only test suites never do.

- **[apache/lucene #16242](https://github.com/apache/lucene/pull/16242)** — new `HangulCompositionCharFilter` for analysis-nori: NFD-form Hangul was silently unanalyzable as Korean ([#16241](https://github.com/apache/lucene/issues/16241)). *(open)*
- **[elastic/elasticsearch #151008](https://github.com/elastic/elasticsearch/pull/151008)** — wildcard queries: re-escape operator characters produced by the normalizer. *(open)*
- **[huggingface/transformers #46530](https://github.com/huggingface/transformers/pull/46530)** — `StopStringCriteria` misses CJK stop strings on byte-level tokenizers ([#46519](https://github.com/huggingface/transformers/issues/46519)). *(open)*
- **[vllm-project/vllm #45168](https://github.com/vllm-project/vllm/pull/45168)** — Hermes tool parser drops tool calls when a literal `</tool_call>` appears inside a JSON string argument ([#45167](https://github.com/vllm-project/vllm/issues/45167)). *(open)*
- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800)** — bf16/fp16 training crash across six learning-to-rank losses. ***(merged)***
- **[explosion/spaCy #13974](https://github.com/explosion/spaCy/pull/13974)** — Korean tokenizer collapsed whitespace runs, breaking `doc.text` round-trips and offsets. *(open)*
- **[run-llama/llama_index #21900](https://github.com/run-llama/llama_index/pull/21900)** — `RecursionError` in text splitters when a single CJK/emoji token exceeds `chunk_size`. *(approved)*

Also: [mlflow #23818](https://github.com/mlflow/mlflow/pull/23818) (OTel retriever-span reassembly), [ragas #2759](https://github.com/vibrantlabsai/ragas/pull/2759), BentoML [#5632](https://github.com/bentoml/BentoML/pull/5632)·[#5633](https://github.com/bentoml/BentoML/pull/5633), and the same tool-parser bug class reported in NAVER's [hcx-vllm-plugin](https://github.com/NAVER-Cloud-HyperCLOVA-X/hcx-vllm-plugin/issues/5).

---

## 🧰 Stack

![Python](https://img.shields.io/badge/Python-3E2723?style=flat-square&logo=python&logoColor=EFEBE9) ![PyTorch](https://img.shields.io/badge/PyTorch-4E342E?style=flat-square&logo=pytorch&logoColor=EFEBE9) ![Elasticsearch / Lucene](https://img.shields.io/badge/Elasticsearch_%2F_Lucene-5D4037?style=flat-square&logo=elasticsearch&logoColor=EFEBE9) ![vLLM](https://img.shields.io/badge/vLLM-6D4C41?style=flat-square) ![MLflow](https://img.shields.io/badge/MLflow-795548?style=flat-square&logo=mlflow&logoColor=EFEBE9)
