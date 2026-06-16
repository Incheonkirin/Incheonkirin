<img src="banner.svg" width="100%" alt="Mingi Jeong" />

**ML/LLM Engineer** — retrieval, LLM serving,
and open-source library internals

Previously 5.5 years on the search team at **42Maru** — Korean hybrid retrieval
(BM25 + dense, learning-to-rank, hard negatives), MRC (machine reading
comprehension), RAG, and open-source LLM fine-tuning.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

---

I work on retrieval and LLM systems where real data, much of it Korean, exposes
quiet failures deep in the stack: embedding losses, RoPE caches, continuous
batching. Tracing those to their source is where my open-source work comes
from.

## 🔧 Upstream contributions

The main testbed is
[**search_system**](https://github.com/Incheonkirin/search_system) — a Korean
insurance-clause retrieval lab with nori BM25 + BGE-M3 hybrid retrieval,
real-query failure cases, analyzer probes, and production-style traces. The
pattern is usually small, but it matters in production:

> **Data that is valid on one side of a representation boundary silently
> breaks the other** — NFD Hangul vs. the analyzer, stop strings vs.
> byte-fragment tokens, a literal `</tool_call>` vs. the tool-call parser,
> bf16 logits vs. a float32 loss. Korean hits these boundaries constantly;
> English-only test suites never do.

Recent fixes have landed upstream in sentence-transformers, transformers,
Elasticsearch, MLflow, and LlamaIndex: embedding-loss correctness, dynamic RoPE
cache resets, continuous-batching output snapshots, nori analyzer docs,
MLflow logging, and CJK text-splitter recursion.

**Retrieval training and embedding losses**

- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800)** — bf16/fp16 training crash across six learning-to-rank losses. ***(merged)***
- **[sentence-transformers #3817](https://github.com/huggingface/sentence-transformers/pull/3817)** — on multi-GPU `gather_across_devices`, gathered positives in `GISTEmbedLoss`/`CachedGISTEmbedLoss` were masked as false negatives, so the cross-entropy target collapsed to `-inf` and the training signal silently vanished on rank > 0. Surfaced with a Korean polarity probe; it also covered a regression the earlier in-batch-negative fix (#3453) had left in the GIST losses. ***(merged)***
- **[sentence-transformers #3816](https://github.com/huggingface/sentence-transformers/pull/3816)** — avoid materializing the full non-FAISS hard-negative mining similarity matrix. ***(merged)***
- **[sentence-transformers #3812](https://github.com/huggingface/sentence-transformers/pull/3812)** — MPS support for cached-loss `RandContext`. ***(merged)***
- **[sentence-transformers #3821](https://github.com/huggingface/sentence-transformers/pull/3821)** — hard-negative mining's relative-margin threshold was sign-dependent and inverted on negative positive-scores; made it sign-independent ([#3819](https://github.com/huggingface/sentence-transformers/issues/3819)). ***(merged)***

**LLM serving and model internals**

- **[huggingface/transformers #46530](https://github.com/huggingface/transformers/pull/46530)** — `StopStringCriteria` misses CJK stop strings on byte-level tokenizers ([#46519](https://github.com/huggingface/transformers/issues/46519)). ***(merged)***
- **[huggingface/transformers #46624](https://github.com/huggingface/transformers/pull/46624)** — dynamic RoPE never reset `inv_freq` on the `layer_type=None` path (it wrote `max_seq_len_cached` to a stray `None_…` attribute), so a long sequence followed by a short one kept the scaled frequencies. ***(merged)***
- **[huggingface/transformers #46670](https://github.com/huggingface/transformers/pull/46670)** — continuous batching's output conversion mutated the active request state and returned live aliases of the growing token/logprob buffers; made it a snapshot. ***(merged)***
- **[run-llama/llama_index #21900](https://github.com/run-llama/llama_index/pull/21900)** — `RecursionError` in text splitters when a single CJK/emoji token exceeds `chunk_size`. ***(merged)***
- **[huggingface/transformers #46643](https://github.com/huggingface/transformers/pull/46643)** — `TopHLogitsWarper` was built without `min_tokens_to_keep`, so with peaked logits and beam sampling top-h could keep a single token while the other warpers kept the beam-safe minimum. *(open)*
- **[vllm-project/vllm #45168](https://github.com/vllm-project/vllm/pull/45168)** — Hermes tool parser drops tool calls when a literal `</tool_call>` appears inside a JSON string argument ([#45167](https://github.com/vllm-project/vllm/issues/45167)). *(open)*
- **[NAVER hcx-vllm-plugin #5](https://github.com/NAVER-Cloud-HyperCLOVA-X/hcx-vllm-plugin/issues/5)** — reported the same parser-boundary bug class for literal `<|im_end|>` inside JSON string arguments. *(open issue)*
- **[vllm-project/vllm #45162](https://github.com/vllm-project/vllm/pull/45162)** — `collect_env.py` aborted with an `AssertionError` on non-Linux platforms. *(open)*

**Search analyzers and query normalization**

- **[elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)** — documented that nori's default `XPN` stop tag silently deletes meaning-bearing Korean prefixes, so 비급여 (non-covered) analyzes to 급여 (covered), from issue [#151094](https://github.com/elastic/elasticsearch/issues/151094). ***(merged)***
- **[apache/lucene #16242](https://github.com/apache/lucene/pull/16242)** — new `HangulCompositionCharFilter` for analysis-nori: NFD-form Hangul was silently unanalyzable as Korean ([#16241](https://github.com/apache/lucene/issues/16241)). *(open)*
- **[elastic/elasticsearch #151008](https://github.com/elastic/elasticsearch/pull/151008)** — wildcard queries: re-escape operator characters produced by the normalizer. *(open)*
- **[explosion/spaCy #13974](https://github.com/explosion/spaCy/pull/13974)** — Korean tokenizer collapsed whitespace runs, breaking `doc.text` round-trips and offsets. *(open)*

**Production tooling, tracing, and vector search**

- **[facebookresearch/faiss #5272](https://github.com/facebookresearch/faiss/issues/5272)** — diagnosed that `musllinux` wheels were dropped during the move to official PyPI wheels (`*-musllinux_*` remained in the `cibuildwheel` skip list) and outlined the restore path; upstream shipped the fix in `faiss-cpu 1.14.3` via [#5299](https://github.com/facebookresearch/faiss/pull/5299). *(resolved upstream)*
- **[mlflow #23957](https://github.com/mlflow/mlflow/pull/23957)** — restored dataset expectation/tag logging in `genai.evaluate(scorers=[])`. ***(merged)***
- **[mlflow #23818](https://github.com/mlflow/mlflow/pull/23818)** — OpenTelemetry retriever-span reassembly on ingest. *(open)*
- **[ragas #2759](https://github.com/vibrantlabsai/ragas/pull/2759)** — make VertexAI imports optional so `import ragas` does not fail without Vertex dependencies. *(open)*
- **BentoML [#5632](https://github.com/bentoml/BentoML/pull/5632) / [#5633](https://github.com/bentoml/BentoML/pull/5633)** — proxy-client configurability and monitoring-log span metadata. *(open)*

---

## 🏢 Enterprise NLP/QA at 42Maru (press)

Closed-source enterprise systems I worked on at 42Maru, with the research and
engineering teams: Korean search quality, semantic QA, retrieval behavior, and
OCR/NLP pipelines for real customer workflows.

- **AI ship-sales design-support system — Daewoo Shipbuilding (DSME)**: semantic QA over ~100K historical records for shipowners' pre-contract technical inquiries. [press](http://www.aitimes.kr/news/articleView.html?idxno=13427)
- **AML / trade-based transaction detection — Hana Bank**: OCR-NLP over cross-border remittance invoices. [press](https://www.venturesquare.net/844917)

---

## 📊 Public artifacts from 42Maru — NIA AI Hub

Government-published Korean NLP artifacts from projects I worked on at 42Maru:
large-scale QA/MRC corpora and LLM instruction data shaped by Korean search,
reasoning, table QA, and domain-document behavior. ~2.3M labeled QA pairs plus
a ~300M-token corpus across five public releases, all downloadable on AI Hub.

- **뉴스 기사 기계독해 (news-article MRC)** — 2021, 42Maru lead. Four answer regimes including *inference* and *unanswerable*, with evidence spans for the hard cases. [data](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=577)
- **국가기록물 초거대 AI 말뭉치 (national-archives LLM corpus)** — 2023, 42Maru lead. Instruction-tuning data for an LLM (Llama2), with four personas (formal/casual × written/spoken) and length-controlled answers. [data](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71788)
- **금융·법률 문서 기계독해 (finance/legal MRC)** — 2022. Text+table multimodal QA with *cell-coordinate table answers* and multiple-choice. [data](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610)
- **숫자연산 기계독해 (numeric-reasoning MRC)** — 2022. Numeric reasoning — arithmetic, ratio, date, and *multi-fact comparison*. [data](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71568)
- **표 정보 질의응답 (table-information QA)** — 2022. Complex-table QA with *unanswerable* cases and complexity tiers, informed by English table-QA benchmarks. [data](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71565)

---

## 🧭 Repo map

- **[search_system](https://github.com/Incheonkirin/search_system)** — Korean clause retrieval lab: nori BM25 + BGE-M3 hybrid retrieval, analyzer probes, real-query failures, and traces that feed the upstream work above.
- **Selected upstream workspaces** — [sentence-transformers](https://github.com/Incheonkirin/sentence-transformers), [transformers](https://github.com/Incheonkirin/transformers), [lucene](https://github.com/Incheonkirin/lucene), [elasticsearch](https://github.com/Incheonkirin/elasticsearch), [vllm](https://github.com/Incheonkirin/vllm): short-lived branches for submitted fixes and repros.
- **Domain probes** — [population-baseline-risk](https://github.com/Incheonkirin/population-baseline-risk) and [insurance-bias-probe](https://github.com/Incheonkirin/insurance-bias-probe): focused artifacts around insurance-domain behavior and model/system bias.

---

## 🧰 Stack

![Python](https://img.shields.io/badge/Python-3E2723?style=flat-square&logo=python&logoColor=EFEBE9) ![PyTorch](https://img.shields.io/badge/PyTorch-4E342E?style=flat-square&logo=pytorch&logoColor=EFEBE9) ![Transformers](https://img.shields.io/badge/Transformers-5D4037?style=flat-square&logo=huggingface&logoColor=EFEBE9) ![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6D4C41?style=flat-square) ![vLLM](https://img.shields.io/badge/vLLM-795548?style=flat-square) ![MLflow](https://img.shields.io/badge/MLflow-4E342E?style=flat-square&logo=mlflow&logoColor=EFEBE9) ![Elasticsearch / Lucene](https://img.shields.io/badge/Elasticsearch_%2F_Lucene-5D4037?style=flat-square&logo=elasticsearch&logoColor=EFEBE9) ![Hybrid Retrieval / RAG](https://img.shields.io/badge/Hybrid_Retrieval_%2F_RAG-3E2723?style=flat-square)
