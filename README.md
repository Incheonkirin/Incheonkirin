# Mingi Jeong (정민기)

**Korean search & retrieval ML engineer** — analyzer correctness, ranking losses, LLM serving · 7y · Python · PyTorch

Korean search, fixed where it breaks — upstream: a Hangul NFD-composition char filter merged into Apache Lucene's nori analyzer ([apache/lucene #16242](https://github.com/apache/lucene/pull/16242)); the meaning-inverting nori XPN default (비급여 *non-covered* → 급여 *covered*) now warned in the official Elasticsearch docs ([elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)); a ListMLE/PListMLE padding fix in sentence-transformers ([sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827); maintainer-measured NanoBEIR nDCG@10 0.39 → 0.53). Previously 5.5y on the search team at 42Maru; now at MetLife on production ML.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

---

> **Data that is valid on one side of a representation boundary silently breaks the other** — NFD Hangul vs. the analyzer, stop strings vs. byte-fragment tokens, bf16 logits vs. a float32 loss. Korean hits these boundaries constantly; English-only test suites never do.

## Search depth

[**search_system**](https://github.com/Incheonkirin/search_system) — a Korean insurance-clause (약관) retrieval lab over 36,983 clause passages with 700 hand-graded queries: nori BM25 + BGE-M3 hybrid retrieval, analyzer probes, real-query failures. For each Korean failure I took upstream, the lab holds a before/after fixture tied to the fix and a regression test:

- **XPN polarity (비급여 → 급여)** — nori's default analyzer drops the meaning-bearing prefix, so 비급여 (non-covered) indexes as 급여 (covered) and opposite-meaning clauses become indistinguishable. Reproduced and pinned; documented upstream ([elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)).
- **NFD Hangul** — NFD-decomposed Hangul reaches `KoreanTokenizer` as conjoining jamo and is silently unanalyzable as Korean. Added an opt-in `HangulCompositionCharFilter` that composes it to precomposed syllables with offset correction, merged into Apache Lucene's nori analyzer ([apache/lucene #16242](https://github.com/apache/lucene/pull/16242)).

The lab is also where I compare offline variants — analyzer choices (형태소 분석기), fusion weights, reranker on/off — on the qrels benchmark, decided by nDCG / Recall. The scorecard harness (nori-BM25 → BGE-M3 → RRF → cross-encoder, human-graded qrels, paired bootstrap) is implemented for runs over the human-graded qrels.

## Across the stack

Built or prototyped in `search_system` / production:

- **Ranking** — LambdaMART / two-tower, late-interaction (ColBERT / MaxSim), hybrid fusion vs. fixed RRF.
- **Serving** — quantized + distilled reranker, p99 cascade budget, Docker / Kubernetes; FP8 dequant ([huggingface/transformers #46763](https://github.com/huggingface/transformers/pull/46763)); Transformers continuous-batching internals; vLLM Hermes tool-parser.
- **LLM** — RAG (MLX / vLLM) with citation / abstention eval, post-training (SFT / DPO / LoRA), LLM for search-quality (query rewriting, relevance judging).
- **Data** — Spark / Databricks embedding, near-real-time index refresh, Elasticsearch / OpenSearch + FAISS (C++) tuning.
- **Recommendation** — cross-sell with online A/B tests (MetLife).

## Upstream contributions

**Korean search & ranking — primary**

- **[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827)** — ListMLE/PListMLE listwise reranker losses mixed padding positions into the Plackett-Luce normalizer; excluded the padding. The maintainer measured NanoBEIR nDCG@10 0.39 → 0.53 (ListMLE). ***(merged)***
- **[apache/lucene #16242](https://github.com/apache/lucene/pull/16242)** — added an opt-in `HangulCompositionCharFilter` to Lucene's nori analyzer: composes NFD-form modern Hangul (conjoining L/V/T jamo) into precomposed syllables before `KoreanTokenizer`, preserving offset correction; intentionally narrow, leaving compatibility/archaic jamo and precomposed text untouched. Reviewed and merged by Robert Muir, who confirmed the constants/formula match Unicode Hangul syllable composition. ***(merged)***
- **[elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)** — found that nori's default analyzer silently strips Korean negation prefixes (비급여 *non-covered* → 급여 *covered*, 부담보 → 담보), so opposite-meaning clauses index identically; traced to the default `XPN` stop tag and now warned in the official Elasticsearch nori docs. ***(merged)***

**Embedding losses & model internals**

- **[sentence-transformers #3817](https://github.com/huggingface/sentence-transformers/pull/3817)** — multi-GPU `gather_across_devices`: gathered positives in `GISTEmbedLoss`/`CachedGISTEmbedLoss` were masked as false negatives, so the cross-entropy target collapsed to `-inf` and the training signal silently vanished on rank > 0. Surfaced with a Korean polarity probe. ***(merged)***
- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800)** — bf16/fp16 training crash across six learning-to-rank losses. ***(merged)***
- **[huggingface/transformers #46530](https://github.com/huggingface/transformers/pull/46530)** — `StopStringCriteria` misses CJK stop strings on byte-level tokenizers ([#46519](https://github.com/huggingface/transformers/issues/46519)). ***(merged)***
- **[huggingface/transformers #46670](https://github.com/huggingface/transformers/pull/46670)** — continuous batching returned live aliases of the growing token/logprob buffers; made it a snapshot. ***(merged)***
- **[huggingface/transformers #46624](https://github.com/huggingface/transformers/pull/46624) / [#46763](https://github.com/huggingface/transformers/pull/46763)** — model/serving numeric internals: dynamic RoPE never reset `inv_freq` on the `layer_type=None` path; round the ue8m0 FP8 scale before quantizing so dequant matches the stored inverse. ***(merged)***
- **[huggingface/transformers #46784](https://github.com/huggingface/transformers/pull/46784)** — Moonshine training loss was shifted twice, so it trained against `labels[..., 1:]`; compute loss against the labels themselves. ***(merged)***
- **[run-llama/llama_index #21900](https://github.com/run-llama/llama_index/pull/21900)** — `RecursionError` in text splitters when a single CJK/emoji token exceeds `chunk_size`. ***(merged)***

**Open / active**

- **[pytorch/pytorch #187779](https://github.com/pytorch/pytorch/pull/187779)** — MPS fused RMSNorm multiplied weights in fp16/bf16; do the weight multiply in fp32 before the final cast to match CPU/CUDA. *(approved, pending merge)*
- **[vllm-project/vllm #45168](https://github.com/vllm-project/vllm/pull/45168)** — Hermes tool parser drops tool calls when a literal `</tool_call>` appears inside a JSON string argument ([#45167](https://github.com/vllm-project/vllm/issues/45167)). *(open)*

**Also (Korean & search infra)** — Korean tokenizer offsets ([explosion/spaCy #13974](https://github.com/explosion/spaCy/pull/13974)), FAISS musllinux wheels restored ([facebookresearch/faiss #5272](https://github.com/facebookresearch/faiss/issues/5272)). Reported issue: [NAVER hcx-vllm-plugin #5](https://github.com/NAVER-Cloud-HyperCLOVA-X/hcx-vllm-plugin/issues/5) (`<|im_end|>` parser boundary). [Full PR list →](https://github.com/search?q=author%3AIncheonkirin+type%3Apr&type=pullrequests)

---

## Production & earlier

**MetLife** (current) — churn, fraud, agent activation, cross-sell on Azure ML / Databricks. Deploy, retrain, monitor; online A/B tests for model rollouts.

**42Maru** — search team, 5.5y. BM25 IR, contrastive retrieval, RAG QA, MRC, SFT / DPO / LoRA, large-scale indexing and crawlers.

### Enterprise NLP/QA at 42Maru (press)

Closed-source enterprise systems I worked on at 42Maru, with the research and engineering teams: Korean search quality, semantic QA, retrieval behavior, and OCR/NLP pipelines for real customer workflows.

- **AI ship-sales design-support system — Daewoo Shipbuilding (DSME)**: semantic QA over ~100K historical records for shipowners' pre-contract technical inquiries. [press](http://www.aitimes.kr/news/articleView.html?idxno=13427)
- **AML / trade-based transaction detection — Hana Bank**: OCR-NLP over cross-border remittance invoices. [press](https://www.venturesquare.net/844917)

### Public artifacts from 42Maru — NIA AI Hub

Government-published Korean NLP artifacts from 42Maru projects I worked on: five AI Hub releases across news MRC, national-archives LLM instruction data, finance/legal MRC, numeric reasoning MRC, and table QA. ~2.3M labeled QA pairs plus a ~300M-token corpus.

[news MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=577) ·
[national-archives LLM corpus](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71788) ·
[finance/legal MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610) ·
[numeric-reasoning MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71568) ·
[table QA](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71565)

---

## Repo map

- **[search_system](https://github.com/Incheonkirin/search_system)** — Korean clause retrieval lab: nori BM25 + BGE-M3 hybrid retrieval, analyzer probes, real-query failures, and traces that feed the upstream work above.
- **Selected upstream workspaces** — [sentence-transformers](https://github.com/Incheonkirin/sentence-transformers), [transformers](https://github.com/Incheonkirin/transformers), [lucene](https://github.com/Incheonkirin/lucene), [elasticsearch](https://github.com/Incheonkirin/elasticsearch), [vllm](https://github.com/Incheonkirin/vllm): short-lived branches for submitted fixes and repros.
- **Domain probes** — [insurance-bias-probe](https://github.com/Incheonkirin/insurance-bias-probe): focused artifacts around insurance-domain behavior and model/system bias.

---

## Stack

![Python](https://img.shields.io/badge/Python-3E2723?style=flat-square&logo=python&logoColor=EFEBE9) ![PyTorch](https://img.shields.io/badge/PyTorch-4E342E?style=flat-square&logo=pytorch&logoColor=EFEBE9) ![Transformers](https://img.shields.io/badge/Transformers-5D4037?style=flat-square&logo=huggingface&logoColor=EFEBE9) ![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6D4C41?style=flat-square) ![vLLM](https://img.shields.io/badge/vLLM-795548?style=flat-square) ![MLflow](https://img.shields.io/badge/MLflow-4E342E?style=flat-square&logo=mlflow&logoColor=EFEBE9) ![Elasticsearch / Lucene](https://img.shields.io/badge/Elasticsearch_%2F_Lucene-5D4037?style=flat-square&logo=elasticsearch&logoColor=EFEBE9) ![Hybrid Retrieval / RAG](https://img.shields.io/badge/Hybrid_Retrieval_%2F_RAG-3E2723?style=flat-square)
