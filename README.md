# Mingi Jeong (정민기)

**Korean search & retrieval ML engineer** — analyzer correctness, ranking losses, LLM serving · 7y · Python · PyTorch

Korean search, fixed where it breaks — upstream: a Hangul NFD-composition char filter merged into Apache Lucene's nori analyzer ([apache/lucene #16242](https://github.com/apache/lucene/pull/16242)); the meaning-inverting nori XPN default (비급여 *non-covered* → 급여 *covered*) now warned in the official Elasticsearch docs ([elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)); a ListMLE/PListMLE padding fix in sentence-transformers ([sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827); maintainer-measured NanoBEIR nDCG@10 0.39 → 0.53). Previously 5.5y on the search team at 42Maru; now at MetLife on production ML.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

I also turn failure modes observed in production MRC/LLM R&D into public data and evaluation assets: I led task design, annotation guidelines, quality review, and baseline evaluation for five NIA AI Hub releases later reused in Korean financial RAG and instruction-tuning research.

---

> **Data that is valid on one side of a representation boundary silently breaks the other** — NFD Hangul vs. the analyzer, stop strings vs. byte-fragment tokens, bf16 logits vs. a float32 loss. Korean hits these boundaries constantly; English-only test suites never do.

## Search depth

**Korean evidence-retrieval lab** — built a private clause-retrieval system over 36,983 evidence passages, using nori BM25, BGE-M3 hybrid retrieval, cross-text reranking, analyzer probes, and real-query failure analysis. On a 544-row silver retrieval study, cross-text reranking lifted `clause@20` from 56.4% to 64.9% (+8.5%p paired bootstrap CI: +5.9 to +11.2). A separate public companion, [**ko-evidence-bench**](https://github.com/Incheonkirin/ko-evidence-bench), publishes the privacy-safe scorecard, synthetic probes, aggregate study, and qid-only run contract.

- **XPN polarity (비급여 → 급여)** — nori's default analyzer drops the meaning-bearing prefix, so 비급여 (non-covered) indexes as 급여 (covered) and opposite-meaning clauses become indistinguishable. Reproduced and pinned; documented upstream ([elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)).
- **NFD Hangul** — NFD-decomposed Hangul reaches `KoreanTokenizer` as conjoining jamo and is silently unanalyzable as Korean. Added an opt-in `HangulCompositionCharFilter` that composes it to precomposed syllables with offset correction, merged into Apache Lucene's nori analyzer ([apache/lucene #16242](https://github.com/apache/lucene/pull/16242)).

The lab is also where I compare offline variants — analyzer choices (형태소 분석기), fusion weights, reranker on/off — with nDCG / Recall and paired bootstrap. It includes a 444-triple polarity stress study that catches opposite-meaning evidence being preferred even when ordinary retrieval looks plausible. The released reports scope these measurements as aggregate silver diagnostics; human answer-quality and source-routing claims are evaluated separately.

## Across the stack

Built or prototyped in `search_system` / production:

- **Ranking** — LambdaMART / two-tower, late-interaction (ColBERT / MaxSim), hybrid fusion vs. fixed RRF.
- **Serving** — quantized + distilled reranker, p99 cascade budget, Docker / Kubernetes; FP8 dequant ([huggingface/transformers #46763](https://github.com/huggingface/transformers/pull/46763)); Transformers continuous-batching internals; vLLM Hermes tool-parser.
- **LLM** — RAG (MLX / vLLM) with citation / abstention eval, post-training (SFT / DPO / LoRA), LLM for search-quality (query rewriting, relevance judging).
- **Data** — Spark / Databricks embedding, near-real-time index refresh, Elasticsearch / OpenSearch + FAISS (C++) tuning.
- **Recommendation** — cross-sell with online A/B tests (MetLife).

## Upstream contributions

**16 merged upstream PRs as of July 2026** across Apache Lucene, Elasticsearch, sentence-transformers, Transformers, MLflow, and LlamaIndex. The detailed record is kept below so both impact and technical breadth are independently inspectable.

**Korean search & ranking — primary**

- **[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827)** — ListMLE/PListMLE listwise reranker losses mixed padding positions into the Plackett-Luce normalizer; excluded the padding. The maintainer measured NanoBEIR nDCG@10 0.39 → 0.53 (ListMLE). ***(merged)***
- **[apache/lucene #16242](https://github.com/apache/lucene/pull/16242)** — added an opt-in `HangulCompositionCharFilter` to Lucene's nori analyzer: composes NFD-form modern Hangul (conjoining L/V/T jamo) into precomposed syllables before `KoreanTokenizer`, preserving offset correction; intentionally narrow, leaving compatibility/archaic jamo and precomposed text untouched. Reviewed and merged by Robert Muir, who confirmed the constants/formula match Unicode Hangul syllable composition. ***(merged)***
- **[elastic/elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)** — found that nori's default analyzer silently strips Korean negation prefixes (비급여 *non-covered* → 급여 *covered*, 부담보 → 담보), so opposite-meaning clauses index identically; traced to the default `XPN` stop tag and now warned in the official Elasticsearch nori docs. ***(merged)***

**Embedding losses & model internals**

- **[sentence-transformers #3817](https://github.com/huggingface/sentence-transformers/pull/3817)** — multi-GPU `gather_across_devices`: gathered positives in `GISTEmbedLoss`/`CachedGISTEmbedLoss` were masked as false negatives, so the cross-entropy target collapsed to `-inf` and the training signal silently vanished on rank > 0. Surfaced with a Korean polarity probe. ***(merged)***
- **[sentence-transformers #3821](https://github.com/huggingface/sentence-transformers/pull/3821)** — made relative-margin filtering sign-independent in hard-negative mining and GIST losses, so negative positive-pair scores no longer invert the intended score ordering. ***(merged)***
- **[sentence-transformers #3816](https://github.com/huggingface/sentence-transformers/pull/3816)** — avoided materializing the full non-FAISS query-corpus similarity matrix during hard-negative mining; a 10K x 100K local run reduced peak RSS from 4.56 GB to 1.09 GB while preserving outputs. ***(merged)***
- **[sentence-transformers #3812](https://github.com/huggingface/sentence-transformers/pull/3812)** — added MPS support to cached-loss random-state handling. ***(merged)***
- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800)** — bf16/fp16 training crash across six learning-to-rank losses. ***(merged)***
- **[huggingface/transformers #46530](https://github.com/huggingface/transformers/pull/46530)** — `StopStringCriteria` misses CJK stop strings on byte-level tokenizers ([#46519](https://github.com/huggingface/transformers/issues/46519)). ***(merged)***
- **[huggingface/transformers #46670](https://github.com/huggingface/transformers/pull/46670)** — `RequestState.to_generation_output()` is the per-request output converter, and streaming continuous batching calls it once per generated token. On `main` it returned the request's own `generated_tokens`/`logprobs`/`timestamps` lists by reference and, on the soft-reset path, rewrote `self.generated_tokens`/`self.initial_tokens` while building the output, so an already-delivered streaming chunk changed as later tokens arrived and a soft-reset request's bookkeeping drifted until it stopped short of `max_new_tokens`. Reproduced with CPU regression tests and a CUDA continuous-batching run under forced cache pressure: unpatched, several of twelve greedy requests stopped at 21-23 of 30 tokens; patched, all reached 30 and delivered chunks stayed fixed. Made the conversion a snapshot. ***(merged)***
- **[huggingface/transformers #46624](https://github.com/huggingface/transformers/pull/46624) / [#46763](https://github.com/huggingface/transformers/pull/46763)** — model/serving numeric internals: dynamic RoPE never reset `inv_freq` on the `layer_type=None` path; round the ue8m0 FP8 scale before quantizing so dequant matches the stored inverse. ***(merged)***
- **[huggingface/transformers #46784](https://github.com/huggingface/transformers/pull/46784)** — Moonshine training loss was shifted twice, so it trained against `labels[..., 1:]`; compute loss against the labels themselves. ***(merged)***
- **[run-llama/llama_index #21900](https://github.com/run-llama/llama_index/pull/21900)** — `RecursionError` in text splitters when a single CJK/emoji token exceeds `chunk_size`. ***(merged)***
- **[mlflow/mlflow #23957](https://github.com/mlflow/mlflow/pull/23957)** — fixed `genai.evaluate()` dropping dataset expectations and tags when invoked with `scorers=[]`. ***(merged)***

**Current / recent**

- **[pytorch/pytorch #187779](https://github.com/pytorch/pytorch/pull/187779)** — MPS fused RMSNorm multiplied weights in fp16/bf16; do the weight multiply in fp32 before the final cast to match CPU/CUDA. *(approved, pending merge)*
- **[elastic/elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931)** — graph phrase queries lost the position holes left by token-removing filters (nori decompound + part-of-speech, or synonym_graph + stop), so `match_phrase` with a document's exact source text could return nothing at slop 0; fixed the misplaced `SpanGap` and the dropped gaps between graph segments, with an `analysis-nori` end-to-end test. ***(merged)***
- **[vllm-project/vllm #45168](https://github.com/vllm-project/vllm/pull/45168)** — reproduced a Hermes tool-parser boundary where a literal `</tool_call>` inside a JSON string argument drops the tool call ([#45167](https://github.com/vllm-project/vllm/issues/45167)). *(closed without merge)*

**Also (Korean & search infra)** — Korean tokenizer offsets ([explosion/spaCy #13974](https://github.com/explosion/spaCy/pull/13974)), FAISS musllinux wheels restored ([facebookresearch/faiss #5272](https://github.com/facebookresearch/faiss/issues/5272)). Reported issue: [NAVER hcx-vllm-plugin #5](https://github.com/NAVER-Cloud-HyperCLOVA-X/hcx-vllm-plugin/issues/5) (`<|im_end|>` parser boundary). [Full PR list →](https://github.com/search?q=author%3AIncheonkirin+type%3Apr&type=pullrequests)

---

## Production & earlier

**MetLife** (current) — production insurance ML on Databricks: churn, fraud-risk, distribution-channel performance, and cross-sell models tied to customer and channel decisions. Build and operate the lifecycle from data/features and training to deployment, retraining, monitoring, and online A/B-tested rollouts.

**42Maru** — search team, 5.5y. Production search, retrieval, and QA systems: BM25 relevance, contrastive retrieval, RAG QA, MRC, large-scale indexing, and crawler pipelines.

### Enterprise NLP/QA at 42Maru (press)

Closed-source enterprise systems I worked on at 42Maru, with the research and engineering teams: Korean search quality, semantic QA, retrieval behavior, and OCR/NLP pipelines for real customer workflows.

- **AI ship-sales design-support system — Daewoo Shipbuilding (DSME)**: semantic QA over ~100K historical records for shipowners' pre-contract technical inquiries. [press](http://www.aitimes.kr/news/articleView.html?idxno=13427)
- **AML / trade-based transaction detection — Hana Bank**: OCR-NLP over cross-border remittance invoices. [press](https://www.venturesquare.net/844917)

### Public data and evaluation assets from 42Maru — NIA AI Hub

Led task design, annotation guidelines, quality review, and baseline evaluation for five government-published Korean NLP releases. The work translated failure modes observed during internal MRC/LLM R&D into public tasks across news MRC, national-archives LLM instruction data, finance/legal MRC, numerical reasoning MRC, and table QA: approximately 2.3M labeled QA pairs plus a 304M-token corpus.

**Verified downstream reuse**

- **[K-FinHallu](https://arxiv.org/abs/2605.29523)** by KAIST AI and KakaoBank uses the finance/legal MRC dataset as the source corpus for a multi-turn Korean financial RAG hallucination benchmark.
- **[FINALE](https://aclanthology.org/2024.finnlp-2.9/)** at ACL FinNLP 2024 reuses the finance/legal, numerical-reasoning, and table-QA datasets to construct 76,433 filtered rationale examples. The paper reports an 8.7% average improvement across nine subtasks over its instruction-tuned baseline.

This is downstream dataset impact, not a claim that I authored those papers: public evidence verifies reuse, while my role was the upstream task, guideline, QA, and baseline design described above.

[news MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=577) ·
[national-archives LLM corpus](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71788) ·
[finance/legal MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610) ·
[numeric-reasoning MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71568) ·
[table QA](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71565)

---

## Repo map

- **[search_system](https://github.com/Incheonkirin/search_system)** — runnable Korean insurance-clause retrieval stack: nori BM25, BGE-M3 embeddings, hybrid fusion, cross-encoder reranking, Elasticsearch, and a FastAPI service.
- **[ko-evidence-bench](https://github.com/Incheonkirin/ko-evidence-bench)** — privacy-safe evaluation companion for Korean evidence retrieval: source routing, abstention, surface robustness, synthetic probes, and aggregate study artifacts.
- **[fraud-dataset-validity](https://github.com/Incheonkirin/fraud-dataset-validity)** — reproducible shortcut and validity audit of public synthetic fraud datasets, with external anchors and model-sensitivity checks. It is a dataset-audit case study, not a claim of real-world insurance-fraud performance.
- **[insurance-bias-probe](https://github.com/Incheonkirin/insurance-bias-probe)** — controlled demographic-consistency probes for Korean insurance answers.
- **Upstream evidence** — the merged fixes are inspected in the [full PR record](https://github.com/search?q=author%3AIncheonkirin+type%3Apr&type=pullrequests); the local upstream mirrors remain private working copies rather than portfolio projects.

---

## Stack

![Python](https://img.shields.io/badge/Python-3E2723?style=flat-square&logo=python&logoColor=EFEBE9) ![PyTorch](https://img.shields.io/badge/PyTorch-4E342E?style=flat-square&logo=pytorch&logoColor=EFEBE9) ![Transformers](https://img.shields.io/badge/Transformers-5D4037?style=flat-square&logo=huggingface&logoColor=EFEBE9) ![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6D4C41?style=flat-square) ![vLLM](https://img.shields.io/badge/vLLM-795548?style=flat-square) ![MLflow](https://img.shields.io/badge/MLflow-4E342E?style=flat-square&logo=mlflow&logoColor=EFEBE9) ![Elasticsearch / Lucene](https://img.shields.io/badge/Elasticsearch_%2F_Lucene-5D4037?style=flat-square&logo=elasticsearch&logoColor=EFEBE9) ![Hybrid Retrieval / RAG](https://img.shields.io/badge/Hybrid_Retrieval_%2F_RAG-3E2723?style=flat-square)
