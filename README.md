# Mingi Jeong (정민기)

**Search & Retrieval Applied Scientist spanning lexical retrieval, dense retrieval, and neural reranking.**

7 years across production search, retrieval ML, evaluation, and insurance ML. I
turn production relevance failures into reproducible benchmarks and upstream
fixes across Apache Lucene, Elasticsearch, sentence-transformers, Transformers,
MLflow, and LlamaIndex. **16 merged upstream contributions as of July 2026.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

## Representative Results

### Neural Ranking: Padding-Safe ListMLE

Found that sentence-transformers `ListMLELoss` and `PListMLELoss` allowed padded
positions to enter the Plackett-Luce normalizer. A query's loss and gradients
therefore depended on the longest other list in the batch; one valid document's
gradient even changed sign.

**Measured evidence:** sentence-transformers maintainer benchmarking reported
NanoBEIR R100 mean nDCG@10 of approximately `0.39 -> 0.529` for ListMLE and
`0.514 -> 0.525` for a matched PListMLE comparison. The older ListMLE baseline
caveat is documented in the technical note.

[Technical case study](case-studies/listmle-padding-correctness.md) ·
[Merged sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827)

### Query Correctness: The Exact Phrase That Returned Zero Results

Traced a Korean `match_phrase` false negative through nori's token graph into two
Elasticsearch query-construction defects. A one-document fixture containing the
exact source text returned `1` hit for `match`, `0` for `match_phrase(slop=0)`,
and `1` for `match_phrase(slop=1)`.

**Upstream result:** fixed misplaced and dropped position gaps across graph phrase
paths, with span-query unit coverage and a nori end-to-end regression.

[Technical case study](https://github.com/Incheonkirin/ko-evidence-bench/blob/main/case_studies/korean-retrieval-correctness/exact-phrase-zero-results.md) ·
[Merged Elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931)

### Language-Aware Engine: Hangul Composition In Lucene

Found that canonically equivalent NFD and NFC Hangul crossed the tokenizer
boundary differently. Added an opt-in, offset-correct
`HangulCompositionCharFilter` to Lucene's nori module, narrowly implementing the
Unicode modern Hangul composition path while leaving compatibility and archaic
jamo out of scope.

**Upstream result:** merged as a new Lucene analysis feature with NFC/NFD
term/POS equivalence, original-offset, randomized composition, and boundary tests.

[Representation-correctness case study](https://github.com/Incheonkirin/ko-evidence-bench/tree/main/case_studies/korean-retrieval-correctness) ·
[Merged Apache Lucene #16242](https://github.com/apache/lucene/pull/16242)

## Search Stack

```text
Korean text and Unicode
        |
        v
Lucene / nori analysis
        |
        v
Elasticsearch query construction
        |
        v
BM25 and dense candidate generation
        |
        v
Hybrid fusion
        |
        v
Cross-encoder reranking and listwise training
        |
        v
Relevance, polarity, sufficiency, and abstention evaluation
```

| Layer | Public evidence |
|---|---|
| Text representation | Lucene `#16242`: NFD/NFC Hangul composition with offset correction |
| Morphological meaning | Elasticsearch `#151157`: `비급여 -> 급여` XPN polarity collapse and configuration guidance |
| Query semantics | Elasticsearch `#152931`: token-graph position-gap preservation |
| Sparse, dense, hybrid | [`search_system`](https://github.com/Incheonkirin/search_system): nori BM25, BGE-M3, RRF, cross-encoder stack |
| Ranking objectives | sentence-transformers `#3827`, `#3817`, `#3821`, `#3800` |
| Retrieval evaluation | [`ko-evidence-bench`](https://github.com/Incheonkirin/ko-evidence-bench): 544-row retrieval study and 444-triple polarity stress test |

## Deep Case Studies

### 1. [When an Exact Phrase Returns Zero Results](https://github.com/Incheonkirin/ko-evidence-bench/blob/main/case_studies/korean-retrieval-correctness/exact-phrase-zero-results.md)

Elasticsearch token graphs and position holes, the compiled span-query tree, why
`slop=1` hid rather than fixed the defect, the two-path upstream change, and the
nori regression that made the failure executable.

### 2. [A Padding Bug That Changed ListMLE Training](case-studies/listmle-padding-correctness.md)

Plackett-Luce normalization, padding-dependent loss and gradient direction, why
post-normalizer masking was insufficient, the relational regression test, and
the maintainer's published MS MARCO/NanoBEIR runs.

### 3. [When a Korean Analyzer Reverses Meaning](https://github.com/Incheonkirin/ko-evidence-bench/blob/main/case_studies/korean-retrieval-correctness/analyzer-reverses-meaning.md)

`비급여 -> 급여`, why topical nDCG can miss polarity failure, analyzer remedies
and their regression risk, and a 444-triple comparison across lexical, dense, and
reranked systems.

Each study follows the same review path:

> Production-shaped failure -> minimal reproduction -> broken invariant -> why
> existing tests missed it -> fix -> measured evidence -> upstream validation.

## Full Upstream Record

The detailed record remains public so both depth and technical breadth are
independently inspectable. The three representative results above are the entry
points; this section is the evidence ledger.

### Korean Search And Ranking

- **[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827)**: excluded padding from the ListMLE/PListMLE Plackett-Luce normalizer and added a padding-invariance regression. Maintainer-reported NanoBEIR R100 mean nDCG@10: ListMLE approximately `0.39 -> 0.529`; matched PListMLE `0.514 -> 0.525`. **Merged.**
- **[Apache Lucene #16242](https://github.com/apache/lucene/pull/16242)**: added an opt-in `HangulCompositionCharFilter` to nori with offset correction and deliberately narrow Unicode scope. **Merged.**
- **[Elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157)**: documented that the default nori `XPN` stop tag can remove meaning-bearing prefixes (`비급여 -> 급여`, `부담보 -> 담보`) and documented configuration remedies. **Merged.**
- **[Elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931)**: preserved position holes in graph phrase queries, fixing exact-source false negatives with span and nori end-to-end tests. **Merged.**

### Embedding Losses And Model Internals

- **[sentence-transformers #3817](https://github.com/huggingface/sentence-transformers/pull/3817)**: fixed gathered positives being masked as false negatives in `GISTEmbedLoss` and `CachedGISTEmbedLoss`, which could collapse the cross-entropy target on distributed ranks. **Merged.**
- **[sentence-transformers #3821](https://github.com/huggingface/sentence-transformers/pull/3821)**: made relative-margin filtering sign-independent in mining and GIST losses so negative positive-pair scores no longer invert the intended ordering. **Merged.**
- **[sentence-transformers #3816](https://github.com/huggingface/sentence-transformers/pull/3816)**: avoided materializing the full non-FAISS query-corpus similarity matrix during hard-negative mining; a 10K x 100K local run reduced peak RSS from 4.56 GB to 1.09 GB while preserving outputs. **Merged.**
- **[sentence-transformers #3812](https://github.com/huggingface/sentence-transformers/pull/3812)**: added MPS support to cached-loss random-state handling. **Merged.**
- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800)**: fixed bf16/fp16 training crashes across six learning-to-rank losses. **Merged.**
- **[Transformers #46530](https://github.com/huggingface/transformers/pull/46530)**: fixed `StopStringCriteria` misses for CJK stop strings on byte-fragment tokenizers. **Merged.**
- **[Transformers #46670](https://github.com/huggingface/transformers/pull/46670)**: made continuous-batching generation output a snapshot rather than a mutable alias, preventing delivered chunks from changing and soft-reset requests from stopping short under cache pressure. **Merged.**
- **[Transformers #46624](https://github.com/huggingface/transformers/pull/46624)**: fixed dynamic RoPE failing to reset `inv_freq` when `layer_type=None`. **Merged.**
- **[Transformers #46763](https://github.com/huggingface/transformers/pull/46763)**: rounded the ue8m0 FP8 scale before quantization so dequantization matches the stored inverse. **Merged.**
- **[Transformers #46784](https://github.com/huggingface/transformers/pull/46784)**: fixed Moonshine training loss being shifted twice and trained against `labels[..., 1:]`. **Merged.**
- **[LlamaIndex #21900](https://github.com/run-llama/llama_index/pull/21900)**: fixed `RecursionError` in text splitters when one CJK or emoji token exceeds `chunk_size`. **Merged.**
- **[MLflow #23957](https://github.com/mlflow/mlflow/pull/23957)**: fixed `genai.evaluate()` dropping dataset expectations and tags when called with `scorers=[]`. **Merged.**

### Current And Additional Evidence

- **[PyTorch #187779](https://github.com/pytorch/pytorch/pull/187779)**: perform MPS fused RMSNorm weight multiplication in fp32 before the final cast to match CPU/CUDA. **Approved, pending merge.**
- **[vLLM #45168](https://github.com/vllm-project/vllm/pull/45168)**: reproduced a Hermes tool-parser boundary where a literal `</tool_call>` inside a JSON string argument drops the tool call. **Closed without merge.**
- Korean tokenizer offsets: [spaCy #13974](https://github.com/explosion/spaCy/pull/13974).
- FAISS musllinux wheels: [FAISS #5272](https://github.com/facebookresearch/faiss/issues/5272).
- HyperCLOVA-X parser boundary: [hcx-vllm-plugin #5](https://github.com/NAVER-Cloud-HyperCLOVA-X/hcx-vllm-plugin/issues/5).
- [Full GitHub PR search](https://github.com/search?q=author%3AIncheonkirin+type%3Apr&type=pullrequests).

## Production And Data Impact

### MetLife

Production insurance ML on Databricks: churn, fraud-risk,
distribution-channel performance, and cross-sell models tied to customer and
channel decisions. Work across data and features, training, deployment,
retraining, monitoring, and online A/B-tested rollouts.

### 42Maru

5.5 years on the search team building production search, retrieval, and QA
systems: BM25 relevance, contrastive retrieval, RAG QA, MRC, large-scale
indexing, and crawler pipelines.

Publicly documented enterprise projects completed with the research and
engineering teams:

- **Daewoo Shipbuilding (DSME):** semantic QA over approximately 100K historical records for shipowners' pre-contract technical inquiries. [Press](http://www.aitimes.kr/news/articleView.html?idxno=13427)
- **Hana Bank:** OCR-NLP for trade-based AML workflows over cross-border remittance invoices. [Press](https://www.venturesquare.net/844917)

### Public Data And Evaluation Assets

Led task design, annotation guidelines, quality review, and baseline evaluation
for five government-published Korean NLP releases. The work translated failure
modes observed during internal MRC/LLM R&D into public tasks spanning news MRC,
national-archives LLM instruction data, finance/legal MRC, numerical reasoning,
and table QA: approximately **2.3M labeled QA pairs and a 304M-token corpus**.

**Verified downstream reuse:**

- **[K-FinHallu](https://arxiv.org/abs/2605.29523)** by KAIST AI and KakaoBank uses the finance/legal MRC dataset as the source corpus for a multi-turn Korean financial RAG hallucination benchmark.
- **[FINALE](https://aclanthology.org/2024.finnlp-2.9/)** at ACL FinNLP 2024 reuses the finance/legal, numerical-reasoning, and table-QA datasets to construct 76,433 filtered rationale examples. The paper reports an 8.7% average improvement across nine subtasks over its instruction-tuned baseline.

This is downstream dataset impact, not a claim that I authored those papers. My
role was the upstream task, guideline, QA, and baseline design.

[News MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=577) ·
[National-archives LLM corpus](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71788) ·
[Finance/legal MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610) ·
[Numerical-reasoning MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71568) ·
[Table QA](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71565)

## Technical Scope

**ATS keywords:** Information Retrieval, Search Relevance, Dense/Sparse
Retrieval, Learning to Rank, Cross-Encoder Reranking, Hard-Negative Mining,
Hybrid Search, Lucene, Elasticsearch, PyTorch, sentence-transformers, BM25,
ANN, nDCG, Recall.

- **Retrieval and ranking:** LambdaMART, two-tower retrieval, ColBERT/MaxSim, BM25, BGE-M3, hybrid fusion, RRF, cross-encoder reranking.
- **Serving:** quantized and distilled rerankers, p99 cascade budgets, Docker, Kubernetes, Transformers continuous batching, vLLM.
- **LLM:** RAG with citation and abstention evaluation, SFT/DPO/LoRA, query rewriting, and relevance judging.
- **Data and MLOps:** Spark, Databricks, Elasticsearch/OpenSearch, FAISS, deployment, retraining, monitoring, and online experiments.

## Repository Map

- **[search_system](https://github.com/Incheonkirin/search_system):** runnable Korean insurance-clause retrieval stack with nori BM25, BGE-M3 embeddings, hybrid fusion, cross-encoder reranking, Elasticsearch, and FastAPI.
- **[ko-evidence-bench](https://github.com/Incheonkirin/ko-evidence-bench):** privacy-safe Korean retrieval evaluation with source routing, abstention, surface robustness, synthetic probes, aggregate studies, and the representation-correctness case study.
- **[fraud-dataset-validity](https://github.com/Incheonkirin/fraud-dataset-validity):** reproducible shortcut and validity audit of public synthetic fraud datasets, with external anchors and model-sensitivity checks.
- **[insurance-bias-probe](https://github.com/Incheonkirin/insurance-bias-probe):** controlled demographic-consistency probes for Korean insurance answers.

![Python](https://img.shields.io/badge/Python-3E2723?style=flat-square&logo=python&logoColor=EFEBE9)
![PyTorch](https://img.shields.io/badge/PyTorch-4E342E?style=flat-square&logo=pytorch&logoColor=EFEBE9)
![Transformers](https://img.shields.io/badge/Transformers-5D4037?style=flat-square&logo=huggingface&logoColor=EFEBE9)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6D4C41?style=flat-square)
![Elasticsearch / Lucene](https://img.shields.io/badge/Elasticsearch_%2F_Lucene-5D4037?style=flat-square&logo=elasticsearch&logoColor=EFEBE9)
![Hybrid Retrieval / RAG](https://img.shields.io/badge/Hybrid_Retrieval_%2F_RAG-3E2723?style=flat-square)
