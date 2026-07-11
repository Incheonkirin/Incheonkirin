# Mingi Jeong (정민기)

**Search & Retrieval Applied Scientist spanning lexical retrieval, dense retrieval, hybrid search, and neural reranking.**

`7 years` · `16 merged upstream PRs` · `42Maru Search 5.5y` · `MetLife Production ML`

I turn production relevance failures into reproducible research, evaluation
protocols, and upstream fixes across Lucene, Elasticsearch,
sentence-transformers, and Transformers.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

## Selected Evidence

**Neural ranking correctness.** Padding changed ListMLE/PListMLE loss and
valid-document gradients. Maintainer-reported NanoBEIR R100 mean nDCG@10:
ListMLE `~0.39 -> 0.529`; matched PListMLE `0.514 -> 0.525`.<br>
[Research study](case-studies/listmle-padding-correctness.md) ·
[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827)

**Query semantics.** An exact Korean source phrase produced `match=1`,
`match_phrase(slop=0)=0`, and `match_phrase(slop=1)=1`. Fixed two graph
position-gap paths.<br>
[Research study](https://github.com/Incheonkirin/ko-evidence-bench/blob/main/case_studies/korean-retrieval-correctness/exact-phrase-zero-results.md) ·
[Elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931)

**Text representation.** Added an offset-correct Hangul composition filter so
canonically equivalent NFD/NFC modern Hangul can receive equivalent nori
analysis.<br>
[Research study](https://github.com/Incheonkirin/ko-evidence-bench/tree/main/case_studies/korean-retrieval-correctness) ·
[Apache Lucene #16242](https://github.com/apache/lucene/pull/16242)

## End-To-End Search Relevance

```text
Korean text -> Lucene/nori -> Elasticsearch query semantics
            -> BM25 + dense retrieval -> hybrid fusion
            -> cross-encoder reranking -> relevance evaluation
```

**[Ranking-objective correctness](case-studies/listmle-padding-correctness.md):**
Can irrelevant batch padding alter a query's optimization signal and downstream
reranker quality?

**[Representation correctness](https://github.com/Incheonkirin/ko-evidence-bench/tree/main/case_studies/korean-retrieval-correctness):**
Which Unicode, morphology, and token-graph boundaries erase distinctions before
ranking?

**[Polarity-aware retrieval](https://github.com/Incheonkirin/ko-evidence-bench/blob/main/case_studies/korean-retrieval-correctness/analyzer-reverses-meaning.md):**
Do lexical, dense, and reranked systems prefer evidence supporting the opposite
proposition?

Each study follows the same evidence chain:

> Production-shaped failure -> minimal reproduction -> broken invariant ->
> controlled experiment -> measured impact -> upstream validation.

## Production And Public Impact

**Production search.** 5.5 years on 42Maru's search team across BM25 relevance,
contrastive retrieval, RAG/MRC QA, large-scale indexing, and crawler systems.
Enterprise evidence: [DSME semantic QA](http://www.aitimes.kr/news/articleView.html?idxno=13427),
[Hana Bank OCR-NLP/AML](https://www.venturesquare.net/844917).

**Production insurance ML.** Current MetLife work across features, training,
deployment, retraining, monitoring, and online-tested churn, fraud-risk, channel,
and cross-sell models.

**Retrieval research lab.** 36,983 evidence passages; 544-row silver retrieval
study; best checked-in `clause@20` `64.9%`; 444-triple polarity stress across
lexical, dense, and reranked systems.
[Reproducible public companion](https://github.com/Incheonkirin/ko-evidence-bench).

**Public data assets.** Led task design, annotation guidelines, QA, and baseline
evaluation for five NIA AI Hub releases totaling approximately **2.3M QA pairs
and a 304M-token corpus**. Downstream reuse:
[K-FinHallu](https://arxiv.org/abs/2605.29523),
[FINALE](https://aclanthology.org/2024.finnlp-2.9/).

**Open-source depth.** 16 merged external contributions across Lucene,
Elasticsearch, sentence-transformers, Transformers, MLflow, and LlamaIndex.
[Full PR search](https://github.com/search?q=author%3AIncheonkirin+type%3Apr&type=pullrequests).

## Technical Scope

`Information Retrieval` · `Search Relevance` · `BM25` · `Dense Retrieval` ·
`Hybrid Search` · `Learning to Rank` · `Cross-Encoder Reranking` ·
`Hard-Negative Mining` · `Lucene` · `Elasticsearch` · `PyTorch` ·
`sentence-transformers` · `ANN` · `nDCG` · `Recall` · `RAG Evaluation`

<details>
<summary><strong>Full upstream record: 16 merged contributions</strong></summary>

### Korean Search And Ranking

- **[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827):** excluded padding from the ListMLE/PListMLE Plackett-Luce normalizer and added a padding-invariance regression. Maintainer-reported NanoBEIR R100 mean nDCG@10: ListMLE approximately `0.39 -> 0.529`; matched PListMLE `0.514 -> 0.525`. **Merged.**
- **[Apache Lucene #16242](https://github.com/apache/lucene/pull/16242):** added an opt-in `HangulCompositionCharFilter` to nori with offset correction and deliberately narrow Unicode scope. **Merged.**
- **[Elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157):** documented that the default nori `XPN` stop tag can remove meaning-bearing prefixes (`비급여 -> 급여`, `부담보 -> 담보`) and documented configuration remedies. **Merged.**
- **[Elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931):** preserved position holes in graph phrase queries, fixing exact-source false negatives with span and nori end-to-end tests. **Merged.**

### Embedding Losses And Model Internals

- **[sentence-transformers #3817](https://github.com/huggingface/sentence-transformers/pull/3817):** fixed gathered positives being masked as false negatives in distributed GIST losses. **Merged.**
- **[sentence-transformers #3821](https://github.com/huggingface/sentence-transformers/pull/3821):** made relative-margin filtering sign-independent in mining and GIST losses. **Merged.**
- **[sentence-transformers #3816](https://github.com/huggingface/sentence-transformers/pull/3816):** avoided materializing the full non-FAISS query-corpus similarity matrix; a 10K x 100K local run reduced peak RSS from 4.56 GB to 1.09 GB while preserving outputs. **Merged.**
- **[sentence-transformers #3812](https://github.com/huggingface/sentence-transformers/pull/3812):** added MPS support to cached-loss random-state handling. **Merged.**
- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800):** fixed bf16/fp16 training crashes across six learning-to-rank losses. **Merged.**
- **[Transformers #46530](https://github.com/huggingface/transformers/pull/46530):** fixed CJK stop-string matching on byte-fragment tokenizers. **Merged.**
- **[Transformers #46670](https://github.com/huggingface/transformers/pull/46670):** made continuous-batching output a snapshot, preventing mutable delivered chunks and soft-reset bookkeeping drift. **Merged.**
- **[Transformers #46624](https://github.com/huggingface/transformers/pull/46624):** fixed dynamic RoPE failing to reset `inv_freq` when `layer_type=None`. **Merged.**
- **[Transformers #46763](https://github.com/huggingface/transformers/pull/46763):** rounded the ue8m0 FP8 scale before quantization so dequantization matches the stored inverse. **Merged.**
- **[Transformers #46784](https://github.com/huggingface/transformers/pull/46784):** fixed Moonshine training loss being shifted twice. **Merged.**
- **[LlamaIndex #21900](https://github.com/run-llama/llama_index/pull/21900):** fixed text-splitter recursion when one CJK or emoji token exceeds `chunk_size`. **Merged.**
- **[MLflow #23957](https://github.com/mlflow/mlflow/pull/23957):** fixed `genai.evaluate()` dropping dataset expectations and tags with `scorers=[]`. **Merged.**

### Additional Evidence

- [PyTorch #187779](https://github.com/pytorch/pytorch/pull/187779): MPS fused RMSNorm fp32 weight multiplication. **Approved, pending merge.**
- [vLLM #45168](https://github.com/vllm-project/vllm/pull/45168): Hermes parser boundary reproduction. **Closed without merge.**
- [spaCy #13974](https://github.com/explosion/spaCy/pull/13974): Korean tokenizer offsets.
- [FAISS #5272](https://github.com/facebookresearch/faiss/issues/5272): musllinux wheels.

</details>

## Repository Map

- **[search_system](https://github.com/Incheonkirin/search_system):** runnable Korean insurance-clause retrieval with nori BM25, BGE-M3, hybrid fusion, cross-encoder reranking, Elasticsearch, and FastAPI.
- **[ko-evidence-bench](https://github.com/Incheonkirin/ko-evidence-bench):** privacy-safe retrieval evaluation, synthetic probes, aggregate studies, and representation-correctness research.
- **[fraud-dataset-validity](https://github.com/Incheonkirin/fraud-dataset-validity):** reproducible validity audit of public synthetic fraud datasets.
- **[insurance-bias-probe](https://github.com/Incheonkirin/insurance-bias-probe):** controlled demographic-consistency probes for Korean insurance answers.

![Python](https://img.shields.io/badge/Python-3E2723?style=flat-square&logo=python&logoColor=EFEBE9)
![PyTorch](https://img.shields.io/badge/PyTorch-4E342E?style=flat-square&logo=pytorch&logoColor=EFEBE9)
![Transformers](https://img.shields.io/badge/Transformers-5D4037?style=flat-square&logo=huggingface&logoColor=EFEBE9)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6D4C41?style=flat-square)
![Elasticsearch / Lucene](https://img.shields.io/badge/Elasticsearch_%2F_Lucene-5D4037?style=flat-square&logo=elasticsearch&logoColor=EFEBE9)
