# Mingi Jeong (정민기)

**Applied ML · Search & Retrieval**

At MetLife, I develop and operate ML models. Before that, I spent 5.5 years
building Korean search and QA systems at 42Maru.

`7 years` · `42Maru Search 2019–2024` · `MetLife Production ML` · `16 merged upstream PRs` · `5 national NLP data initiatives`

My public work traces search, ranking, and ML-system failures to reproducible
tests and upstream fixes across Lucene, Elasticsearch, sentence-transformers,
and Transformers. I also helped turn production MRC/RAG failure modes into
five nationally funded public data assets later reused by KAIST/KakaoBank and
ACL FinNLP research.

[![Blog](https://img.shields.io/badge/Blog-4E342E?style=flat-square)](https://incheonkirin.github.io)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-3E2723?style=flat-square)](https://www.linkedin.com/in/mingi-jeong-8a9210180/)
[![Email](https://img.shields.io/badge/Email-5D4037?style=flat-square&logo=gmail&logoColor=EFEBE9)](mailto:incheonkirin@gmail.com)

## Selected Evidence

**Neural ranking correctness.** Padding changed ListMLE/PListMLE loss and
valid-document gradients. Maintainer-benchmarked NanoBEIR R100 mean nDCG@10:
ListMLE `~0.39 -> 0.529` (vs. an older-release no-fix baseline); matched
PListMLE `0.514 -> 0.525`.<br>
[Why padding broke the loss](case-studies/listmle-padding-correctness.md) ·
[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827)

**Query semantics.** An exact Korean source phrase produced `match=1`,
`match_phrase(slop=0)=0`, and `match_phrase(slop=1)=1`. Fixed two graph
position-gap paths; merged with a nori end-to-end regression.<br>
[The phrase that returned zero results](https://github.com/Incheonkirin/ko-evidence-bench/blob/main/case_studies/korean-retrieval-correctness/exact-phrase-zero-results.md) ·
[Elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931)

**Text representation.** Added an offset-correct Hangul composition filter so
canonically equivalent NFD/NFC modern Hangul receives equivalent nori
analysis. Merged same-day by the Lucene PMC member who independently
verified the constants against Unicode Hangul syllable composition.<br>
[NFD Hangul vs. the analyzer](https://github.com/Incheonkirin/ko-evidence-bench/tree/main/case_studies/korean-retrieval-correctness) ·
[Apache Lucene #16242](https://github.com/apache/lucene/pull/16242)

**Serving correctness.** Streaming continuous batching converts request state
to output once per generated token; the converter returned live buffers by
reference and mutated the request on the soft-reset path, so already-delivered
chunks changed and soft-reset requests stopped short of `max_new_tokens`.
Under forced cache pressure on CUDA, unpatched greedy runs stopped at 21-23
of 30 tokens; patched, all twelve reached 30 with first chunks stable.<br>
[Snapshotting generation output](https://incheonkirin.github.io/posts/2026-07-14-snapshotting-generation-output-in-transformers-continuous-batching) ·
[Transformers #46670](https://github.com/huggingface/transformers/pull/46670)

## End-To-End Search Relevance

```text
Korean text -> Lucene/nori -> Elasticsearch query semantics
            -> BM25 + dense retrieval -> hybrid fusion
            -> cross-encoder reranking -> relevance evaluation
```

Every study starts from a failure I hit in production, gets reduced to a
minimal reproduction, and ends with an upstream fix or a measured limitation.

## Production And Public Impact

**Production search, 42Maru (2019–2024).** 5.5 years on the search team
across BM25 relevance, contrastive retrieval, RAG/MRC QA, large-scale
indexing, and crawler systems. In the press:
[DSME semantic QA, 2019](http://www.aitimes.kr/news/articleView.html?idxno=13427) ·
[Hana Bank OCR-NLP/AML, 2021](https://www.venturesquare.net/844917).

**Production insurance ML, MetLife (current).** New-business risk, fraud
detection, agent activation, and cross-sell models on Azure ML and Databricks.

**Public benchmark leadership.** Turned failure modes observed while developing
production MRC and LLM systems into five NIA AI Hub data initiatives. Initiated
the project proposals, secured their selection, and led task design, annotation
guidelines, quality assurance, baseline development, and release. The resulting
public assets contain approximately **2.27M QA pairs and a 304M-token corpus**.

**Downstream adoption.** The datasets were subsequently reused by
[K-FinHallu](https://arxiv.org/abs/2605.29523) from KAIST and KakaoBank,
[FINALE](https://aclanthology.org/2024.finnlp-2.9/) at ACL FinNLP 2024, and
SKKU RoSeLLa. As of July 2026, the five releases total **7,738 downloads,
468 likes, and 245K+ page views** on AI Hub.

[News MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=577) ·
[National-archives LLM corpus](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71788) ·
[Finance/legal MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610) ·
[Numerical-reasoning MRC](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71568) ·
[Table QA](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71565)

**Retrieval research lab.** 36,983 evidence passages. On a 544-row silver
retrieval study, cross-text reranking lifted `clause@20` from `56.4%` to
`64.9%` (+8.5pp, paired bootstrap CI +5.9 to +11.2); a 444-triple polarity
stress test compares lexical, dense, and reranked systems. Reported as
aggregate silver diagnostics.
[Reproducible public companion](https://github.com/Incheonkirin/ko-evidence-bench).

**Open-source depth.** 16 merged external contributions across Lucene,
Elasticsearch, sentence-transformers, Transformers, MLflow, and LlamaIndex —
from ranking losses to a hard-negative-mining memory fix (peak RSS
`4.56 GB -> 1.09 GB`).
[Full PR search](https://github.com/search?q=author%3AIncheonkirin+type%3Apr&type=pullrequests).

## Technical Scope

`Information Retrieval` · `Search Relevance` · `BM25` · `Dense Retrieval` ·
`Hybrid Search` · `Vector Search` · `Learning to Rank` ·
`Cross-Encoder Reranking` · `Hard-Negative Mining` · `Embeddings` · `Lucene` ·
`Elasticsearch` · `OpenSearch` · `FAISS` · `PyTorch` ·
`sentence-transformers` · `ANN` · `nDCG` · `Recall` · `RAG Evaluation`

<details>
<summary><strong>Full upstream record: 16 merged contributions</strong></summary>

### Korean Search And Ranking

- **[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827):** excluded padding from the ListMLE/PListMLE Plackett-Luce normalizer and added a padding-invariance regression. Maintainer-reported NanoBEIR R100 mean nDCG@10: ListMLE approximately `0.39 -> 0.529` (older-release no-fix baseline); matched PListMLE `0.514 -> 0.525`. **Merged.**
- **[Apache Lucene #16242](https://github.com/apache/lucene/pull/16242):** added an opt-in `HangulCompositionCharFilter` to nori with offset correction and deliberately narrow Unicode scope; constants verified against the Unicode spec by the reviewing PMC member. **Merged.**
- **[Elasticsearch #151157](https://github.com/elastic/elasticsearch/pull/151157):** documented that the default nori `XPN` stop tag can remove meaning-bearing prefixes (`비급여 -> 급여`, `부담보 -> 담보`) and documented configuration remedies. **Merged.**
- **[Elasticsearch #152931](https://github.com/elastic/elasticsearch/pull/152931):** preserved position holes in graph phrase queries, fixing exact-source false negatives with span and nori end-to-end tests. **Merged.**

### Embedding Losses And Model Internals

- **[sentence-transformers #3817](https://github.com/huggingface/sentence-transformers/pull/3817):** fixed gathered positives being masked as false negatives in distributed GIST losses; surfaced with a Korean polarity probe. **Merged.**
- **[sentence-transformers #3821](https://github.com/huggingface/sentence-transformers/pull/3821):** made relative-margin filtering sign-independent in mining and GIST losses. **Merged.**
- **[sentence-transformers #3816](https://github.com/huggingface/sentence-transformers/pull/3816):** avoided materializing the full non-FAISS query-corpus similarity matrix in hard-negative mining; a 10K x 100K run cut peak RSS from `4.56 GB` to `1.09 GB` with identical outputs. **Merged.**
- **[sentence-transformers #3812](https://github.com/huggingface/sentence-transformers/pull/3812):** added MPS support to cached-loss random-state handling. **Merged.**
- **[sentence-transformers #3800](https://github.com/huggingface/sentence-transformers/pull/3800):** fixed bf16/fp16 training crashes across six learning-to-rank losses. **Merged.**
- **[Transformers #46530](https://github.com/huggingface/transformers/pull/46530):** fixed CJK stop-string matching on byte-fragment tokenizers. **Merged.**
- **[Transformers #46670](https://github.com/huggingface/transformers/pull/46670):** made continuous-batching output a snapshot, preventing mutable delivered chunks and soft-reset bookkeeping drift. **Merged.**
- **[Transformers #46624](https://github.com/huggingface/transformers/pull/46624):** fixed dynamic RoPE failing to reset `inv_freq` when `layer_type=None`. **Merged.**
- **[Transformers #46763](https://github.com/huggingface/transformers/pull/46763):** rounded the ue8m0 FP8 scale before quantization so dequantization matches the stored inverse. **Merged.**
- **[Transformers #46784](https://github.com/huggingface/transformers/pull/46784):** fixed Moonshine training loss being shifted twice. **Merged.**
- **[LlamaIndex #21900](https://github.com/run-llama/llama_index/pull/21900):** fixed text-splitter recursion when one CJK or emoji token exceeds `chunk_size`. **Merged.**
- **[MLflow #23957](https://github.com/mlflow/mlflow/pull/23957):** fixed `genai.evaluate()` dropping dataset expectations and tags with `scorers=[]`. **Merged.**

### In Review

- **[PyTorch #187779](https://github.com/pytorch/pytorch/pull/187779):** MPS fused RMSNorm fp32 weight multiplication to match CPU/CUDA. **Approved, pending merge.**
- **[spaCy #13974](https://github.com/explosion/spaCy/pull/13974):** Korean tokenizer whitespace preservation. **Open.**

### Reports And Reproductions

- **[FAISS #5272](https://github.com/facebookresearch/faiss/issues/5272):** reported broken musllinux wheels; wheels restored.
- **[vLLM #45168](https://github.com/vllm-project/vllm/pull/45168):** reproduced a Hermes tool-parser boundary where a literal `</tool_call>` inside a JSON string argument drops the call.

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
