# A Padding Bug That Changed ListMLE Training

**Why the same query received different loss and gradients depending on another
query's list length**

`ListMLELoss` and `PListMLELoss` in sentence-transformers supported batches of
variable-length document lists. Shorter lists were padded to the longest list in
the batch. Those padding positions were excluded from the final loss, but they
were still included in the Plackett-Luce normalizer used to compute every real
document's log-probability.

The result violated a basic training invariant:

> Adding padding to a query must not change that query's loss or valid-document
> gradients.

The fix was merged in
[sentence-transformers #3827](https://github.com/huggingface/sentence-transformers/pull/3827).

## Minimal Counterexample

Take one query with two real documents:

```text
labels = [2, 1]
logits = [1.5, 0.3]
```

Only the padding width changes:

| `max_docs` | Padding slots | Loss | Gradient excerpt |
|---:|---:|---:|---|
| 2 | 0 | 0.2633 | `[-0.2315, +0.2315]` |
| 3 | 1 | 0.9759 | `[-0.3440, -0.2280]` |
| 4 | 2 | 1.4671 | changed again |
| 6 | 4 | 2.1627 | changed again |

With one padding slot, the second real document's gradient changes sign. The
same query is therefore trained in a different direction depending on how many
documents the longest other query in the batch happens to contain.

## Where The Extra Probability Mass Came From

For a ranked list, ListMLE uses a Plackett-Luce likelihood. At each rank, the
selected document is normalized against the documents that remain:

```math
P(\pi \mid s) = \prod_{j=1}^{n}
\frac{\exp(s_{\pi_j})}
{\sum_{k=j}^{n}\exp(s_{\pi_k})}
```

The implementation scattered each list into a `(batch_size, max_docs)` tensor and
used `1e-16` as the padding logit. It then computed:

```python
scores = sorted_logits.exp()
normalizers = scores.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
```

The padding value looks negligible in logit space, but:

```text
exp(1e-16) ≈ 1
```

Every padding slot therefore added roughly one unit of mass to the reverse
cumulative denominator. Applying the padding mask after the cumulative sum could
remove a padded position's own loss term, but could not remove the mass that had
already polluted earlier real positions.

## The Fix

Padding must have zero probability mass before the normalizer is constructed:

```python
scores = sorted_logits.exp().masked_fill(~sorted_mask, 0.0)
```

This makes padding equivalent to a `-inf` logit before exponentiation. Lists with
no padding are unchanged.

The patch changed five implementation lines and added a deterministic regression
covering both affected losses and both `respect_input_order` modes.

## The Regression Test

The useful property test does not assert one hard-coded loss value. It compares
the same queries under two batching conditions:

1. compute each query's loss separately;
2. batch a two-document query with a three-document query;
3. assert that the batched loss equals the mean of the two independent losses.

On the unpatched implementation all four parametrized cases failed. With the fix
all four passed.

This catches the actual invariant: unrelated batch padding must not change a
query's objective.

## External Training Evidence

Sentence-transformers maintainer Tom Aarsen trained MiniLM rerankers on MS MARCO
v1.1 and published the runs. The reported NanoBEIR R100 mean nDCG@10 at the best
checkpoint was:

| Loss | Without fix | With PR #3827 |
|---|---:|---:|
| ListMLE | approximately 0.39 | **0.529** |
| PListMLE | 0.514 | **0.525** |

The maintainer noted that the ListMLE no-fix number came from an older
sentence-transformers/Transformers release rather than a newly rerun matched
baseline. PListMLE used the same setup with the fix reverted. The safe conclusion
is therefore:

- the padding behavior was a demonstrated correctness defect;
- matched PListMLE training improved from `0.514` to `0.525`;
- the maintainer observed a much larger ListMLE gap, with an explicit
  older-baseline caveat.

Published runs:

- [ListMLE with the fix](https://huggingface.co/tomaarsen/reranker-msmarco-v1.1-MiniLM-L12-H384-uncased-listmle-pr3827)
- [PListMLE with the fix](https://huggingface.co/tomaarsen/reranker-msmarco-v1.1-MiniLM-L12-H384-uncased-plistmle-pr3827)
- [PListMLE matched no-fix run](https://huggingface.co/tomaarsen/reranker-msmarco-v1.1-MiniLM-L12-H384-uncased-plistmle-pr3827-baseline)
- [Older ListMLE no-fix run](https://huggingface.co/tomaarsen/reranker-msmarco-v1.1-MiniLM-L12-H384-uncased-listmle)

## Why Ordinary Tests Missed It

- fixed-length batches contain no padding;
- forward-only tests can check finiteness without checking gradient direction;
- masking the final loss makes padded positions appear excluded;
- average training metrics do not identify batch-composition dependence;
- a single expected loss value does not express padding invariance.

The broader lesson is that variable-length learning objectives need relational
tests. The same example should produce the same objective under irrelevant batch
packing choices.

## Upstream Validation

- [Merged pull request](https://github.com/huggingface/sentence-transformers/pull/3827)
- merge commit [`7763e94154f6`](https://github.com/huggingface/sentence-transformers/commit/7763e94154f60cde77e93d4b50d32bc3a364860e)
- implementation and regression were merged after the maintainer's external
  training runs and CI completion

## Takeaway

> Padding was not merely wasting computation. It was changing the probability
> model and reversing a valid document's training signal.
