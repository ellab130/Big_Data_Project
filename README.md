
# Efficient Semantic Joins with Hybrid Value Matching

**NYU Tandon School of Engineering — Big Data Project**

**Team:** Game of Joins  
**Members:** Ella Bourassa, Rajdeep Dey, Jingrao Chen

## 1. Project Overview

Traditional database joins require exact agreement between join keys. However, real-world datasets often contain different names for the same entity, including spelling variations, abbreviations, formatting differences, and alternative names.

This project implements a hybrid semantic join system that identifies matching values across two datasets even when their string representations differ.

The system combines:

- Token-based candidate blocking
- Sentence-transformer embeddings
- Top-k semantic candidate retrieval
- Optional character n-gram retrieval
- Fuzzy string similarity
- Optional cardinality constraints
- PySpark joins for reconstructing matched rows

The goal is to reduce unnecessary candidate comparisons while maintaining useful matching accuracy.

## 2. System Architecture

The semantic join pipeline consists of five stages.

### Stage 1: Extract Distinct Values

Extract unique, non-null join-key values from both input tables.

Matching distinct values avoids repeatedly computing similarity for duplicate strings.

### Stage 2: Candidate Generation

Generate candidate value pairs using a combination of:

1. **Token blocking:** Retrieve pairs that share informative tokens.
2. **Embedding retrieval:** Retrieve the top-k semantically similar values.
3. **Optional character n-gram retrieval:** Retrieve additional candidates based on overlapping character sequences.

Candidate generation limits the pairs considered by subsequent verification.

### Stage 3: Candidate Verification

Use a pretrained sentence-transformer model:

`sentence-transformers/all-MiniLM-L6-v2`

Compute embedding similarity and fuzzy string similarity for candidate pairs.

Under the evaluated configuration, candidates are accepted when their embedding similarity exceeds the specified threshold or their normalized values match exactly.

Optional year and location filters can be enabled for domain-specific experiments.

### Stage 4: Cardinality Constraints

The system supports three matching modes:

- `many_to_many`: Allow multiple mappings on either side.
- `right_unique`: Allow each distinct right-side value to match at most one left-side value.
- `one_to_one`: Allow each distinct value on either side to participate in at most one mapping.

Cardinality constraints are applied greedily using match scores.

They can substantially improve precision when the dataset's entity relationships justify the constraint.

### Stage 5: Join Reconstruction

After matching distinct values, the system joins the resulting mappings back to the original tables.

The PySpark wrapper broadcasts the mapping table and performs distributed row joins.

This preserves duplicate source rows and their corresponding matching combinations.

## 3. Repository Structure

```text
Big_Data_Project/
├── code/
│   ├── semantic_join.py
│   ├── spark_semantic_join.py
│   ├── blocking.py
│   ├── final_benchmark.py
│   ├── heldout_benchmark.py
│   ├── evaluate_datasets.py
│   ├── test_semantic_join.py
│   ├── test_spark_semantic_join.py
│   ├── test_spark_benchmark.py
│   ├── experiments/
│   └── other diagnostic and test scripts
├── results/
│   ├── final_benchmark_results.csv
│   ├── heldout_benchmark_results.csv
│   └── additional experimental results
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. Installation

The project was developed using Python 3.12.

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

The sentence-transformer model may be downloaded automatically during the first execution.

PySpark also requires a compatible Java installation. The local development environment used Java 21.

## 5. Running the Experiments

From the repository root:

```bash
cd code
```

Run the development benchmark:

```bash
python final_benchmark.py
```

Run the held-out evaluation:

```bash
python heldout_benchmark.py
```

Results are written to the `results/` directory.

Run the basic semantic join test:

```bash
python test_semantic_join.py
```

Run the PySpark integration test:

```bash
python test_spark_semantic_join.py
```

These scripts provide examples of how to use the implementation.

## 6. Experimental Methodology

We evaluated the system on entity-matching datasets provided by the Automatic Fuzzy Join (AutoFJ) benchmark.

### Development Datasets

- TennisTournament
- Museum
- Drug
- Stadium

These datasets were used during algorithm development and error analysis.

### Held-Out Datasets

- Artwork
- Hospital

These datasets were evaluated after the main algorithm and matching parameters had been selected.

### Configurations

**Original Hybrid**

Token blocking and embedding top-k retrieval with unrestricted many-to-many matching.

**Hybrid + Character**

Adds ranked character n-gram retrieval to the original hybrid system.

**Hybrid + Cardinality**

Uses hybrid retrieval with a right-unique matching constraint.

**Full Improved Hybrid**

Combines hybrid retrieval, ranked character retrieval, and right-unique matching.

### Metrics

We report:

- Precision
- Recall
- F1 score
- Runtime
- Candidate counts and reduction

Precision, recall, and F1 are computed by comparing predicted source-record ID pairs with the benchmark ground truth.

## 7. Development Results

| Dataset | Original Hybrid F1 | Hybrid + Cardinality F1 |
|---|---:|---:|
| TennisTournament | 0.292 | 0.679 |
| Museum | 0.357 | 0.702 |
| Drug | 0.284 | 0.413 |
| Stadium | 0.188 | 0.491 |

The right-unique constraint improved F1 across all four development datasets.

## 8. Held-Out Results

| Dataset | Configuration | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Artwork | Original Hybrid | 0.670 | 0.739 | 0.703 |
| Artwork | Hybrid + Cardinality | 0.910 | 0.702 | 0.793 |
| Artwork | Full Improved Hybrid | 0.910 | 0.702 | 0.793 |
| Hospital | Original Hybrid | 0.189 | 0.728 | 0.300 |
| Hospital | Hybrid + Cardinality | 0.724 | 0.623 | 0.669 |
| Hospital | Full Improved Hybrid | 0.724 | 0.623 | 0.669 |

Cardinality constraints improved F1 on both held-out datasets, primarily by increasing precision.

Adding character n-gram retrieval did not improve final F1 under the evaluated verification threshold.

Full experimental outputs are available in:

- `results/final_benchmark_results.csv`
- `results/heldout_benchmark_results.csv`

## 9. Key Findings

### Cardinality Constraints Matter

Semantic similarity alone can generate many plausible but incorrect matches.

When right-side uniqueness is appropriate, enforcing it substantially improves precision and overall F1.

### Candidate Recall Does Not Guarantee Final Recall

Adding retrieval methods can recover additional true candidate pairs, but these pairs may still be rejected during verification.

Candidate generation and verification must therefore be evaluated separately.

### Semantic Similarity Is Not Entity Identity

The system struggles with some historical renamings, aliases, and domain-specific identifiers.

Two names may refer to the same entity without having sufficiently similar embeddings.

Conversely, semantically similar strings may describe different entities.

## 10. Limitations

The current implementation has several important limitations:

1. **Local semantic matching:** Distinct values are collected to the driver, and embedding-based candidate matching runs locally.
2. **Dense similarity matrix:** The embedding retrieval stage computes a dense cross-table similarity matrix, requiring quadratic space in the number of distinct values.
3. **Greedy cardinality selection:** Cardinality constraints are enforced greedily rather than through globally optimal assignment.
4. **Domain-dependent constraints:** Right-unique matching is inappropriate for datasets that genuinely require multiple left-side matches per right-side value.
5. **Alias detection:** Historical renamings and unrelated-looking aliases remain difficult.
6. **Limited scalability evaluation:** Local benchmark runtimes do not establish large-scale distributed matching performance.

PySpark distributes the final row-joining stage, but the current semantic matching stage is not fully distributed.

## 11. Future Work

Potential improvements include:

- Approximate nearest-neighbor indexing for embedding retrieval
- Distributed candidate generation and verification
- Improved alias and entity-resolution models
- Globally optimized matching under cardinality constraints
- Larger-scale benchmarking
- More systematic threshold calibration

## 12. References

- Li et al., Automatic Fuzzy Join (AutoFJ): https://github.com/chu-data-lab/AutomaticFuzzyJoin
- Sentence Transformers: https://www.sbert.net/
- Apache Spark: https://spark.apache.org/
- RapidFuzz: https://github.com/rapidfuzz/RapidFuzz
