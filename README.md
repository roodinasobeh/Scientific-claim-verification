# Scientific Claim Verification Pipeline

A three-stage NLP pipeline for **automated fact-checking of scientific claims** against a corpus of research paper abstracts. Given a claim, the system retrieves the most relevant papers, selects the sentences that best support or refute the claim, and classifies the claim as **SUPPORT**, **CONTRADICT**, or **NOT ENOUGH INFO**.

This project is built and evaluated on a SciFact-style dataset (`corpus.jsonl` + `claims_train/dev/test.jsonl`).

## Pipeline Overview

The system is organized into three sequential stages:

```
Claim ──▶ 1. Document Retrieval ──▶ 2. Evidence Selection ──▶ 3. Verdict Classification ──▶ Verdict
```

1. **Document Retrieval** — Given a claim, find the most relevant papers in the corpus.
2. **Evidence Selection** — Within the retrieved papers, identify the specific sentences that serve as evidence for the claim.
3. **Verdict Classification (NLI)** — Given the claim and its evidence, decide whether the evidence supports, contradicts, or gives insufficient information about the claim.

## 1. Document Retrieval

Three retrieval methods were implemented and compared:

| Method | Description |
|---|---|
| **TF-IDF** | Vectorizes claims and documents with TF-IDF (unigrams + bigrams) and ranks papers by cosine similarity. |
| **BM25** | Classic lexical ranking function (via `rank_bm25`), scored on tokenized title + abstract text. |
| **Semantic (SPECTER)** | Uses the `allenai-specter` sentence-transformer model to embed claims and documents, ranked by cosine similarity of embeddings. |

Retrieval quality is evaluated using:
- **Recall@1**, **Recall@5** — whether a gold (correct) document appears in the top-1 / top-5 results.
- **MRR (Mean Reciprocal Rank)** — how high the first correct document is ranked, on average.

## 2. Evidence Selection

Once candidate papers are retrieved, the pipeline scores each sentence in their abstracts to decide whether it is relevant evidence for the claim.

**Features used per (claim, sentence) pair:**
- BM25 score of the sentence against the claim
- Word overlap ratio between claim and sentence
- Sentence position within the abstract
- Sentence length

**Models compared:**
- Logistic Regression (`class_weight="balanced"`)
- Random Forest (`n_estimators=200`, `class_weight="balanced"`) — saved as `evidence_selector_rf.joblib`

Performance is evaluated with:
- Classification report (precision / recall / F1 for "Evidence" vs "Not Evidence")
- **Evidence Recall@1 / @3 / @5** — whether the correct evidence sentence is among the top-k selected sentences.

## 3. Verdict Classification (Fine-tuned NLI Model)

The final stage fine-tunes a transformer model to classify a (claim, evidence) pair into one of three classes:

- `SUPPORT`
- `CONTRADICT`
- `NOT_ENOUGH_INFO`

**Base model:** [`allenai/scibert_scivocab_uncased`](https://huggingface.co/allenai/scibert_scivocab_uncased) — a BERT model pretrained on scientific text, fine-tuned here for sequence classification.

**Fine-tuning strategies compared:**

| Strategy | Description | F1 (macro) |
|---|---|---|
| Full freeze | Only the classification head is trained | 0.588 |
| LoRA | Low-rank adapters added to the frozen backbone | 0.619 |
| **Partial fine-tuning (used)** | Last two encoder layers + pooler + classifier unfrozen | **0.696** |

The final model unfreezes `encoder.layer.10`, `encoder.layer.11`, the pooler, and the classification head, and is trained with the Hugging Face `Trainer` API (8 epochs, learning rate `3e-5`, mixed precision `fp16`, best checkpoint selected by macro F1).

The fine-tuned model and tokenizer are saved to `./final_model`.

## Repository / Notebook Structure

The entire pipeline lives in a single Jupyter notebook, organized into these sections:

1. **Load Libraries** — imports (scikit-learn, `rank_bm25`, `sentence-transformers`, `transformers`, `datasets`, `torch`).
2. **Retriever and Paper Selection** — loading the corpus, building TF-IDF / BM25 / Semantic retrievers, and evaluating them.
3. **Evidence Selection** — building (claim, sentence) training pairs, feature engineering, training the Logistic Regression / Random Forest evidence selectors, and evaluating evidence recall.
4. **Fine-tuning & Model Selection** — building NLI examples, tokenizing, fine-tuning SciBERT, and evaluating on dev/train/test sets.

## Data Format

The pipeline expects the following files inside a `data/` directory:

| File | Description |
|---|---|
| `data/corpus.jsonl` | One JSON object per line: `{"doc_id": int, "title": str, "abstract": [list of sentences]}` |
| `data/claims_train.jsonl` | Training claims with gold evidence annotations |
| `data/claims_dev.jsonl` | Development/validation claims with gold evidence annotations |
| `data/claims_test.jsonl` | Test claims (evidence may be withheld) |

Each **claim** object has the form:
```json
{
  "claim": "Prematurity affects cerebral white matter development.",
  "evidence": {
    "<doc_id>": [
      {"sentences": [0, 1], "label": "SUPPORT"}
    ]
  }
}
```

## Requirements

Install the dependencies with:

```bash
pip install numpy scikit-learn rank_bm25 sentence-transformers torch transformers datasets joblib
```

## How to Try the System

The system can be tested in two ways:

### 1. User Mode

For users who want to try the system normally:

1. Open `deployment/gradio_app.ipynb`.
2. Run all cells in the notebook.
3. The Gradio interface will launch automatically.
4. Enter a scientific claim.
5. The system will retrieve relevant scientific papers, select supporting evidence, and return the predicted verdict:

   * SUPPORT
   * CONTRADICT
   * NOT ENOUGH INFO

### 2. Developer Mode

For developers who want to test the full pipeline:

1. Open the full pipeline notebook.
2. Choose the dataset split to work with (Train/Test).
3. Run the complete pipeline from preprocessing and retrieval to evidence verification and prediction.
4. Select samples from the dataset and run predictions.
5. Evaluate and inspect the system's results across the selected data.

## User Interface

### Gradio Interface
<img width="1912" height="872" alt="image" src="https://github.com/user-attachments/assets/c23a4adb-45fc-454f-b9ab-3d9cade84eb1" />

## Future Improvements

* **Deployment** — Deploy the system online to make it accessible without requiring local setup.



