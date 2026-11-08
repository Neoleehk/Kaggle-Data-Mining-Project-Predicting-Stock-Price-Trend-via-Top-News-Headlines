# News Alpha Project

This workspace now includes an executable starter pipeline for:

1. loading and cleaning DJIA news data,
2. creating fixed temporal splits from the agreed blueprint,
3. running a first baseline experiment (TF-IDF text classifier),
4. building FinBERT + Sentence-BERT embedding cache,
5. running embedding-based relevance/top-k/EWMA experiments.

## Structure

- `src/news_alpha/data.py`: dataset loading, text cleaning, target construction.
- `src/news_alpha/splits.py`: fixed expanding-window CV and holdout split indices.
- `src/news_alpha/baseline.py`: TF-IDF + Logistic/SVM baseline training/evaluation.
- `scripts/prepare_dataset.py`: export prepared dataset and split metadata.
- `scripts/run_tfidf_baseline.py`: run baseline and save results JSON.
- `src/news_alpha/textcnn.py`: lightweight TextCNN model and training utilities.
- `src/news_alpha/light_filter.py`: lightweight relevance filter (finance keywords + TF-IDF anchors).
- `scripts/run_textcnn_experiments.py`: CPU-friendly TextCNN experiments.
- `src/news_alpha/embeddings.py`: embedding cache build/load utilities.
- `src/news_alpha/relevance.py`: hybrid relevance scoring.
- `src/news_alpha/features.py`: F0-F5 feature generation (including EWMA).
- `scripts/build_embedding_cache.py`: generate embedding cache artifacts.
- `scripts/run_embedding_experiments.py`: run embedding feature experiments.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

Prepare dataset + split metadata:

```bash
python scripts/prepare_dataset.py
```

Run same-day baseline:

```bash
python scripts/run_tfidf_baseline.py --target-mode same_day --output-json outputs/baseline_same_day.json
```

Run next-day robustness baseline:

```bash
python scripts/run_tfidf_baseline.py --target-mode next_day --output-json outputs/baseline_next_day.json
```

Run fast TextCNN experiments (recommended when transformer path is too heavy):

```bash
python scripts/run_textcnn_experiments.py --target-mode same_day --keyword-weight 0.4 --tfidf-weight 0.6 --output-json outputs/textcnn_same_day.json
python scripts/run_textcnn_experiments.py --target-mode next_day --keyword-weight 0.4 --tfidf-weight 0.6 --output-json outputs/textcnn_next_day.json
```

Build embedding cache:

```bash
python scripts/build_embedding_cache.py --strategy two_stage --finbert-top-n 5 --output-dir outputs/embedding_cache
```

Run same-day embedding experiments:

```bash
python scripts/run_embedding_experiments.py --target-mode same_day --cache-dir outputs/embedding_cache --output-json outputs/embedding_same_day.json
```

Run next-day embedding robustness experiments:

```bash
python scripts/run_embedding_experiments.py --target-mode next_day --cache-dir outputs/embedding_cache --output-json outputs/embedding_next_day.json
```

Notes:
- `two_stage` mode reduces CPU time by scoring all headlines with SBERT first, then computing FinBERT embeddings only for shortlisted headlines.
- `--finbert-top-n` controls shortlist size. Smaller is faster.
