#!/usr/bin/env python
"""LDA Topic Discovery Script.

Reads a CSV file, preprocesses English text, runs Latent Dirichlet Allocation
using collapsed Gibbs sampling, and outputs discovered topics with document
assignments.

Usage:
    python discover_topics.py data.csv --text-column merged_text --n-topics 10
"""

import argparse
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover topics in text data using LDA"
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument(
        "--text-column", default="merged_text",
        help="Name of the column containing text (default: merged_text)"
    )
    parser.add_argument(
        "--n-topics", type=int, default=10,
        help="Number of topics to discover (default: 10)"
    )
    parser.add_argument(
        "--n-iter", type=int, default=2000,
        help="Number of Gibbs sampling iterations (default: 2000)"
    )
    parser.add_argument(
        "--n-top-words", type=int, default=10,
        help="Number of top words to display per topic (default: 10)"
    )
    parser.add_argument(
        "--max-features", type=int, default=None,
        help="Maximum vocabulary size (default: no limit)"
    )
    parser.add_argument(
        "--min-df", type=int, default=2,
        help="Minimum document frequency for a term (default: 2)"
    )
    parser.add_argument(
        "--max-df", type=float, default=0.95,
        help="Maximum document frequency for a term (default: 0.95)"
    )
    parser.add_argument(
        "--output", default="topic_results",
        help="Output file prefix (default: topic_results)"
    )
    parser.add_argument(
        "--random-state", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Dirichlet prior on document-topic distributions (default: 0.1)"
    )
    parser.add_argument(
        "--eta", type=float, default=0.01,
        help="Dirichlet prior on topic-word distributions (default: 0.01)"
    )
    return parser.parse_args()


def load_data(csv_file, text_column):
    """Load CSV and validate the text column exists."""
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Error: pandas is required. Install with: pip install pandas")

    df = pd.read_csv(csv_file)
    if text_column not in df.columns:
        sys.exit(
            f"Error: column '{text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows with missing or empty text
    mask = df[text_column].notna() & (df[text_column].astype(str).str.strip() != "")
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        logger.warning("Dropped %d rows with empty text", n_dropped)
    df = df[mask].reset_index(drop=True)

    if len(df) < 2:
        sys.exit("Error: need at least 2 documents after filtering")

    return df


def build_dtm(documents, max_features, min_df, max_df):
    """Build document-term matrix using CountVectorizer."""
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        sys.exit(
            "Error: scikit-learn is required. Install with: pip install scikit-learn"
        )

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )
    dtm = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()

    # Remove all-zero rows (documents with no terms after filtering)
    row_sums = np.array(dtm.sum(axis=1)).flatten()
    nonzero_mask = row_sums > 0
    n_empty = (~nonzero_mask).sum()
    if n_empty > 0:
        logger.warning(
            "%d documents have no terms after vocabulary filtering; "
            "they will be excluded from LDA", n_empty
        )
    dtm_filtered = dtm[nonzero_mask]

    return dtm_filtered, vocab, nonzero_mask


def run_lda(dtm, n_topics, n_iter, alpha, eta, random_state):
    """Fit LDA model and return model instance."""
    import lda

    model = lda.LDA(
        n_topics=n_topics,
        n_iter=n_iter,
        alpha=alpha,
        eta=eta,
        random_state=random_state,
    )
    model.fit(dtm)
    return model


def print_topics(model, vocab, n_top_words):
    """Print top words for each topic to stdout."""
    print(f"\n{'='*60}")
    print(f"Discovered {model.n_topics} Topics")
    print(f"{'='*60}\n")

    for i, topic_dist in enumerate(model.topic_word_):
        top_indices = np.argsort(topic_dist)[::-1][:n_top_words]
        top_words = [(vocab[j], topic_dist[j]) for j in top_indices]
        words_str = ", ".join(f"{w} ({p:.4f})" for w, p in top_words)
        print(f"Topic {i}: {words_str}")


def save_results(df, model, vocab, nonzero_mask, n_top_words, output_prefix):
    """Save topic assignments and topic descriptions to CSV files."""
    import pandas as pd

    # --- Document assignments ---
    dominant_topics = np.full(len(df), -1, dtype=int)
    topic_probs = np.full(len(df), np.nan)

    doc_topic = model.doc_topic_
    valid_indices = np.where(nonzero_mask)[0]
    for local_idx, global_idx in enumerate(valid_indices):
        dominant_topics[global_idx] = np.argmax(doc_topic[local_idx])
        topic_probs[global_idx] = doc_topic[local_idx].max()

    df_out = df.copy()
    df_out["dominant_topic"] = dominant_topics
    df_out["topic_probability"] = topic_probs

    assignments_path = f"{output_prefix}_assignments.csv"
    df_out.to_csv(assignments_path, index=False)
    print(f"\nDocument assignments saved to: {assignments_path}")

    # --- Topic descriptions ---
    rows = []
    for i, topic_dist in enumerate(model.topic_word_):
        top_indices = np.argsort(topic_dist)[::-1][:n_top_words]
        row = {"topic_id": i}
        for rank, j in enumerate(top_indices):
            row[f"word_{rank+1}"] = vocab[j]
            row[f"prob_{rank+1}"] = round(float(topic_dist[j]), 6)
        rows.append(row)

    topics_path = f"{output_prefix}_topics.csv"
    pd.DataFrame(rows).to_csv(topics_path, index=False)
    print(f"Topic descriptions saved to: {topics_path}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Loading data from: {args.csv_file}")
    df = load_data(args.csv_file, args.text_column)
    documents = df[args.text_column].astype(str).tolist()
    print(f"Loaded {len(documents)} documents")

    print("Building document-term matrix...")
    dtm, vocab, nonzero_mask = build_dtm(
        documents, args.max_features, args.min_df, args.max_df
    )
    print(f"Corpus: {dtm.shape[0]} documents, {dtm.shape[1]} terms")

    print(f"Running LDA with {args.n_topics} topics, {args.n_iter} iterations...")
    model = run_lda(
        dtm, args.n_topics, args.n_iter, args.alpha, args.eta, args.random_state
    )

    print_topics(model, vocab, args.n_top_words)
    save_results(df, model, vocab, nonzero_mask, args.n_top_words, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
