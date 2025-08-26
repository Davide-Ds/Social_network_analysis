import os
import re
import json
from collections import Counter
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from data_processing.import_data import load_tweets_and_labels
from typing import Optional, Tuple, List


def clean_text(t: str) -> str:
    """
    Clean the input text by lowercasing, removing URLs, mentions, hashtags, and extra spaces.

    Args:
        t (str): Raw text of a tweet.

    Returns:
        str: Cleaned text suitable for NLP feature extraction.
    """
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)  # Remove URLs
    t = re.sub(r"@\w+", " ", t)  # Remove mentions
    t = re.sub(r"#\w+", " ", t)  # Remove hashtags
    t = re.sub(r"\s+", " ", t).strip()  # Normalize whitespace
    return t


def map_label(raw) -> Optional[int]:
    """
    Map raw labels to binary classification: 
    1 = fake/rumor/false/unverified
    0 = real/non-rumor/true
    Other labels are ignored (None).

    Args:
        raw (str): Original label string.

    Returns:
        Optional[int]: Mapped label (0 or 1) or None if invalid.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"fake", "rumor", "false", "unverified"}:
        return 1
    if s in {"real", "non-rumor", "true"}:
        return 0
    return None


def build_dataset(tweets: dict) -> Tuple[List[str], List[int]]:
    """
    Construct lists of cleaned tweet texts and corresponding binary labels.

    Supports multiple key names: 'text', 'text_content', 'tweet_label', 'label'.

    Args:
        tweets (dict): Dictionary of tweets, where each value contains text and label info.

    Returns:
        Tuple[List[str], List[int]]: 
            - texts: List of cleaned tweet texts.
            - labels: Corresponding list of integer labels (0 or 1).
    """
    texts, labels = [], []
    for t in tweets.values():
        text = t.get("text") or t.get("text_content") or t.get("content") or ""
        y = map_label(t.get("tweet_label") or t.get("label"))
        if text and y is not None:
            texts.append(clean_text(text))
            labels.append(y)
    return texts, labels


def train_and_evaluate(path_source_tweets: str, path_labels: str, artifacts_dir: str = "artifacts") -> None:
    """
    Train a Logistic Regression classifier on tweet data and evaluate performance.
    Saves model, vectorizer, classification report, and confusion matrix to disk.

    Args:
        path_source_tweets (str): Path to the file containing raw tweets.
        path_labels (str): Path to the file containing corresponding labels.
        artifacts_dir (str, optional): Directory to save trained artifacts. Defaults to "artifacts".

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load and preprocess tweets
    tweets = load_tweets_and_labels(path_source_tweets, path_labels)
    texts, labels = build_dataset(tweets)

    if not texts:
        print("No valid data available for classification.")
        return

    print(f"{len(texts)} usable examples found.")
    counts = Counter(labels)
    print(f"Class distribution: {counts}")

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train Logistic Regression classifier
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1-score: {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["real", "fake"]))

    # Save artifacts: model, vectorizer, report, confusion matrix
    joblib.dump(clf, os.path.join(artifacts_dir, "model_logreg.joblib"))
    joblib.dump(vectorizer, os.path.join(artifacts_dir, "tfidf_vectorizer.joblib"))

    with open(os.path.join(artifacts_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, target_names=["real", "fake"]))

    with open(os.path.join(artifacts_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"confusion_matrix": confusion_matrix(y_test, y_pred).tolist(), "labels": ["real", "fake"]},
            f,
            indent=2
        )

    print("Training and evaluation completed. Artifacts saved to:", artifacts_dir)
