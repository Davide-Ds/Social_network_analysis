import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from utils.neo4j_utils import compute_and_save_tweet_embeddings
from analysis.graph_analysis import create_complete_gds_graph


def generate_graphsage_embeddings(driver, graph_name: str, model_name: str, dim: int = 128, node_label: str = "User"):
    """
    Generate GraphSAGE embeddings for nodes of a given type in the Neo4j graph.

    This updated version will sanitize numeric feature properties (e.g. pagerank) by
    setting NULL or NaN values to 0 before calling the GDS procedures. It also wraps
    train/stream calls with try/except to provide clearer diagnostics when the GDS
    procedures fail due to invalid feature values.

    Args:
        driver (neo4j.Driver): Neo4j driver instance.
        graph_name (str): Name of the in-memory GDS graph.
        model_name (str): Name for the embedding model.
        dim (int): Embedding dimension.
        node_label (str): Label of nodes to embed ("User" or "Tweet").

    Returns:
        dict: Dictionary mapping node IDs to embedding vectors (numpy arrays).

    Example:
        embeddings = generate_graphsage_embeddings(driver, "myGraph", "UserSAGE", 128, node_label="User")
        user_emb = embeddings[user_id]
    """
    # Choose the appropriate feature property based on node type
    if node_label == "User":
        feature_properties = ["pagerank"]
    elif node_label == "Tweet":
        feature_properties = ["text_embedding"]
    else:
        raise ValueError("node_label must be 'User' or 'Tweet'")

    with driver.session() as session:
        # If using pagerank as a numeric feature, ensure no NULL or NaN values remain
        if "pagerank" in feature_properties:
            # Set NULL pagerank values to 0
            try:
                session.run("MATCH (u:User) WHERE u.pagerank IS NULL SET u.pagerank = 0 RETURN count(u) AS updated")
            except Exception:
                # ignore if permission or APOC not available; training will fail later if values invalid
                pass
            # Some NaN values may be stored; detect NaN by comparing value to itself (NaN != NaN)
            try:
                session.run("MATCH (u:User) WHERE u.pagerank <> u.pagerank SET u.pagerank = 0 RETURN count(u) AS fixed")
            except Exception:
                pass

        # Step 1: Train GraphSAGE model
        # Drop model if it already exists (ignore failure)
        drop_query = f"CALL gds.beta.model.drop('{model_name}') YIELD modelName"
        try:
            session.run(drop_query)
        except Exception:
            pass

        train_query = f"""
        CALL gds.beta.graphSage.train('{graph_name}',
            {{
                modelName: '{model_name}',
                nodeLabels: ['{node_label}'],
                relationshipTypes: ['RETWEET','QUOTE','INTERACTION','CREATES','RETWEETED_FROM'],
                featureProperties: {feature_properties},
                embeddingDimension: $dim,
                randomSeed: 42
            }}
        )
        YIELD modelInfo
        """
        try:
            session.run(train_query, {"dim": dim}).consume()
        except Exception as e:
            # Re-raise with clearer context
            raise RuntimeError(f"GraphSAGE training failed for model {model_name}: {e}") from e

        # Step 2: Generate embeddings
        embed_query = f"""
        CALL gds.beta.graphSage.stream('{graph_name}',
            {{
                nodeLabels: ['{node_label}'],
                modelName: '{model_name}'
            }}
        )
        YIELD nodeId, embedding
        RETURN gds.util.asNode(nodeId).{node_label.lower()}_id AS node_id, embedding
        """
        try:
            result = session.run(embed_query)
        except Exception as e:
            raise RuntimeError(f"GraphSAGE streaming failed for model {model_name}: {e}") from e

        # Step 3: Convert embeddings to numpy arrays
        embeddings = {}
        for record in result:
            node_id = record.get("node_id")
            if node_id is not None:
                emb_vector = np.array(record.get("embedding"), dtype=np.float32)
                embeddings[node_id] = emb_vector

        return embeddings

def build_link_prediction_dataset(
    driver,
    user_embeddings,
    tweet_embeddings,
    negative_ratio=1.0,
    random_seed=42
):
    """
    Builds a dataset for link prediction (User -> Tweet) using node embeddings.

    This function constructs a supervised learning dataset for the link prediction task,
    where the goal is to predict whether a user will retweet a tweet. It generates
    positive samples (existing RETWEET relationships) and negative samples (user-tweet
    pairs without a RETWEET relationship), and creates feature vectors by concatenating
    the embeddings of the user and the tweet.

    The function queries the Neo4j database to extract all positive pairs, identifies
    active users and popular tweets for more challenging negative sampling, and ensures
    reproducibility via a random seed. The resulting dataset can be used to train and
    evaluate machine learning models for link prediction.

    Args:
        driver (neo4j.Driver): 
            An active Neo4j driver instance for database connection.
        user_embeddings (dict): 
            Dictionary mapping user_id (str or int) to their embedding vector (np.ndarray).
        tweet_embeddings (dict): 
            Dictionary mapping tweet_id (str or int) to their embedding vector (np.ndarray).
        negative_ratio (float, optional): 
            Ratio of negative samples to positive samples. 
            For example, 1.0 means the same number of negatives as positives. Default is 1.0.
        random_seed (int, optional): 
            Random seed for reproducibility of negative sampling. Default is 42.

    Returns:
        tuple:
            X (np.ndarray): 
                Feature matrix of shape (n_samples, embedding_dim * 2), where each row is the
                concatenation of a user and tweet embedding.
            y (np.ndarray): 
                Label vector of shape (n_samples,), where 1 indicates a positive (RETWEET exists)
                and 0 indicates a negative (no RETWEET).

    Example:
        >>> X, y = build_link_prediction_dataset(driver, user_embeddings, tweet_embeddings, negative_ratio=1.0)

    Notes:
        - Positive pairs are all (user, tweet) pairs with a RETWEET relationship in the graph.
        - Negative pairs are sampled from active users and popular tweets, excluding existing RETWEETs.
        - The function prints progress and dataset statistics for transparency.
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Step 1: Query positive pairs (existing RETWEET relationships)
    pos_query = """
    MATCH (u:User)-[:RETWEET]->(t:Tweet)
    RETURN u.user_id AS user_id, t.tweet_id AS tweet_id
    """
    with driver.session() as session:
        pos_result = session.run(pos_query)
        positive_pairs = [(record["user_id"], record["tweet_id"]) for record in pos_result]

        # Identify active users (users who have retweeted at least once)
        active_users = [
            record["user_id"] for record in session.run(
                "MATCH (u:User)-[:RETWEET]->() RETURN DISTINCT u.user_id AS user_id"
            )
        ]

        # Identify popular tweets (tweets that have been retweeted at least once)
        popular_tweets = [
            record["tweet_id"] for record in session.run(
                "MATCH ()-[:RETWEET]->(t:Tweet) RETURN DISTINCT t.tweet_id AS tweet_id"
            )
        ]

    positive_set = set(positive_pairs)
    num_negatives = int(len(positive_pairs) * negative_ratio)
    negative_pairs = set()
    attempts = 0
    max_attempts = num_negatives * 20  # Prevents infinite loops in dense graphs

    # Step 2: Sample negative pairs (user-tweet pairs with no RETWEET)
    print("Building negative RETWEET pairs...")
    while len(negative_pairs) < num_negatives and attempts < max_attempts:
        u = random.choice(active_users)
        t = random.choice(popular_tweets)
        if (u, t) not in positive_set and (u, t) not in negative_pairs:
            negative_pairs.add((u, t))
        attempts += 1
    negative_pairs = list(negative_pairs)

    print(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative samples.")

    # Step 3: Build feature vectors and labels
    X = []
    y = []

    print("Building positive RETWEET pairs...")
    for u, t in positive_pairs:
        if u in user_embeddings and t in tweet_embeddings:
            X.append(np.concatenate([user_embeddings[u], tweet_embeddings[t]]))
            y.append(1)

    print("Building negative RETWEET pairs...")
    for u, t in negative_pairs:
        if u in user_embeddings and t in tweet_embeddings:
            X.append(np.concatenate([user_embeddings[u], tweet_embeddings[t]]))
            y.append(0)

    print(f"Final dataset size: {len(y)} samples.")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


def run_link_prediction(driver, cross_validation_folds=5):
    """
    Execute the complete link prediction workflow (Task 3).
    
    This function:
    1. Computes text embeddings for tweets if not already present.
    2. Creates a complete GDS graph with User and Tweet nodes.
    3. Generates GraphSAGE embeddings for User and Tweet nodes.
    4. Builds a link prediction dataset from positive/negative pairs.
    5. Trains a Random Forest classifier.
    6. Evaluates the model with metrics and cross-validation.
    
    Args:
        driver (neo4j.Driver): Neo4j driver instance.
    
    Returns:
        None
    """
    
    print("\nRunning Link Prediction using GraphSAGE embeddings...")
    print("Computing embeddings for tweets using all-MiniLM-L6-v2 model if not already present...")
    
    # Check if tweet embeddings exist
    if not driver.session().run("MATCH (t:Tweet) WHERE t.text_embedding IS NOT NULL RETURN t LIMIT 1").single():
        compute_and_save_tweet_embeddings(driver, model_name='all-MiniLM-L6-v2', text_property='text', embedding_property='text_embedding')
    
    print("Creating complete GDS graph with User and Tweet nodes...")
    create_complete_gds_graph(driver)
    
    # ---------------------------
    # Step 1: Generate embeddings
    # ---------------------------
    print("\nGenerating GraphSAGE embeddings for Users...")
    user_sage_result = driver.session().run("CALL gds.model.exists('UserSAGE') YIELD exists RETURN exists").single()
    if user_sage_result is not None and user_sage_result.value():
        driver.session().run("CALL gds.model.drop('UserSAGE')")
    user_embeddings = generate_graphsage_embeddings(
        driver,
        graph_name="fullGraph_analysis",
        model_name="UserSAGE",
        dim=128,
        node_label="User"
    )
    
    print("\nGenerating GraphSAGE embeddings for Tweets...")
    tweet_sage_result = driver.session().run("CALL gds.model.exists('TweetSAGE') YIELD exists RETURN exists").single()
    if tweet_sage_result is not None and tweet_sage_result.value():
        driver.session().run("CALL gds.model.drop('TweetSAGE')")
    tweet_embeddings = generate_graphsage_embeddings(
        driver,
        graph_name="fullGraph_analysis",
        model_name="TweetSAGE",
        dim=128,
        node_label="Tweet"
    )
    
    # ---------------------------
    # Step 2: Build link prediction dataset
    # ---------------------------
    print("\nBuilding link prediction dataset...")
    X, y = build_link_prediction_dataset(driver, user_embeddings, tweet_embeddings)
    
    # ---------------------------
    # Step 3: Train-test split
    # ---------------------------
    print("\nSplitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ---------------------------
    # Step 4: Train classifier
    # ---------------------------
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print(f"\nTraining samples: {len(y_train)}, Test samples: {len(y_test)}")
    print("\nFitting model...")
    clf.fit(X_train, y_train)
    
    # ---------------------------
    # Step 5: Predict and evaluate
    # ---------------------------
    
    # Evaluation on training set
    y_train_pred = clf.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Evaluation on test set
    y_test_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("\nMetrics confrontation:")
    print(f"\nTraining set - F1-score: {train_f1:.4f}, Accuracy: {train_acc:.4f}")
    print(f"\nTest set     - F1-score: {test_f1:.4f}, Accuracy: {test_acc:.4f}")
    print("\nEvaluating model...")
    y_pred = clf.predict(X_test)
    
    # Compute metrics
    metrics = {
        "F1-score": [f1_score(y_test, y_pred)],
        "Accuracy": [accuracy_score(y_test, y_pred)],
        "Precision": [precision_score(y_test, y_pred)],
        "Recall": [recall_score(y_test, y_pred)]
    }
    
    df_metrics = pd.DataFrame(metrics)
    print("\nEvaluation Metrics:\n")
    print(df_metrics.to_string(index=False))
    
    # Confusion Matrix
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    

    if cross_validation_folds > 1:
        print("\nPerforming {}-fold cross-validation...".format(cross_validation_folds))
        cv = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True, random_state=42)
        
        scoring = ['accuracy', 'f1', 'precision', 'recall']
        results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)
        
        # Print scores for each fold and their averages
        print("\nCross-validation results:")
        for metric in scoring:
            scores = results[f'test_{metric}']
            print(f"{metric.capitalize()} per fold: {scores}")
            print(f"{metric.capitalize()} medio: {scores.mean():.4f} (+/- {scores.std():.4f})")