import numpy as np
from neo4j import GraphDatabase
import random


def generate_graphsage_embeddings(driver, graph_name: str, model_name: str, dim: int = 128, node_label: str = "User"):
    """
    Generate GraphSAGE embeddings for nodes of a given type in the Neo4j graph.

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
    # Scegli la property giusta in base al tipo di nodo
    if node_label == "User":
        feature_properties = ["pagerank"]
    elif node_label == "Tweet":
        feature_properties = ["text_embedding"]
    else:
        raise ValueError("node_label deve essere 'User' o 'Tweet'")

    with driver.session() as session:
        # Step 1: Train GraphSAGE model
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
        session.run(train_query, {"dim": dim})

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
        result = session.run(embed_query)

        # Step 3: Convert embeddings to numpy arrays
        import numpy as np
        embeddings = {}
        for record in result:
            node_id = record["node_id"]
            if node_id is not None:
                emb_vector = np.array(record["embedding"], dtype=np.float32)
                embeddings[node_id] = emb_vector

        return embeddings

def build_link_prediction_dataset(driver, user_embeddings, tweet_embeddings, negative_ratio=1.0, random_seed=42):
    """
    Build dataset for link prediction (User -> Tweet) using embeddings only.

    Args:
        driver (neo4j.Driver): Neo4j driver instance.
        user_embeddings (dict): {user_id: embedding vector}
        tweet_embeddings (dict): {tweet_id: embedding vector}
        negative_ratio (float): ratio of negative samples to positive samples
        random_seed (int): random seed for reproducibility

    Returns:
        X (np.ndarray): feature matrix (concatenated embeddings)
        y (np.ndarray): labels (1=RETWEET exists, 0=no RETWEET)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Step 1: Get positive pairs from Neo4j
    pos_query = """
    MATCH (u:User)-[:RETWEET]->(t:Tweet)
    RETURN u.user_id AS user_id, t.tweet_id AS tweet_id
    """
    with driver.session() as session:
        pos_result = session.run(pos_query)
        positive_pairs = [(record["user_id"], record["tweet_id"]) for record in pos_result]

    # Step 2: Generate negative pairs
    all_users = list(user_embeddings.keys())
    all_tweets = list(tweet_embeddings.keys())
    num_negatives = int(len(positive_pairs) * negative_ratio)
    negative_pairs = set()

    while len(negative_pairs) < num_negatives:
        u = random.choice(all_users)
        t = random.choice(all_tweets)
        if (u, t) not in positive_pairs:
            negative_pairs.add((u, t))
    negative_pairs = list(negative_pairs)

    # Step 3: Build feature vectors and labels
    X = []
    y = []

    for u, t in positive_pairs:
        if u in user_embeddings and t in tweet_embeddings:
            X.append(np.concatenate([user_embeddings[u], tweet_embeddings[t]]))
            y.append(1)

    for u, t in negative_pairs:
        if u in user_embeddings and t in tweet_embeddings:
            X.append(np.concatenate([user_embeddings[u], tweet_embeddings[t]]))
            y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y
