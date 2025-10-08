import logging
import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


def get_neo4j_driver():
    """
    Create and return a Neo4j driver using environment variables for configuration.
    
    Args:
        None
    """
    port = os.getenv("NEO4J_PORT", "7687")  # Default: 7687 se non impostata
    uri = f"bolt://localhost:{port}"
    username = os.getenv("NEO4J_USERNAME", "neo4j")  # Default: neo4j
    password = os.getenv("NEO4J_PASSWORD", "password!")  # Default: password!
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver

def create_indexes(driver):
    """
    Create indexes on User(user_id) and Tweet(tweet_id) if they do not exist.
    
    Args:
        driver: Neo4j driver.
    """
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:Tweet) ON (t.tweet_id)")
    logging.info("Created indexes on User(user_id) and Tweet(tweet_id)")
    
def serialize_path(path):
    """
    Serialize a Neo4j path object into a JSON-serializable format.

    Args:
        path: Neo4j path object.
        
    Returns: 
        JSON-serializable representation of the path.
    """
    return {
        "nodes": [dict(node) | {"labels": list(node.labels)} for node in path.nodes],
        "relationships": [dict(rel) | {
            "type": rel.type,
            "start_node_id": rel.start_node.id,
            "end_node_id": rel.end_node.id
        } for rel in path.relationships]
}
    
def compute_and_save_tweet_embeddings(driver, model_name='all-MiniLM-L6-v2', text_property='text', embedding_property='text_embedding'):
    """
    Compute and save text embeddings for Tweet nodes using a SentenceTransformer model.
    
    Args:
        driver: Neo4j driver
        model_name: name of the SentenceTransformer model to use
        text_property: name of the text property to embed
        embedding_property: name of the property to save the embedding
    
    Returns: 
        None
    """

    model = SentenceTransformer(model_name)
    tweet_features = []

    # Extract tweet texts
    with driver.session() as session:
        tweets = session.run(f"MATCH (t:Tweet) RETURN t.tweet_id AS id, t.{text_property} AS text")
        for record in tweets:
            tweet_id = record["id"]
            text = record["text"] or ""
            emb = model.encode(text).tolist()
            tweet_features.append({"tweet_id": tweet_id, "embedding": emb})

    # Save embeddings to Tweet nodes
    with driver.session() as session:
        for feat in tweet_features:
            session.run(
                f"MATCH (t:Tweet {{tweet_id: $id}}) SET t.{embedding_property} = $embedding",
                id=feat["tweet_id"], embedding=feat["embedding"]
            )
    print(f"Saved embeddings '{embedding_property}' for all Tweets.")
