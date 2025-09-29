import logging
import os
from neo4j import GraphDatabase

def get_neo4j_driver():
    port = os.getenv("NEO4J_PORT", "7687")  # Default: 7687 se non impostata
    uri = f"bolt://localhost:{port}"
    username = os.getenv("NEO4J_USERNAME", "neo4j")  # Default: neo4j
    password = os.getenv("NEO4J_PASSWORD", "password!")  # Default: password!
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver

def create_indexes(driver):
    with driver.session() as session:
        session.run("CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (t:Tweet) ON (t.tweet_id)")
    logging.info("Indici creati su User(user_id) e Tweet(tweet_id)")   
    
def serialize_path(path):
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
    Calcola l'embedding della proprietà 'text' per ogni nodo Tweet e lo salva come property 'text_embedding' nel nodo stesso.
    Args:
        driver: Neo4j driver
        model_name: nome del modello SentenceTransformer da usare
        text_property: nome della property testuale da embeddare
        embedding_property: nome della property dove salvare l'embedding
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    tweet_features = []

    # Estrai i testi dei tweet
    with driver.session() as session:
        tweets = session.run(f"MATCH (t:Tweet) RETURN t.tweet_id AS id, t.{text_property} AS text")
        for record in tweets:
            tweet_id = record["id"]
            text = record["text"] or ""
            emb = model.encode(text).tolist()
            tweet_features.append({"tweet_id": tweet_id, "embedding": emb})

    # Salva l'embedding nei nodi Tweet
    with driver.session() as session:
        for feat in tweet_features:
            session.run(
                f"MATCH (t:Tweet {{tweet_id: $id}}) SET t.{embedding_property} = $embedding",
                id=feat["tweet_id"], embedding=feat["embedding"]
            )
    print(f"Salvati gli embedding '{embedding_property}' per tutti i Tweet.")    
