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